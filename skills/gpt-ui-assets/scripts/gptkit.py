#!/usr/bin/env python3
"""gptkit: design-to-assets pipeline on any OpenAI-compatible API — UI mockups, asset inventory, 1:1 from-scratch
redraws, SVG icons, Android VectorDrawables, animation frames, Lottie and app launcher icon sets. Run `gptkit.py -h`.

Config (environment variables win over the config file):
  GPTKIT_BASE_URL / OPENAI_BASE_URL   default https://api.openai.com/v1 (or a gateway that mirrors /images/* multipart
                                      and /responses with reasoning, web_search and strict json_schema, e.g. LiteLLM)
  GPTKIT_API_KEY  / OPENAI_API_KEY    required
  GPTKIT_IMAGE_MODEL                  default gpt-image-2
  GPTKIT_TEXT_MODEL                   default gpt-6 (vision + structured outputs; `doctor` lists the ids your endpoint
                                      offers, e.g. gpt-6-luna / gpt-6-sol / gpt-6.1-sol when the bare alias is not listed)
  GPTKIT_CONFIG                       config file path, default ~/.config/gptkit/env (KEY=value lines)
  GPTKIT_RETRIES                      retries per API call on 408/429/5xx and transport errors, default 2
  GPTKIT_DEBUG                        1: re-raise errors with the traceback instead of the one-line "gptkit <cmd>: ..."
  GPTKIT_MODERATION                   low | auto: `moderation` for image calls (dropped when a gateway rejects it)
  GPTKIT_CHROME                       Chrome/Chromium binary for SVG/Lottie/CSS rendering (none disables)
  GPTKIT_CHROME_NO_SANDBOX            1: pass --no-sandbox to headless Chrome (added automatically when running as root)
  GPTKIT_EBSYNTH                      ebsynth binary for `sequence propagate`
"""
import argparse
import base64
import concurrent.futures as cf
import functools
import hashlib
import http.server
import io
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from datetime import date, datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import numpy as np
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat

SVG_NS = "{http://www.w3.org/2000/svg}"
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

PRICES_CHECKED = "2026-10-18"
PRICES = {
    "gpt-image-2.5-flare": {"text_in": 5, "image_in": 8, "image_out": 30},
    "gpt-image-2.5-sunburst": {"text_in": 5, "image_in": 8, "image_out": 30},
    "gpt-image-2": {"text_in": 5, "image_in": 8, "image_out": 30},
    "gpt-image-1.5": {"text_in": 5, "image_in": 8, "image_out": 32, "text_out": 10},
    "gpt-6-sol": {"in": 2, "out": 10}, "gpt-6.1-sol": {"in": 2, "out": 10},
    "gpt-6-luna": {"in": 0.1, "out": 0.5}, "web_search": {"per_call": 0.01},
}
_LEDGER_LOCK = threading.Lock()
_LEDGER_WARNED = False
_LEDGER_CMD, _LEDGER_DIR = "", None


def _ledger_record(endpoint, model, body, *, image=False, quality=None, size=None):
    """Append parsed successful usage; a failed local ledger must never fail a paid command."""
    global _LEDGER_WARNED
    path = os.environ.get("GPTKIT_LEDGER")
    if path == "off":
        return
    path = Path(path).expanduser() if path else Path.home() / ".cache/gptkit/ledger.jsonl"
    usage = body.get("usage") if isinstance(body, dict) else None
    estimated = not isinstance(usage, dict)
    unknown = model not in PRICES
    prices = PRICES.get(model, PRICES["gpt-image-2" if image else "gpt-6-sol"])
    cost = 0.0
    if image:
        if estimated:
            quality = quality or "high"
            output = {"low": 196, "medium": 439, "high": 1756}.get(quality, 1756)
            try:
                w, h = [int(x) for x in (size or "1024x1024").split("x")]
                output *= w * h / (1024 * 1024)
            except (ValueError, AttributeError):
                pass
            cost = output * prices.get("image_out", 30) / 1_000_000
        else:
            details = usage.get("input_tokens_details") or {}
            cost += (details.get("text_tokens") or 0) * prices.get("text_in", 5) / 1_000_000
            cost += (details.get("image_tokens") or 0) * prices.get("image_in", 8) / 1_000_000
            cost += (usage.get("output_tokens") or 0) * prices.get("image_out", 30) / 1_000_000
    else:
        if estimated:
            cost = (1000 * prices.get("in", 2) + 1000 * prices.get("out", 10)) / 1_000_000
        else:
            cost = ((usage.get("input_tokens") or 0) * prices.get("in", 2) +
                    (usage.get("output_tokens") or 0) * prices.get("out", 10)) / 1_000_000
        if model in ("gpt-image-1.5",):
            cost = ((usage or {}).get("input_tokens", 0) * prices.get("text_in", 5) +
                    (usage or {}).get("output_tokens", 0) * prices.get("text_out", 10)) / 1_000_000
    outputs = body.get("output") or [] if isinstance(body, dict) else []
    web_calls = sum(1 for item in outputs if isinstance(item, dict) and item.get("type") == "web_search_call")
    if not image:
        cost += web_calls * PRICES["web_search"]["per_call"]
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "cmd": _LEDGER_CMD or "unknown",
             "dir": str(Path(_LEDGER_DIR).expanduser().resolve()) if _LEDGER_DIR else None,
             "endpoint": endpoint, "model": model, "quality": quality if image else None,
             "size": size if image else None, "usage": usage, "web_search_calls": web_calls,
             "est_usd": round(cost, 8), "estimated": bool(estimated or unknown)}
    try:
        with _LEDGER_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError as e:
        with _LEDGER_LOCK:
            warn = not _LEDGER_WARNED
            _LEDGER_WARNED = True
        if warn:
            print(f"gptkit: couldn't write ledger {path}: {e}", file=sys.stderr)


def _plan(image_calls=0, text_calls=0, model=None, *, image_model=None, task="concept", background="auto",
          size="1024x1024", edits=False, tokens_in=20000, tokens_out=5000, web_calls=0, dry_run=False, what=None):
    """Print (and, with GPTKIT_BUDGET_USD, enforce) a list-price estimate before a command starts paid work.
    An image call is priced as high-quality output scaled by area, plus ~1,500 image-input tokens when it sends a
    reference (edits) and ~300 prompt tokens; rough by design, it is meant to catch a 10x surprise, not to bill."""
    if not image_calls and not text_calls:
        return 0.0
    text_model = model or _config()["text_model"]
    if image_calls:
        image_model = pick_image_model(image_model, task, background)
    ip = PRICES.get(image_model, PRICES["gpt-image-2"])
    tp = PRICES.get(text_model, PRICES["gpt-6-sol"])
    try:
        w, h = [int(x) for x in str(size).split("x")]
    except ValueError:
        w, h = 1024, 1024
    per_image = (1756 * w * h / 1024 ** 2 * ip.get("image_out", 30) + 300 * ip.get("text_in", 5)
                 + (1500 * ip.get("image_in", 8) if edits else 0)) / 1e6
    estimate = image_calls * per_image + text_calls * (tokens_in * tp.get("in", 2) + tokens_out * tp.get("out", 10)) / 1e6
    estimate += web_calls * PRICES["web_search"]["per_call"]
    parts = ([f"~{image_calls} image call{'s' if image_calls != 1 else ''} ({image_model})"] if image_calls else []) + \
            ([f"{text_calls} text call{'s' if text_calls != 1 else ''} ({text_model})"] if text_calls else [])
    detail = " + ".join(parts) + f" ≈ ${estimate:.2f} list price"
    budget = os.environ.get("GPTKIT_BUDGET_USD")
    if budget is not None:
        try:
            limit = float(budget)
        except ValueError:
            sys.exit("gptkit: GPTKIT_BUDGET_USD must be a number")
        if estimate > limit:
            sys.exit(f"gptkit: plan {detail} exceeds GPTKIT_BUDGET_USD={limit:.2f}; raise GPTKIT_BUDGET_USD or reduce "
                     "--candidates/--variants/--only (nothing was called)")
    print("plan: " + detail)
    return estimate


# ---------------------------------------------------------------- shared helpers (plan contract C1; owner g1-api-core)
# Verbatim block: every group that calls one of these inserts the WHOLE block, unchanged, at this anchor.

CHROME_PAD = 200  # --headless=new lays out ~87 px less than --window-size's height: pad, then crop to (0, 0, w, h)

_JAVA_KEYWORDS = frozenset("abstract assert boolean break byte case catch char class const continue default do double else "
                           "enum extends final finally float for goto if implements import instanceof int interface long "
                           "native new package private protected public return short static strictfp super switch "
                           "synchronized this throw throws transient try void volatile while true false null".split())


def _chrome_flags(w, h):
    """Headless-Chrome argv shared by every capture (no executable, url, --screenshot or --virtual-time-budget).
    The window is padded by CHROME_PAD; callers crop the PNG to (0, 0, w, h)."""
    flags = ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
             "--default-background-color=00000000", "--use-mock-keychain", f"--window-size={w},{h + CHROME_PAD}"]
    if os.environ.get("GPTKIT_CHROME_NO_SANDBOX") == "1" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        flags.append("--no-sandbox")
    return flags


def _chrome_err(r):
    """The most useful stderr line of a finished Chrome CompletedProcess (str or bytes stderr), <= 200 chars."""
    err = r.stderr if isinstance(r.stderr, str) else (r.stderr or b"").decode("utf8", "replace")
    lines = [l.strip() for l in err.splitlines() if l.strip()]
    hit = next((l for l in reversed(lines) if "ERROR" in l or "sandbox" in l.lower()), lines[-1] if lines else "")
    return re.sub(r"^\[\d+:\d+:\d{4}/\d+\.\d+:", "[", hit)[:200] or f"exit code {r.returncode}, no stderr"


@functools.cache
def _chrome_probe():
    """(True, '') when headless Chrome can actually render here, else (False, reason). One 64x64 shot, cached."""
    exe = _chrome()
    if not exe:
        return False, "no Chrome/Chromium found (set GPTKIT_CHROME)"
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "p.png"
        try:
            r = subprocess.run([exe, *_chrome_flags(64, 64), f"--user-data-dir={td}/prof", "--virtual-time-budget=500",
                                f"--screenshot={png}", "data:text/html,<body style=background:red>"],
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"{exe}: {type(e).__name__}"
        if not png.exists():
            return False, f"{exe}: {_chrome_err(r)}"
    return True, ""


def _pos_int(s):
    """argparse type: integer >= 1."""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {s!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {v}")
    return v


def _pos_float(s):
    """argparse type: finite number > 0."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {s!r}")
    if not v > 0 or math.isinf(v):
        raise argparse.ArgumentTypeError(f"expected a positive number, got {s!r}")
    return v


def _unit_float(s):
    """argparse type: number in 0..1."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number between 0 and 1, got {s!r}")
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError(f"expected a number between 0 and 1, got {s!r}")
    return v


def _res_name(name, prefix="img_"):
    """Android file-based resource name: [a-z0-9_], starts with a letter, not a Java keyword (AGP rejects the rest)."""
    n = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", str(name).lower())).strip("_")
    if not n or not n[0].isalpha() or n in _JAVA_KEYWORDS:
        n = prefix + n
    return n


def _need(path, what, hint=""):
    """Path(path) when it exists, else a one-line exit naming what is missing and how to get it."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"gptkit: {what} not found: {p}" + (f" ({hint})" if hint else ""))
    return p


def _open_image(path, what="image", hint=""):
    """_rgba(path) with a one-line exit for a missing or undecodable file."""
    p = _need(path, what, hint)
    try:
        return _rgba(p)
    except (OSError, ValueError) as e:  # PIL.UnidentifiedImageError is an OSError
        sys.exit(f"gptkit: cannot read {what} {p}: {e}")


def _read_json(path, what="file", required=()):
    """json.loads(path) with one-line exits for a missing or corrupt file and for missing top-level keys."""
    p = _need(path, what)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        sys.exit(f"gptkit: {what} {p} is not valid JSON ({e})")
    missing = [k for k in required if not isinstance(data, dict) or k not in data]
    if missing:
        sys.exit(f"gptkit: {what} {p} is missing {', '.join(missing)} (written by an older version, or hand-edited?)")
    return data


def _clip_bbox(b, w, h):
    """{x,y,w,h} as ints clamped to a w x h image; None when the box is unreadable or has no area inside the image."""
    try:
        x0, y0 = int(round(float(b["x"]))), int(round(float(b["y"])))
        x1, y1 = x0 + int(round(float(b["w"]))), y0 + int(round(float(b["h"])))
    except (KeyError, TypeError, ValueError, OverflowError):  # OverflowError: inf (json.loads('1e400'))
        return None
    x0, y0, x1, y1 = max(0, min(x0, x1)), max(0, min(y0, y1)), min(w, max(x0, x1)), min(h, max(y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


# ---------------------------------------------------------------- config + API

_TRUE = ("1", "true", "yes", "on")


@functools.cache
def _config():
    cfg = {}
    path = Path(os.environ.get("GPTKIT_CONFIG", "~/.config/gptkit/env")).expanduser()
    if path.is_dir():
        sys.exit(f"gptkit: GPTKIT_CONFIG points at a directory, not a file: {path} (did you mean {path / 'env'}?)")
    if path.exists():
        try:
            text = path.read_text()
        except (OSError, ValueError) as e:
            sys.exit(f"gptkit: can't read config file {path}: {e}")
        for line in text.splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                cfg[k.strip()] = v.strip().strip('"').strip("'")

    def pick(*names, default=None):
        for n in names:
            if os.environ.get(n):
                return os.environ[n]
        for n in names:
            if cfg.get(n):
                return cfg[n]
        return default
    retries = pick("GPTKIT_RETRIES", default="2")
    if not (retries.isdigit() and int(retries) >= 0):
        sys.exit(f"gptkit: GPTKIT_RETRIES must be an integer >= 0, got {retries!r}")
    moderation = pick("GPTKIT_MODERATION")
    if moderation and moderation not in ("low", "auto"):
        sys.exit(f"gptkit: GPTKIT_MODERATION must be low or auto, got {moderation!r}")
    return {"base": pick("GPTKIT_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1").rstrip("/"),
            "key": pick("GPTKIT_API_KEY", "OPENAI_API_KEY"),
            "image_model": pick("GPTKIT_IMAGE_MODEL", default="gpt-image-2"),
            "image_model_pinned": bool(pick("GPTKIT_IMAGE_MODEL")),
            "text_model": pick("GPTKIT_TEXT_MODEL", default="gpt-6"),
            "retries": int(retries), "debug": pick("GPTKIT_DEBUG", default="").lower() in _TRUE,
            "moderation": moderation, "config_path": str(path)}


def _auth():
    c = _config()
    if not c["key"]:
        sys.exit("gptkit: no API key. Set GPTKIT_API_KEY (or OPENAI_API_KEY), or add GPTKIT_API_KEY=... to "
                 f"{c['config_path']}")
    return c["base"], {"Authorization": f"Bearer {c['key']}"}


def _redact(s):
    """Server text with any bearer token (and the configured key) blanked, on one line."""
    s = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", str(s))
    info = getattr(_config, "cache_info", None)
    key = _config()["key"] if info and info().currsize else None
    if key and len(key) > 8:
        s = s.replace(key, "[redacted]")
    return " ".join(s.split())


_NOTED = set()


def _note_once(key, msg):
    """Print a stderr note the first time `key` is seen in this process."""
    if key not in _NOTED:
        _NOTED.add(key)
        print(msg, file=sys.stderr)


class ApiError(RuntimeError):
    """One failed API call. status: HTTP status (0 = transport failure or exhausted retries); code: error.code from
    the JSON body or a kit code (refusal, incomplete, empty, not_json, no_image, bad_image, exhausted); etype, param:
    error.type / error.param; body: the parsed JSON body or None; prompt: first 200 chars of the prompt that failed."""

    def __init__(self, status, msg, *, code=None, etype=None, param=None, body=None, prompt=None):
        super().__init__(msg)
        self.status, self.code, self.etype, self.param, self.body, self.prompt = status, code, etype, param, body, prompt

    def __str__(self):
        return _redact(self.args[0] if self.args else "") + (f" | prompt: {self.prompt[:200]!r}" if self.prompt else "")


RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


def _error_fields(r):
    """(code, etype, param, message, body) of an error response; message falls back to the raw text."""
    try:
        body = r.json()
    except ValueError:
        body = None
    err = body.get("error") if isinstance(body, dict) else None
    err = {"message": err} if isinstance(err, str) else err if isinstance(err, dict) else {}
    msg = err.get("message") or (getattr(r, "text", "") or "")[:400] or f"HTTP {r.status_code}"
    fix = lambda v: str(v) if v not in (None, "") else None
    return fix(err.get("code")), fix(err.get("type")), fix(err.get("param")), str(msg)[:400], body


def _retry_delay(r, attempt):
    """Retry-After (s) or retry-after-ms when present (capped at 300 s), else exponential backoff with jitter."""
    import random
    h = getattr(r, "headers", None) or {}
    for name, div in (("Retry-After", 1.0), ("retry-after-ms", 1000.0)):
        v = h.get(name)
        if v is not None:
            try:
                return min(300.0, max(0.0, float(v) / div))
            except ValueError:
                pass
    return min(60, 4 * 2 ** attempt) + random.random()


def _call(path, *, json_body=None, files=None, data=None, timeout=900, retries=None):
    """POST to the API; returns the parsed JSON body. Raises only ApiError. Retries 408/429/5xx and transport errors
    (not 429 insufficient_quota/billing_*), honouring Retry-After; retries defaults to GPTKIT_RETRIES (2)."""
    base, headers = _auth()
    if retries is None:
        retries = _config()["retries"]
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(base + path, headers=headers, json=json_body, files=files, data=data, timeout=timeout)
        except requests.RequestException as e:
            last = ApiError(0, f"{path} [exhausted]: {type(e).__name__}: {e}", code="exhausted")
            if attempt < retries:
                delay = min(60, 4 * 2 ** attempt)
                print(f"  ({path}: {type(e).__name__}, retry {attempt + 1}/{retries} in {delay:.0f} s)", file=sys.stderr)
                time.sleep(delay)
            continue
        if r.status_code < 400:
            try:
                return r.json()
            except ValueError:
                raise ApiError(r.status_code, f"HTTP {r.status_code} {path} [not_json]: the response body is not JSON: "
                               f"{(getattr(r, 'text', '') or '')[:200]!r}", code="not_json")
        code, etype, param, msg, body = _error_fields(r)
        msg = _redact(msg)
        if code == "model_not_found" or param == "model" or (r.status_code == 404 and "model" in msg.lower()):
            msg += " (set GPTKIT_TEXT_MODEL / GPTKIT_IMAGE_MODEL or --model to an id `doctor` lists, e.g. gpt-6-luna)"
        retryable = r.status_code in RETRY_STATUSES and not (code == "insufficient_quota" or str(code).startswith("billing"))
        if not retryable:
            raise ApiError(r.status_code, f"HTTP {r.status_code} {path} [{code or etype or 'error'}]: {msg}",
                           code=code, etype=etype, param=param, body=body)
        last = ApiError(r.status_code, f"HTTP {r.status_code} {path} [exhausted]: {code or etype or 'error'}: {msg} "
                        f"(after {attempt + 1} attempt{'s' if attempt else ''})", code="exhausted", etype=etype, param=param, body=body)
        if attempt < retries:
            delay = _retry_delay(r, attempt)
            print(f"  (HTTP {r.status_code} on {path}, retry {attempt + 1}/{retries} in {delay:.1f} s)", file=sys.stderr)
            time.sleep(delay)
    raise last


OPTIONAL_PARAMS = ("input_fidelity", "quality", "output_format", "background", "moderation")
RESPONSES_OPTIONAL = ("reasoning", "tools", "store")
_DROPPED = set()


def _with_fallback(send, params, optional=OPTIONAL_PARAMS):
    """Retry without an optional param the provider rejects with a 400 (e.g. input_fidelity on a model that lacks
    it). error.param decides; the message text is searched only when the body names no param and the code is a
    parameter-style one (never for moderation_blocked and other coded non-parameter errors)."""
    params = dict(params)
    while True:
        try:
            return send(params)
        except ApiError as e:
            if e.status != 400:
                raise
            if e.param:
                root = re.split(r"[.\[]", e.param, 1)[0]
                dropped = root if root in params and root in optional else None
            elif e.code is None or "param" in e.code or "value" in e.code or e.code == "invalid_request_error":
                err = (e.body or {}).get("error") if isinstance(e.body, dict) else None
                text = err.get("message", "") if isinstance(err, dict) else str(e)
                dropped = next((p for p in optional if p in params and p in text), None)
            else:  # a coded non-parameter error (moderation_blocked, ...) whose prose happens to name a param
                dropped = None
            if not dropped:
                raise
            if dropped not in _DROPPED:
                _DROPPED.add(dropped)
                print(f"  (provider rejected '{dropped}', retrying without it)", file=sys.stderr)
            params.pop(dropped)


def _size_for(w, h):
    if w > h * 1.2:
        return "1536x1024"
    if h > w * 1.2:
        return "1024x1536"
    return "1024x1024"


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _image_result(resp):
    """(png_bytes, meta) from an images response. The bytes always decode as PNG (JPEG/WebP from a gateway is
    re-encoded and meta['source_format'] says so); no/undecodable image raises ApiError no_image/bad_image."""
    import binascii
    resp = resp if isinstance(resp, dict) else {}
    meta = {k: resp.get(k) for k in ("quality", "size", "background")}
    meta["usage"] = resp.get("usage")
    data = resp.get("data") or []
    item = data[0] if data and isinstance(data[0], dict) else None
    if not item:
        raise ApiError(200, "the image API returned no image (data: []): the provider may have filtered the request; "
                            "check the prompt for policy-sensitive content", code="no_image", body=resp)
    if item.get("b64_json"):
        try:
            raw = base64.b64decode(item["b64_json"], validate=True)
        except (binascii.Error, ValueError) as e:
            raise ApiError(200, f"the image API returned unreadable base64 ({e})", code="bad_image")
    elif item.get("url"):
        try:
            r = requests.get(item["url"], timeout=120)
            r.raise_for_status()
            raw = r.content
        except requests.RequestException as e:
            raise ApiError(0, f"could not fetch the image URL the API returned: {e}", code="no_image")
    else:
        raise ApiError(200, "the image API item has neither b64_json nor url", code="no_image", body=resp)
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (OSError, ValueError) as e:
        raise ApiError(200, f"the image API returned bytes that are not an image (starts {raw[:16]!r}): {e}", code="bad_image")
    meta["source_format"] = im.format
    if im.format != "PNG":
        _note_once("source_format", f"  (provider returned {im.format}: no alpha channel, background/output_format were ignored)")
        raw = _png_bytes(im if im.mode in ("RGB", "RGBA", "L", "LA", "P") else im.convert("RGBA"))
    return raw, meta


# ---- image model routing: the best listed model for the job unless the user pinned one
# concept: new art from a prompt (mockups, style, icon concepts): speed matters, Flare matches gpt-image-2 quality at ~half the latency.
# precise: edits that must keep a reference intact (forge/fix, sequence and frame renders, icon layers): Sunburst is tuned for that.
# gpt-image-2 only takes background auto/opaque, so it moves to the back of the chain when transparency is needed.
IMAGE_ROUTES = {
    "concept": ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2", "gpt-image-1.5"),
    "precise": ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare", "gpt-image-2", "gpt-image-1.5"),
}
RETIREMENTS = {
    "gpt-image-1": ("2026-10-23", "gpt-image-2.5-sunburst or gpt-image-2.5-flare"),
    "gpt-image-1-mini": ("2026-12-01", "gpt-image-2.5-sunburst or gpt-image-2.5-flare"),
    "gpt-image-1.5": ("2026-12-01", "gpt-image-2.5-sunburst or gpt-image-2.5-flare"),
    "chatgpt-image-latest": ("2026-12-01", "gpt-image-2.5-sunburst or gpt-image-2.5-flare"),
    "gpt-5.1": ("2027-04-01", "gpt-6-sol"),
    "gpt-5.3-codex": ("2027-04-01", "gpt-6-sol"),
    "gpt-5.4-nano": ("2027-04-01", "gpt-6-luna"),
    "gpt-5-2025-08-07": ("2026-12-11", "gpt-6-sol"),
    "gpt-5-mini-2025-08-07": ("2026-12-11", "gpt-6-luna"),
    "gpt-5-nano-2025-08-07": ("2026-12-11", "gpt-6-luna"),
    "gpt-5-pro-2025-10-06": ("2026-12-11", "gpt-6-sol"),
}
NO_TRANSPARENCY = ("gpt-image-2",)


def _today():
    """Current date, isolated so retirement messaging can be tested deterministically."""
    return time.strftime("%Y-%m-%d")


def _warn_retired_model(model):
    retirement = RETIREMENTS.get(model)
    if retirement and _today() > retirement[0]:
        _note_once(("retired-model", model), f"  (model {model} retired on {retirement[0]}; replacement: {retirement[1]})")


def image_chain(task, background="auto"):
    chain = IMAGE_ROUTES.get(task, IMAGE_ROUTES["concept"])
    if background == "transparent":
        chain = tuple(m for m in chain if m not in NO_TRANSPARENCY) + tuple(m for m in chain if m in NO_TRANSPARENCY)
    return chain


@functools.lru_cache(maxsize=None)
def _listed_models(base, key):
    try:
        r = requests.get(base + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
        r.raise_for_status()
        return frozenset(m["id"] for m in r.json().get("data", []))
    except Exception:
        return None


def pick_image_model(model=None, task="concept", background="auto"):
    """--model wins, then a pinned GPTKIT_IMAGE_MODEL, then the first model of the task's chain the endpoint lists,
    then the configured default (also used when the endpoint's model list cannot be read)."""
    c = _config()
    if model:
        _warn_retired_model(model)
        return model
    if c["image_model_pinned"] or not c["key"]:
        _warn_retired_model(c["image_model"])
        return c["image_model"]
    ids = _listed_models(c["base"], c["key"])
    if ids:
        for m in image_chain(task, background):
            if m in ids:
                _note_once(("image-model", task, background, m), f"  (image model {m}: auto-picked for {task}"
                           f"{', transparent' if background == 'transparent' else ''}; pin one with --model or GPTKIT_IMAGE_MODEL)")
                _warn_retired_model(m)
                return m
    _warn_retired_model(c["image_model"])
    return c["image_model"]


def generate(prompt, size="1024x1024", background="auto", model=None, task="concept"):
    c = _config()
    body = {"model": pick_image_model(model, task, background), "prompt": prompt, "size": size, "output_format": "png", "quality": "high"}
    if background != "auto":
        body["background"] = background
    if c["moderation"]:
        body["moderation"] = c["moderation"]
    try:
        resp = _with_fallback(lambda b: _call("/images/generations", json_body=b), body)
        result = _image_result(resp)
        _ledger_record("/images/generations", body["model"], resp, image=True, quality=result[1].get("quality") or body["quality"],
                       size=result[1].get("size") or size)
        return result
    except ApiError as e:
        e.prompt = e.prompt or prompt[:200]
        raise


def edit(ref_png: bytes, prompt, size="1024x1024", background="auto", model=None, fidelity="high", task="precise"):
    # One reference image per call: multi-image edits returned the reference itself on the sub2api gateway the kit
    # was probed against (the official API accepts up to 16; unverified here, see references/api-findings.md).
    c = _config()
    data = {"model": pick_image_model(model, task, background), "prompt": prompt, "size": size, "output_format": "png",
            "quality": "high", "input_fidelity": fidelity}
    if background != "auto":
        data["background"] = background
    if c["moderation"]:
        data["moderation"] = c["moderation"]
    files = [("image", ("ref.png", ref_png, "image/png"))]
    try:
        resp = _with_fallback(lambda d: _call("/images/edits", data=d, files=files), data)
        result = _image_result(resp)
        _ledger_record("/images/edits", data["model"], resp, image=True, quality=result[1].get("quality") or data["quality"],
                       size=result[1].get("size") or size)
        return result
    except ApiError as e:
        e.prompt = e.prompt or prompt[:200]
        raise


def _data_url(img_or_bytes):
    raw = img_or_bytes if isinstance(img_or_bytes, bytes) else _png_bytes(img_or_bytes)
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def _responses(body, timeout=900):
    """POST /responses with the optional fields dropped when a gateway rejects them; a 400 on text.format retries once
    without strict (callers check required keys locally)."""
    def send(b):
        try:
            return _call("/responses", json_body=b, timeout=timeout)
        except ApiError as e:
            fmt = (b.get("text") or {}).get("format") or {}
            if e.status == 400 and fmt.get("strict") and (e.param or "").startswith("text.format"):
                _note_once("strict", "  (provider rejected the strict json_schema format, retrying with strict: false; "
                                     "required keys are checked locally)")
                b["text"] = {**b["text"], "format": {**fmt, "strict": False}}
                return _call("/responses", json_body=b, timeout=timeout)
            raise
    resp = _with_fallback(send, body, RESPONSES_OPTIONAL)
    _ledger_record("/responses", body.get("model") or _config()["text_model"], resp)
    return resp


def _responses_text(resp, schema=None):
    """(text or parsed JSON, [url citations]) of a /responses body. A refusal, an incomplete response, an empty output
    or non-JSON text in schema mode raise ApiError (codes refusal / incomplete / empty / not_json)."""
    text, cites, refusal = "", [], None
    resp = resp if isinstance(resp, dict) else {}  # a list/str body from a gateway is "empty", not an AttributeError
    for o in resp.get("output") or []:
        if o.get("type") == "message":
            for c in o.get("content") or []:
                if c.get("type") == "refusal":
                    refusal = c.get("refusal") or "(no reason given)"
                    continue
                text += c.get("text") or ""
                for x in c.get("annotations") or []:
                    if x.get("type") == "url_citation":
                        cite = {"title": x.get("title", ""), "url": x.get("url", "")}
                        cite.update({k: x[k] for k in ("start_index", "end_index") if k in x})
                        cites.append(cite)
    if resp.get("status") == "incomplete":
        why = (resp.get("incomplete_details") or {}).get("reason") or "unknown"
        hint = ": lower --effort or shorten the prompt" if why == "max_output_tokens" else ""
        raise ApiError(200, f"/responses: incomplete response ({why}{hint})", code="incomplete", body=resp)
    if refusal and not text.strip():
        raise ApiError(200, f"/responses: the model refused: {refusal[:300]}", code="refusal", body=resp)
    if not text.strip():
        raise ApiError(200, "/responses: empty response (no message text, refusal or incomplete marker: unexpected "
                            "gateway shape)", code="empty", body=resp)
    if schema:
        try:
            return json.loads(text), cites
        except ValueError as e:
            raise ApiError(200, f"/responses: the model's text is not JSON ({e}): {text[:120]!r} (the endpoint may not "
                                "honour json_schema; try --model gpt-6-sol or another gateway)", code="not_json", body=resp)
    return text, cites


def ask(parts, schema=None, name="result", model=None):
    """parts: list of str | PIL.Image | bytes. Returns parsed JSON when a schema is given, else text.
    schema="json" (schema-less json_object mode) is deprecated: pass a real JSON schema."""
    content = []
    for p in parts:
        if isinstance(p, str):
            content.append({"type": "input_text", "text": p})
        else:
            content.append({"type": "input_image", "image_url": _data_url(p)})
    chosen_model = model or _config()["text_model"]
    _warn_retired_model(chosen_model)
    body = {"model": chosen_model, "store": False, "input": [{"role": "user", "content": content}]}
    if schema == "json":
        body["text"] = {"format": {"type": "json_object"}}
    elif schema:
        body["text"] = {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}
    try:
        return _responses_text(_responses(body), schema)[0]
    except ApiError as e:
        e.prompt = e.prompt or next((p for p in parts if isinstance(p, str)), "")[:200]
        raise


# ---------------------------------------------------------------- image utils

def _rgba(p):
    """Open as RGBA: EXIF orientation applied, 16-bit greyscale (I;16*, I) scaled to 8 bits first (plan contract C5)."""
    from PIL import ImageOps
    im = ImageOps.exif_transpose(Image.open(p))
    if im.mode == "I" or im.mode.startswith("I;16"):
        a = np.asarray(im).astype(np.float64)
        if a.size and a.max() > 255:
            a = a / 257.0  # 16-bit range -> 8-bit (65535/257 = 255)
        im = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8), "L")
    return im.convert("RGBA")


def content_box(img, thr=24):
    return img.getchannel("A").point(lambda v: 255 if v > thr else 0).getbbox()


def trim(img, thr=8, pad=0.0):
    bb = content_box(img, thr) or (0, 0, img.width, img.height)
    c = img.crop(bb)
    if pad:
        p = int(max(c.size) * pad)
        out = Image.new("RGBA", (c.width + 2 * p, c.height + 2 * p), (0, 0, 0, 0))
        out.alpha_composite(c, (p, p))
        return out
    return c


def ref_crop(mock, bbox, margin=0.6, long_side=1024, center=False):
    """The element with `margin` context on each side, long side resized to long_side. center=True pads the crop
    with the design's backdrop colour where the mockup edge cuts the margin, so the element is always centred."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    full = (int(x - w * margin), int(y - h * margin), int(x + w * (1 + margin)), int(y + h * (1 + margin)))
    box = (max(0, full[0]), max(0, full[1]), min(mock.width, full[2]), min(mock.height, full[3]))
    c = mock.crop(box).convert("RGB")
    if center and box != full:
        padded = Image.new("RGB", (full[2] - full[0], full[3] - full[1]), _backdrop(mock, bbox))
        padded.paste(c, (box[0] - full[0], box[1] - full[1]))
        c = padded
    s = long_side / max(c.size)
    return c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)


def checker(w, h, c=16):
    im = Image.new("RGB", (w, h), "#bdbdbd")
    d = ImageDraw.Draw(im)
    for y in range(0, h, c):
        for x in range(0, w, c):
            if (x // c + y // c) % 2:
                d.rectangle([x, y, x + c - 1, y + c - 1], fill="#ececec")
    return im


def _tile(img, T, label=None, bg="checker"):
    im = img.convert("RGBA").copy() if img is not None else Image.new("RGBA", (1, 1))
    im.thumbnail((T, T))
    base = checker(T, T) if bg == "checker" else Image.new("RGB", (T, T), bg)
    base.paste(im, ((T - im.width) // 2, (T - im.height) // 2), im)
    if label:
        d = ImageDraw.Draw(base)
        lines, rest = [], str(label)
        while rest and len(lines) < 2:
            cut = len(rest)
            while cut > 1 and d.textlength(rest[:cut]) > T - 6:
                cut = rest.rfind(" ", 0, cut) if " " in rest[:cut - 1] else cut - 1
            lines.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        d.rectangle([0, 0, T, 2 + 12 * len(lines)], fill="white")
        for i, ln in enumerate(lines):
            d.text((3, 1 + 12 * i), ln, fill="black")
    return base


def sheet(rows, out, T=220):
    """rows: list of lists of (label, PIL image)."""
    cols = max(len(r) for r in rows)
    im = Image.new("RGB", (cols * T, len(rows) * T), "white")
    for r, row in enumerate(rows):
        for c, (label, img) in enumerate(row):
            im.paste(_tile(img, T, label), (c * T, r * T))
    im.save(out)
    return out


# ---------------------------------------------------------------- scoring

def _mean_diff(a, b):
    return sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3


def _edge_var(img):
    return ImageStat.Stat(img.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]


_M_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])
WORK = 128  # long side of the working view used by the perceptual metrics


def _lab(rgb):
    c = np.asarray(rgb, np.float64)[..., :3] / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M_XYZ.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    c = np.clip(xyz @ np.linalg.inv(_M_XYZ).T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def _morph(mask, *ops):
    im = Image.fromarray(np.uint8(mask) * 255)
    for op in ops:
        im = im.filter(ImageFilter.MinFilter(3) if op == "min" else ImageFilter.MaxFilter(3))
    return np.asarray(im) > 127


def _clean(mask):
    return _morph(mask, "min", "max", "max", "min")


def _backdrop(mock, bbox, k=4):
    """The design's backdrop colour: median of a k-px ring OUTSIDE the bbox (clamped to the mockup). Only when the
    bbox is the whole mockup does the box's own edge serve, so tight or tiny bboxes never sample the element itself."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    M = mock.convert("RGB")
    box = (max(0, x - k), max(0, y - k), min(M.width, x + w + k), min(M.height, y + h + k))
    a = np.asarray(M.crop(box)).astype(int)
    inner = np.zeros(a.shape[:2], bool)
    inner[y - box[1]:y - box[1] + h, x - box[0]:x - box[0] + w] = True
    ring = a[~inner] if (~inner).any() else np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(ring, 0))


def _backdrop_edge_frac(asset, backdrop, thr=36):
    """Fraction of the asset's silhouette boundary (alpha > 200 minus a 5 px erosion at 128 px) within thr RGB of
    the backdrop. A kept card/circle puts backdrop colour on the outline; a light icon with its own outline does not."""
    g = asset.crop(content_box(asset, 40) or (0, 0, asset.width, asset.height))
    g.thumbnail((WORK, WORK))
    a = np.asarray(g).astype(int)
    solid = a[..., 3] > 200
    eroded = np.asarray(Image.fromarray(np.uint8(solid) * 255).filter(ImageFilter.MinFilter(5))) > 127
    edge = solid & ~eroded
    if edge.sum() < 8:
        return 0.0
    near = np.abs(a[..., :3] - np.array(backdrop)).sum(-1) < thr
    return float((near & edge).sum() / edge.sum())


def _design_view(mock, bbox):
    """The design element at WORK scale in Lab, its backdrop and a foreground mask segmented from it."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    s = WORK / max(w, h)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    arr = np.asarray(mock.crop((x, y, x + w, y + h)).convert("RGB").resize(size, Image.LANCZOS))
    backdrop = _backdrop(mock, bbox)
    L, bl = _lab(arr), _lab(np.array(backdrop))
    dist = np.linalg.norm(L - bl, axis=-1)
    thr = min(22.0, max(8.0, 0.45 * float(np.percentile(dist, 95))))
    return {"L": L, "mask": _clean(dist > thr), "backdrop": backdrop, "bl": bl, "thr": thr, "size": size, "up": s}


def _align_view(asset, view):
    """Scale/offset of the asset (composited on the design backdrop) that best overlaps the design silhouette."""
    w, h = view["size"]
    M = max(w, h) // 5
    g = asset.crop(content_box(asset, 40) or (0, 0, asset.width, asset.height))
    if max(g.size) > 4 * WORK:
        k = 4 * WORK / max(g.size)
        g = g.resize((max(1, round(g.width * k)), max(1, round(g.height * k))), Image.LANCZOS)
    s0 = min(w / g.width, h / g.height)
    dm = view["mask"]
    search = dm.sum() >= 16
    best = None
    for s in (np.arange(0.88, 1.14, 0.025) if search else [1.0]):
        gw, gh = max(1, round(g.width * s0 * s)), max(1, round(g.height * s0 * s))
        big = Image.new("RGBA", (w + 2 * M, h + 2 * M), view["backdrop"] + (255,))
        big.alpha_composite(g.resize((gw, gh), Image.LANCZOS), ((w + 2 * M - gw) // 2, (h + 2 * M - gh) // 2))
        big = big.convert("RGB")
        if view["up"] > 1.25:  # blur like the upscaled low-res design so colours at edges compare fairly
            small = (max(1, round(big.width / view["up"])), max(1, round(big.height / view["up"])))
            big = big.resize(small, Image.LANCZOS).resize(big.size, Image.LANCZOS)
        BL = _lab(np.asarray(big))
        BM = _clean(np.linalg.norm(BL - view["bl"], axis=-1) > view["thr"])
        if search:  # IoU of every offset at once: FFT correlation for |am & dm|, an integral image for |am|
            lo, hi = M + (-M // 2), M + M // 2 + 1
            inter = np.rint(_corr_valid(BM.astype(np.float64), dm.astype(np.float64)))[lo:hi, lo:hi]
            S = np.pad(BM.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            area = (S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w])[lo:hi, lo:hi]
            iou = inter / np.maximum(area + dm.sum() - inter, 1)
            k = int(np.argmax(iou))  # first maximum in (dy, dx) row-major order, like the former loop
            dy, dx = divmod(k, iou.shape[1])
            dy, dx, val = dy + (-M // 2), dx + (-M // 2), float(iou.flat[k])
        else:
            dy = dx = 0
            am0 = BM[M:M + h, M:M + w]
            val = (am0 & dm).sum() / max((am0 | dm).sum(), 1)
        if best is None or val > best[0]:
            best = (val, BL[M + dy:M + dy + h, M + dx:M + dx + w], BM[M + dy:M + dy + h, M + dx:M + dx + w])
    return best


def _corr_valid(A, B):
    """sum(A[i:i+h, j:j+w] * B) for every valid (i, j), via a circular FFT whose tail is wrap-free."""
    (H, W), (h, w) = A.shape, B.shape
    full = np.fft.irfft2(np.fft.rfft2(A, (H, W)) * np.fft.rfft2(B[::-1, ::-1], (H, W)), (H, W))
    return full[h - 1:, w - 1:]


def _qemd(a, b, n=32):
    """Mean 1-D earth mover's distance of the L, a, b distributions (quantile form)."""
    q = np.linspace(0.02, 0.98, n)
    return float(np.mean([np.abs(np.quantile(a[:, i], q) - np.quantile(b[:, i], q)).mean() for i in range(3)]))


def _fill_view(asset, view):
    """An opaque asset cropped to the element's aspect and resized onto the design view, in Lab."""
    w, h = view["size"]
    tw, th = asset.size
    if tw / th > w / h:
        nw = round(th * w / h)
        asset = asset.crop(((tw - nw) // 2, 0, (tw - nw) // 2 + nw, th))
    else:
        nh = round(tw * h / w)
        asset = asset.crop((0, (th - nh) // 2, tw, (th - nh) // 2 + nh))
    return _lab(np.asarray(asset.convert("RGB").resize((w, h), Image.LANCZOS)))


def perceptual(mock, bbox, asset, transparent=True):
    """iou: silhouette overlap after alignment; de: mean CIELAB distance; col: Lab distribution distance."""
    view = _design_view(mock, bbox)
    L, dm = view["L"], view["mask"]
    if transparent:
        iou, AL, am = _align_view(asset, view)
        if dm.sum() >= 16:
            return {"iou": float(iou), "de": float(np.linalg.norm(AL - L, axis=-1)[am | dm].mean()),
                    "col": _qemd(AL[am], L[dm]) if am.sum() > 8 else 50.0}
    else:
        AL = _fill_view(asset, view)
    # opaque art, or an element that can't be separated from its backdrop: compare the whole box
    return {"iou": None, "de": float(np.linalg.norm(AL - L, axis=-1).mean()), "col": _qemd(AL.reshape(-1, 3), L.reshape(-1, 3))}


def _fid(diff, p):
    return 0.3 * diff + 0.5 * p["de"] + p["col"] + (40 * (1 - p["iou"]) if p["iou"] is not None else 0)


def color_match(mock, bbox, asset, strength=0.75):
    """Reinhard-style transfer of the design element's Lab mean/std onto a transparent asset; alpha is kept."""
    view = _design_view(mock, bbox)
    _, AL, am = _align_view(asset, view)
    src, dst = AL[_morph(am, "min")], view["L"][_morph(view["mask"], "min")]
    if len(src) < 16 or len(dst) < 16:
        return asset
    k = np.clip(dst.std(0) / np.maximum(src.std(0), 1e-3), 0.75, 1.33).astype(np.float32)
    a = np.asarray(asset.convert("RGBA"))
    # the transfer is a per-colour map: compute it on the unique colours only (a 4096^2 asset no longer needs 3 GB)
    keys = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    rgb = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], -1).astype(np.uint8)
    L = _lab(rgb).astype(np.float32)
    L = L + np.float32(strength) * (dst.mean(0).astype(np.float32) + (L - src.mean(0).astype(np.float32)) * k - L)
    out = np.dstack([_lab_to_rgb(L)[inv.reshape(a.shape[:2])], a[..., 3]])
    out[a[..., 3] == 0, :3] = 0
    return Image.fromarray(out, "RGBA")


def score(mock, bbox, asset, ref=None, transparent=True):
    """Fidelity vs the mockup element plus copy/crop red flags. Lower 'fid' (and 'diff') = closer to the design."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    crop = mock.crop((x, y, x + w, y + h)).convert("RGB")
    backdrop = _backdrop(mock, bbox)
    flags = []
    A = asset.getchannel("A")
    transp = sum(A.histogram()[:8]) / (asset.width * asset.height)
    cb = content_box(asset, 40)
    empty = cb is None or (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.002 * asset.width * asset.height
    bb = cb or (0, 0, asset.width, asset.height)
    g = asset.crop(bb)
    if empty:
        flags.append("EMPTY")
    if transparent:
        if transp < 0.15:
            flags.append("OPAQUE")
        if not empty and (bb[0] <= 2 or bb[1] <= 2 or bb[2] >= asset.width - 2 or bb[3] >= asset.height - 2):
            flags.append("CUT-OFF")
        if not empty and _backdrop_edge_frac(asset, backdrop) > 0.5:
            flags.append("BACKDROP")
    s = min(w / g.width, h / g.height)
    gs = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), backdrop + (255,))
    canvas.alpha_composite(gs, ((w - gs.width) // 2, (h - gs.height) // 2))
    diff = _mean_diff(canvas, crop)
    flat = Image.new("RGBA", asset.size, backdrop + (255,))
    flat.alpha_composite(asset)
    small = flat.convert("RGB").resize((64, 64), Image.LANCZOS)
    if _mean_diff(small, mock.convert("RGB").resize((64, 64), Image.LANCZOS)) < 10:
        flags.append("COPY:returned-mockup")
    if ref is not None and _mean_diff(small, ref.convert("RGB").resize((64, 64), Image.LANCZOS)) < 8:
        flags.append("COPY:returned-reference")
    sharp = _edge_var(g) / max(_edge_var(crop.resize(g.size, Image.BICUBIC)), 1e-6)
    warn = ["soft"] if sharp < 2 else []
    p = perceptual(mock, bbox, asset, transparent)
    if p["iou"] is not None and p["iou"] < 0.5:
        warn.append("silhouette")
    return {"fid": round(_fid(diff, p), 1), "diff": round(diff, 1), "iou": None if p["iou"] is None else round(p["iou"], 3),
            "de": round(p["de"], 1), "col": round(p["col"], 1), "flags": flags, "warn": warn,
            "sharp_ratio": round(sharp, 1), "transparent_frac": round(transp, 3),
            "aspect_err": round(abs((g.width / g.height) / (w / h) - 1), 3)}


FLAG_WEIGHT = {"EMPTY": 1000, "COPY": 100, "OPAQUE": 10, "BACKDROP": 10, "CUT-OFF": 3}


def _flag_weight(flags):
    return sum(FLAG_WEIGHT.get(f.split(":")[0], 1) for f in flags)


def _rank(sc, meta):
    """Sort key: flag severity (EMPTY 1000 > COPY 100 > OPAQUE, BACKDROP 10 > CUT-OFF 3), then fid (+6 for low quality)."""
    return (_flag_weight(sc["flags"]), sc.get("fid", sc["diff"]) + (6 if (meta or {}).get("quality") == "low" else 0))


def _unusable(sc):
    """The flag that makes a candidate unshippable (EMPTY or COPY:*), or None."""
    return next((f for f in sc["flags"] if f == "EMPTY" or f.startswith("COPY")), None)


# ---------------------------------------------------------------- prompts

MOCKUP_WRAP = ("High-fidelity production UI design render of a single app screen on a {aspect} canvas. {prompt} "
               "Flat, pixel-crisp screen render at 1:1, no device frame, no hands, no perspective, no watermark. "
               "Real, legible UI text. Consistent spacing grid and one cohesive icon style.")

PLATFORMS = {
    "android": {
        "label": "native Android app (Material 3 / Material 3 Expressive, Jetpack Compose)",
        "system_ui": "Android status bar at the top (time, signal, battery) and the gesture navigation handle at the bottom; edge-to-edge layout",
        "components": ("TopAppBar (small/center-aligned/medium/large), NavigationBar (3-5 destinations, pill active indicator), "
                       "NavigationRail (tablets), FloatingActionButton / ExtendedFAB, Cards (elevated/filled/outlined), "
                       "Buttons (filled/tonal/outlined/text), SegmentedButton, Chips (assist/filter/input/suggestion), "
                       "Switch, Checkbox, RadioButton, Slider, SearchBar, Tabs, ListItem, Badge, ModalBottomSheet, "
                       "Dialog, Snackbar, progress indicators"),
        "metrics": "4 dp baseline / 8 dp spacing grid, 48 dp minimum touch targets, 24 dp icons, "
                   "corner radii 4/8/12/16/28 dp (extra-small..extra-large)",
        "type": "Material 3 type scale: Display/Headline/Title/Body/Label in Large/Medium/Small (e.g. Headline Small 24 sp, "
                "Title Large 22 sp, Body Large 16 sp, Label Large 14 sp)",
        "icons": "Material Symbols (Outlined or Rounded, consistent weight), filled variant for the selected tab",
        "screen_dp": 412,
    },
    "ios": {
        "label": "native iOS app (Human Interface Guidelines, iOS 26 Liquid Glass, SwiftUI)",
        "system_ui": "iOS status bar with Dynamic Island at the top and the home indicator at the bottom; respect safe areas",
        "components": ("NavigationStack bar with large title, floating Liquid Glass TabBar (up to 5 tabs), Toolbar, "
                       "List (inset grouped), Section, sheet with detents, Menu, context menu, Alert, "
                       "confirmation dialog, Picker / segmented control, Toggle, Slider, Stepper, searchable field, "
                       "DatePicker, ProgressView, swipe actions"),
        "metrics": "44 pt minimum touch targets, 16-20 pt margins, 8 pt spacing rhythm, continuous-corner rounded rectangles",
        "type": "SF Pro with Dynamic Type styles: Large Title 34, Title 1 28, Title 2 22, Title 3 20, Headline 17 semibold, "
                "Body 17, Callout 16, Subheadline 15, Footnote 13, Caption 12/11",
        "icons": "SF Symbols (matching weight to adjacent text), filled variant for the selected tab",
        "screen_dp": 393,
    },
    "web": {
        "label": "responsive web app (semantic HTML/CSS, accessible)",
        "system_ui": "no device chrome; a browser viewport",
        "components": ("header with primary navigation, sidebar or top nav, breadcrumbs, buttons, links, form fields with "
                       "labels, selects, checkboxes, toggles, cards, tables, tabs, modals/dialogs, toasts, pagination"),
        "metrics": "8 px spacing grid, 44 px minimum touch targets on mobile, visible focus states, "
                   "breakpoints 390 / 768 / 1280 / 1440 px",
        "type": "a web font pairing with a clear rem-based type scale (e.g. 14/16/20/24/32/40 px)",
        "icons": "one consistent icon set (e.g. Material Symbols or Lucide)",
        "screen_dp": 390,
    },
}

PLATFORM_RULE = ("Target platform: {label}. Build the screen from this platform's native components wherever one fits: "
                 "{components}. Show the {system_ui}. Metrics: {metrics}. Typography: {type}. Icons: {icons}. "
                 "Anything custom must still look and behave native to this platform.")


def _platform_rule(platform):
    return " " + PLATFORM_RULE.format(**PLATFORMS[platform]) if platform else ""


def _design_meta(path):
    """design.json written by `mockup` next to a mockup (platform, screen width), if any."""
    for d in (Path(path).parent, Path(path).parent.parent):
        f = d / "design.json"
        if f.exists():
            try:
                meta = json.loads(f.read_text())
            except (OSError, ValueError) as e:
                print(f"  ! ignoring unreadable {f}: {e}")
                return {}
            meta = meta if isinstance(meta, dict) else {}
            if meta.get("platform") is not None and meta["platform"] not in PLATFORMS:
                print(f"  ! {f}: unknown platform {meta.pop('platform')!r} ignored (use {', '.join(PLATFORMS)})")
            return meta
    return {}


def _screen_dp(platform, given=None):
    return given or PLATFORMS.get(platform or "", {}).get("screen_dp", 390)


STYLE_REF_PROMPT = ("The attached image is an APPROVED screen of this app. Design a DIFFERENT screen of the same app on a "
                    "{aspect} canvas: {prompt} Use the attached screen ONLY as the style guide: the same design language "
                    "(palette, typefaces and weights, text colours, eyebrow/label treatment, corner radii, surfaces, "
                    "borders, glow, icon style, illustration style, spacing rhythm) and the same shared chrome (app name, "
                    "navigation bar with the same items and icons) where this screen has it. Do NOT copy its layout, "
                    "sections, cards or copy: compose a new layout for this screen's purpose, with new content. "
                    "Flat, pixel-crisp screen render at 1:1, no device frame. Real, legible UI text.")

CRITIQUE_PROMPT = """Do design QA on this app UI mockup ({W}x{H} px; the screen is {dp}dp wide, so 1dp = {k:.2f}px;
target platform: {platform}). It will be built in code. Report only real defects a developer or user would hit, most
severe first:
- text: every string that is garbled, misspelled, nonsense, truncated or oddly cased (quote it exactly).
- alignment/spacing: misaligned edges or baselines, uneven gaps between repeated items, inconsistent margins.
- icons: mixed icon styles (outline vs filled, stroke weight, corners), ambiguous or malformed glyphs.
- contrast/accessibility: text or icons below WCAG AA (4.5:1 body, 3:1 large text and icons), text illegibly small
  RELATIVE to the rest of the screen (not absolute sizes: the drawing scale is inexact and is calibrated later),
  controls that cannot get a {touch}dp touch target (a smaller visible pill or icon with room around it is fine).
- overlap: a floating element (FAB, snackbar, sheet, badge, mini player) covering text, numbers, list rows or
  artwork; elements colliding, or hidden under the navigation bar or system bars. High when it hides content.
- platform: {platform} guideline problems (navigation pattern, tab count, safe areas, standard controls), and system
  UI that belongs to another platform (e.g. an iOS "9:41" status bar or Dynamic Island on Android, Android status
  icons or a gesture pill on iOS).
- implementability: what code can't build cleanly (text baked into artwork, impossible overlaps, inconsistent radii,
  content cut off).{ref_rule}
Report each root cause ONCE: group repeated instances (e.g. several labels with the same size problem) into one
issue. Per issue: severity (high = broken, illegible or blocks the build; medium = visible defect; low = nit),
category, a bbox around the problem (or its first instance) in this image's pixels, the problem in one sentence, and
ONE edit instruction (<= 30 words) that fixes only that issue. No taste preferences. A clean design gets few or no
issues."""

CRITIQUE_REF_RULE = """
- consistency: the SECOND image is an approved screen of the same app. Flag differences in app name, navigation
  items/icons, typefaces, palette and component styling (bbox refers to the first image)."""

ISSUE_ITEM = {"type": "object", "additionalProperties": False,
              "required": ["severity", "category", "bbox", "problem", "edit"],
              "properties": {
                  "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                  "category": {"type": "string", "enum": ["text", "alignment", "icons", "contrast", "accessibility",
                                                          "overlap", "platform", "implementability", "consistency"]},
                  "bbox": {"type": "object", "additionalProperties": False, "required": ["x", "y", "w", "h"],
                           "properties": {k: {"type": "integer"} for k in "xywh"}},
                  "problem": {"type": "string"}, "edit": {"type": "string"}}}

CRITIQUE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["issues"],
                   "properties": {"issues": {"type": "array", "items": ISSUE_ITEM}}}

VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["checks", "new_issues"],
                 "properties": {
                     "checks": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                           "required": ["id", "fixed"],
                                                           "properties": {"id": {"type": "integer"},
                                                                          "fixed": {"type": "boolean"}}}},
                     "new_issues": {"type": "array", "items": ISSUE_ITEM}}}

SEVERITY = {"high": 3, "medium": 2, "low": 1}


ASSET_PROMPT = ("The attached image is a zoomed, low-resolution screenshot region from an APPROVED app design. "
                "Recreate ONLY the {kind} \"{name}\" ({description}) from scratch as a clean, high-resolution "
                "standalone production asset that matches the design 1:1: identical silhouette, proportions, "
                "number of layers/parts, colour values{colors}, gradient directions, highlights and glow. "
                "The target is the element in the CENTRE of the attached image (about the middle 45% of its width and "
                "height); everything around it is context only. Do NOT upscale, trace or copy the screenshot pixels. "
                "Do NOT include the backdrop, container shape, text, or any neighbouring element. No watermark, "
                "signature or drop shadow unless the design has one. {bg}")

BACKGROUND_PROMPT = ("The attached image is a cropped region of an APPROVED app UI design. Recreate ONLY the {kind} "
                     "\"{name}\" ({description}) as a standalone full-bleed {aspect}, from scratch at full resolution, "
                     "with exactly the framing of the attached crop. REMOVE every UI element drawn on top of it: all text, "
                     "buttons, chips, icons, badges, rounded corners and borders. Where UI covered it, continue the "
                     "artwork naturally. Keep the exact composition, colours and lighting of the design.")

# per-candidate emphasis (c1, c2, c3, then repeating) so best-of-N samples different failure modes
CANDIDATE_FOCUS = ["",
                   " PRIORITY: the exact colour values, saturation, lightness, gradient directions and glow strength as "
                   "in the reference; do not brighten, saturate, warm, cool or shift any hue.",
                   " PRIORITY: the exact silhouette, proportions and relative sizes of every part as in the reference "
                   "(how big each part is relative to the others, stroke/ring thickness, spacing); do not enlarge, "
                   "simplify or add parts."]


def _aspect_words(w, h):
    r = w / h
    if r > 1.2:
        return f"wide landscape image (about {r:.1f}:1)"
    if r < 1 / 1.2:
        return f"tall portrait image (about 1:{1 / r:.1f})"
    return "square image"

MANIFEST_PROMPT = """This is an APPROVED mobile/web UI design ({W}x{H} px). Inventory every visual ASSET a developer
needs as a separate file to build it in code: icons, illustrations, images/photos, logos, decorative graphics, and
artwork that sits behind UI (kind "background"). Rules:
- Exclude plain text, solid/gradient fills, simple shapes, dividers, shadows and standard containers code can draw.
- Exclude system UI the OS draws: status bar (time, wifi, signal, battery, notification icons), Dynamic Island,
  home indicator, gesture handle, system navigation buttons.
- One entry per distinct graphic. Identical repeats (same icon used twice) appear once.
- bbox is TIGHT around the graphic itself in this image's pixels (exclude any circle/card behind an icon unless that
  backdrop is part of the artwork).
- format "svg" for flat/line/monochrome/simple-gradient glyphs (tab bar, toolbar, chevrons, UI icons); "png" for
  painterly, textured, glowing, 3D, photographic or detailed illustrations.
- transparent=false only for full-bleed artwork/photos (kind background/photo).
- animated=true if it is a natural candidate for motion (flame, loader, mascot, weather, pulsing element).
- name: short snake_case noun for the graphic itself (no file extension, no format/kind suffix).
- description: precise enough to redraw it from scratch — shapes, parts, layer order, colours, gradients, stroke weight.
- colors: main hex colours sampled from the design.
- material_symbol / sf_symbol: for an icon that is a standard glyph, the closest Google Material Symbols name
  (snake_case, e.g. home, settings, search, favorite) and Apple SF Symbols name (e.g. house.fill, gearshape,
  magnifyingglass, heart); "" for custom glyphs and for every non-icon."""

SVG_RULES = """Write each asset as a standalone SVG that matches the design 1:1 (shape, proportions, stroke weight,
corner rounding, colours). Hard rules so it converts to Android VectorDrawable and normalises onto a 24 grid:
- Root: <svg xmlns="http://www.w3.org/2000/svg" viewBox="..."> (the xmlns is required).
- Elements allowed: path, circle, ellipse, rect, line, polyline, polygon, g, defs, linearGradient, radialGradient,
  stop. No text, image, filter, mask, clipPath, use, symbol, style, CSS classes, marker, pattern, nested svg.
- No transform attribute on ANY element (bake positions into the coordinates), no gradientTransform.
- Plain numbers only: no % and no units (px, pt, em) in coordinates, sizes, stroke widths or opacities.
- Colours as hex. Gradients use gradientUnits="userSpaceOnUse" with explicit numeric coordinates: every
  linearGradient has x1 y1 x2 y2, every radialGradient has cx cy r (no fx/fy), and each has at least two stops.
- No stroke-dasharray, vector-effect, paint-order or marker-start/mid/end (VectorDrawable cannot express them).
- Square viewBox (e.g. "0 0 24 24") for icons unless the graphic is clearly non-square.
- Keep stroke-based icons stroke-based (stroke, stroke-width, stroke-linecap, stroke-linejoin); fill="none" on them."""

VECTOR_PROMPT = """Vectorise this asset from an APPROVED design for a looping vector animation (Lottie and Android
AnimatedVectorDrawable). The MOVING part is: {moving}. The STATIC part is: {static}. Motion: {motion}, one loop is
{seconds}s. {strength} Rules:
- Every path "d" uses ONLY absolute M, L, C and Z commands (no H/V/S/Q/A, no relative commands).
- Shape change (flame, liquid, cloth, smoke): an animated layer with {k} keyframes that all have the IDENTICAL command
  sequence (same count and order of M/L/C/Z) and morph into each other. The loop runs back from the last keyframe to
  the first, so never repeat the first keyframe at the end.
- Rigid motion (bounce, sway, tilt, spin, pulse, pop, fade, expanding rings): the layer's "motion". anchor_x/anchor_y
  is the pivot in viewBox units. keys: t (0..1 of the loop, first key t=0), dx/dy offset, sx/sy scale factor (1 = rest),
  rotate (degrees, clockwise), opacity (multiplier 0..1), ease (curve towards the next key). If the last key is before
  t=1 the loop eases back to the first key. keys: [] means the layer does not move. A layer can morph AND move.
- glows: soft radial halos (centre, radii, colour, peak opacity) drawn as a gradient fading to transparent. Add one for
  every glow or light in the design (behind a flame, around a lamp) and animate its opacity/scale; otherwise [].
- Several copies of a moving part (ripples, sparks, notes) are separate layers with time-shifted keys.
- fill is a colour, url(#id) of a gradient in "defs" (gradientUnits="userSpaceOnUse", explicit coordinates in viewBox
  units, at least 2 stops, no gradientTransform) or "none"; stroke is a colour or "none", stroke_width in viewBox units
  (0 without stroke). Colours are #RRGGBB only; transparency goes in opacity (0..1), never in the colour.
- Names: lowercase snake_case, unique across layers and glows. Coordinates: at most 1 decimal.
- Draw only the asset itself: no background, backdrop, card, plate or shadow behind it.
- Match the design's silhouette, proportions and colours. layer_order lists every layer and glow, bottom to top."""

MOTION_STRENGTH = {
    "subtle": "Keep the motion calm: moving parts shift or deform by about 2-4% of their size.",
    "normal": "The motion must read clearly at icon size: moving parts shift, deform or scale by about 6-12% of their size.",
    "strong": "Make the motion pronounced and lively: moving parts shift, deform or scale by 12-25% of their size "
              "(flame tips sway widely, pulses scale by 20%).",
}
MOTION_MIN = {"subtle": 0.015, "normal": 0.04, "strong": 0.08}

EASES = {"linear": (0, 0, 1, 1), "ease_in_out": (0.42, 0, 0.58, 1), "ease_out": (0, 0, 0.58, 1),
         "ease_in": (0.42, 0, 1, 1), "overshoot": (0.34, 1.56, 0.64, 1)}

_NUMBER_T = {"type": "number"}
_UNIT_T = {"type": "number", "minimum": 0, "maximum": 1}
_POS_T = {"type": "number", "minimum": 0}
# pattern / minimum / maximum / minItems are part of OpenAI strict structured outputs; a gateway that rejects them
# answers 400 on text.format, which ask() (g1) retries without strict mode, and _canon_vec/_validate_vec re-check here.
_NAME_T = {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}
_COLOR_T = {"type": "string", "pattern": "^(#[0-9A-Fa-f]{6}|url\\(#[A-Za-z_][\\w-]*\\)|none)$"}
_HEX_T = {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"}
_MOTION = {"type": "object", "additionalProperties": False, "required": ["anchor_x", "anchor_y", "keys"],
           "properties": {"anchor_x": _NUMBER_T, "anchor_y": _NUMBER_T, "keys": {"type": "array", "items": {
               "type": "object", "additionalProperties": False,
               "required": ["t", "dx", "dy", "sx", "sy", "rotate", "opacity", "ease"],
               "properties": {**{k: _NUMBER_T for k in ("dx", "dy", "sx", "sy", "rotate")}, "t": _UNIT_T, "opacity": _UNIT_T,
                              "ease": {"type": "string", "enum": list(EASES)}}}}}}
_PAINT = {"fill": _COLOR_T, "opacity": _UNIT_T, "stroke": _COLOR_T, "stroke_width": _POS_T,
          "fill_rule": {"type": "string", "enum": ["nonzero", "evenodd"]},
          "cap": {"type": "string", "enum": ["butt", "round", "square"]}, "join": {"type": "string", "enum": ["miter", "round", "bevel"]}}

VECTOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["viewBox", "defs", "static_layers", "animated_layers", "glows", "layer_order"],
    "properties": {
        "viewBox": {"type": "string", "pattern": "^-?[0-9.]+ -?[0-9.]+ [0-9.]+ [0-9.]+$"}, "defs": {"type": "string"},
        "layer_order": {"type": "array", "items": _NAME_T},
        "static_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "paths", "motion"],
            "properties": {"name": _NAME_T, "motion": _MOTION, "paths": {"type": "array", "minItems": 1, "items": {
                "type": "object", "additionalProperties": False, "required": ["d", *_PAINT],
                "properties": {"d": {"type": "string"}, **_PAINT}}}}}},
        "animated_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", *_PAINT, "keyframes", "motion"],
            "properties": {"name": _NAME_T, **_PAINT, "motion": _MOTION,
                           "keyframes": {"type": "array", "minItems": 2, "items": {"type": "string"}}}}},
        "glows": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "cx", "cy", "rx", "ry", "color", "opacity", "motion"],
            "properties": {"name": _NAME_T, "cx": _NUMBER_T, "cy": _NUMBER_T, "rx": {"type": "number", "exclusiveMinimum": 0},
                           "ry": {"type": "number", "exclusiveMinimum": 0}, "opacity": _UNIT_T, "color": _HEX_T, "motion": _MOTION}}},
    },
}

MANIFEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["assets"],
    "properties": {"assets": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["name", "kind", "bbox", "format", "transparent", "animated", "description", "colors",
                     "material_symbol", "sf_symbol"],
        "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["icon", "illustration", "photo", "logo", "decoration", "background"]},
            "bbox": {"type": "object", "additionalProperties": False, "required": ["x", "y", "w", "h"],
                     "properties": {k: {"type": "integer"} for k in "xywh"}},
            "format": {"type": "string", "enum": ["svg", "png"]},
            "transparent": {"type": "boolean"}, "animated": {"type": "boolean"},
            "description": {"type": "string"}, "colors": {"type": "array", "items": {"type": "string"}},
            "material_symbol": {"type": "string"}, "sf_symbol": {"type": "string"}}}}},
}

SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["palette", "typography", "radii", "spacing", "components"],
    "properties": {
        "palette": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                               "required": ["name", "hex", "usage"],
                                               "properties": {"name": {"type": "string"}, "hex": {"type": "string"},
                                                              "usage": {"type": "string"}}}},
        "typography": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["role", "sample", "family_guess", "size_px", "weight", "letter_spacing_em", "color", "platform_style"],
                                                  "properties": {"role": {"type": "string"}, "sample": {"type": "string"},
                                                                 "platform_style": {"type": "string"},
                                                                 "family_guess": {"type": "string"}, "size_px": {"type": "number"},
                                                                 "weight": {"type": "integer"}, "letter_spacing_em": {"type": "number"},
                                                                 "color": {"type": "string"}}}},
        "radii": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["element", "px"],
                                             "properties": {"element": {"type": "string"}, "px": {"type": "number"}}}},
        "spacing": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["between", "px"],
                                               "properties": {"between": {"type": "string"}, "px": {"type": "number"}}}},
        "components": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["name", "bbox", "native", "notes"],
                                                  "properties": {"name": {"type": "string"}, "notes": {"type": "string"},
                                                                 "native": {"type": "string"},
                                                                 "bbox": {"type": "object", "additionalProperties": False,
                                                                          "required": ["x", "y", "w", "h"],
                                                                          "properties": {k: {"type": "integer"} for k in "xywh"}}}}},
    },
}


# ---------------------------------------------------------------- commands

ASSET_DEFAULTS = (("kind", "illustration"), ("format", "png"), ("description", ""), ("colors", []))


def _load(d):
    """(dir, manifest, mockup RGBA) with one-line exits and normalised manifest entries (plan contract C5):
    missing screen_dp is filled from the platform, per-asset defaults are applied and printed, every bbox becomes
    ints clamped inside the mockup (clamps printed; no overlap or no name/bbox -> exit naming the asset)."""
    d = Path(d)
    if not d.is_dir():
        sys.exit(f"gptkit: project dir not found: {d} (run manifest first)")
    mf = d / "assets.json"
    m = _read_json(mf, "manifest", required=("mockup", "width", "height", "assets"))
    if not isinstance(m["assets"], list):
        sys.exit(f"gptkit: manifest {mf}: 'assets' must be a list of assets")
    if not isinstance(m["mockup"], str) or not m["mockup"]:
        sys.exit(f"gptkit: manifest {mf}: 'mockup' must be the mockup file name")
    mock = _open_image(d / m["mockup"], "mockup", f"named by {mf}")
    W, H = mock.size
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0  # noqa: E731
    for k in ("width", "height", "screen_dp"):  # hand-edited values: a wrong type would surface after the API calls
        if k in m and not num(m[k]):
            sys.exit(f"gptkit: {mf}: '{k}' must be a positive number, got {m[k]!r}")
    if (m["width"], m["height"]) != (W, H):  # bboxes are mockup pixels: the mockup's size is the truth
        print(f"  ~ {mf.name}: width/height {m['width']}x{m['height']} differ from the mockup {W}x{H}; using the mockup's")
        m["width"], m["height"] = W, H
    if "screen_dp" not in m:
        plat = m.get("platform") if m.get("platform") in PLATFORMS else None
        m["screen_dp"] = _screen_dp(plat)
        print(f"  ~ {mf}: no screen_dp; assuming {m['screen_dp']} dp ({plat or 'no known platform'}; run spec to fix)")
    defaults, seen = [], set()
    for i, x in enumerate(m["assets"]):
        if not isinstance(x, dict) or not isinstance(x.get("name"), str) or not x["name"] or "bbox" not in x:
            who = x.get("name") if isinstance(x, dict) and x.get("name") else f"#{i + 1}"
            sys.exit(f"gptkit: {mf}: asset {who} needs 'name' and 'bbox' {{x,y,w,h}}")
        n = x["name"]
        if n in seen:
            sys.exit(f"gptkit: {mf}: duplicate asset name '{n}' (rename one of them)")
        seen.add(n)
        added = []
        for k, v in ASSET_DEFAULTS:
            if k not in x:
                x[k] = list(v) if isinstance(v, list) else v
                added.append(f"{k}={v!r}")
        if "transparent" not in x:
            x["transparent"] = x["kind"] not in ("background", "photo")
            added.append(f"transparent={x['transparent']}")
        if added:
            defaults.append(f"{n}: {', '.join(added)}")
        b = x["bbox"]
        nb = _clip_bbox(b, W, H)
        if nb is None:
            sys.exit(f"gptkit: {mf}: asset '{n}' has an invalid bbox {b}: x,y,w,h must be numbers with w,h >= 1 "
                     f"overlapping the {W}x{H} mockup")
        try:
            asked = {k: int(round(float(b[k]))) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            asked = None
        if asked != nb:
            was = " ".join(f"{k}={b.get(k)}" for k in ("x", "y", "w", "h"))
            print(f"  ! {mf.name}: {n} bbox clamped to the {W}x{H} mockup (was {was})")
        elif any(not isinstance(b[k], int) for k in ("x", "y", "w", "h")):
            print(f"  ~ {mf.name}: {n} bbox normalised to ints")
        x["bbox"] = nb
    if defaults:
        print(f"  ~ {mf.name}: defaults applied: " + "; ".join(defaults))
    return d, m, mock


def _known(m):
    return ", ".join(a["name"] for a in m["assets"]) or "none"


def _select(m, only):
    """The manifest assets named by a comma list (all when empty); unknown names exit with the known ones."""
    if not only:
        return list(m["assets"])
    names = [n.strip() for n in only.split(",") if n.strip()]
    unknown = [n for n in names if n not in {a["name"] for a in m["assets"]}]
    if unknown:
        sys.exit(f"gptkit: unknown asset(s): {', '.join(unknown)} (known: {_known(m)})")
    return [a for a in m["assets"] if a["name"] in names]


def _asset(m, name):
    """One manifest asset by name; exits naming the known ones."""
    for a in m["assets"]:
        if a["name"] == name:
            return a
    sys.exit(f"gptkit: unknown asset: {name} (known: {_known(m)})")


ASPECTS = {"1024x1536": "portrait 2:3 (phone screen, taller than wide)", "1536x1024": "landscape 3:2 (desktop/tablet, wider than tall)",
           "1024x1024": "square 1:1"}


def _aspect_ok(img_bytes, size):
    w, h = Image.open(io.BytesIO(img_bytes)).size
    tw, th = map(int, size.split("x"))
    return (w > h) == (tw > th) and (w < h) == (tw < th)


def _layout_diff(a, b):
    return round(_mean_diff(a.convert("L").resize((64, 96)), b.convert("L").resize((64, 96))), 1)


def _prep_ref(path, long_side=1536, what="reference image"):
    """PNG bytes of a reference image for /images/edits: any mode/format, downscaled to long_side (a 4000x6000 photo
    would otherwise be uploaded at full size and may exceed the endpoint's limits)."""
    img = _open_image(path, what)
    s = long_side / max(img.size)
    if s < 1:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return _png_bytes(img)


def cmd_mockup(a):
    if a.edit and a.style_ref:
        sys.exit("gptkit: use --edit (revise a screen) or --style-ref (new screen in its style), not both")
    if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", a.size or ""):
        sys.exit(f"gptkit: --size must be WIDTHxHEIGHT, e.g. 1024x1536 (portrait), 1536x1024 or 1024x1024; got {a.size!r}")
    edit_png = _prep_ref(a.edit, what="--edit mockup") if a.edit else None
    ref_png = _prep_ref(a.style_ref, what="--style-ref image") if a.style_ref else None
    ref = Image.open(io.BytesIO(ref_png)) if ref_png else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    aspect = ASPECTS.get(a.size, a.size)
    platform = a.platform or (_design_meta(a.edit or a.style_ref).get("platform") if (a.edit or a.style_ref) else None)
    if not platform:
        print("  (no --platform: designing platform-neutral; pass --platform android|ios|web for native components)")
    _plan(a.variants, 0, image_model=a.model, size=a.size, edits=bool(a.edit or a.style_ref))
    rule = _platform_rule(platform)
    prompt = a.prompt if a.raw else MOCKUP_WRAP.format(prompt=a.prompt, aspect=aspect) + rule
    edit_prompt = f"{a.prompt} Keep the {aspect} canvas and everything not mentioned exactly as it is."
    style_prompt = a.prompt if a.raw else STYLE_REF_PROMPT.format(prompt=a.prompt, aspect=aspect) + rule
    if platform:
        meta_path = out / "design.json"
        meta = _design_meta(meta_path) if meta_path.exists() else {}
        meta.update({"platform": platform, "screen_dp": _screen_dp(platform)})
        meta_path.write_text(json.dumps(meta, indent=2))

    def one(i):
        # the backend picks aspect from the wording, not from `size`, and a style-ref edit can return the reference
        for _ in range(2):
            if edit_png is not None:
                img, meta = edit(edit_png, edit_prompt, a.size, model=a.model, task="concept")
            elif ref_png is not None:
                img, meta = edit(ref_png, style_prompt, a.size, model=a.model, task="concept")
            else:
                img, meta = generate(prompt, a.size, model=a.model)
            meta["layout_diff"] = _layout_diff(ref, Image.open(io.BytesIO(img))) if ref is not None else None
            copied = ref is not None and meta["layout_diff"] < 10
            if copied:
                print(f"  ! variant {i + 1} copied the style reference (layout_diff {meta['layout_diff']}), retrying")
            if _aspect_ok(img, a.size) and not copied:
                break
        if not _aspect_ok(img, a.size):
            got = "x".join(map(str, Image.open(io.BytesIO(img)).size))
            print(f"  ! variant {i + 1}: backend returned {got} instead of the requested {a.size} aspect after 2 attempts; "
                  "saved anyway (state the aspect in the prompt or crop/pad)")
        if copied:
            print(f"  ! variant {i + 1} still copies the style reference after 2 attempts; saved anyway (reword the prompt)")
        p = out / f"{a.name}_v{i + 1}.png"
        p.write_bytes(img)
        return p, meta
    ex = cf.ThreadPoolExecutor(a.variants)
    futs = {ex.submit(one, i): i for i in range(a.variants)}
    made, last_err = 0, None
    try:
        for f in cf.as_completed(futs):
            try:
                p, meta = f.result()
            except ApiError as e:
                last_err = e
                if a.variants > 1:
                    print(f"  ! variant {futs[f] + 1} failed: {e}")
                continue
            made += 1
            extra = f" layout_diff={meta['layout_diff']}" if ref is not None else ""
            print(f"{p}  quality={meta.get('quality')} size={meta.get('size')}{extra}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if not made:
        if a.variants == 1 or last_err is None:
            raise last_err or RuntimeError("mockup produced no image")
        sys.exit(f"gptkit: mockup produced no image (all {a.variants} variants failed, see above; last: {last_err})")


def _lum(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ [0.2126, 0.7152, 0.0722]


def measured_contrast(img, bbox):
    """WCAG ratio estimate inside bbox: perimeter median as background vs the most distinct 5% of pixels."""
    x0, y0 = min(max(0, bbox["x"]), img.width - 2), min(max(0, bbox["y"]), img.height - 2)
    x1, y1 = min(img.width, max(x0 + 2, bbox["x"] + bbox["w"])), min(img.height, max(y0 + 2, bbox["y"] + bbox["h"]))
    px = np.asarray(img.convert("RGB").crop((x0, y0, x1, y1)), float)
    edge = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]])
    bg = _lum(np.median(edge, 0))
    lum = _lum(px.reshape(-1, 3))
    far = np.abs(lum - bg)
    fg = np.median(lum[far >= np.quantile(far, 0.95)])
    hi, lo = max(fg, bg), min(fg, bg)
    return round((hi + 0.05) / (lo + 0.05), 1)


def critique(img, dp=390, platform="android", style_ref=None, note=None):
    rule = CRITIQUE_REF_RULE if style_ref is not None else ""
    touch = 48 if platform == "android" else 44
    label = PLATFORMS[platform]["label"] if platform in PLATFORMS else platform
    text = CRITIQUE_PROMPT.format(W=img.width, H=img.height, dp=dp, k=img.width / dp, platform=label, touch=touch,
                                  ref_rule=rule) + (f"\nThe user also says: {note}" if note else "")
    parts = [text, img.convert("RGB")] + ([style_ref.convert("RGB")] if style_ref is not None else [])
    issues = _clean_issues(ask(parts, CRITIQUE_SCHEMA, "design_qa"), "issues", img)
    for i, x in enumerate(issues):
        x["id"] = i + 1
        if x["category"] == "contrast" and x["bbox"]["w"] > 2 and x["bbox"]["h"] > 2:
            x["measured_contrast"] = measured_contrast(img, x["bbox"])
    return issues


def _clean_issues(res, key, img):
    """res[key] from a critique/verify response, each issue's bbox clipped to img (ints inside the image). Issues whose
    bbox is unreadable or lies outside the image are dropped and printed; a missing list exits (the call was wasted)."""
    items = res.get(key) if isinstance(res, dict) else None
    if not isinstance(items, list):
        sys.exit(f"gptkit: the model's design-QA response has no '{key}' list (non-conforming gateway?): {str(res)[:200]}")
    out = []
    for x in items:
        if not isinstance(x, dict):
            continue
        b = _clip_bbox(x.get("bbox"), img.width, img.height)
        if b is None:
            print(f"  ! dropped an issue whose bbox {x.get('bbox')} is not inside the {img.width}x{img.height} image: "
                  f"[{x.get('severity')}] {x.get('problem', '')}")
            continue
        out.append({"category": "", "problem": "", "edit": "", **x, "bbox": b})
    return out


_SEV_WARNED = set()


def _sev(x):
    """Weight of an issue's severity. An unknown one (lax gateway, older log) counts as medium, with one warning."""
    s = x.get("severity")
    if s in SEVERITY:
        return SEVERITY[s]
    if repr(s) not in _SEV_WARNED:
        _SEV_WARNED.add(repr(s))
        print(f"  ! unknown issue severity {s!r} from the model: counted as medium (weight 2)")
    return 2


def _weight(issues):
    return sum(_sev(x) for x in issues)


def _issue_overlay(img, issues, out):
    ov = img.convert("RGB")
    d = ImageDraw.Draw(ov)
    col = {"high": "#ff2d55", "medium": "#ff9f0a", "low": "#0a84ff"}
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for x in issues:
        b = _clip_bbox(x.get("bbox"), ov.width, ov.height)  # already clipped by critique(); older logs may not be
        if b is None:
            continue
        c = col.get(x.get("severity"), col["medium"])
        d.rectangle([b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1], outline=c, width=4)
        ty = max(0, b["y"] - 36)
        d.rectangle([b["x"], ty, b["x"] + 40, ty + 36], fill=c)
        d.text((b["x"] + 6, ty + 3), str(x.get("id", "")), fill="white", font=font)
    ov.save(out)


def _print_issues(issues):
    for x in issues:
        mc = f"  [measured {x['measured_contrast']}:1]" if "measured_contrast" in x else ""
        print(f"  {x.get('id', ''):>3}. {str(x.get('severity')):6s} {str(x.get('category', '')):16s} {x.get('problem', '')}"
              f"{mc}\n      edit: {x.get('edit', '')}")


def cmd_critique(a):
    src = Path(a.mockup)
    out = Path(a.out) if a.out else src.parent
    cur_path, cur = src, _open_image(src, "mockup")
    ref = _open_image(a.style_ref, "--style-ref image") if a.style_ref else None
    out.mkdir(parents=True, exist_ok=True)
    a.platform = a.platform or _design_meta(src).get("platform") or "android"
    a.screen_dp = _screen_dp(a.platform, a.screen_dp)
    issues = critique(cur, a.screen_dp, a.platform, ref, a.note)
    log = {"mockup": str(src), "rounds": [{"image": str(src), "issues": issues, "weight": _weight(issues)}]}
    print(f"{src.name}: {len(issues)} issues, weight {_weight(issues)}")
    _print_issues(issues)
    _issue_overlay(cur, issues, out / f"{src.stem}_critique.png")
    min_w = SEVERITY[a.min_severity]
    for rnd in range(a.rounds if a.apply else 0):
        todo = [x for x in issues if _sev(x) >= min_w]
        if not todo:
            print("nothing at or above --min-severity to apply")
            break
        aspect = ASPECTS.get(_size_for(*cur.size), "same")
        fixes = "\n".join(f"{i + 1}. {x['edit']}" for i, x in enumerate(todo))
        prompt = (f"Revise the attached app UI design. Apply ONLY these fixes:\n{fixes}\nKeep everything else exactly as "
                  f"it is: layout, content, copy, colours, typefaces, illustrations, icons and the {aspect} canvas.")
        raw, _ = edit(_png_bytes(cur.convert("RGB")), prompt, _size_for(*cur.size), model=a.model)
        p = out / f"{src.stem}_qa{rnd + 1}.png"
        p.write_bytes(raw)
        new = _rgba(p)
        listing = "\n".join(f"{x['id']}. [{x.get('severity')}] {x['problem']}" for x in todo)
        res = ask([f"FIRST image: a UI design. SECOND image: its revision, which was asked to fix these issues:\n{listing}\n"
                   "For each id say whether the SECOND image fixed it. Then list NEW defects the revision introduced "
                   "(changed layout or copy, garbled text, lost elements, style drift) in the same issue format, "
                   "bbox in the SECOND image's pixels. Do not re-list issues that exist in both.", cur.convert("RGB"),
                   new.convert("RGB")], VERIFY_SCHEMA, "design_qa_verify")
        new_issues = _clean_issues(res, "new_issues", new)
        fixed = {c.get("id") for c in res.get("checks") or [] if isinstance(c, dict) and c.get("fixed") is True
                 and isinstance(c.get("id"), int)}
        left = [x for x in todo if x["id"] not in fixed]
        before, after = _weight(todo), _weight(left) + _weight(new_issues)
        drift = round(_mean_diff(cur.resize((128, 192)), new.resize((128, 192))), 1)
        kept = after < before
        fixed &= {x["id"] for x in todo}
        print(f"round {rnd + 1}: fixed {len(fixed)}/{len(todo)}, new {len(new_issues)}, weight {before} -> {after}, "
              f"drift {drift} -> {'KEPT' if kept else 'discarded'}  ({p})")
        _print_issues([dict(x, id=f"n{i + 1}") for i, x in enumerate(new_issues)])
        log["rounds"].append({"image": str(p), "fixed": sorted(fixed), "new_issues": new_issues,
                              "weight_before": before, "weight_after": after, "drift": drift, "kept": kept})
        if not kept:
            break
        rest = [x for x in issues if _sev(x) < min_w]
        issues = [dict(x, id=i + 1) for i, x in enumerate(left + new_issues + rest)]
        cur_path, cur = p, new
    log["best"] = str(cur_path)
    (out / f"{src.stem}_critique.json").write_text(json.dumps(log, indent=2))
    print(f"-> {out / (src.stem + '_critique.json')}, overlay {out / (src.stem + '_critique.png')}, best: {cur_path}")


MANIFEST_REQUIRED = ("name", "kind", "bbox", "format", "transparent", "animated", "description")


def _manifest_assets(res, W, H):
    """The model's asset list made safe to write: required fields present, bboxes clipped to the WxH mockup (int, inside),
    boxes mostly outside it dropped, names normalised and unique (flame, flame_2, flame_3). Every drop/clamp is printed."""
    items = res.get("assets") if isinstance(res, dict) else None
    if not isinstance(items, list):
        sys.exit(f"gptkit: the model's manifest response has no 'assets' list (non-conforming gateway?): {str(res)[:200]}")
    out, seen = [], set()
    for x in items:
        missing = [k for k in MANIFEST_REQUIRED if not isinstance(x, dict) or k not in x]
        if missing:
            print(f"  ! dropped an asset missing {missing} from the model's response (non-conforming gateway?): "
                  f"{(x.get('name') if isinstance(x, dict) else x)!r}")
            continue
        b0 = x["bbox"]
        b = _clip_bbox(b0, W, H)
        try:
            claimed = abs(float(b0["w"]) * float(b0["h"]))
        except (KeyError, TypeError, ValueError):
            claimed = 0
        if b is None or b["w"] * b["h"] < 0.25 * claimed:
            print(f"  ! dropped '{x['name']}': bbox mostly outside the image ({b0} on {W}x{H})")
            continue
        if any(b[k] != b0.get(k) for k in "xywh"):
            print(f"  ! '{x['name']}': bbox clamped to the image ({b0} -> {b['x']},{b['y']},{b['w']},{b['h']})")
        n = re.sub(r"_(svg|png|icon|asset)$", "", _norm_name(str(x["name"]))).strip("_") or "asset"
        base, i = n, 2
        while n in seen:
            n, i = f"{base}_{i}", i + 1
        seen.add(n)
        out.append(dict(x, name=n, bbox=b))
    return out


def cmd_manifest(a):
    mock_path = Path(a.mockup).resolve()
    mock = _open_image(mock_path, "mockup")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    local = out / "mockup.png"
    if mock_path != local.resolve():
        mock.save(local)
    platform = a.platform or _design_meta(mock_path).get("platform")
    screen_dp = _screen_dp(platform, a.screen_dp)
    native = (f"\n- Target platform: {PLATFORMS[platform]['label']}. Native components ({PLATFORMS[platform]['components']}) "
              "are built in code with the platform toolkit: never list a component itself (tab bar, app bar, switch, slider, "
              "chip, button, sheet) as an asset, only custom glyphs/artwork inside it (e.g. the tab icons, not the tab bar).") \
        if platform else ""
    res = ask([MANIFEST_PROMPT.format(W=mock.width, H=mock.height) + native, mock], MANIFEST_SCHEMA, "asset_manifest",
              model=a.model)
    assets = _manifest_assets(res, mock.width, mock.height)
    system = re.compile(r"wifi|wi_fi|signal|cellular|battery|status_bar|clock|dynamic_island|home_indicator|gesture")
    dropped = [x["name"] for x in assets if system.search(x["name"]) and
               (x["bbox"]["y"] + x["bbox"]["h"] < 0.05 * mock.height or x["bbox"]["y"] > 0.95 * mock.height)]
    if dropped:
        print(f"  (dropped system UI the OS draws: {', '.join(dropped)})")
        assets = [x for x in assets if x["name"] not in dropped]
    for x in assets:
        x["material_symbol"], x["sf_symbol"] = _symbol_names(str(x.get("material_symbol") or ""),
                                                             str(x.get("sf_symbol") or ""))
        x["dp"] = {"w": round(x["bbox"]["w"] * screen_dp / mock.width, 1),
                   "h": round(x["bbox"]["h"] * screen_dp / mock.width, 1)}
    manifest = {"mockup": "mockup.png", "width": mock.width, "height": mock.height, "platform": platform,
                "screen_dp": screen_dp, "assets": assets}
    (out / "assets.json").write_text(json.dumps(manifest, indent=2))
    ov = mock.convert("RGB")
    d = ImageDraw.Draw(ov)
    for x in assets:
        b = x["bbox"]
        col = "#ff2d55" if x["format"] == "png" else "#34c759"
        d.rectangle([b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1], outline=col, width=3)
        d.text((b["x"] + 2, max(0, b["y"] - 12)), x["name"], fill=col)
    ov.save(out / "manifest_overlay.png")
    for x in assets:
        b = x["bbox"]
        print(f"{x['name']:34s} {str(x['kind']):12s} {x['format']}  bbox={b['x']},{b['y']},{b['w']},{b['h']}  "
              f"~{x['dp']['w']}x{x['dp']['h']}dp  transparent={x['transparent']} animated={x['animated']}")
    print(f"\n{len(assets)} assets -> {out / 'assets.json'}   overlay -> {out / 'manifest_overlay.png'} "
          f"(red=png, green=svg)")


def _platform_styles(platform):
    """The text-style names `tokens` maps onto: M3 type-scale slots on android, HIG text styles on ios, none elsewhere."""
    if platform == "android":
        return tuple(n for n, _, _ in _M3_SCALE)
    if platform == "ios":
        return tuple(n for n, _ in _HIG_SCALE)
    return ()


def _spec_schema(platform):
    """SPEC_SCHEMA with `platform_style` restricted to _platform_styles(platform) when the platform has fixed names."""
    sch = json.loads(json.dumps(SPEC_SCHEMA))
    names = _platform_styles(platform)
    if names:
        sch["properties"]["typography"]["items"]["properties"]["platform_style"]["enum"] = list(names)
    return sch


def cmd_spec(a):
    if a.recalibrate:
        return cmd_recalibrate(a.dir, a.scale)
    d, m, mock = _load(a.dir)
    platform = m.get("platform")
    styles = _platform_styles(platform)
    target = (f" Target platform: {PLATFORMS[platform]['label']}. For every component set 'native' to the native "
              f"component that should implement it (from: {PLATFORMS[platform]['components']}; add its variant, e.g. "
              "'Card (elevated)'), 'system' for UI the OS draws (status bar, home indicator), or 'custom' when none fits. For every text style set 'platform_style' to the closest "
              f"style of {PLATFORMS[platform]['type']}" + (f", named as one of: {', '.join(styles)}." if styles else ".")) if platform else \
        " Set 'native' to 'custom' and 'platform_style' to '' (no target platform given)."
    screen_dp = m.get("screen_dp") or _screen_dp(platform)
    res = ask([f"This is an APPROVED UI design ({mock.width}x{mock.height}px; the screen is {screen_dp}dp wide). "
               "Extract the design spec a developer needs to rebuild it 1:1: palette (hex), every text style "
               "(role, a sample string, likely font family, size in px of THIS image, weight, letter spacing, colour), "
               "corner radii, key spacings, and the component list with pixel bboxes. Every bbox covers the WHOLE component "
               "(for text, the full block including every line, not one word)." + target, mock], _spec_schema(platform),
              "design_spec", model=a.model)
    if not isinstance(res, dict):
        sys.exit(f"gptkit: the model's spec response is not a JSON object (non-conforming gateway?): {str(res)[:200]}")
    comps = []
    for c in res.get("components") or []:
        b = _clip_bbox(c.get("bbox"), mock.width, mock.height) if isinstance(c, dict) else None
        if b is None:
            print(f"  ! dropped component {(c.get('name') if isinstance(c, dict) else c)!r}: bbox "
                  f"{c.get('bbox') if isinstance(c, dict) else None} is not inside the {mock.width}x{mock.height} mockup")
            continue
        comps.append({"name": "", "native": "", "notes": "", **c, "bbox": b})
    res["components"] = comps
    res["platform"] = platform
    res["px_to_dp"] = round(screen_dp / mock.width, 5)
    _calibrate(d, res, m, mock.width, a.scale)
    (d / "design_spec.json").write_text(json.dumps(res, indent=2))
    print(f"-> {d / 'design_spec.json'}  (multiply px values by px_to_dp={res['px_to_dp']} for dp/pt)")


# native widgets whose platform size is fixed: (pattern on spec 'native', dp, measured bbox side)
NATIVE_ANCHORS = {
    "android": [(r"navigation ?bar", 80, "h"), (r"extended ?(fab|floating)", 56, "h"),
                (r"^(?!.*extended).*(\bfab\b|floating ?action)", 56, "h"),
                (r"top ?app ?bar(?!.*(medium|large))", 64, "h"), (r"search ?bar", 56, "h"), (r"navigation ?rail", 80, "w")],
    # iOS: UISwitch 31 pt, segmented control 32 pt, tab bar 49 pt (above the home-indicator inset), inline navigation
    # bar 44 pt; a large-title bar grows with its title, so it is never an anchor
    "ios": [(r"toggle|switch", 31, "h"), (r"segmented|picker \(segmented", 32, "h"), (r"tab ?bar", 49, "h"),
            (r"^(?!.*large).*(navigation ?stack|nav(igation)? ?bar)", 44, "h")],
}


def _calibrate(d, spec, m, width, scale="auto"):
    """Scale px -> dp from native widgets of known size; GPT draws them smaller than the platform does."""
    design_k = round((m.get("screen_dp") or _screen_dp(m.get("platform"))) / width, 5)
    spec["px_to_dp_design"] = design_k
    spec["px_to_dp"] = design_k
    anchors = []
    for c in spec.get("components") or []:
        for pat, dp, side in NATIVE_ANCHORS.get(m.get("platform") or "", []):
            px = c.get("bbox", {}).get(side) if isinstance(c, dict) and isinstance(c.get("bbox"), dict) else None
            if not isinstance(px, (int, float)) or isinstance(px, bool):
                break
            if px > 4 and re.search(pat, str(c.get("native", "")), re.I):
                anchors.append({"component": c.get("name", ""), "native": c["native"], "px": px, "dp": dp, "k": dp / px})
                break
    if not anchors:
        spec["native_scale"] = {"factor": 1.0, "anchors": [], "applied": False}
        if scale == "native":
            print("  ! --scale native: no native anchor widgets (nav bar, FAB, app bar...) found; kept the design scale")
        return
    k = float(np.median([x["k"] for x in anchors]))
    factor = k / design_k
    apply = scale == "native" or (scale == "auto" and abs(factor - 1) > 0.12)
    spec["native_scale"] = {"factor": round(factor, 3), "px_to_dp_native": round(k, 5), "anchors": anchors, "applied": apply}
    if apply:
        spec["px_to_dp"] = round(k, 5)
        spec["effective_width_dp"] = round(width * k)
        for x in m.get("assets") or []:
            if isinstance(x, dict) and isinstance(x.get("bbox"), dict) and {"w", "h"} <= x["bbox"].keys():
                x["dp"] = {"w": round(x["bbox"]["w"] * k, 1), "h": round(x["bbox"]["h"] * k, 1)}
        m["px_to_dp"] = spec["px_to_dp"]
        m["effective_width_dp"] = spec["effective_width_dp"]
        (d / "assets.json").write_text(json.dumps(m, indent=2))
        print(f"  native scale: GPT drew {', '.join(a['component'] for a in anchors[:3])} at {1 / factor:.2f}x platform size; "
              f"px_to_dp {design_k} -> {spec['px_to_dp']} (design depicts a ~{spec['effective_width_dp']}dp-wide screen; "
              "asset dp estimates updated). `spec --scale design` keeps the proportional scale.")
    elif scale == "design":
        print(f"  native scale: kept the proportional px_to_dp {design_k} (--scale design; native factor {factor:.2f})")
    else:
        print(f"  native scale: design matches platform sizes (factor {factor:.2f}); kept px_to_dp {design_k}")


def cmd_recalibrate(d, scale):
    d = Path(d)
    _need(d / "design_spec.json", "design spec", "run `spec DIR` first")
    spec = _read_json(d / "design_spec.json", "design spec")
    m = _read_json(d / "assets.json", "assets.json", required=("width",))
    if not isinstance(spec, dict) or not isinstance(m["width"], (int, float)) or m["width"] <= 0:
        sys.exit(f"gptkit: {d / 'design_spec.json'} or {d / 'assets.json'} is malformed (re-run `manifest` / `spec`)")
    _calibrate(d, spec, m, m["width"], scale)
    (d / "design_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"-> {d / 'design_spec.json'}  px_to_dp={spec['px_to_dp']}")


FIX_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["differences", "edit_instruction"],
              "properties": {"differences": {"type": "array", "items": {"type": "string"}},
                             "edit_instruction": {"type": "string"}}}

_warned_format = set()  # image formats a provider returned instead of PNG, announced once per process


def _png_verified(raw):
    """(png_bytes, source_format): the image result decoded with PIL and re-encoded as PNG when it was not one
    (belt and braces with the API layer). ValueError names undecodable bytes (an HTML error page, a cut-off download)."""
    try:
        im = Image.open(io.BytesIO(raw or b""))
        im.load()
    except Exception as e:  # PIL raises UnidentifiedImageError/OSError/ValueError/SyntaxError
        head = (raw or b"")[:40]
        raise ValueError(f"the provider returned no decodable image ({type(e).__name__}; starts with {head!r})") from None
    fmt = im.format or "?"
    if fmt == "PNG":
        return raw, fmt
    if fmt not in _warned_format:
        _warned_format.add(fmt)
        print(f"  ! provider returned {fmt} instead of PNG: no alpha channel, background/output_format were ignored by "
              "this endpoint/model (the file is re-encoded as PNG)", file=sys.stderr)
    return _png_bytes(im.convert("RGBA")), fmt


def _forge_job(mock, asset, cand, model, vary=True):
    b = asset["bbox"]
    colors = f" ({', '.join(asset['colors'])})" if asset.get("colors") else ""
    if asset["kind"] in ("background", "photo") and not asset["transparent"]:
        # the region's own crop kept the design's composition closer than the full mockup did
        prompt = BACKGROUND_PROMPT.format(kind=asset["kind"], name=asset["name"], description=asset["description"],
                                          aspect=_aspect_words(b["w"], b["h"]))
        ref = ref_crop(mock, b, margin=0.02, long_side=1536)
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]), "opaque", model)
    else:
        ref = ref_crop(mock, b, center=True)
        bg = ("Transparent background (real alpha). Centered with about 10% empty padding on every side."
              if asset["transparent"] else
              "Fill the canvas edge to edge with the element's own artwork (no letterbox, no empty padding).")
        prompt = ASSET_PROMPT.format(kind=asset["kind"], name=asset["name"].replace("_", " "),
                                     description=asset["description"], colors=colors, bg=bg)
        prompt += CANDIDATE_FOCUS[cand % len(CANDIDATE_FOCUS)] if vary else ""
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", model)
    img, fmt = _png_verified(img)
    meta = dict(meta or {})
    meta.setdefault("source_format", fmt)
    return asset["name"], cand, img, meta, ref


# ---- forge_report.json: dir-relative paths, re-read + merge + atomic replace (parallel forge/fix/pick runs are safe)

def _rep_rel(d, p):
    """A path as stored in forge_report.json: relative to the project dir when it lies inside it."""
    try:
        return Path(p).resolve().relative_to(Path(d).resolve()).as_posix()
    except ValueError:
        return str(p)


def _rep_abs(d, p):
    """A stored report path resolved: d/path (current reports), as-is (older cwd-relative reports), d/raw/<basename>."""
    d = Path(d)
    for c in (d / p, Path(p), d / "raw" / Path(p).name):
        if c.exists():
            return c
    return d / p


def _read_report(d):
    p = Path(d) / "forge_report.json"
    if not p.exists():
        return {}
    r = _read_json(p, "forge report")
    r = r if isinstance(r, dict) else {}
    ok = lambda e: isinstance(e, dict) and isinstance(e.get("candidates", []), list) and all(
        isinstance(c, dict) and isinstance(c.get("path"), str) for c in e.get("candidates", []))
    bad = [n for n, e in r.items() if not ok(e)]
    if bad:  # hand-edited or truncated entries: treated as never forged instead of crashing forge/fix/pick
        print(f"  ! {p}: ignoring damaged entries: {', '.join(bad)} (forge {d} --only {','.join(bad)} --force redoes them)")
    return {n: e for n, e in r.items() if n not in bad}


def _update_report(d, entries):
    """Merge entries into forge_report.json: re-read, replace only these names, write a temp file, os.replace."""
    p = Path(d) / "forge_report.json"
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text())
            cur = cur if isinstance(cur, dict) else {}
        except ValueError:
            print(f"  ! {p} was not valid JSON; rewritten with this run's entries")
    cur.update(entries)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, p)
    return cur


def _entry(d, report, name):
    """The report entry of a forged asset; exits when forge never produced a chosen candidate for it."""
    e = report.get(name)
    if isinstance(e, dict) and e.get("chosen") and not (isinstance(e["chosen"], dict) and isinstance(e["chosen"].get("path"), str)):
        sys.exit(f"gptkit: {Path(d) / 'forge_report.json'}: the '{name}' entry's chosen candidate is damaged "
                 f"({e['chosen']!r:.60}); run pick {d} {name} N or forge {d} --only {name} --force")
    if not isinstance(e, dict) or not e.get("chosen"):
        sys.exit(f"gptkit: no forged candidate for '{name}' in {Path(d) / 'forge_report.json'}: run forge {d} --only {name} first")
    return e


def _current(d, name, entry):
    """Path of the image behind an asset's final file: the colour-matched copy when one was kept, else the chosen
    candidate; falls back to assets/<name>.png with a warning when the raw file is gone."""
    cm = entry.get("color_match") or {}
    p = _rep_abs(d, cm["path"]) if cm.get("applied") and cm.get("path") else _rep_abs(d, entry["chosen"]["path"])
    if p.exists():
        return p
    alt = Path(d) / "assets" / f"{name}.png"
    if alt.exists():
        print(f"  ! {name}: {p} is gone (raw/ cleaned?); using {alt}")
        return alt
    sys.exit(f"gptkit: {name}: neither {p} nor {alt} exists; run forge {d} --only {name} --force")


def _color_final(mock, asset, path, sc, strength, d):
    """Colour-match a chosen candidate into d/raw/<stem>_cm.png, kept only if it scores better without new flags."""
    # opaque backgrounds are skipped: UI drawn over them skews the stats and desaturated the accents in tests
    if strength <= 0 or not asset["transparent"]:
        return None
    if sc["flags"]:
        print(f"  ~ {asset['name']}: colour match skipped (flagged {' '.join(sc['flags'])})")
        return None
    src = _rgba(path)
    img = color_match(mock, asset["bbox"], src, strength)
    if img is src:
        print(f"  ~ {asset['name']}: colour match skipped (too few opaque pixels to match)")
        return None
    new = score(mock, asset["bbox"], img, None, True)
    info = {"strength": strength, "fid_before": sc["fid"], "fid_after": new["fid"],
            "applied": new["fid"] < sc["fid"] and not new["flags"]}
    if new["flags"]:
        info["note"] = "introduced " + " ".join(new["flags"])
    if info["applied"]:
        p = Path(d) / "raw" / f"{Path(path).stem}_cm.png"
        img.save(p)
        info.update(path=_rep_rel(d, p), score=new)
    return info


def _finalize(asset, img):
    if asset["transparent"]:
        return trim(img, pad=0.02)
    b = asset["bbox"]
    tw, th = img.size
    target = b["w"] / b["h"]
    if tw / th > target:
        nw = round(th * target)
        return img.crop(((tw - nw) // 2, 0, (tw - nw) // 2 + nw, th))
    nh = round(tw / target)
    return img.crop((0, (th - nh) // 2, tw, (th - nh) // 2 + nh))


def _design_tile(mock, bbox, T=220):
    """The design element with 25% context for the review sheet: tiny elements upscaled NEAREST, the bbox outlined."""
    b = bbox
    pad = int(max(b["w"], b["h"]) * 0.25)
    box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(mock.width, b["x"] + b["w"] + pad), min(mock.height, b["y"] + b["h"] + pad))
    t = mock.crop(box).convert("RGBA")
    k = 1
    if max(t.size) < T // 2:
        k = max(1, (T // 2) // max(t.size))
        t = t.resize((t.width * k, t.height * k), Image.NEAREST)
    ImageDraw.Draw(t).rectangle([(b["x"] - box[0]) * k, (b["y"] - box[1]) * k, (b["x"] + b["w"] - box[0]) * k - 1,
                                 (b["y"] + b["h"] - box[1]) * k - 1], outline="#FF00FF", width=max(1, k // 2))
    return t


def _mid(s, n):
    return s if len(s) <= n else s[:n // 2 - 1] + "~" + s[-(n - n // 2):]


def _cand_label(c, best):
    return f"{'BEST ' if c is best else ''}c{c.get('cand', '?')} fid={c['score']['fid']} {' '.join(c['score']['flags'])}".rstrip()


def _incomplete(e):
    return isinstance(e, dict) and not e.get("complete", bool(e.get("chosen")))


def cmd_forge(a):
    d, m, mock = _load(a.dir)
    todo = [x for x in _select(m, a.only) if x["format"] == "png" or a.include_svg]
    if not todo:
        sys.exit(f"forge: no png-format assets{' among ' + a.only if a.only else ''} in {d / 'assets.json'} "
                 "(svg-format ones: `svg`, or forge --include-svg)")
    report = _read_report(d)
    keep = {}  # name -> already scored candidates of an interrupted run (their raw files still exist)
    if not a.force:
        rest = []
        for x in todo:
            e = report.get(x["name"]) or {}
            if (d / "assets" / f"{x['name']}.png").exists() and not _incomplete(e) and not e.get("needs_attention"):
                continue
            if _incomplete(e):
                keep[x["name"]] = [c for c in e.get("candidates", [])
                                   if isinstance(c.get("cand"), int) and _rep_abs(d, c["path"]).exists()]
            rest.append(x)
        if not rest:
            print("nothing to forge: every asset has a final file without flags (use --force to redo, --only NAME for one)")
            return
        todo = rest
    (d / "raw").mkdir(exist_ok=True)
    (d / "assets").mkdir(exist_ok=True)
    by_name = {x["name"]: x for x in todo}
    results = {x["name"]: list(keep.get(x["name"], [])) for x in todo}
    jobs = [(x, c) for x in todo for c in range(a.candidates) if c + 1 not in {k["cand"] for k in results[x["name"]]}]
    _plan(len(jobs), 0, image_model=a.model, task="precise", background="transparent", edits=True)
    if getattr(a, "dry_run", False):
        print("would render: " + (", ".join(f"{x['name']} c{c + 1}" for x, c in jobs) or "nothing"))
        return
    pending = {x["name"]: sum(1 for j, _ in jobs if j is x) for x in todo}
    reused = sum(len(v) for v in results.values())
    print(f"forging {len(todo)} assets x {a.candidates} candidates = {len(jobs)} image calls, {a.jobs} parallel"
          + (f" ({reused} scored candidates of an interrupted run reused)" if reused else ""))
    rows, nocand, unusable, flagged = [], [], [], []

    def finish(name):
        asset, cands = by_name[name], results[name]
        entry = dict(report.get(name) or {}, candidates=cands, complete=True, dp=asset.get("dp"))
        cm = None
        if not cands:
            entry.update(chosen=None, color_match=None, final=None, needs_attention=True)
            nocand.append(name)
        else:
            cands.sort(key=lambda c: _rank(c["score"], c["meta"]))
            best = cands[0]
            entry.update(chosen=best, picked_manually=False)
            bad = _unusable(best["score"])
            if bad:
                entry.update(color_match=None, final=None, needs_attention=True, unusable=bad)
                print(f"  ! {name}: no usable candidate (the best is {bad}) - forge {d} --only {name} --force --candidates 3")
                unusable.append(name)
            else:
                entry.pop("unusable", None)
                cm = _color_final(mock, asset, _rep_abs(d, best["path"]), best["score"], a.color_match, d)
                entry["color_match"] = cm
                if cm:
                    print(f"  {name}: colour match {cm['strength']} fid {cm['fid_before']} -> {cm['fid_after']} "
                          f"({'kept' if cm['applied'] else 'discarded'}{'; ' + cm['note'] if cm.get('note') else ''})")
                final = _finalize(asset, _rgba(_current(d, name, entry)))
                out = d / "assets" / f"{name}.png"
                final.save(out)
                entry.update(final=_rep_rel(d, out), final_size=list(final.size), needs_attention=bool(best["score"]["flags"]))
                if entry["needs_attention"]:
                    flagged.append(name)
            rows.append([(f"{_mid(name, 24)} (design)", _design_tile(mock, asset["bbox"]))]
                        + [(_cand_label(c, best), _rgba(_rep_abs(d, c["path"]))) for c in cands]
                        + ([(f"BEST+colour fid={cm['fid_after']}", _rgba(_rep_abs(d, cm["path"])))] if cm and cm["applied"] else []))
        report[name] = entry
        _update_report(d, {name: entry})

    for name in [n for n, k in pending.items() if k == 0]:
        finish(name)
    ex = cf.ThreadPoolExecutor(max(1, min(a.jobs, len(jobs))))
    futs = {ex.submit(_forge_job, mock, x, c, a.model, not a.same_prompt): (x["name"], c + 1) for x, c in jobs}
    try:
        for f in cf.as_completed(futs):
            name, cn = futs[f]
            try:
                name, cand, raw, meta, ref = f.result()
                asset = by_name[name]
                p = d / "raw" / f"{name}_c{cand + 1}.png"
                p.write_bytes(raw)
                sc = score(mock, asset["bbox"], _rgba(p), ref, asset["transparent"])
                results[name].append({"cand": cand + 1, "path": _rep_rel(d, p), "score": sc,
                                      "meta": {k: meta.get(k) for k in ("quality", "size", "source_format")}})
                print(f"  {name} c{cand + 1}: fid={sc['fid']} (diff={sc['diff']} iou={sc['iou']} de={sc['de']} col={sc['col']}) "
                      f"flags={' '.join(sc['flags']) or '-'} warn={' '.join(sc['warn']) or '-'} quality={meta.get('quality')}")
            except Exception as e:  # ApiError (carries the prompt after the API-layer fix), undecodable bytes, scoring
                print(f"  ! {name} c{cn} failed: {e}")
            pending[name] -= 1
            if pending[name] == 0:
                finish(name)
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        partial = {n: dict(report.get(n) or {}, candidates=results[n], complete=False) for n, k in pending.items() if k > 0}
        if partial:
            _update_report(d, partial)
        print(f"\n  ! interrupted: {sum(len(v['candidates']) for v in partial.values())} scored candidates of "
              f"{', '.join(partial) or 'no asset'} kept; re-run forge {d} to draw only the missing ones")
        raise  # main() exits 130 without waiting for the calls still in flight
    except BaseException:  # a fatal error in finish(): do not start the queued image calls before exiting
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    if rows:
        (d / "review").mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i in range(0, len(rows), 6):
            print("review sheet ->", sheet(rows[i:i + 6], d / "review" / f"forge_{stamp}_{i // 6 + 1}.png"))
    bad = nocand + unusable
    ok = [n for n in results if n not in bad]
    print(f"forged {len(ok)}/{len(results)} assets" + (f"; failed: {', '.join(bad)} (re-run: forge {d} --only {','.join(bad)})" if bad else ""))
    if flagged:
        print(f"NEEDS ATTENTION (every candidate flagged): {', '.join(flagged)} -> read review/forge_*.png, then "
              f"pick {d} <name> <N>, fix {d} <name>, or forge {d} --only {','.join(flagged)} --force --candidates 3")
    if bad:
        sys.exit(1)


def cmd_fix(a):
    d, m, mock = _load(a.dir)
    asset = _asset(m, a.name)
    report = _read_report(d)
    entry = _entry(d, report, a.name)
    cur = _rgba(_current(d, a.name, entry))
    cur_score = score(mock, asset["bbox"], cur, None, asset["transparent"])
    b = asset["bbox"]
    is_bg = asset["kind"] in ("background", "photo") and not asset["transparent"]
    for rnd in range(a.rounds):
        T = 512
        pad = 6
        tgt = mock.crop((max(0, b["x"] - pad), max(0, b["y"] - pad), min(mock.width, b["x"] + b["w"] + pad),
                         min(mock.height, b["y"] + b["h"] + pad)))
        tgt = tgt.resize((max(1, round(tgt.width * T / tgt.height)), T), Image.LANCZOS).convert("RGB")
        g = trim(cur)
        cv = Image.new("RGBA", g.size, _backdrop(mock, b) + (255,))
        cv.alpha_composite(g)
        cv = cv.resize((max(1, round(cv.width * T / cv.height)), T), Image.LANCZOS).convert("RGB")
        sbs = Image.new("RGB", (tgt.width + cv.width + 30, T), "white")
        sbs.paste(tgt, (0, 0))
        sbs.paste(cv, (tgt.width + 30, 0))
        extra = f" The user also says: {a.note}" if a.note else ""
        left = ("LEFT is the design region with UI (text, buttons, cards) drawn on top: ignore all UI; compare only the "
                "artwork behind it. RIGHT is the recreated full-bleed artwork. " if is_bg else
                "LEFT is the target graphic from an approved design (low-res screenshot on its backdrop). RIGHT is a "
                "recreated asset on a similar backdrop. ")
        res = ask([left + "List every visible difference RIGHT must fix to match LEFT 1:1 (silhouette, proportions, "
                   "parts, colours with hex, gradients, glow, relative sizes). Ignore resolution/sharpness"
                   + (" and the backdrop" if not is_bg else " and the UI") + ". Then write ONE concise corrective edit "
                   f"instruction (<=100 words) limited to the real differences; both empty if nothing differs.{extra}",
                   sbs], FIX_SCHEMA, "fix")
        diffs = [str(x).strip() for x in (res.get("differences") or []) if str(x).strip()]
        instruction = str(res.get("edit_instruction") or "").strip()
        if not diffs or not instruction:
            print(f"round {rnd + 1}: no visible differences - nothing to fix")
            break
        print(f"round {rnd + 1} differences:", *[f"  - {x}" for x in diffs], sep="\n")
        head = "Keep everything that already matches; change only what is listed. "
        if is_bg:
            prompt = ("Revise the attached full-bleed background artwork. " + head +
                      "Clean high-resolution redraw; keep it full-bleed, no UI, no text. Corrections: ")
        elif asset["transparent"]:
            prompt = ("Revise the attached standalone asset. " + head +
                      "Clean high-resolution redraw, transparent background, no backdrop, no text. Corrections: ")
        else:
            prompt = ("Revise the attached standalone asset. " + head + "Clean high-resolution redraw. Fill the canvas "
                      "edge to edge with the element's own artwork (no letterbox), no text. Corrections: ")
        raw, meta = edit(_png_bytes(cur), prompt + instruction, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", a.model)
        raw, fmt = _png_verified(raw)
        n = 1 + sum(1 for c in entry.get("candidates", []) if str(c.get("cand", "")).startswith("fix"))
        p = d / "raw" / f"{a.name}_fix{n}.png"
        while p.exists():
            n += 1
            p = d / "raw" / f"{a.name}_fix{n}.png"
        p.write_bytes(raw)
        new = _rgba(p)
        sc = score(mock, b, new, None, asset["transparent"])
        cm = _color_final(mock, asset, p, sc, a.color_match, d)
        final_img, final_sc = (_rgba(_rep_abs(d, cm["path"])), cm["score"]) if cm and cm["applied"] else (new, sc)
        bad = _unusable(final_sc)
        better = _rank(final_sc, meta) < _rank(cur_score, None)
        kept = (better or a.accept) and not bad
        cand = {"cand": f"fix{n}", "path": _rep_rel(d, p), "score": sc,
                "meta": {"quality": (meta or {}).get("quality"), "size": (meta or {}).get("size"), "source_format": fmt}}
        entry.setdefault("candidates", []).append(cand)
        why = "KEPT" if kept else (f"discarded ({bad})" if bad else "discarded (worse)")
        print(f"round {rnd + 1}: fid {cur_score['fid']} -> {final_sc['fid']} (diff {cur_score['diff']} -> {final_sc['diff']}"
              f"{', colour-matched' if final_img is not new else ''}) flags={' '.join(sc['flags']) or '-'} -> {why}  ({p})")
        if kept:
            cur, cur_score = final_img, final_sc
            entry.update(chosen=cand, color_match=cm, picked_manually=False)
            final = _finalize(asset, final_img)
            out = d / "assets" / f"{a.name}.png"
            final.save(out)
            entry.update(final=_rep_rel(d, out), final_size=list(final.size), needs_attention=bool(sc["flags"]), complete=True)
            entry.pop("unusable", None)
        _update_report(d, {a.name: entry})


_warned_render = False


@functools.cache
def _cairosvg():
    try:
        import cairosvg
        return cairosvg
    except Exception:  # missing package or missing libcairo
        return None


@functools.cache
def _renderer_name():
    """The SVG rasteriser that works here: chrome (only when the probe render succeeds), cairosvg, rsvg-convert,
    qlmanage, or None."""
    if _chrome() and _chrome_probe()[0]:
        return "chrome"
    if _cairosvg():
        return "cairosvg"
    return next((t for t in ("rsvg-convert", "qlmanage") if shutil.which(t)), None)


def _ensure_xmlns(txt):
    """SVG text with xmlns="http://www.w3.org/2000/svg" on the root (browsers ignore an <img> SVG without it)."""
    m_ = re.search(r"<svg\b[^>]*>", txt)
    if m_ and not re.search(r"""\sxmlns\s*=\s*["']http://www\.w3\.org/2000/svg["']""", m_.group(0)):
        return txt[:m_.start()] + m_.group(0).replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1) + txt[m_.end():]
    return txt


def _sized_svg(txt, size):
    """The SVG with width/height set to `size` (ElementTree round-trip: single quotes and duplicate attributes are safe)."""
    root = ET.fromstring(_ensure_xmlns(txt))
    root.set("width", str(size))
    root.set("height", str(size))
    return ET.tostring(root, encoding="unicode")


_warned_chrome = False


def _chrome_render(svgs, size):
    """Rasterise sized SVG texts with headless Chrome, one grid page per <= 256 icons; [RGBA Image or None]."""
    global _warned_chrome
    out = []
    per = max(1, min(256, (2048 // size) * (8192 // size)))  # keep each page under ~2048 x 8192 px
    for k in range(0, len(svgs), per):
        chunk = svgs[k:k + per]
        cols = max(1, min(len(chunk), 2048 // size))
        rows = math.ceil(len(chunk) / cols)
        W, H = cols * size, rows * size
        cells = "".join(f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}" style="position:absolute;'
                        f'left:{i % cols * size}px;top:{i // cols * size}px;width:{size}px;height:{size}px">'
                        for i, s in enumerate(chunk))
        grid = None
        with tempfile.TemporaryDirectory() as td:
            html, png = Path(td) / "r.html", Path(td) / "r.png"
            html.write_text(f"<!doctype html><style>html,body{{margin:0;background:transparent}}</style>{cells}")
            try:
                r = subprocess.run([_chrome(), *_chrome_flags(W, H), "--virtual-time-budget=3000", f"--screenshot={png}",
                                    html.as_uri()], capture_output=True, text=True, timeout=120)
                err = _chrome_err(r) if not png.exists() else ""
            except (OSError, subprocess.TimeoutExpired) as e:
                err = f"{type(e).__name__}: {e}"
            if png.exists():
                grid = Image.open(png).convert("RGBA").crop((0, 0, W, H))
        if grid is None:
            if not _warned_chrome:
                print(f"  ! headless Chrome produced no screenshot: {err}")
                _warned_chrome = True
            out += [None] * len(chunk)
            continue
        out += [grid.crop((i % cols * size, i // cols * size, i % cols * size + size, i // cols * size + size))
                for i in range(len(chunk))]
    return out


def _matte(on_white, on_black):
    """Recover RGBA from two opaque renders on white and on black."""
    w = np.asarray(on_white.convert("RGB").resize(on_black.size), dtype=float)
    b = np.asarray(on_black.convert("RGB"), dtype=float)
    a = np.clip(255 - (w - b).mean(2), 0, 255)
    rgb = np.clip(b * 255 / np.maximum(a, 1)[..., None], 0, 255)
    return Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")


def _render_one(sized, size):
    if _cairosvg():
        try:
            return _rgba(io.BytesIO(_cairosvg().svg2png(bytestring=sized.encode(), output_width=size, output_height=size)))
        except Exception as e:
            print(f"  ! cairosvg failed: {str(e).splitlines()[0]}")
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "i.svg", Path(td) / "o.png"
        if shutil.which("rsvg-convert"):
            src.write_text(sized)
            subprocess.run(["rsvg-convert", "-w", str(size), "-h", str(size), "-o", str(out), str(src)], capture_output=True)
            return _rgba(out) if out.exists() else None
        if shutil.which("qlmanage"):
            # Quick Look has no alpha and pads/clips its thumbnail: render smaller on white and black, matte, re-frame
            vb = re.search(r'viewBox="([^"]+)"', sized)
            x, y, w, h = [float(v) for v in re.split(r"[\s,]+", vb.group(1).strip())] if vb else (0, 0, size, size)
            small = _sized_svg(sized, int(size * 0.75))
            shots = []
            for bg in ("#ffffff", "#000000"):
                src.write_text(re.sub(r"(<svg\b[^>]*>)", rf'\1<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{bg}"/>',
                                      small, count=1))
                subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", td, str(src)], capture_output=True)
                ql = Path(td) / "i.svg.png"
                if not ql.exists():
                    return None
                shots.append(Image.open(ql).convert("RGB"))
                ql.unlink()
            frame = ImageChops.difference(*shots).convert("L").point(lambda v: 255 if v > 128 else 0).getbbox()
            if not frame:
                return None
            frame = (frame[0] + 1, frame[1] + 1, frame[2] - 1, frame[3] - 1)
            return _matte(shots[0].crop(frame), shots[1].crop(frame)).resize((size, size), Image.LANCZOS)
    return None


def render_svgs(svgs, size=256):
    """Rasterise SVG texts with real alpha (headless Chrome, cairosvg, rsvg-convert, or Quick Look + matting);
    one Image or None per input (unparseable SVG, or no working renderer: said once)."""
    global _warned_render
    sized = []
    for s in svgs:
        try:
            sized.append(_sized_svg(s, size) if s else None)
        except (ET.ParseError, AttributeError, TypeError):
            sized.append(None)
    ok = [i for i, s in enumerate(sized) if s]
    out = [None] * len(svgs)
    if not ok:
        return out
    name = _renderer_name()
    if name == "chrome":
        for i, im in zip(ok, _chrome_render([sized[i] for i in ok], size)):
            out[i] = im
        return out
    if not name:
        if not _warned_render:
            print(f"  (no SVG renderer: {_chrome_probe()[1]}; install Chrome/Chromium (GPTKIT_CHROME) or pip install cairosvg)")
            _warned_render = True
        return out
    for i in ok:
        out[i] = _render_one(sized[i], size)
    return out


def _render_svg(svg_path, out_dir, size=256):
    """Rasterise one SVG file to out_dir/<stem>.png with transparency; returns the path or None."""
    svg_path, out_dir = Path(svg_path), Path(out_dir)
    im = render_svgs([svg_path.read_text()], size)[0]
    if im is None:
        return None
    final = out_dir / f"{svg_path.stem}.png"
    im.save(final)
    return final




def _element_mask(mock, bbox):
    """Segment a design element from its backdrop: (soft alpha, backdrop rgb, crop rgb) for bbox plus a margin."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    m = max(3, round(max(w, h) * 0.12))
    c = np.asarray(mock.crop((max(0, x - m), max(0, y - m), min(mock.width, x + w + m),
                              min(mock.height, y + h + m))).convert("RGB"), dtype=float)
    ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    bg = np.median(ring, 0)
    dist = np.linalg.norm(c - bg, axis=2)
    noise = np.percentile(np.linalg.norm(ring - bg, axis=1), 90)
    core = max(np.percentile(dist, 99), noise + 20)
    return np.clip((dist - noise) / (core - noise), 0, 1), bg, c


def _abox(a, thr=0.3):
    ys, xs = np.nonzero(a > thr)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def svg_fidelity(mock, bbox, render, visual=False):
    """Registered comparison of an SVG render with the design element (rel: 0 = perfect, 1 = as bad as empty)."""
    ad, bg, crop = _element_mask(mock, bbox)
    z = max(1.0, 96 / max(ad.shape))
    if z > 1:
        size = (round(ad.shape[1] * z), round(ad.shape[0] * z))
        crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize(size, Image.BICUBIC), dtype=float)
        ad = np.asarray(Image.fromarray((ad * 255).astype(np.uint8)).resize(size, Image.BICUBIC), dtype=float) / 255
    db, rb = _abox(ad), _abox(np.asarray(render.getchannel("A"), dtype=float) / 255)
    if not db or not rb:
        sc = {"fit": 0.0, "iou": 0.0, "ink": 0.0, "dE": 99.0, "diff": 99.0, "rel": 1.0, "aspect_err": 1.0,
              "design_hex": "", "svg_hex": "", "backdrop_hex": ""}
        return (sc, Image.fromarray(crop.astype(np.uint8))) if visual else sc
    dw, dh = db[2] - db[0], db[3] - db[1]
    g0 = render.crop(rb)
    s0 = min(dw / g0.width, dh / g0.height)
    H, W = ad.shape
    r = max(1, round(z))
    best = None
    for f in (0.95, 0.975, 1.0, 1.025, 1.05):
        g = g0.resize((max(1, round(g0.width * s0 * f)), max(1, round(g0.height * s0 * f))), Image.BOX)
        ga = g.getchannel("A")
        x0, y0 = round((db[0] + db[2] - g.width) / 2), round((db[1] + db[3] - g.height) / 2)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                m_ = Image.new("L", (W, H), 0)
                m_.paste(ga, (x0 + dx, y0 + dy))
                ar = np.asarray(m_, dtype=float) / 255
                iou = np.minimum(ad, ar).sum() / max(np.maximum(ad, ar).sum(), 1e-6)
                if best is None or iou > best[0]:
                    best = (iou, g, (x0 + dx, y0 + dy))
    iou, g, pos = best
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(g, pos) if pos[0] >= 0 and pos[1] >= 0 else layer.paste(g, pos)
    lr = np.asarray(layer, dtype=float)
    ar = lr[..., 3] / 255
    k = 2 * max(1, round(max(dw, dh) / 48)) + 1
    grow = lambda a: np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k)), dtype=float) / 255
    prec = np.minimum(ar, grow(ad)).sum() / max(ar.sum(), 1e-6)
    rec = np.minimum(ad, grow(ar)).sum() / max(ad.sum(), 1e-6)
    fit = 2 * prec * rec / max(prec + rec, 1e-6)
    comp = lr[..., :3] * ar[..., None] + bg * (1 - ar[..., None])
    core_d, core_r = ad > 0.85, ar > 0.85
    if core_d.sum() >= 3 and core_r.sum() >= 3:
        cd, cr = np.median(crop[core_d], 0), np.median(lr[..., :3][core_r], 0)
    else:
        cd, cr = np.average(crop.reshape(-1, 3), 0, ad.ravel() + 1e-6), np.average(lr[..., :3].reshape(-1, 3), 0, ar.ravel() + 1e-6)
    hexc = lambda c: "#" + "".join(f"{int(round(v)):02X}" for v in c)
    err = np.abs(comp - crop).mean()
    sc = {"fit": round(float(fit), 3), "iou": round(float(iou), 3), "ink": round(float(ar.sum() / max(ad.sum(), 1e-6)), 2),
          "dE": round(float(np.linalg.norm(_lab(cd) - _lab(cr))), 1), "diff": round(float(err), 1),
          "rel": round(float(err / max(np.abs(bg - crop).mean(), 1e-6)), 3),
          "aspect_err": round(float(abs((rb[2] - rb[0]) / (rb[3] - rb[1]) / (dw / dh) - 1)), 3),
          "design_hex": hexc(cd), "svg_hex": hexc(cr), "backdrop_hex": hexc(bg)}
    if not visual:
        return sc
    T = 256
    left, right = (Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)) for x in (crop, comp))
    left, right = (im.resize((round(im.width * T / im.height), T), Image.LANCZOS) for im in (left, right))
    sbs = Image.new("RGB", (left.width * 2 + 16, T), "white")
    sbs.paste(left, (0, 0))
    sbs.paste(right, (left.width + 16, 0))
    return sc, sbs


# ---- SVG -> Android VectorDrawable

_PARAMS = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7, "Z": 0}
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
PRES_ATTRS = ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "fill-rule",
              "fill-opacity", "stroke-opacity", "stroke-dasharray", "color", "display", "visibility", "clip-rule",
              "marker-start", "marker-mid", "marker-end", "vector-effect", "paint-order")
NAMED_COLORS = {"transparent": "#00000000", **{k: "#" + v for k, v in (kv.split("=") for kv in (
    "aliceblue=f0f8ff antiquewhite=faebd7 aqua=00ffff aquamarine=7fffd4 azure=f0ffff beige=f5f5dc bisque=ffe4c4 black=000000 "
    "blanchedalmond=ffebcd blue=0000ff blueviolet=8a2be2 brown=a52a2a burlywood=deb887 cadetblue=5f9ea0 chartreuse=7fff00 "
    "chocolate=d2691e coral=ff7f50 cornflowerblue=6495ed cornsilk=fff8dc crimson=dc143c cyan=00ffff darkblue=00008b "
    "darkcyan=008b8b darkgoldenrod=b8860b darkgray=a9a9a9 darkgreen=006400 darkgrey=a9a9a9 darkkhaki=bdb76b darkmagenta=8b008b "
    "darkolivegreen=556b2f darkorange=ff8c00 darkorchid=9932cc darkred=8b0000 darksalmon=e9967a darkseagreen=8fbc8f "
    "darkslateblue=483d8b darkslategray=2f4f4f darkslategrey=2f4f4f darkturquoise=00ced1 darkviolet=9400d3 deeppink=ff1493 "
    "deepskyblue=00bfff dimgray=696969 dimgrey=696969 dodgerblue=1e90ff firebrick=b22222 floralwhite=fffaf0 forestgreen=228b22 "
    "fuchsia=ff00ff gainsboro=dcdcdc ghostwhite=f8f8ff gold=ffd700 goldenrod=daa520 gray=808080 green=008000 greenyellow=adff2f "
    "grey=808080 honeydew=f0fff0 hotpink=ff69b4 indianred=cd5c5c indigo=4b0082 ivory=fffff0 khaki=f0e68c lavender=e6e6fa "
    "lavenderblush=fff0f5 lawngreen=7cfc00 lemonchiffon=fffacd lightblue=add8e6 lightcoral=f08080 lightcyan=e0ffff "
    "lightgoldenrodyellow=fafad2 lightgray=d3d3d3 lightgreen=90ee90 lightgrey=d3d3d3 lightpink=ffb6c1 lightsalmon=ffa07a "
    "lightseagreen=20b2aa lightskyblue=87cefa lightslategray=778899 lightslategrey=778899 lightsteelblue=b0c4de lightyellow=ffffe0 "
    "lime=00ff00 limegreen=32cd32 linen=faf0e6 magenta=ff00ff maroon=800000 mediumaquamarine=66cdaa mediumblue=0000cd "
    "mediumorchid=ba55d3 mediumpurple=9370db mediumseagreen=3cb371 mediumslateblue=7b68ee mediumspringgreen=00fa9a "
    "mediumturquoise=48d1cc mediumvioletred=c71585 midnightblue=191970 mintcream=f5fffa mistyrose=ffe4e1 moccasin=ffe4b5 "
    "navajowhite=ffdead navy=000080 oldlace=fdf5e6 olive=808000 olivedrab=6b8e23 orange=ffa500 orangered=ff4500 orchid=da70d6 "
    "palegoldenrod=eee8aa palegreen=98fb98 paleturquoise=afeeee palevioletred=db7093 papayawhip=ffefd5 peachpuff=ffdab9 "
    "peru=cd853f pink=ffc0cb plum=dda0dd powderblue=b0e0e6 purple=800080 rebeccapurple=663399 red=ff0000 rosybrown=bc8f8f "
    "royalblue=4169e1 saddlebrown=8b4513 salmon=fa8072 sandybrown=f4a460 seagreen=2e8b57 seashell=fff5ee sienna=a0522d "
    "silver=c0c0c0 skyblue=87ceeb slateblue=6a5acd slategray=708090 slategrey=708090 snow=fffafa springgreen=00ff7f "
    "steelblue=4682b4 tan=d2b48c teal=008080 thistle=d8bfd8 tomato=ff6347 turquoise=40e0d0 violet=ee82ee wheat=f5deb3 "
    "white=ffffff whitesmoke=f5f5f5 yellow=ffff00 yellowgreen=9acd32").split())}}


def parse_path(d):
    """SVG path data -> [(cmd, [floats])], one entry per segment (implicit repeats and compact arc flags expanded)."""
    out, i, cmd, n = [], 0, None, len(d or "")
    while True:
        while i < n and (d[i].isspace() or d[i] == ","):
            i += 1
        if i >= n:
            return out
        if d[i].isalpha():
            cmd = d[i]
            i += 1
            if cmd.upper() not in _PARAMS:
                raise ValueError(f"unknown path command {cmd!r}")
            if cmd in "Zz":
                out.append((cmd, []))
                continue
        elif cmd is None or cmd in "Zz":
            raise ValueError(f"path data has numbers without a command near {d[i:i + 12]!r}")
        vals = []
        for j in range(_PARAMS[cmd.upper()]):
            while i < n and (d[i].isspace() or d[i] == ","):
                i += 1
            if cmd in "Aa" and j in (3, 4):
                if i >= n or d[i] not in "01":
                    raise ValueError(f"bad arc flag near {d[i:i + 12]!r}")
                vals.append(float(d[i]))
                i += 1
                continue
            m_ = _NUM.match(d, i)
            if not m_:
                raise ValueError(f"expected a number near {d[i:i + 12]!r}")
            vals.append(float(m_.group()))
            i = m_.end()
        out.append((cmd, vals))
        cmd = {"M": "L", "m": "l"}.get(cmd, cmd)


def _fmt(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path_str(segs):
    return " ".join(c + " ".join(_fmt(v) for v in vals) for c, vals in segs)


def transform_path(segs, s, tx=0.0, ty=0.0):
    """Scale by s and translate every coordinate of parsed path segments."""
    out = []
    for idx, (c, v) in enumerate(segs):
        rel = c.islower()
        if idx == 0 and c == "m":
            c, rel = "M", False
        U = c.upper()
        if U == "A":
            v = [v[0] * s, v[1] * s, v[2], v[3], v[4]] + ([v[5] * s, v[6] * s] if rel else [v[5] * s + tx, v[6] * s + ty])
        elif U == "H":
            v = [v[0] * s + (0 if rel else tx)]
        elif U == "V":
            v = [v[0] * s + (0 if rel else ty)]
        else:
            v = [x * s + (0 if rel else (tx if j % 2 == 0 else ty)) for j, x in enumerate(v)]
        out.append((c, v))
    return out


def _num(v, default=0.0):
    if v is None:
        return default
    return float(re.sub(r"[a-z%]+$", "", str(v).strip()) or default)


def _style_dict(el):
    return {k.strip(): v.strip() for k, _, v in (kv.partition(":") for kv in (el.get("style") or "").split(";")) if v}


def _css_rules(root):
    """Simple rules from every <style>: [(tag, class, id, {prop: value})] for selectors like `.st0`, `path`, `#a`,
    `circle.b` and comma lists; anything more complex (combinators, pseudo-classes) is ignored."""
    css = "".join(el.text or "" for el in root.iter() if el.tag.replace(SVG_NS, "") == "style")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = []
    for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        decl = {k.strip(): v.strip() for k, _, v in (kv.partition(":") for kv in body.split(";")) if v.strip()}
        for one in sel.split(","):
            m_ = re.fullmatch(r"\s*([a-zA-Z][\w-]*)?(?:\.([\w-]+))?(?:#([\w-]+))?\s*", one)
            if m_ and decl and any(m_.groups()):
                rules.append((m_.group(1), m_.group(2), m_.group(3), decl))
    return rules


def _css_props(el, rules):
    """Declarations that apply to el, lowest specificity first (tag < .class < #id)."""
    tag, classes, eid = el.tag.replace(SVG_NS, ""), set((el.get("class") or "").split()), el.get("id")
    out = {}
    # key= only: two rules with the same selector must not fall through to comparing their dicts (TypeError); the stable
    # sort keeps document order within a specificity, so the later rule wins as in CSS
    for t, c, i, decl in sorted(rules, key=lambda r: (bool(r[2]), bool(r[1]), bool(r[0]))):
        if (t is None or t == tag) and (c is None or c in classes) and (i is None or i == eid):
            out.update(decl)
    return out


def _opacity(v, dflt=1.0):
    """An SVG opacity value ("0.5", "50%") as a float."""
    if v is None:
        return dflt
    v = str(v).strip()
    return _num(v) / 100 if v.endswith("%") else _num(v, dflt)


def _inherit(el, parent, css=None):
    """Effective presentation attributes of el: attribute < <style> rule < inline style; opacity multiplies down the
    tree; `inherit` and empty values keep the parent's; `!important` is stripped."""
    attrs = dict(parent)
    rule = _css_props(el, css) if css else {}
    style = {**rule, **_style_dict(el)}
    for k in PRES_ATTRS:
        v = style.get(k, el.get(k))
        if v is not None:
            v = re.sub(r"\s*!important\s*$", "", v.strip())
            if v and v != "inherit":
                attrs[k] = v
    op = style.get("opacity", el.get("opacity"))
    attrs["opacity"] = _opacity(parent.get("opacity"), 1.0) * (_opacity(op, 1.0) if op is not None else 1.0)
    return attrs


def _color(v, opacity=1.0, current=None):
    """CSS colour -> #RRGGBB / #AARRGGBB (hex, rgb()/rgba(), hsl()/hsla(), the 148 named colours, currentColor)."""
    v = re.sub(r"\s*!important\s*$", "", (v or "").strip().lower())
    if v == "currentcolor":
        v = (current or "#000000").strip().lower()
    v = NAMED_COLORS.get(v, v)
    m_ = re.match(r"(rgba?|hsla?)\(([^)]*)\)", v)
    if m_:
        parts = [p.strip() for p in re.split(r"[\s,/]+", m_.group(2).strip()) if p.strip()]
        if len(parts) < 3:
            raise ValueError(f"unsupported colour {v!r}")
        a = (_num(parts[3]) / 100 if parts[3].endswith("%") else _num(parts[3])) if len(parts) > 3 else 1.0
        if m_.group(1).startswith("hsl"):
            unit = re.sub(r"^[-+0-9.eE]+", "", parts[0])  # CSS hue units: deg (default), turn, rad, grad
            h = _num(parts[0]) * {"turn": 360.0, "rad": 180 / math.pi, "grad": 0.9}.get(unit, 1.0)
            h = (h % 360) / 360
            sat, lig = min(1.0, max(0.0, _num(parts[1]) / 100)), min(1.0, max(0.0, _num(parts[2]) / 100))
            import colorsys
            rgb = [int(c * 255 + 0.5 + 1e-6) for c in colorsys.hls_to_rgb(h, lig, sat)]  # CSS rounds half up
        else:
            rgb = [round(_num(p) * 2.55) if p.endswith("%") else round(_num(p)) for p in parts[:3]]
        v = "#" + "".join(f"{max(0, min(255, c)):02x}" for c in rgb) + f"{round(max(0.0, min(1.0, a)) * 255):02x}"
    if not v.startswith("#") or len(v) not in (4, 5, 7, 9) or not re.fullmatch(r"#[0-9a-f]+", v):
        raise ValueError(f"unsupported colour {v!r}")
    h = v[1:]
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    alpha = round((int(h[6:8], 16) if len(h) == 8 else 255) * opacity)
    return f"#{alpha:02X}{h[:6].upper()}" if alpha < 255 else f"#{h[:6].upper()}"


def _shape_path(el):
    tag = el.tag.replace(SVG_NS, "")
    g = lambda k, dflt=0.0: _num(el.get(k), dflt)
    if tag == "path":
        return el.get("d")
    if tag in ("circle", "ellipse"):
        cx, cy = g("cx"), g("cy")
        rx = g("r") if tag == "circle" else g("rx", g("ry"))
        ry = g("r") if tag == "circle" else g("ry", rx)
        if rx <= 0 or ry <= 0:
            return None
        return f"M{cx - rx},{cy}a{rx},{ry} 0 1,0 {2 * rx},0a{rx},{ry} 0 1,0 {-2 * rx},0Z"
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        rx = g("rx", g("ry"))
        ry = g("ry", rx)
        if not rx or not ry:
            return f"M{x},{y}h{w}v{h}h{-w}Z"
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        return (f"M{x + rx},{y}h{w - 2 * rx}a{rx},{ry} 0 0,1 {rx},{ry}v{h - 2 * ry}a{rx},{ry} 0 0,1 {-rx},{ry}"
                f"h{-(w - 2 * rx)}a{rx},{ry} 0 0,1 {-rx},{-ry}v{-(h - 2 * ry)}a{rx},{ry} 0 0,1 {rx},{-ry}Z")
    if tag == "line":
        return f"M{g('x1')},{g('y1')}L{g('x2')},{g('y2')}"
    if tag in ("polyline", "polygon"):
        pts = _NUM.findall(el.get("points", ""))
        pairs = [f"{pts[i]},{pts[i + 1]}" for i in range(0, len(pts) - 1, 2)]
        return ("M" + "L".join(pairs) + ("Z" if tag == "polygon" else "")) if pairs else None
    return None


def _transforms(s):
    """SVG transform list -> VectorDrawable <group> attribute dicts, outermost first."""
    groups = []
    counts = {"translate": (1, 2), "scale": (1, 2), "rotate": (1, 3), "matrix": (6, 6)}
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM.findall(args)]
        lo, hi = counts.get(name, (0, 0))
        if name in counts and not lo <= len(v) <= hi:
            raise ValueError(f"bad {name}() arguments: {args.strip()!r}")
        if name == "translate":
            groups.append({"translateX": v[0], "translateY": v[1] if len(v) > 1 else 0})
        elif name == "scale":
            groups.append({"scaleX": v[0], "scaleY": v[1] if len(v) > 1 else v[0]})
        elif name == "rotate":
            groups.append({"rotation": v[0], "pivotX": v[1] if len(v) > 2 else 0, "pivotY": v[2] if len(v) > 2 else 0})
        elif name == "matrix":
            a, b, c, d, e, f = v
            if abs(a * c + b * d) > 1e-6:
                raise ValueError("transform matrix with skew is not expressible in VectorDrawable")
            sx = math.hypot(a, b)
            if not sx:
                raise ValueError("transform matrix with zero scale")
            groups.append({"translateX": e, "translateY": f, "rotation": math.degrees(math.atan2(b, a)),
                           "scaleX": sx, "scaleY": (a * d - b * c) / sx})
        else:
            raise ValueError(f"transform {name}() is not expressible in VectorDrawable")
    return groups


def _viewbox(root):
    """(x, y, w, h) of an <svg> root: its viewBox, else width/height; ValueError when malformed or without area."""
    raw = root.get("viewBox", "").strip()
    if raw:
        vb = [float(v) for v in re.split(r"[\s,]+", raw) if v]
        if len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0:
            raise ValueError(f"viewBox must be 4 numbers with positive size, got {raw!r}")
        return vb
    w, h = _num(root.get("width"), 24), _num(root.get("height"), 24)
    if w <= 0 or h <= 0:
        raise ValueError(f"width/height must be positive, got {root.get('width')!r} x {root.get('height')!r}")
    return [0, 0, w, h]


_UNIT_PX = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}


def _length_px(v):
    """An SVG length ("24", "24px", "18pt", "24mm") in CSS px; None for missing, percentage or unknown units."""
    m_ = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z%]*)\s*", str(v or ""))
    if not m_ or m_.group(2) not in _UNIT_PX:
        return None
    return float(m_.group(1)) * _UNIT_PX[m_.group(2)]


def _pct(v, ref, dflt):
    """A gradient coordinate: percentages against ref (viewBox width/height/diagonal), else user units."""
    if v is None:
        v = dflt
    v = str(v).strip()
    return _num(v) / 100 * ref if v.endswith("%") else _num(v)


_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _tf_affine(tf):
    """SVG transform list -> (a, b, c, d, e, f); _matrix parses skewX/skewY itself, so this is only the name the
    converter uses for it."""
    return _matrix(tf)


def _mmul(p, q):
    """Affine (a, b, c, d, e, f) product p . q (q is applied first)."""
    return (p[0] * q[0] + p[2] * q[1], p[1] * q[0] + p[3] * q[1], p[0] * q[2] + p[2] * q[3],
            p[1] * q[2] + p[3] * q[3], p[0] * q[4] + p[2] * q[5] + p[4], p[1] * q[4] + p[3] * q[5] + p[5])


def _mapply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _abs_path(d, m=_IDENTITY):
    """Path data as absolute L/C segments with matrix m applied (arcs become cubics)."""
    subs = _subpaths(d)
    if m != _IDENTITY:
        for sp in subs:
            sp["pts"] = [_mapply(m, *pt) for pt in sp["pts"]]
            sp["segs"] = [(sg[0], *[_mapply(m, *pt) for pt in sg[1:]]) for sg in sp["segs"]]
    return _d(subs)


def _path_bbox(d):
    """(x0, y0, x1, y1) hull of the path's anchor and control points (exact for straight edges)."""
    pts = _points(_subpaths(d))
    if not pts:
        return None
    return min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts)


def _inline_uses(root, ids):
    """Replace every <use> with a copy of what it references (a <symbol> becomes a <g> scaled to the use's size)."""
    import copy
    for _ in range(8):
        uses = [(par, ch) for par in root.iter() for ch in par if ch.tag.replace(SVG_NS, "") == "use"]
        if not uses:
            return
        for par, use in uses:
            href = (use.get("href") or use.get("{http://www.w3.org/1999/xlink}href") or "").strip()
            if not href.startswith("#") or href[1:] not in ids:
                raise ValueError(f"<use> references {href!r}, which is not in this file")
            src = copy.deepcopy(ids[href[1:]])
            src.attrib.pop("id", None)
            tf = f"translate({_num(use.get('x')):g} {_num(use.get('y')):g})"
            if src.tag.replace(SVG_NS, "") in ("symbol", "svg"):
                vb = src.get("viewBox")
                if vb and use.get("width") and use.get("height"):
                    x, y, w, h = [float(v) for v in re.split(r"[\s,]+", vb.strip()) if v]
                    tf += f" scale({_num(use.get('width')) / w:g} {_num(use.get('height')) / h:g}) translate({-x:g} {-y:g})"
                src.tag = SVG_NS + "g"
                for k in ("viewBox", "width", "height", "preserveAspectRatio"):
                    src.attrib.pop(k, None)
            g = ET.Element(SVG_NS + "g", {k: v for k, v in use.attrib.items()
                                          if k not in ("x", "y", "width", "height", "href", "transform", "id")
                                          and not k.startswith("{http://www.w3.org/1999/xlink}")})
            g.set("transform", (use.get("transform", "") + " " + tf).strip())
            g.append(src)
            par[list(par).index(use)] = g
    raise ValueError("<use> chain nested more than 8 levels deep")


def _rewind_evenodd(d):
    """Path data with every subpath wound by nesting depth (a hole runs against its outer loop) so the nonzero rule
    keeps what evenodd cut out: VectorDrawable <clip-path> has no fillType. Exact for nested, non-crossing subpaths."""
    subs = _subpaths(d)
    loops = [np.array(_curve_pts([sp])) for sp in subs]
    out = []
    for i, (sp, P) in enumerate(zip(subs, loops)):
        depth = sum(_inside(P[0], Q) for j, Q in enumerate(loops) if j != i and len(Q) >= 3 and abs(_area(Q)) > abs(_area(P)))
        if len(P) < 3 or (_area(P) > 0) == (depth % 2 == 0):
            out.append(sp)
            continue
        ends = [sp["pts"][0]] + [s[-1] for s in sp["segs"]]
        segs = [("L", ends[k]) if s[0] == "L" else ("C", s[2], s[1], ends[k]) for k, s in reversed(list(enumerate(sp["segs"])))]
        out.append({"pts": [ends[-1]], "segs": segs, "closed": sp["closed"]})
    return _d(out)


def svg_to_vd(svg_text, width_dp=None, warnings=None):
    """Convert an SVG to VectorDrawable XML. Unsupported features raise ValueError; lossy ones append to warnings.
    Handles shapes, groups and transforms (skews baked into coordinates), <style> classes, <use>/<symbol>, clip-path,
    linear/radial gradients (userSpaceOnUse and objectBoundingBox, gradientTransform), CSS colours.
    Default size: the root width/height (px; pt/mm/cm/in converted), else the viewBox, at the viewBox's aspect."""
    warns = warnings if warnings is not None else []
    root = ET.fromstring(svg_text)
    vx, vy, vw, vh = _viewbox(root)
    if width_dp:
        wdp = float(width_dp)
    else:
        w, h = _length_px(root.get("width")), _length_px(root.get("height"))
        wdp = w if w and w > 0 else (h * vw / vh if h and h > 0 else vw)
        if wdp > 200:
            warns.append(f"{wdp:g}dp > 200dp (Android vectors should stay <= 200dp: pass --dp)")
    hdp = round(wdp * vh / vw, 2)
    ids = {el.get("id"): el for el in root.iter() if el.get("id")}
    _inline_uses(root, ids)
    if not root.get("viewBox", "").strip() and (root.get("width") is None or root.get("height") is None):
        warns.append("no viewBox and no width/height: a 24x24 user-unit viewport is assumed (anything past 24 is clipped)")
    pct_ref = {"x": vw, "cx": vw, "x1": vw, "x2": vw, "width": vw, "rx": vw, "y": vh, "cy": vh, "y1": vh, "y2": vh, "height": vh, "ry": vh,
               "r": math.sqrt((vw * vw + vh * vh) / 2)}  # percentage lengths on shapes resolve against the viewBox (SVG 1.1 7.10)
    for el in root.iter():
        if el.tag.replace(SVG_NS, "") in _SHAPES:
            for k, ref in pct_ref.items():
                v = (el.get(k) or "").strip()
                if v.endswith("%"):
                    el.set(k, f"{_num(v) / 100 * ref:g}")
    css = _css_rules(root)
    tag_of = lambda el: el.tag.replace(SVG_NS, "") if el.tag.startswith(SVG_NS) or "{" not in el.tag else None
    # skews (and matrices with skew) are not expressible as <group>: bake every transform into coordinates first
    for el in root.iter():
        if el.get("transform"):
            try:
                _transforms(el.get("transform"))
            except ValueError as e:
                if "skew" not in str(e) and "not expressible" not in str(e):
                    raise
                baked = _bake_transforms(ET.tostring(root, encoding="unicode"))
                if baked is None:
                    raise ValueError(f"{e}; could not bake it into path coordinates (non-uniform scale on a stroked shape?)")
                root = ET.fromstring(baked)
                ids = {el.get("id"): el for el in root.iter() if el.get("id")}
                warns.append("skew transform baked into path coordinates")
                break
    diag = math.sqrt((vw * vw + vh * vh) / 2)
    grads = {}
    for gel in root.iter():
        tag = tag_of(gel)
        if tag not in ("linearGradient", "radialGradient"):
            continue
        src = gel
        href = gel.get("href") or gel.get("{http://www.w3.org/1999/xlink}href")
        if not any(tag_of(s) == "stop" for s in gel) and href and href[1:] in ids:
            src = ids[href[1:]]
        get = lambda k, dflt=None: gel.get(k) if gel.get(k) is not None else src.get(k, dflt)
        gid = gel.get("id")
        obb = get("gradientUnits") != "userSpaceOnUse"
        stops, prev = [], 0.0
        for s in src:
            if tag_of(s) == "stop":
                st = {**_css_props(s, css), **_style_dict(s)}
                col = st.get("stop-color", s.get("stop-color", "#000000"))
                op = _opacity(st.get("stop-opacity", s.get("stop-opacity")), 1.0)
                off = s.get("offset", "0")
                off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
                prev = min(1.0, max(0.0, off, prev))
                stops.append((prev, _color(col, op)))
        gt = _tf_affine(get("gradientTransform")) if get("gradientTransform") else _IDENTITY
        if tag == "linearGradient":
            rw, rh = (1.0, 1.0) if obb else (vw, vh)
            geo = {"startX": _pct(get("x1"), rw, "0%"), "startY": _pct(get("y1"), rh, "0%"),
                   "endX": _pct(get("x2"), rw, "100%"), "endY": _pct(get("y2"), rh, "0%")}
        else:
            # objectBoundingBox: r% is against the unit box's normalised diagonal sqrt((1+1)/2) = 1 (SVG 1.1 7.10), so
            # the default 50% is 0.5 bbox units; the bbox matrix scales it by sqrt(bw*bh) in geometry()
            rw, rh, rd = (1.0, 1.0, 1.0) if obb else (vw, vh, diag)
            geo = {"centerX": _pct(get("cx"), rw, "50%"), "centerY": _pct(get("cy"), rh, "50%"),
                   "gradientRadius": _pct(get("r"), rd, "50%")}
            fx, fy = get("fx"), get("fy")
            if (fx is not None and abs(_pct(fx, rw, 0) - geo["centerX"]) > 1e-6) or \
               (fy is not None and abs(_pct(fy, rh, 0) - geo["centerY"]) > 1e-6):
                warns.append(f"gradient #{gid}: focal point fx/fy ignored (VectorDrawable radial gradients are concentric)")
        grads[gid] = (tag, geo, stops, {"reflect": "mirror", "repeat": "repeat"}.get(get("spreadMethod")), obb, gt, gid)
    used_grad = False

    def geometry(ref, d):
        """Gradient geometry in the path's user space: objectBoundingBox mapped through the path's bbox, then gradientTransform."""
        tag, geo, _, _, obb, gt, gid = grads[ref]
        m = gt
        if obb:
            bb = _path_bbox(d)
            if not bb:
                raise ValueError(f"gradient #{gid}: objectBoundingBox needs a path with area")
            bw, bh = max(bb[2] - bb[0], 1e-9), max(bb[3] - bb[1], 1e-9)
            m = _mmul((bw, 0.0, 0.0, bh, bb[0], bb[1]), gt)
            if any(sg[0] == "C" for sp in _subpaths(d) for sg in sp["segs"]):
                warns.append(f"gradient #{gid}: objectBoundingBox mapped through the path's control-point box (curves: approximate)")
        if m == _IDENTITY:
            return dict(geo)
        a, b, c, d_ = m[:4]
        sx, sy = math.hypot(a, b), math.hypot(c, d_)
        similar = abs(sx - sy) < 1e-6 * max(sx, 1) and abs(a * c + b * d_) < 1e-6
        if tag == "linearGradient":
            p1, p2 = (geo["startX"], geo["startY"]), (geo["endX"], geo["endY"])
            q1, q2 = _mapply(m, *p1), _mapply(m, *p2)
            if not similar:  # iso-lines map to lines: keep the mapped direction perpendicular to them (exact)
                iso = (-(p2[1] - p1[1]), p2[0] - p1[0])
                iso = (a * iso[0] + c * iso[1], b * iso[0] + d_ * iso[1])
                nrm = math.hypot(*iso) or 1.0
                n_ = (iso[1] / nrm, -iso[0] / nrm)
                t = (q2[0] - q1[0]) * n_[0] + (q2[1] - q1[1]) * n_[1]
                q2 = (q1[0] + n_[0] * t, q1[1] + n_[1] * t)
            return {"startX": q1[0], "startY": q1[1], "endX": q2[0], "endY": q2[1]}
        cx, cy = _mapply(m, geo["centerX"], geo["centerY"])
        if not similar:
            warns.append(f"gradient #{gid}: elliptical radial gradient approximated by a circle")
        r = geo["gradientRadius"] * math.sqrt(abs(a * d_ - b * c)) if not similar else geo["gradientRadius"] * sx
        return {"centerX": cx, "centerY": cy, "gradientRadius": r}

    def paint(v):
        """SVG paint -> ("none", None) | ("solid", colour) | ("grad", id)."""
        if v in ("none", "transparent", ""):
            return "none", None
        m_ = re.match(r"""url\(\s*['"]?#([^)'"]+)['"]?\s*\)""", v)
        if not m_:
            return "solid", v
        ref = m_.group(1)
        if ref not in grads:
            raise ValueError(f"gradient #{ref} not found")
        stops = grads[ref][2]
        if not stops:
            return "none", None
        if len(stops) == 1:
            return "solid", stops[0][1]
        return "grad", ref

    def grad_xml(ref, attr, indent, d):
        nonlocal used_grad
        used_grad = True
        tag, _, stops, tile, *_ = grads[ref]
        geo = geometry(ref, d)
        if tag == "radialGradient" and geo["gradientRadius"] <= 0:
            raise ValueError(f"gradient #{ref}: radius 0")
        head = f'android:type="{"linear" if tag == "linearGradient" else "radial"}" ' + \
            " ".join(f'android:{k}="{_fmt(v)}"' for k, v in geo.items())
        if tile:
            head += f' android:tileMode="{tile}"'
        items = "".join(f'\n{indent}      <item android:offset="{o:g}" android:color="{c}"/>' for o, c in stops)
        return (f'\n{indent}  <aapt:attr name="android:{attr}">\n{indent}    <gradient {head}>{items}\n'
                f'{indent}    </gradient>\n{indent}  </aapt:attr>')

    def path_xml(d, attrs, ind):
        fk, fv = paint(attrs.get("fill", "#000000"))
        sk, sv = paint(attrs.get("stroke", "none"))
        if any(attrs.get(k) not in (None, "none") for k in ("marker-start", "marker-mid", "marker-end")):
            warns.append("marker-start/mid/end dropped (VectorDrawable has no markers: draw arrowheads as paths)")
        if fk == "none" and sk == "none":
            return ""
        op = attrs.get("opacity", 1.0)
        cur = attrs.get("color")
        out = [f'{ind}<path', f'{ind}    android:pathData="{path_str(parse_path(d))}"']
        children = ""
        if fk != "none":
            if fk == "grad":
                children += grad_xml(fv, "fillColor", ind, d)
            else:
                out.append(f'{ind}    android:fillColor="{_color(fv, 1.0, cur)}"')
            fo = _opacity(attrs.get("fill-opacity"), 1.0) * op
            if fo < 1:
                out.append(f'{ind}    android:fillAlpha="{fo:g}"')
            if attrs.get("fill-rule") == "evenodd":
                out.append(f'{ind}    android:fillType="evenOdd"')
        if sk != "none":
            if sk == "grad":
                children += grad_xml(sv, "strokeColor", ind, d)
            else:
                out.append(f'{ind}    android:strokeColor="{_color(sv, 1.0, cur)}"')
            out.append(f'{ind}    android:strokeWidth="{_num(attrs.get("stroke-width"), 1.0):g}"')
            so = _opacity(attrs.get("stroke-opacity"), 1.0) * op
            if so < 1:
                out.append(f'{ind}    android:strokeAlpha="{so:g}"')
            if attrs.get("stroke-linecap") in ("round", "square"):
                out.append(f'{ind}    android:strokeLineCap="{attrs["stroke-linecap"]}"')
            if attrs.get("stroke-linejoin") in ("round", "bevel"):
                out.append(f'{ind}    android:strokeLineJoin="{attrs["stroke-linejoin"]}"')
            if attrs.get("stroke-miterlimit"):
                out.append(f'{ind}    android:strokeMiterLimit="{_num(attrs["stroke-miterlimit"]):g}"')
            if attrs.get("stroke-dasharray") not in (None, "none", "0"):
                warns.append("stroke-dasharray dropped (VectorDrawable has no dashes; stroke drawn solid)")
            if attrs.get("vector-effect") == "non-scaling-stroke":
                warns.append("vector-effect=non-scaling-stroke ignored (the stroke scales with its group)")
            if "stroke" in (attrs.get("paint-order") or "") and (attrs.get("paint-order") or "").split()[0] == "stroke":
                warns.append("paint-order=stroke ignored (VectorDrawable always paints the stroke over the fill)")
        return "\n".join(out) + (f">{children}\n{ind}</path>" if children else " />")

    def wrap(body, tf, ind):
        groups = _transforms(tf)
        if "strokeWidth" in body and any(abs(g.get("scaleX", 1) - g.get("scaleY", 1)) > 1e-6 for g in groups):
            warns.append("non-uniform scale on a stroked path: Android scales the stroke width approximately")
        for g in reversed(groups):
            attrs = " ".join(f'android:{k}="{v:g}"' for k, v in g.items() if v or k.startswith("scale"))
            body = f"{ind}<group {attrs}>\n" + "\n".join("    " + ln for ln in body.split("\n")) + f"\n{ind}</group>"
        return body

    def clip_of(ref, el_paths, ind, body):
        """<group> with a <clip-path> around body; None when the clip is a rect covering the whole viewBox."""
        cel = ids.get(ref)
        if cel is None or tag_of(cel) != "clipPath":
            raise ValueError(f"clip-path #{ref} not found")
        obb = cel.get("clipPathUnits") == "objectBoundingBox"
        base = _IDENTITY
        if obb:
            boxes = [b for b in (_path_bbox(d) for d in el_paths) if b]
            if not boxes:
                raise ValueError(f"clip-path #{ref}: objectBoundingBox needs content with area")
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            base = (x1 - x0, 0.0, 0.0, y1 - y0, x0, y0)
            warns.append(f"clip-path #{ref}: objectBoundingBox mapped through the content's control-point box (approximate)")
        parts = []
        for ch in cel:
            ctag = tag_of(ch)
            if ctag not in _SHAPES:
                if ctag and ctag not in ("title", "desc"):
                    warns.append(f"clip-path #{ref}: <{ctag}> ignored")
                continue
            cattrs = _inherit(ch, _inherit(cel, {}, css), css)
            evenodd = cattrs.get("clip-rule") == "evenodd" or cattrs.get("fill-rule") == "evenodd"
            if ch.get("clip-path") or cel.get("clip-path"):
                warns.append(f"clip-path #{ref}: nested clip-path ignored")
            m = _mmul(base, _tf_affine(ch.get("transform"))) if ch.get("transform") else base
            d = _shape_path(ch)
            if not d:
                continue
            if ctag == "rect" and m == _IDENTITY and not ch.get("rx") and not ch.get("ry") and len(list(cel)) == 1:
                x, y = _num(ch.get("x")), _num(ch.get("y"))
                w, h = _num(ch.get("width")), _num(ch.get("height"))
                if x <= vx + 1e-6 and y <= vy + 1e-6 and x + w >= vx + vw - 1e-6 and y + h >= vy + vh - 1e-6:
                    return None
            part = _abs_path(d, m) if m != _IDENTITY or ctag != "path" else path_str(parse_path(d))
            if evenodd and len(_subpaths(part)) > 1:  # <clip-path> is always nonzero: rewind the holes instead
                part = _rewind_evenodd(part)
                warns.append(f"clip-path #{ref}: evenodd clip-rule: subpaths rewound by nesting for the nonzero clip "
                             "(exact unless subpaths cross)")
            parts.append(part)
        if not parts:
            warns.append(f"clip-path #{ref} has no shapes: everything it clips is invisible")
            return ""
        inner = "\n".join("    " + ln for ln in body.split("\n"))
        return f'{ind}<group>\n{ind}    <clip-path android:pathData="{" ".join(parts)}"/>\n{inner}\n{ind}</group>'

    def clip_ref(el):
        v = _style_dict(el).get("clip-path") or _css_props(el, css).get("clip-path") or el.get("clip-path")
        m_ = re.match(r"""url\(\s*['"]?#([^)'"]+)['"]?\s*\)""", (v or "").strip())
        return m_.group(1) if m_ else None

    def walk(el, inherited, ind):
        lines = []
        children = list(el)
        if tag_of(el) == "switch":
            first = [ch for ch in children if tag_of(ch) in _SHAPES + ("g", "a", "svg", "use", "text", "image")][:1]
            if len(children) > len(first):
                warns.append("<switch>: only its first renderable child is converted")
            children = first
        for ch in children:
            tag = tag_of(ch)
            if tag is None or tag in ("defs", "linearGradient", "radialGradient", "title", "desc", "metadata", "style",
                                      "clipPath", "symbol", "marker", "script"):
                continue
            if tag in ("text", "image", "mask", "filter", "pattern", "foreignObject"):
                raise ValueError(f"<{tag}> is not convertible")
            attrs = _inherit(ch, inherited, css)
            if attrs.get("display") == "none" or _style_dict(ch).get("display") == "none":
                continue
            if _style_dict(ch).get("mask") or _css_props(ch, css).get("mask") or ch.get("mask") or ch.get("filter") or \
               _style_dict(ch).get("filter"):
                raise ValueError(f"<{tag}> uses mask/filter (not convertible)")
            paths = []
            if tag in ("g", "a", "switch"):
                body = walk(ch, attrs, ind)
                paths = [x.get("d") or _shape_path(x) for x in ch.iter() if tag_of(x) in _SHAPES]
            elif tag in _SHAPES:
                if attrs.get("visibility") in ("hidden", "collapse"):
                    continue
                d = _shape_path(ch)
                paths = [d] if d else []
                body = path_xml(d, attrs, ind) if d else ""
            else:
                warns.append(f"<{tag}> ignored (not convertible: nested svg, foreignObject, markers, ...)"
                             if tag in ("svg",) else f"<{tag}> ignored")
                continue
            if body and clip_ref(ch):
                clipped = clip_of(clip_ref(ch), [p_ for p_ in paths if p_], ind, body)
                body = body if clipped is None else clipped
            if body:
                lines.append(wrap(body, ch.get("transform"), ind) if ch.get("transform") else body)
        return "\n".join(lines)
    body = walk(root, _inherit(root, {}, css), "    ")
    if clip_ref(root):
        clipped = clip_of(clip_ref(root), [x.get("d") or _shape_path(x) for x in root.iter() if tag_of(x) in _SHAPES], "    ", body)
        body = body if clipped is None else clipped
    if root.get("transform"):
        body = wrap(body, root.get("transform"), "    ")
    if vx or vy:
        body = wrap(body, f"translate({-vx:g} {-vy:g})", "    ")
    ns = ' xmlns:aapt="http://schemas.android.com/aapt"' if used_grad else ""
    api24 = [k for k, v in (("evenOdd fill", 'android:fillType="evenOdd"' in body), ("gradient", used_grad)) if v]
    if api24:
        warns.append(f"needs API 24+ or AppCompat srcCompat: {' / '.join(api24)}")
    return (f'<?xml version="1.0" encoding="utf-8"?>\n<vector xmlns:android="http://schemas.android.com/apk/res/android"{ns}\n'
            f'    android:width="{wdp:g}dp"\n    android:height="{hdp:g}dp"\n'
            f'    android:viewportWidth="{vw:g}"\n    android:viewportHeight="{vh:g}">\n' + body + "\n</vector>\n")


AND = "{http://schemas.android.com/apk/res/android}"


def vd_to_svg(xml):
    """Render-equivalent SVG of a VectorDrawable (used to verify conversions)."""
    root = ET.fromstring(xml)
    g = lambda el, k, dflt=None: el.get(AND + k, dflt)
    defs, n = [], [0]

    def colour(v, alpha):
        v = v.lstrip("#")
        a = int(v[:2], 16) / 255 if len(v) == 8 else 1.0
        return "#" + v[-6:], a * alpha

    def paint(el, attr, alpha):
        grad = next((ga for ga in el if ga.tag.endswith("attr") and ga.get("name") == f"android:{attr}"), None)
        if grad is not None:
            gr = grad[0]
            n[0] += 1
            stops = "".join(f'<stop offset="{g(it, "offset")}" stop-color="{colour(g(it, "color"), 1)[0]}" '
                            f'stop-opacity="{colour(g(it, "color"), 1)[1]:g}"/>' for it in gr)
            if g(gr, "type") == "radial":
                head = f'<radialGradient id="p{n[0]}" gradientUnits="userSpaceOnUse" cx="{g(gr, "centerX")}" cy="{g(gr, "centerY")}" r="{g(gr, "gradientRadius")}"'
                tagn = "radialGradient"
            else:
                head = f'<linearGradient id="p{n[0]}" gradientUnits="userSpaceOnUse" x1="{g(gr, "startX")}" y1="{g(gr, "startY")}" x2="{g(gr, "endX")}" y2="{g(gr, "endY")}"'
                tagn = "linearGradient"
            spread = {"mirror": "reflect", "repeat": "repeat"}.get(g(gr, "tileMode"), "pad")
            defs.append(f'{head} spreadMethod="{spread}">{stops}</{tagn}>')
            return f"url(#p{n[0]})", alpha
        v = g(el, attr)
        return colour(v, alpha) if v else ("none", 1.0)

    def walk(el):
        out, open_clips = [], 0
        for ch in el:
            if ch.tag == "group":
                tx, ty = float(g(ch, "translateX", 0)), float(g(ch, "translateY", 0))
                px, py = float(g(ch, "pivotX", 0)), float(g(ch, "pivotY", 0))
                tf = (f"translate({tx + px:g} {ty + py:g}) rotate({g(ch, 'rotation', 0)}) "
                      f"scale({g(ch, 'scaleX', 1)} {g(ch, 'scaleY', 1)}) translate({-px:g} {-py:g})")
                out.append(f'<g transform="{tf}">{walk(ch)}</g>')
            elif ch.tag == "clip-path":  # clips the siblings that follow it, like VectorDrawable does
                n[0] += 1
                defs.append(f'<clipPath id="c{n[0]}"><path d="{g(ch, "pathData")}"/></clipPath>')
                out.append(f'<g clip-path="url(#c{n[0]})">')
                open_clips += 1
            elif ch.tag == "path":
                fill, fa = paint(ch, "fillColor", float(g(ch, "fillAlpha", 1)))
                stroke, sa = paint(ch, "strokeColor", float(g(ch, "strokeAlpha", 1)))
                out.append(f'<path d="{g(ch, "pathData")}" fill="{fill}" fill-opacity="{fa:g}" stroke="{stroke}" '
                           f'stroke-opacity="{sa:g}" stroke-width="{g(ch, "strokeWidth", 0)}" '
                           f'stroke-linecap="{g(ch, "strokeLineCap", "butt")}" stroke-linejoin="{g(ch, "strokeLineJoin", "miter")}" '
                           f'stroke-miterlimit="{g(ch, "strokeMiterLimit", 4)}" '
                           f'fill-rule="{"evenodd" if g(ch, "fillType") == "evenOdd" else "nonzero"}"/>')
        return "".join(out) + "</g>" * open_clips
    body = walk(root)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {g(root, "viewportWidth")} {g(root, "viewportHeight")}">'
            f'<defs>{"".join(defs)}</defs>{body}</svg>')


def vd_roundtrip(pairs, size=128):
    """[(svg, vd xml)] -> per pair {"mean": mean premultiplied RGBA difference (0-255), "strong": pixels whose largest
    channel difference exceeds 64} between the SVG and its VectorDrawable rendered, or None without a renderer.
    Correct conversions measure strong == 0 at 128 px; dropped or moved content measures 90+ even when mean < 2."""
    ims = render_svgs([s for s, _ in pairs] + [vd_to_svg(x) for _, x in pairs], size)
    out = []
    for a, b in zip(ims[:len(pairs)], ims[len(pairs):]):
        if a is None or b is None:
            out.append(None)
            continue
        pa, pb = (np.asarray(im, dtype=float) for im in (a, b))
        pa[..., :3] *= pa[..., 3:] / 255
        pb[..., :3] *= pb[..., 3:] / 255
        diff = np.abs(pa - pb)
        out.append({"mean": round(float(diff.mean()), 2), "strong": int((diff.max(2) > 64).sum())})
    return out


@functools.cache
def _android_sdk():
    homes = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"), "~/Library/Android/sdk", "~/Android/Sdk"]
    for h in homes:
        if h and Path(h).expanduser().is_dir():
            sdk = Path(h).expanduser()
            ver = lambda p: [int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", p.parent.name.replace("android-", ""))]
            tools = sorted(sdk.glob("build-tools/*/aapt2"), key=ver)
            jars = sorted(sdk.glob("platforms/android-*/android.jar"), key=ver)
            if tools:
                return str(tools[-1]), str(jars[-1]) if jars else None
    return (shutil.which("aapt2"), None) if shutil.which("aapt2") else None


def aapt2_level():
    """"link" when aapt2 and a platforms/android-*/android.jar are available, "compile" with aapt2 alone, else None."""
    sdk = _android_sdk()
    return None if not sdk else ("link" if sdk[1] else "compile")


def _aapt2_line(n=None):
    """The one-line aapt2 verdict for `n` files that passed every available step (plan contract C4 wording)."""
    lvl, files = aapt2_level(), f": {n} file(s)" if n is not None else ""
    if lvl == "link":
        return f"aapt2 compile+link ok ({Path(_android_sdk()[1]).parent.name}){files}"
    if lvl == "compile":
        return ("aapt2 compile ok; link skipped (no $ANDROID_HOME/platforms/android-*/android.jar: attribute names, enums and "
                'dimensions NOT checked; sdkmanager "platforms;android-35")' + files)
    return "aapt2 not found (set ANDROID_HOME): not validated"


_VD_ATTRS = {
    "vector": {"width", "height", "viewportWidth", "viewportHeight", "tint", "tintMode", "autoMirrored", "alpha", "name",
               "opticalInsetLeft", "opticalInsetTop", "opticalInsetRight", "opticalInsetBottom"},
    "group": {"name", "rotation", "pivotX", "pivotY", "scaleX", "scaleY", "translateX", "translateY"},
    "path": {"name", "pathData", "fillColor", "strokeColor", "strokeWidth", "strokeAlpha", "fillAlpha", "trimPathStart",
             "trimPathEnd", "trimPathOffset", "strokeLineCap", "strokeLineJoin", "strokeMiterLimit", "fillType"},
    "clip-path": {"name", "pathData"},
    "gradient": {"type", "startX", "startY", "endX", "endY", "centerX", "centerY", "gradientRadius", "startColor",
                 "centerColor", "endColor", "tileMode"},
    "item": {"offset", "color"},
}
_VD_ENUMS = {"fillType": {"evenOdd", "nonZero"}, "strokeLineCap": {"butt", "round", "square"},
             "strokeLineJoin": {"miter", "round", "bevel"}, "tileMode": {"clamp", "repeat", "mirror", "disabled"},
             "type": {"linear", "radial", "sweep"}, "autoMirrored": {"true", "false"},
             "tintMode": {"add", "multiply", "screen", "src_atop", "src_in", "src_over"}}
_VD_COLORS = {"fillColor", "strokeColor", "startColor", "centerColor", "endColor", "color", "tint"}
_VD_FLOATS = {"viewportWidth", "viewportHeight", "rotation", "pivotX", "pivotY", "scaleX", "scaleY", "translateX", "translateY",
              "strokeWidth", "strokeAlpha", "fillAlpha", "trimPathStart", "trimPathEnd", "trimPathOffset", "strokeMiterLimit",
              "alpha", "startX", "startY", "endX", "endY", "centerX", "centerY", "gradientRadius", "offset"}
AAPT_NS = "{http://schemas.android.com/aapt}"


def _vd_lint(xml_text):
    """Offline VectorDrawable check (what `aapt2 link` would reject, plus runtime crashes aapt2 never sees): tag and
    attribute whitelist, enums, dimensions with units, colour formats, numbers, positive viewport, nan/inf and syntax
    in pathData, radial radius > 0, at least two gradient stops. [] = clean."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return [f"not well-formed XML: {e}"]
    errs = []
    if root.tag != "vector":
        return [f"root element must be <vector>, got <{root.tag}>"]
    dim = re.compile(r"^-?(\d+\.?\d*|\.\d+)(dp|dip|px|sp|pt|mm|in)$|^@(android:)?dimen/|^\?")
    col = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$|^@(android:)?color/|^\?")

    def check(el, parent):
        tag = el.tag.replace(AAPT_NS, "aapt:")
        allowed = _VD_ATTRS.get(tag)
        if tag == "aapt:attr":
            if set(el.attrib) - {"name"} or not re.fullmatch(r"android:(fillColor|strokeColor)", el.get("name", "")):
                errs.append(f'<aapt:attr> needs exactly name="android:fillColor|strokeColor"')
            if len(el) != 1 or el[0].tag != "gradient":
                errs.append("<aapt:attr> must contain exactly one <gradient>")
        elif allowed is None:
            errs.append(f"<{tag}> is not a VectorDrawable element" + (f" (inside <{parent}>)" if parent else ""))
            return
        else:
            for k, v in el.attrib.items():
                if not k.startswith(AND):
                    if not k.startswith("{"):
                        errs.append(f"<{tag}> {k}: attribute needs the android: namespace")
                    continue
                k = k[len(AND):]
                if k not in allowed:
                    errs.append(f"<{tag}> android:{k}: unknown attribute")
                    continue
                if k in _VD_ENUMS and v not in _VD_ENUMS[k]:
                    errs.append(f'<{tag}> {k}="{v}": expected {"|".join(sorted(_VD_ENUMS[k]))}')
                elif k in ("width", "height") and not dim.match(v):
                    errs.append(f'<{tag}> {k}="{v}": needs a dimension with unit (e.g. {v}dp)')
                elif k in _VD_COLORS and not col.match(v):
                    errs.append(f'<{tag}> {k}="{v}": not a colour (#RGB, #ARGB, #RRGGBB, #AARRGGBB or @color/)')
                elif k in _VD_FLOATS:
                    try:
                        f = float(v)
                        if math.isnan(f) or math.isinf(f):
                            raise ValueError
                    except ValueError:
                        errs.append(f'<{tag}> {k}="{v}": not a number')
                        continue
                    if k in ("viewportWidth", "viewportHeight") and f <= 0:
                        errs.append(f'<{tag}> {k}="{v}": must be > 0')
                    if k == "gradientRadius" and f <= 0:
                        errs.append(f'<gradient> gradientRadius="{v}": must be > 0 (radial gradients crash at inflation otherwise)')
                elif k == "pathData":
                    if re.search(r"\b(nan|inf)", v, re.I):
                        errs.append(f"<{tag}> pathData contains nan/inf")
                    else:
                        try:
                            parse_path(v)
                        except ValueError as e:
                            errs.append(f"<{tag}> pathData: {e}")
        if tag == "gradient":
            stops = [c for c in el if c.tag == "item"]
            if len(stops) < 2 and not (el.get(AND + "startColor") and el.get(AND + "endColor")):
                errs.append(f"<gradient> has {len(stops)} stop(s): Android needs at least two colours")
            if el.get(AND + "type") == "radial" and el.get(AND + "gradientRadius") is None:
                errs.append("<gradient android:type=\"radial\"> without gradientRadius")
        if tag == "vector":
            for k in ("width", "height", "viewportWidth", "viewportHeight"):
                if el.get(AND + k) is None:
                    errs.append(f"<vector> is missing android:{k}")
        for ch in el:
            if tag in ("gradient",) and ch.tag != "item":
                errs.append(f"<{ch.tag}> inside <gradient>")
                continue
            check(ch, tag)
    check(root, None)
    return errs


def aapt2_check(xml_files):
    """Compile (+ link when an android.jar exists: see aapt2_level) VectorDrawable files with aapt2.
    {file name: first real error line}; {} = every available step passed; None = no aapt2."""
    sdk = _android_sdk()
    if not sdk:
        return None
    aapt2, jar = sdk
    errs = {}
    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "res" / "drawable"
        res.mkdir(parents=True)
        for f in xml_files:
            shutil.copy(f, res / Path(f).name)
        (Path(td) / "AndroidManifest.xml").write_text(
            '<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="gptkit.check"/>')
        steps = [[aapt2, "compile", "-o", f"{td}/c.zip", "--dir", str(res.parent)]]
        if jar:
            steps.append([aapt2, "link", "-o", f"{td}/o.apk", "-I", jar, "--manifest", f"{td}/AndroidManifest.xml", f"{td}/c.zip"])
        generic = ("file failed to compile", "failed linking", "failed processing")
        for cmd in steps:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                return {"*": f"aapt2 did not run: {e}"}
            for line in (r.stderr + r.stdout).splitlines():
                m_ = re.search(r"drawable/([^/:]+\.xml)(?::\d+)?: error: (.*)", line)
                if m_ and (m_.group(1) not in errs or errs[m_.group(1)].startswith(generic)):
                    errs[m_.group(1)] = m_.group(2).strip()
            if r.returncode and not errs:
                errs["*"] = (r.stderr or r.stdout).strip()[:300] or f"aapt2 {cmd[1]} exited {r.returncode}"
            if errs:
                break
    return errs


def _aapt2_tree(res_dir, manifest=None, work=None, extra_res=None):
    """aapt2 over a whole res/ tree (drawables next to their PNGs, mipmaps, fonts): compile always, link when a platform
    android.jar exists (see aapt2_level). One line per contract C4: _aapt2_line() when every available step passed,
    'aapt2 compile|link FAILED: <first error text>' otherwise. `manifest` replaces the minimal AndroidManifest.xml used
    for link; `work` keeps the compiled zip/apk instead of a temp dir; `extra_res` ({"values/x.xml": text}) is added to a
    temporary copy of the tree (what a library such as androidx.core contributes at build time), never to res_dir."""
    sdk = _android_sdk()
    if not sdk:
        return _aapt2_line()
    aapt2, jar = sdk
    with tempfile.TemporaryDirectory() as td:
        w = Path(work) if work else Path(td)
        w.mkdir(parents=True, exist_ok=True)
        if extra_res:
            shutil.copytree(res_dir, Path(td) / "res")
            res_dir = Path(td) / "res"
            for rel, text in extra_res.items():
                (res_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                (res_dir / rel).write_text(text)
        steps = [("compile", [aapt2, "compile", "-o", str(w / "res.zip"), "--dir", str(res_dir)])]
        if jar:
            (w / "AndroidManifest.xml").write_text(
                manifest or '<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="gptkit.check"/>')
            steps.append(("link", [aapt2, "link", "-o", str(w / "check.apk"), "-I", jar, "--manifest",
                                   str(w / "AndroidManifest.xml"), str(w / "res.zip")]))
        for step, cmd in steps:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                return f"aapt2 {step} FAILED: aapt2 did not run ({e})"
            if r.returncode:
                return f"aapt2 {step} FAILED: " + (r.stderr or r.stdout).strip()[:300]
    return _aapt2_line()


def _report_dp(p):
    """Per-icon dp written by `svg` into svg_report.json next to (or one level above) the SVG, or None."""
    for d in (p.parent, p.parent.parent):
        rp = d / "svg_report.json"
        if rp.exists():
            try:
                dp = json.loads(rp.read_text()).get(p.stem, {}).get("dp")
                dp = float(dp) if dp else None
                if dp is not None and not 0 < dp < 1e5:  # also nan
                    raise ValueError(dp)
                return dp
            except (OSError, ValueError, AttributeError, TypeError):
                print(f"  ! {rp}: no usable dp for {p.stem} (hand-edited?); pass --dp")
                return None
    return None


def convert_vd(paths, out_dir, dp=None, prefix="ic_", check=True):
    """Convert SVG files to out_dir/<res name>.xml with the offline lint, render round-trip and aapt2; returns the number
    of files that failed (not written, lint or aapt2 error) after printing one line per file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, names, failed = [], {}, 0
    for p in paths:
        p, warns = Path(p), []
        try:
            svg = p.read_text(encoding="utf-8-sig")
            xml = svg_to_vd(svg, dp or _report_dp(p), warns)
        except OSError as e:
            print(f"  ! {p}: cannot read ({e.strerror or e})")
            failed += 1
            continue
        except (ValueError, ET.ParseError) as e:
            print(f"  ! {p.name}: {e}")
            failed += 1
            continue
        name = _res_name(prefix + p.stem, "ic_")
        if name in names:
            base, k = name, 2
            while name in names:
                name, k = f"{base}_{k}", k + 1
            print(f"  ! {p}: resource name {base} is already used by {names[base]}; writing {name}.xml")
        elif name != prefix + p.stem:
            print(f"  renamed {p.name} -> {name}.xml (Android resource names: [a-z0-9_], starting with a letter)")
        names[name] = p
        out = out_dir / f"{name}.xml"
        out.write_text(xml)
        done.append((out, svg, xml, warns))
    rts = vd_roundtrip([(s, x) for _, s, x, _ in done]) if done and check else [None] * len(done)
    for (out, _, xml, warns), rt in zip(done, rts):
        notes = [f"  ! {w}" for w in dict.fromkeys(warns) if not w.startswith("needs API")]
        notes += [f"  ({w})" for w in dict.fromkeys(warns) if w.startswith("needs API")]
        lint = _vd_lint(xml)
        if lint:
            failed += 1
        if rt is not None and (rt["mean"] > 2 or rt["strong"] > 16):
            notes.append(f"  ! render differs from the SVG (mean diff {rt['mean']}, {rt['strong']} px far off)")
        match = f"  (render match {rt['mean']})" if rt is not None and rt["mean"] <= 2 and rt["strong"] <= 16 else ""
        print(f"  {out}{match}{''.join(notes)}")
        for e in lint:
            print(f"      ! lint: {e}")
    if done and check and all(rt is None for rt in rts):
        why = "no SVG renderer" if not _renderer_name() else "renderer failed"
        print(f"  (render check skipped: {why}{': ' + _chrome_probe()[1] if not _renderer_name() else ''})")
    written = [x[0] for x in done]
    if written and check:
        errs = aapt2_check(written)
        if errs is None:
            print(f"  {_aapt2_line()}")
        elif errs:
            for k, v in errs.items():
                print(f"  ! aapt2: {k}: {v}")
            failed += len(errs)
        else:
            print(f"  {_aapt2_line(len(written))}")
    return failed


def cmd_vd(a):
    paths = []
    for f in a.svgs:
        p = Path(f)
        if p.is_dir():
            found = sorted(p.glob("*.svg"))
            if not found:
                print(f"  ! {p}: no .svg files in this directory")
            paths += found
        else:
            paths.append(p)
    if not paths:
        sys.exit("vd: no SVG files to convert")
    if convert_vd(paths, a.out, a.dp, a.prefix, not a.no_check):
        sys.exit(1)


# ---- SVG icons: fidelity retry, 24-grid normalisation, icon-set spec + consistency, new icons

SVG_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["files"], "properties": {"files": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "svg"],
                               "properties": {"name": {"type": "string"}, "svg": {"type": "string"}}}}}}

ICON_SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["style", "stroke_px", "stroke_24", "linecap", "linejoin", "corner_radius_24", "traits", "states", "icons"],
    "properties": {
        "style": {"type": "string", "enum": ["outlined", "filled", "mixed", "duotone"]},
        "stroke_px": {"type": "number"}, "stroke_24": {"type": "number"},
        "linecap": {"type": "string", "enum": ["round", "butt", "square"]},
        "linejoin": {"type": "string", "enum": ["round", "miter", "bevel"]},
        "corner_radius_24": {"type": "number"}, "traits": {"type": "string"},
        "states": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["state", "treatment", "color"],
            "properties": {"state": {"type": "string"}, "treatment": {"type": "string", "enum": ["outlined", "filled"]},
                           "color": {"type": "string"}}}},
        "icons": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "state", "treatment", "material_symbol", "sf_symbol"],
            "properties": {"name": {"type": "string"}, "state": {"type": "string"},
                           "treatment": {"type": "string", "enum": ["outlined", "filled"]},
                           "material_symbol": {"type": "string"}, "sf_symbol": {"type": "string"}}}}},
}

ICON_SPEC_PROMPT = """This is an APPROVED UI design ({W}x{H} px). After it come zoomed crops of its icons, each labelled with
its name. Describe the icon SET's shared style precisely enough that new icons can be drawn to match:
- style, stroke width in design px for the most common icon size, and the same stroke on a 24-unit grid whose live
  area is 20 units (stroke_px * 20 / icon size in px), line caps and joins, typical corner radius on that 24 grid;
- traits: geometry (geometric/organic), terminal shapes, level of detail, how active/selected icons differ;
- states: each visual state (e.g. "tab inactive", "tab active", "on accent button") with its treatment and hex colour;
- icons: for every listed icon, its state and treatment, plus the closest standard glyph name in Google Material
  Symbols (snake_case, e.g. home, settings, favorite, person, explore, play_arrow, chevron_right, graphic_eq, air) and
  Apple SF Symbols (e.g. house.fill, gearshape, heart, person, safari, play.fill, chevron.right, waveform, wind).
  Use "" when no standard glyph is a close match."""

SVG_FIX_PROMPT = """You wrote these SVGs for assets of an APPROVED design; each one either does not match the design closely
enough or breaks a VectorDrawable rule listed below. For each asset the image (when present) shows LEFT the design element
(zoomed screenshot on its backdrop) and RIGHT your SVG rendered at the same scale on the same backdrop; then come the
measured problems and your current SVG. Rewrite each SVG so RIGHT matches LEFT 1:1 and every rule holds: fix the listed
problems, keep what already matches, do not simplify, return the same names."""

ICONS_PROMPT = """Design NEW icons for the app in the first image (an APPROVED design). They must be members of the SAME icon
set as its existing icons (the second image shows them; their SVG sources follow): same grid, stroke weight, caps/joins,
corner rounding, level of detail and optical size, so nobody can tell which icons were added later.
Concepts: {concepts}.
Draw each on a 24x24 grid: viewBox="0 0 24 24", artwork inside the 2..22 live area and optically centred, {treatment},
stroke-width="{sw}", stroke-linecap="{cap}", stroke-linejoin="{join}", colour {color}. Use the simplest recognisable
metaphor (the one Material Symbols / SF Symbols use for it). No text, no background shape. Name each file after its
concept in snake_case."""

_SHAPES = ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon")


def _vb(root):
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if v]
    return vb if len(vb) == 4 else [0, 0, _num(root.get("width"), 24), _num(root.get("height"), 24)]


def _units_box(render, vb, par="xMidYMid meet"):
    """Alpha bounding box of a render (made by render_svgs: the SVG sized to a square) in viewBox units, honouring the
    root's preserveAspectRatio for non-square viewBoxes."""
    bb = content_box(render, 25)
    if not bb:
        return None
    R = render.width
    words = (par or "xMidYMid meet").split()
    align, slice_ = words[0], "slice" in words
    if align == "none":
        sx, sy, ox, oy = R / vb[2], R / vb[3], 0.0, 0.0
    else:
        sx = sy = R / (min(vb[2], vb[3]) if slice_ else max(vb[2], vb[3]))
        pos = lambda key, span: 0.0 if key + "min" in align.lower() else (R - span) if key + "max" in align.lower() else (R - span) / 2
        ox, oy = pos("x", vb[2] * sx), pos("y", vb[3] * sy)
    return (vb[0] + (bb[0] - ox) / sx, vb[1] + (bb[1] - oy) / sy, vb[0] + (bb[2] - ox) / sx, vb[1] + (bb[3] - oy) / sy)


def _bake_transforms(txt):
    """The SVG with every transform multiplied into absolute path coordinates (arcs become cubics), so the grid
    normaliser can rescale it. None when it cannot be done exactly: non-uniform scale or skew on stroked shapes,
    gradientTransform, nested <svg>, <use>, or percent units."""
    root = ET.fromstring(txt)
    if not any(el.get("transform") for el in root.iter()):
        return txt
    if any(el.get("gradientTransform") for el in root.iter()) or \
       any(el.tag.replace(SVG_NS, "") in ("svg", "use", "symbol") for el in list(root.iter())[1:]):
        return None
    mul, apply = _mmul, lambda m, pt: _mapply(m, *pt)  # the one affine helper set (no private copies)
    ok = [True]

    def walk(el, m):
        for ch in list(el):
            tag = ch.tag.replace(SVG_NS, "")
            cm = mul(m, _tf_affine(ch.get("transform"))) if ch.get("transform") else m
            if ch.get("transform"):
                del ch.attrib["transform"]
            if tag in ("g", "a", "switch", "defs", "clipPath", "mask", "marker", "pattern"):
                walk(ch, cm)
                continue
            if tag.endswith("Gradient"):
                if cm != _IDENTITY:
                    ok[0] = False
                continue
            if tag not in _SHAPES or cm == _IDENTITY:
                continue
            a, b, c, d = cm[:4]
            uniform = abs(math.hypot(a, b) - math.hypot(c, d)) < 1e-6 and abs(a * c + b * d) < 1e-6
            st = _style_dict(ch)
            stroked = (st.get("stroke") or ch.get("stroke") or (el.get("stroke") if tag != "g" else None) or "none") != "none"
            if not uniform and (stroked or root.get("stroke") not in (None, "none")):
                ok[0] = False
                return
            dpath = _shape_path(ch)
            if not dpath or any("%" in (ch.get(k) or "") for k in ch.attrib):
                ok[0] = False
                return
            subs = _subpaths(dpath)
            for sp in subs:
                sp["pts"] = [apply(cm, pt) for pt in sp["pts"]]
                sp["segs"] = [(sg[0], *[apply(cm, pt) for pt in sg[1:]]) for sg in sp["segs"]]
            k = math.hypot(a, b)
            for key in ("stroke-width", "stroke-dashoffset"):
                if st.get(key):
                    st[key] = _fmt(_num(st[key]) * k)
                elif ch.get(key):
                    ch.set(key, _fmt(_num(ch.get(key)) * k))
                elif key == "stroke-width" and stroked and abs(k - 1) > 1e-6:
                    ch.set(key, _fmt(_num(el.get(key) or root.get(key), 1.0) * k))
            if st.get("stroke-dasharray") not in (None, "none"):
                st["stroke-dasharray"] = " ".join(_fmt(float(v) * k) for v in _NUM.findall(st["stroke-dasharray"]))
            elif ch.get("stroke-dasharray") not in (None, "none"):
                ch.set("stroke-dasharray", " ".join(_fmt(float(v) * k) for v in _NUM.findall(ch.get("stroke-dasharray"))))
            if st:
                ch.set("style", ";".join(f"{k_}:{v}" for k_, v in st.items()))
            for key in ("cx", "cy", "r", "rx", "ry", "x", "y", "width", "height", "x1", "y1", "x2", "y2", "points"):
                ch.attrib.pop(key, None)
            ch.tag = SVG_NS + "path" if ch.tag.startswith(SVG_NS) else "path"
            ch.set("d", _d(subs))
    walk(root, _IDENTITY)
    return ET.tostring(root, encoding="unicode") if ok[0] else None


def normalize_svg(txt, box, grid=24, pad=2.0, scale=None):
    """Scale coordinates (never adding transforms) so content box `box` fits a grid x grid viewBox inside `pad`;
    `scale` overrides the fit factor (set-level normalisation: one scale per size group), the box is still centred."""
    baked = _bake_transforms(txt)
    if baked is None:
        return None
    root = ET.fromstring(baked)
    s = scale or (grid - 2 * pad) / max(box[2] - box[0], box[3] - box[1], 1e-6)
    tx, ty = grid / 2 - s * (box[0] + box[2]) / 2, grid / 2 - s * (box[1] + box[3]) / 2
    fx = {"x": lambda v: _fmt(_num(v) * s + tx), "y": lambda v: _fmt(_num(v) * s + ty), "l": lambda v: _fmt(_num(v) * s)}
    attrs = {"circle": {"cx": "x", "cy": "y", "r": "l"}, "ellipse": {"cx": "x", "cy": "y", "rx": "l", "ry": "l"},
             "rect": {"x": "x", "y": "y", "width": "l", "height": "l", "rx": "l", "ry": "l"},
             "line": {"x1": "x", "y1": "y", "x2": "x", "y2": "y"},
             "linearGradient": {"x1": "x", "y1": "y", "x2": "x", "y2": "y"},
             "radialGradient": {"cx": "x", "cy": "y", "r": "l", "fx": "x", "fy": "y", "fr": "l"}}
    stroked = False
    for el in root.iter():
        tag = el.tag.replace(SVG_NS, "")
        if tag.endswith("Gradient") and el.get("gradientUnits") != "userSpaceOnUse":
            continue
        for k, kind in attrs.get(tag, {}).items():
            if el.get(k) is not None:
                if "%" in el.get(k):
                    return None
                el.set(k, fx[kind](el.get(k)))
        if tag == "path" and el.get("d"):
            el.set("d", path_str(transform_path(parse_path(el.get("d")), s, tx, ty)))
        if tag in ("polyline", "polygon"):
            pts = [float(v) for v in _NUM.findall(el.get("points", ""))]
            el.set("points", " ".join(f"{_fmt(pts[i] * s + tx)},{_fmt(pts[i + 1] * s + ty)}" for i in range(0, len(pts) - 1, 2)))
        st = _style_dict(el)
        for k in ("stroke-width", "stroke-dashoffset"):
            if el.get(k) is not None:
                el.set(k, fx["l"](el.get(k)))
            if k in st:
                st[k] = fx["l"](st[k])
        if el.get("stroke-dasharray") not in (None, "none"):
            el.set("stroke-dasharray", " ".join(_fmt(float(v) * s) for v in _NUM.findall(el.get("stroke-dasharray"))))
        if st.get("stroke-dasharray") not in (None, "none"):
            st["stroke-dasharray"] = " ".join(_fmt(float(v) * s) for v in _NUM.findall(st["stroke-dasharray"]))
        if st:
            el.set("style", ";".join(f"{k}:{v}" for k, v in st.items()))
        stroked |= (st.get("stroke") or el.get("stroke") or "none") != "none"
    if stroked and root.get("stroke-width") is None and "stroke-width" not in _style_dict(root):
        root.set("stroke-width", _fmt(s))
    for k in ("width", "height"):
        root.set(k, _fmt(grid))
    root.set("viewBox", f"0 0 {_fmt(grid)} {_fmt(grid)}")
    return ET.tostring(root, encoding="unicode")


def svg_style(txt):
    """Effective style facts of an SVG: viewBox size, stroke widths, caps/joins, fill/stroke use, colours."""
    root = ET.fromstring(txt)
    vb = _vb(root)
    css = _css_rules(root)
    info = {"viewBox": [vb[2], vb[3]], "strokes": [], "caps": set(), "joins": set(), "filled": False,
            "stroked": False, "colors": set(), "transform": False}

    def walk(el, attrs):
        for ch in el:
            tag = ch.tag.replace(SVG_NS, "")
            at = _inherit(ch, attrs, css)
            info["transform"] |= bool(ch.get("transform"))
            if tag in ("g", "a", "switch"):
                walk(ch, at)
            if tag not in _SHAPES:
                continue
            fill, stroke = at.get("fill", "#000000"), at.get("stroke", "none")
            if stroke not in ("none", "transparent"):
                info["stroked"] = True
                info["strokes"].append(_num(at.get("stroke-width"), 1.0))
                segs = parse_path(ch.get("d")) if tag == "path" else []
                open_ = tag in ("line", "polyline") or (tag == "path" and segs and segs[-1][0] not in "Zz")
                if open_:
                    info["caps"].add(at.get("stroke-linecap", "butt"))
                if tag in ("path", "polyline", "polygon") or (tag == "rect" and not (ch.get("rx") or ch.get("ry"))):
                    info["joins"].add(at.get("stroke-linejoin", "miter"))
                info["colors"].add(stroke.lower())
            if fill not in ("none", "transparent") and tag != "line":
                info["filled"] = True
                info["colors"].add(fill.lower())
    walk(root, _inherit(root, {}, css))
    return info


def _stroke24(st):
    return round(float(np.median(st["strokes"])) * 24 / max(st["viewBox"]), 2) if st["strokes"] else None


def icon_set_report(entries, tol=0.15):
    """entries: [{name, svg, size (design px or None)}] -> consistency notes (stroke outliers per size group, caps, viewBox)."""
    rows, notes = [], []
    for e in entries:
        try:
            st = svg_style(e["svg"])
        except ET.ParseError as ex:
            notes.append(f"{e['name']}: invalid XML ({ex})")
            continue
        except ValueError as ex:
            notes.append(f"{e['name']}: unreadable path data ({ex})")
            continue
        rows.append({**e, "style": st, "sw": _stroke24(st)})
    vbs = {tuple(r["style"]["viewBox"]) for r in rows}
    if len(vbs) > 1:
        notes.append("mixed viewBoxes: " + ", ".join(f"{r['name']} {r['style']['viewBox'][0]:g}x{r['style']['viewBox'][1]:g}"
                                                     for r in rows))
    groups = []
    for r in sorted(rows, key=lambda r: r["size"] or 0):
        g = next((g for g in groups if (g[0]["size"] is None) == (r["size"] is None) and
                  (r["size"] is None or r["size"] <= g[0]["size"] * 1.25)), None)
        (g.append(r) if g is not None else groups.append([r]))
    for g in groups:
        strokes = [r for r in g if r["sw"]]
        if len(strokes) >= 2:
            med = float(np.median([r["sw"] for r in strokes]))
            for r in strokes:
                if abs(r["sw"] - med) / med > tol:
                    notes.append(f"{r['name']}: stroke {r['sw']} on the 24 grid vs {med:.2f} for its size group "
                                 f"({', '.join(x['name'] for x in strokes)})")
        for key in ("caps", "joins"):
            vals = [v for r in strokes for v in r["style"][key]]
            if vals:
                top = max(set(vals), key=vals.count)
                for r in strokes:
                    if r["style"][key] - {top}:
                        notes.append(f"{r['name']}: {key[:-1]} {'/'.join(sorted(r['style'][key]))} vs {top} in its group")
    for r in rows:
        if r["style"]["transform"]:
            notes.append(f"{r['name']}: uses transform attributes")
    return rows, notes


def _load_spec(d):
    p = Path(d) / "icon_spec.json"
    spec = _read_json(p, "icon_spec.json") if p.exists() else None
    if p.exists() and not isinstance(spec, dict):
        sys.exit(f"gptkit: icon_spec.json {p} is not a JSON object (hand-edited?): delete it or re-run `iconspec {d}`")
    return spec


def _spec_text(spec):
    if not spec:
        return ""
    states = "; ".join(f"{s['state']}: {s['treatment']} {s['color']}" for s in spec.get("states", []))
    return (f"\nICON SET SPEC (every icon must follow it so the set stays uniform): {spec['style']} style; stroke about "
            f"{spec['stroke_px']:g} px in the design ({spec['stroke_24']:g} on a 24-unit grid); {spec['linecap']} caps, "
            f"{spec['linejoin']} joins; corner radius about {spec['corner_radius_24']:g} on the 24 grid. {spec['traits']} "
            f"States: {states}.")


def _svg_issues(sc):
    out = []
    if sc["fit"] < 0.9 or sc["rel"] > 0.45:
        out.append(f"the silhouette differs from the design (overlap {sc['iou']:.2f}): check parts, proportions, positions")
    if sc["ink"] > 1.3:
        out.append(f"about {round((sc['ink'] - 1) * 100)}% heavier than the design: thinner strokes or smaller filled areas")
    elif sc["ink"] < 0.77:
        out.append(f"about {round((1 - sc['ink']) * 100)}% lighter than the design: thicker strokes or larger/filled areas")
    if sc["dE"] > 12:
        out.append(f"colour is off: design {sc['design_hex']}, yours {sc['svg_hex']}")
    if sc["aspect_err"] > 0.15:
        out.append("the width/height ratio differs from the design")
    return out


def _evaluate_svgs(mock, assets, svgs):
    """{name: svg} -> {name: {svg, score, issues, image}} with VectorDrawable-compatibility and fidelity checks."""
    res, texts = {}, {}
    for name, svg in svgs.items():
        issues = []
        try:
            ET.fromstring(svg)
            warns = []
            svg_to_vd(svg, 24, warns)
            issues += [f"not VectorDrawable-safe: {w}" for w in dict.fromkeys(warns) if not w.startswith("needs API")]
            texts[name] = svg
        except ET.ParseError as e:
            issues.append(f"invalid XML: {e}")
        except ValueError as e:
            issues.append(f"not convertible to VectorDrawable: {e}")
            texts[name] = svg
        res[name] = {"svg": svg, "score": None, "issues": issues, "image": None}
    names = list(texts)
    for name, r in zip(names, render_svgs([texts[n] for n in names], 256)):
        if r is None:
            continue
        a = assets.get(name)
        if a:
            sc, img = svg_fidelity(mock, a["bbox"], r, visual=True)
            res[name].update(score=sc, image=img, render=r)
            res[name]["issues"] += _svg_issues(sc)
        else:
            res[name]["render"] = r
    return res


def _svg_rank(e):
    hard = sum(1 for i in e["issues"] if i.startswith(("invalid", "not convertible")))
    return hard, e["score"]["rel"] if e["score"] else 9.0


def _norm_name(n):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", n.lower().removesuffix(".svg"))).strip("_")


def _size_groups(sizes):
    """Indices grouped like icon_set_report: sorted by size, sizes within 1.25x of the group's smallest; None sizes alone."""
    groups = []
    for i in sorted(range(len(sizes)), key=lambda i: sizes[i] or 0):
        g = next((g for g in groups if (sizes[g[0]] is None) == (sizes[i] is None) and
                  (sizes[i] is None or sizes[i] <= sizes[g[0]] * 1.25)), None)
        (g.append(i) if g is not None and sizes[i] is not None else groups.append([i]))
    return groups


def _grid_normalize(svgs, grid=24, pad=2.0, sizes=None, dps=None, with_dp=False):
    """[svg] -> [svg rescaled onto the grid, or None], from one batch render that measures each content box.
    With `sizes` (each icon's long side in design px) icons are normalised per size group: the group's largest icon
    fills the live area and the others keep their relative size, so one stroke and one display dp serve the group.
    `dps` (design dp per icon) gives that dp; with_dp=True returns (svg, dp) pairs."""
    n = len(svgs)
    boxes, vbs = [None] * n, [None] * n
    for i, (s, r) in enumerate(zip(svgs, render_svgs(svgs, 512))):
        try:
            root = ET.fromstring(s)
            vbs[i] = _vb(root)
            boxes[i] = r and _units_box(r, vbs[i], root.get("preserveAspectRatio"))
        except (ValueError, ET.ParseError):
            pass
    scales, dp_out = [None] * n, [None] * n
    long = lambda b: max(b[2] - b[0], b[3] - b[1], 1e-6)
    for g in (_size_groups(sizes) if sizes else [[i] for i in range(n)]):
        g = [i for i in g if boxes[i]]
        if not g:
            continue
        ref = max(g, key=lambda i: sizes[i] if sizes else 0)
        ref_px = sizes[ref] if sizes and sizes[ref] else None
        for i in g:
            if ref_px and sizes[i]:
                scales[i] = (grid - 2 * pad) / (long(boxes[i]) * ref_px / sizes[i])
            else:
                scales[i] = (grid - 2 * pad) / long(boxes[i])
            if dps and dps[ref]:
                dp_out[i] = round(dps[ref] * grid / (grid - 2 * pad), 1)
    out = []
    for i, s in enumerate(svgs):
        norm = None
        if boxes[i]:
            try:
                norm = normalize_svg(s, boxes[i], grid, pad, scales[i])
            except (ValueError, ET.ParseError):
                norm = None
        out.append((norm, dp_out[i]) if with_dp else norm)
    return out


def _svg_schema(names):
    """SVG_SCHEMA with `name` restricted to the requested names (strict mode then stops renamed and extra files)."""
    sch = json.loads(json.dumps(SVG_SCHEMA))
    if names:
        sch["properties"]["files"]["items"]["properties"]["name"]["enum"] = list(names)
    return sch


_LAST_BATCH_ERROR = None  # the last ApiError an _ask_batches() request raised (cmd_svg names it when icons stay missing)


def _ask_batches(fn, batches, what, fatal=True):
    """Run fn(batch) -> files in a pool; {returned name: svg} plus the batches whose request failed (printed). When every
    batch failed and `fatal`, the last ApiError is raised (the first request of a run); retries pass fatal=False."""
    got, failed = {}, []
    global _LAST_BATCH_ERROR
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(fn, b): b for b in batches}
        for fut in cf.as_completed(futs):
            b = futs[fut]
            try:
                for f in fut.result():
                    got[_norm_name(f["name"])] = _ensure_xmlns(f["svg"])
            except ApiError as e:
                _LAST_BATCH_ERROR = e
                if len(batches) > 1:
                    print(f"  ! {what} failed for {', '.join(n for n, *_ in b) if b and isinstance(b[0], tuple) else ', '.join(x['name'] for x in b)}: {e}")
                failed.append(b)
    if fatal and failed and len(failed) == len(batches):
        raise _LAST_BATCH_ERROR   # nothing came back at all: the error itself is the one-line exit (main() formats it)
    return got, failed


def cmd_svg(a):
    if not (math.isfinite(a.grid) and 0 <= a.grid <= 1024):
        sys.exit(f"svg: --grid {a.grid:g} must be a number of viewBox units, 0-1024 (24 = Material/SF grid, 0 = keep as drawn)")
    if not (math.isfinite(a.pad) and 0 <= a.pad and (not a.grid or a.pad < a.grid / 2)):
        sys.exit(f"svg: --pad {a.pad:g} must be >= 0 and less than half of --grid {a.grid:g}")
    d, m, mock = _load(a.dir)
    if a.only:
        known = {x["name"] for x in m["assets"]}
        unknown = [n for n in a.only.split(",") if n and n not in known]
        if unknown:
            sys.exit(f"svg: unknown asset(s): {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    todo = [x for x in _select(m, a.only) if x["format"] == "svg" or a.only]
    if not todo:
        sys.exit(f"svg: no svg-format assets in {d / 'assets.json'} (set \"format\": \"svg\" on the icons, or pass --only NAME)")
    for sub in ("preview", "raw"):
        (d / "svg" / sub).mkdir(parents=True, exist_ok=True)
    spec = _load_spec(d)
    per_icon = {i["name"]: i for i in (spec or {}).get("icons", [])}
    by_name = {x["name"]: x for x in todo}
    aliases = {_norm_name(n): n for n in by_name}  # model names come back normalised; assets.json names may not be
    batches = [todo[i:i + 6] for i in range(0, len(todo), 6)]
    if spec:
        print(f"using icon set spec {d / 'icon_spec.json'}")
    if a.check:
        return _svg_check(d, m, mock, by_name)

    def run(batch):
        parts = [f"The first image is an APPROVED UI design. After it come zoomed crops of specific graphics. {SVG_RULES}"
                 f"{_spec_text(spec)}\nReturn one file per requested asset, name exactly as given.", mock.convert("RGB")]
        for x in batch:
            b = x["bbox"]
            st = per_icon.get(x["name"])
            state = f" State: {st['state']}, {st['treatment']}." if st else ""
            parts += [f"Asset \"{x['name']}\" at bbox x={b['x']},y={b['y']},w={b['w']},h={b['h']}: {x['description']} "
                      f"Colours: {', '.join(x.get('colors', []))}.{state}", ref_crop(mock, b, margin=0.25, long_side=512)]
        return ask(parts, _svg_schema([x["name"] for x in batch]), "svg_files")["files"]

    def fix(batch):
        parts = [SVG_FIX_PROMPT + "\n" + SVG_RULES + _spec_text(spec)]
        for name, e in batch:
            parts += [f"Asset \"{name}\": {by_name[name]['description']}"]
            if e["image"] is not None:
                parts.append(e["image"])
            parts.append("Problems: " + "; ".join(e["issues"]) + f"\nCurrent SVG:\n{e['svg']}")
        return ask(parts, _svg_schema([n for n, _ in batch]), "svg_files")["files"]

    def keep(got):
        out = {}
        for n, svg in got.items():
            if n in aliases:
                out[aliases[n]] = svg
            else:
                print(f"  ! model returned an unrequested file {n!r}: not written")
        return out
    got, _ = _ask_batches(run, batches, "svg request")
    first = keep(got)
    missing = [n for n in by_name if n not in first]
    if missing:
        print(f"  ! not returned: {', '.join(missing)}; asking once more for them")
        got, _ = _ask_batches(run, [[by_name[n] for n in missing[i:i + 6]] for i in range(0, len(missing), 6)], "svg request", fatal=False)
        first.update(keep(got))
        missing = [n for n in by_name if n not in first]
    evals = _evaluate_svgs(mock, by_name, first)
    history = {n: [e] for n, e in evals.items()}
    todo_fix = [(n, e) for n, e in evals.items() if e["issues"] and n in by_name] if a.retry else []
    if todo_fix:
        print(f"retrying {len(todo_fix)} below threshold: {', '.join(n for n, _ in todo_fix)}")
        fixed, _ = _ask_batches(fix, [todo_fix[i:i + 4] for i in range(0, len(todo_fix), 4)], "svg retry", fatal=False)
        fixed = keep(fixed)
        for n, e in _evaluate_svgs(mock, by_name, {k: v for k, v in fixed.items() if k in history}).items():
            history[n].append(e)
    report_path = d / "svg_report.json"
    try:  # read after the API calls: a damaged old report must not throw this run's icons away
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
    except (OSError, ValueError) as e:
        print(f"  ! {report_path} is unreadable ({e}); rewritten with this run's icons")
        report = {}
    report = report if isinstance(report, dict) else {}
    rows = []
    best = {n: min(t, key=_svg_rank) for n, t in history.items()}
    final = {n: b["svg"] for n, b in best.items()}
    grid_ok = [n for n, b in best.items() if a.grid and by_name.get(n, {}).get("kind") == "icon" and
               not any(i.startswith("invalid") for i in b["issues"])]
    dps = {}
    px_per_dp = m["width"] / _screen_dp(m.get("platform"), m.get("screen_dp"))
    long_px = lambda x: max(x["bbox"]["w"], x["bbox"]["h"])
    long_dp = lambda x: max(x["dp"]["w"], x["dp"]["h"]) if isinstance(x.get("dp"), dict) else long_px(x) / px_per_dp
    for n, (norm, dp) in zip(grid_ok, _grid_normalize([final[n] for n in grid_ok], a.grid, a.pad,
                                                       sizes=[long_px(by_name[n]) for n in grid_ok],
                                                       dps=[long_dp(by_name[n]) for n in grid_ok], with_dp=True)):
        if norm:
            final[n] = norm
        if dp:
            dps[n] = dp
    previews = dict(zip(final, render_svgs(list(final.values()), 256)))
    for n in grid_ok:
        sc0 = best[n]["score"]
        if final[n] != best[n]["svg"] and sc0 and previews[n] is not None:
            if svg_fidelity(mock, by_name[n]["bbox"], previews[n])["rel"] > sc0["rel"] + 0.05:
                print(f"  ! {n}: grid normalisation changed the render, kept the original viewBox")
                final[n] = best[n]["svg"]
                previews[n] = render_svgs([final[n]], 256)[0]
    for name, tries in history.items():
        for i, e in enumerate(tries):
            (d / "svg" / "raw" / f"{name}_try{i + 1}.svg").write_text(e["svg"])
        b, asset, svg, prev = best[name], by_name.get(name, {}), final[name], previews[name]
        grid = a.grid if svg != b["svg"] else None
        p = d / "svg" / f"{name}.svg"
        p.write_text(svg)
        if prev:
            prev.save(d / "svg" / "preview" / f"{name}.png")
        sc = b["score"]
        line = f"  {p}"
        if sc:
            line += f"  rel={sc['rel']} fit={sc['fit']} ink={sc['ink']} dE={sc['dE']}"
        if len(tries) > 1:
            r0, r1 = (t["score"]["rel"] if t["score"] else "-" for t in tries[:2])
            line += f"  retry rel {r0} -> {r1}: {'KEPT' if b is tries[-1] else 'discarded'}"
        dp = dps.get(name)
        if grid and dp:
            line += f"  (24-grid; use at {dp:g}dp)"
        if asset.get("material_symbol") or asset.get("sf_symbol"):
            line += f"  [material: {asset.get('material_symbol') or '-'}, sf: {asset.get('sf_symbol') or '-'}]"
        print(line)
        for i in b["issues"]:
            print(f"      ! {i}")
        report[name] = {"svg": str(p), "grid": grid, "dp": dp if grid else None, "score": sc, "issues": b["issues"],
                        "tries": len(tries), "kept_try": tries.index(b) + 1}
        if sc and prev:
            rows.append([(f"{name}: design | svg", b["image"])] +
                        [(f"try {i + 1}: rel={t['score']['rel']}" + (" KEPT" if t is b else ""), t["render"])
                         for i, t in enumerate(tries) if t["score"]] +
                        [(f"final ink={sc['ink']} dE={sc['dE']}", prev)])
    for n in missing:
        print(f"  ! {n}: model returned no SVG (re-run: svg {d} --only {n})")
        report[n] = {"error": "missing"}
    for n in list(report):
        if n not in {x["name"] for x in m["assets"]}:
            del report[n]  # stale entries for assets no longer in assets.json
    report_path.write_text(json.dumps(report, indent=2))
    if rows:
        (d / "review").mkdir(exist_ok=True)
        stamp = time.strftime("%H%M%S")
        for i in range(0, len(rows), 8):
            print("review sheet ->", sheet(rows[i:i + 8], d / "review" / f"svg_{stamp}_{i // 8 + 1}.png", T=240))
    entries = [{"name": x["name"], "svg": (d / "svg" / f"{x['name']}.svg").read_text(),
                "size": max(x["bbox"]["w"], x["bbox"]["h"])}
               for x in m["assets"] if x["format"] == "svg" and x.get("kind") == "icon" and (d / "svg" / f"{x['name']}.svg").exists()]
    _, notes = icon_set_report(entries)
    print("icon set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")
    if missing:
        sys.exit(f"svg: {len(missing)} icon(s) missing ({', '.join(missing)}): re-run svg {d} --only {','.join(missing)}"
                 + (f"; last request error: {_LAST_BATCH_ERROR}" if _LAST_BATCH_ERROR else ""))


def _svg_check(d, m, mock, by_name):
    have = {n: (d / "svg" / f"{n}.svg").read_text() for n in by_name if (d / "svg" / f"{n}.svg").exists()}
    if not have:
        sys.exit(f"svg --check: no svg files to check in {d / 'svg'} (run svg first)")
    rows = []
    for n, e in _evaluate_svgs(mock, by_name, have).items():
        sc = e["score"]
        print(f"  {n:22s} " + (f"rel={sc['rel']} fit={sc['fit']} ink={sc['ink']} dE={sc['dE']}" if sc else "not scored"))
        for i in e["issues"]:
            print(f"      ! {i}")
        if sc:
            rows.append([(f"{n}: design | svg", e["image"]), (f"rel={sc['rel']} ink={sc['ink']}", e["render"])])
    if rows:
        (d / "review").mkdir(exist_ok=True)
        print("review sheet ->", sheet(rows, d / "review" / f"svgcheck_{time.strftime('%H%M%S')}.png", T=240))
    missing = [n for n in by_name if n not in have]
    if missing:
        print(f"  ! not in {d / 'svg'}: {', '.join(missing)}")
    _, notes = icon_set_report([{"name": n, "svg": s, "size": max(by_name[n]["bbox"]["w"], by_name[n]["bbox"]["h"])}
                                for n, s in have.items() if by_name[n].get("kind") == "icon"])
    print("icon set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")


def cmd_iconspec(a):
    d, m, mock = _load(a.dir)
    icons = [x for x in m["assets"] if x.get("format") == "svg"]
    if not icons:
        sys.exit(f"gptkit: iconspec: no format \"svg\" assets in {d / 'assets.json'} (run `manifest` first, or set "
                 "\"format\": \"svg\" on the icon entries)")
    parts = [ICON_SPEC_PROMPT.format(W=mock.width, H=mock.height), mock.convert("RGB")]
    for x in icons:
        parts += [f"Icon \"{x['name']}\" ({str(x.get('description') or '')[:160]})",
                  ref_crop(mock, x["bbox"], margin=0.25, long_side=256)]
    spec = ask(parts, ICON_SPEC_SCHEMA, "icon_spec", model=a.model)
    if not isinstance(spec, dict) or not isinstance(spec.get("icons"), list):
        sys.exit(f"gptkit: the model's icon-spec response has no 'icons' list (non-conforming gateway?): {str(spec)[:200]}")
    spec["icons"] = [i for i in spec["icons"] if isinstance(i, dict) and i.get("name")]
    for i in spec["icons"]:
        i["name"] = _norm_name(str(i["name"]))
        i["material_symbol"], i["sf_symbol"] = _symbol_names(str(i.get("material_symbol") or ""),
                                                             str(i.get("sf_symbol") or ""))
    (d / "icon_spec.json").write_text(json.dumps(spec, indent=2))
    names = {i["name"]: i for i in spec["icons"]}
    for x in m["assets"]:
        if x["name"] in names:
            x["material_symbol"], x["sf_symbol"] = names[x["name"]]["material_symbol"], names[x["name"]]["sf_symbol"]
    (d / "assets.json").write_text(json.dumps(m, indent=2))
    g = lambda k: spec.get(k, "?")
    num = lambda k: f"{spec[k]:g}" if isinstance(spec.get(k), (int, float)) else "?"
    print(f"{g('style')} icons, stroke {num('stroke_px')}px ({num('stroke_24')} on 24 grid), {g('linecap')} caps, "
          f"{g('linejoin')} joins, radius {num('corner_radius_24')}")
    print(f"traits: {g('traits')}")
    for s in spec.get("states") or []:
        if isinstance(s, dict):
            print(f"  state {s.get('state')}: {s.get('treatment')} {s.get('color')}")
    for i in spec["icons"]:
        print(f"  {i['name']:22s} {str(i.get('state')):18s} {str(i.get('treatment')):9s} "
              f"material={i['material_symbol'] or '-':16s} sf={i['sf_symbol'] or '-'}")
    missing = sorted({x["name"] for x in icons} - names.keys())
    if missing:
        print(f"  ! the model didn't describe: {', '.join(missing)} (their symbol names in assets.json are unchanged)")
    extra = sorted(names.keys() - {x["name"] for x in m["assets"]})
    if extra:
        print(f"  ! the model described icons that are not in assets.json: {', '.join(extra)} (kept in icon_spec.json only)")
    print(f"-> {d / 'icon_spec.json'} (symbol names merged into assets.json)")


def _symbol_names(material, sf):
    material = material.strip() if re.fullmatch(r"[a-z0-9_]+", material.strip()) else ""
    sf = sf.strip() if re.fullmatch(r"[a-z0-9]+(\.[a-z0-9]+)*", sf.strip()) else ""
    return material, sf


def _snap_stroke(svg, sw):
    """Every stroke width in the SVG (attribute, style, and the inherited default on the root) set to sw."""
    root = ET.fromstring(svg)
    for el in root.iter():
        st = _style_dict(el)
        if "stroke-width" in st:
            st["stroke-width"] = _fmt(sw)
            el.set("style", ";".join(f"{k}:{v}" for k, v in st.items()))
        elif el.get("stroke-width") is not None or el is root:
            el.set("stroke-width", _fmt(sw))
    return ET.tostring(root, encoding="unicode")


def cmd_icons(a):
    d, m, mock = _load(a.dir)
    concepts = [c.strip() for c in a.concepts.split(",") if c.strip()]
    if not concepts:
        sys.exit('icons: no concepts given (e.g. "notifications, sleep timer, download")')
    slugs = {_norm_name(c): c for c in concepts}
    spec = _load_spec(d)
    out = Path(a.out) if a.out else d / "icons"
    out.mkdir(parents=True, exist_ok=True)
    have = [x for x in m["assets"] if x["format"] == "svg" and x.get("kind") == "icon" and (d / "svg" / f"{x['name']}.svg").exists()]
    if not have:
        sys.exit("icons: no existing svg icons in DIR/svg; run `svg` first so the new icons can match them")
    existing = []
    for x in have:
        try:
            src = _ensure_xmlns((d / "svg" / f"{x['name']}.svg").read_text())
            svg_style(src)
            existing.append({"asset": x, "svg": src})
        except (ET.ParseError, ValueError, OSError) as e:
            print(f"  ! {x['name']}.svg: unreadable ({e}), left out of the reference set")
    if not existing:
        sys.exit("icons: none of the existing icons in DIR/svg is readable")
    sizes = [max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) for e in existing]
    for e, n in zip(existing, _grid_normalize([e["svg"] for e in existing], sizes=sizes)):
        e["svg"] = n or e["svg"]
        e["style"] = svg_style(e["svg"])
        e["sw"] = _stroke24(e["style"])
    size = float(np.median(sizes))
    group = [e for e in existing if max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) <= size * 1.25 and
             max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) >= size / 1.25] or existing
    outlined = [e for e in group if e["style"]["stroked"] and not e["style"]["filled"]]
    treatment = a.treatment if a.treatment != "auto" else (
        spec["style"] if spec and spec["style"] in ("outlined", "filled") else ("outlined" if outlined else "filled"))
    pool = outlined if treatment == "outlined" and outlined else group
    sw = a.stroke or (float(np.median([e["sw"] for e in pool if e["sw"]])) if any(e["sw"] for e in pool)
                      else spec["stroke_24"] if spec else 2)
    caps = [c for e in pool for c in e["style"]["caps"]] or [spec["linecap"] if spec else "round"]
    joins = [j for e in pool for j in e["style"]["joins"]] or [spec["linejoin"] if spec else "round"]
    colors = [c for e in pool for c in e["style"]["colors"] if c.startswith("#")]
    color = a.color or (max(set(colors), key=colors.count).upper() if colors else "#000000")
    cap, join = max(set(caps), key=caps.count), max(set(joins), key=joins.count)
    ref = sheet([[(e["asset"]["name"], ref_crop(mock, e["asset"]["bbox"], 0.25, 160)) for e in existing[i:i + 6]]
                 for i in range(0, len(existing), 6)], out / "_existing.png", T=160)
    tr = ("outlined: fill=\"none\" on stroked parts, filled only where the set fills small details" if treatment == "outlined"
          else "filled shapes like the set's filled icons")
    named = ", ".join(f'{c} (file name "{n}")' for n, c in slugs.items())
    prompt = ICONS_PROMPT.format(concepts=named, treatment=tr, sw=f"{sw:g}", cap=cap, join=join, color=color)
    exemplars = "\n".join(f"--- {e['asset']['name']}.svg\n{e['svg']}" for e in pool[:5])
    parts = [prompt + "\n" + SVG_RULES + _spec_text(spec), mock.convert("RGB"), _rgba(ref).convert("RGB"),
             f"Existing icon SVG sources, on the same 24 grid:\n{exemplars}"]
    files, bad, broken = {}, {}, {}
    for attempt in range(2):
        try:
            returned = ask(parts, _svg_schema(list(slugs)), "svg_files")["files"]
        except ApiError as e:
            print(f"  ! icons: request failed: {e}")
            break
        for f in returned:
            n = _norm_name(f["name"])
            if n not in slugs:
                print(f"  ! model returned an unrequested icon {n!r}: ignored")
                continue
            svg = _ensure_xmlns(f["svg"])
            try:
                ET.fromstring(svg)
            except ET.ParseError as e:
                (out / "raw").mkdir(exist_ok=True)
                (out / "raw" / f"{n}_try{attempt + 1}.svg").write_text(svg)
                broken[n] = f"invalid XML: {e}"
                continue
            files[n] = svg
            broken.pop(n, None)
            try:
                svg_to_vd(svg, 24)
                bad.pop(n, None)
            except ValueError as e:
                bad[n] = str(e)
        missing = [n for n in slugs if n not in files]
        redo = {**bad, **{n: broken.get(n, "not returned") for n in missing}}
        if not redo:
            break
        print("  retrying:", "; ".join(f"{k}: {v}" for k, v in redo.items()))
        parts = parts + [f"These icons were invalid or missing, return them again (only these, same names): {redo}"]
    if not files:
        sys.exit(f"icons: no usable icons returned; missing: {', '.join(slugs)}" +
                 (f" ({'; '.join(f'{k}: {v}' for k, v in broken.items())})" if broken else ""))
    for n, s in files.items():
        vbx = _vb(ET.fromstring(s))
        if vbx[2:] != [24.0, 24.0]:
            norm = _grid_normalize([s])[0]
            if norm:
                s = norm
            else:
                print(f"  ~ {n}: kept as drawn ({vbx[2]:g}x{vbx[3]:g} viewBox; transform/percent units), not on the 24 grid")
        if a.snap:
            s = _snap_stroke(s, sw * max(_vb(ET.fromstring(s))[2:]) / 24)
        files[n] = s
        (out / f"{n}.svg").write_text(s)
    renders = render_svgs([e["svg"] for e in existing] + list(files.values()), 192)
    hexbg = lambda x: "#" + "".join(f"{int(v):02X}" for v in _element_mask(mock, x["bbox"])[1])
    bg = hexbg(pool[0]["asset"])
    old_r, new_r = renders[:len(existing)], renders[len(existing):]
    tiles = [(f"{e['asset']['name']} (existing)", r, hexbg(e["asset"])) for e, r in zip(existing, old_r)]
    names = list(files)
    rows = [tiles[i:i + 6] for i in range(0, len(tiles), 6)]
    rows += [[(f"{n} (new)", r, bg) for n, r in zip(names[i:i + 6], new_r[i:i + 6])] for i in range(0, len(names), 6)]
    T = 150
    im = Image.new("RGB", (6 * T, len(rows) * T), "white")
    for ri, row in enumerate(rows):
        for ci, (label, r, b) in enumerate(row):
            im.paste(_tile(r, T, label, bg=b), (ci * T, ri * T))
    im.save(out / "preview.png")
    (out / "_existing.png").unlink(missing_ok=True)
    _, notes = icon_set_report([{"name": e["asset"]["name"], "svg": e["svg"], "size": None} for e in pool] +
                               [{"name": n, "svg": s, "size": None} for n, s in files.items()])
    for n in names:
        try:
            st = svg_style(files[n])
            print(f"  {out / (n + '.svg')}  stroke={_stroke24(st)} caps={'/'.join(sorted(st['caps'])) or '-'} "
                  f"{'filled' if st['filled'] else ''}{' outlined' if st['stroked'] else ''}")
        except ValueError as e:
            print(f"  {out / (n + '.svg')}  (unreadable path data: {e})")
    print(f"target: {treatment}, stroke {sw:g}, {cap} caps, {join} joins, {color}; preview -> {out / 'preview.png'}")
    print("set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")
    for n, err in bad.items():
        print(f"  ! {n}: {err} (kept out of --vd)")
    still = [n for n in slugs if n not in files]
    if still:
        print(f"  ! missing: {', '.join(still)}" + "".join(f" ({n}: {broken[n]})" for n in still if n in broken))
    good = [n for n in names if n not in bad]
    failed_vd = convert_vd([out / f"{n}.svg" for n in good], a.vd, a.dp) if a.vd and good else 0
    if failed_vd:
        print(f"  ! {failed_vd} VectorDrawable(s) failed validation (see above); the SVGs are written, fix them and run `vd`")
    if still or failed_vd:
        sys.exit(1)


# ---- animation frames

def _runs(v):
    """(start, end) pairs of the True runs of a 1-D array."""
    d = np.diff(np.concatenate([[0], np.asarray(v, np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _remove_dividers(img):
    """Clear drawn grid lines: thin lines that span nearly the whole sheet."""
    arr = np.array(img.convert("RGBA"))
    solid = arr[..., 3] > 100
    for axis in (0, 1):
        cov = solid.mean(axis=axis)
        n = cov.size
        for s, e in _runs(cov >= 0.9):
            side = cov[[i for i in (s - 3, e + 2) if 0 <= i < n]]
            if e - s <= max(4, n // 150) and (side < 0.6).all():
                sl = slice(max(0, s - 1), min(n, e + 1))
                arr[(slice(None), sl) if axis == 0 else sl] = 0
    return Image.fromarray(arr)


def _bands(occ, gap):
    """Occupied runs of a 1-D profile; gaps under `gap` are bridged and slivers join their nearest neighbour."""
    runs = []
    for s, e in _runs(occ):
        if runs and s - runs[-1][1] < gap:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    while len(runs) > 1:
        sizes = [e - s for s, e in runs]
        i = int(np.argmin(sizes))
        if sizes[i] >= 0.3 * max(sizes):
            break
        left = runs[i][0] - runs[i - 1][1] if i else math.inf
        right = runs[i + 1][0] - runs[i][1] if i + 1 < len(runs) else math.inf
        j = i - 1 if left <= right else i
        runs[j:j + 2] = [(runs[j][0], runs[j + 1][1])]
    return runs


def _gutter_cells(A):
    """XY cut along empty gutters: row bands, then column bands inside each row, in reading order."""
    H, W = A.shape
    m = A > 64
    rows = _bands(m.sum(1) > max(2, W * 0.005), max(4, H // 60))
    cells = []
    for r, (y0, y1) in enumerate(rows):
        top = 0 if r == 0 else (rows[r - 1][1] + y0) // 2
        bot = H if r == len(rows) - 1 else (y1 + rows[r + 1][0]) // 2
        cols = _bands(m[top:bot].sum(0) > max(2, (bot - top) * 0.005), max(4, W // 60))
        for c, (x0, x1) in enumerate(cols):
            left = 0 if c == 0 else (cols[c - 1][1] + x0) // 2
            right = W if c == len(cols) - 1 else (x1 + cols[c + 1][0]) // 2
            cells.append((left, top, right, bot))
    return cells


def _reading_order(boxes):
    boxes = sorted(boxes, key=lambda b: b[1] + b[3])
    tol = 0.5 * float(np.median([b[3] - b[1] for b in boxes]))
    rows = []
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        if rows and abs(cy - rows[-1][0]) < tol:
            rows[-1][1].append(b)
        else:
            rows.append((cy, [b]))
    return [b for _, r in rows for b in sorted(r, key=lambda b: b[0])]


def _blob_cells(A, count):
    """(box, mask) cells from connected blobs, for sheets whose frames are not separated by straight gutters."""
    f = max(1, max(A.shape) // 400)
    m = A[::f, ::f] > 96
    h, w = m.shape
    ml, lab, boxes = m.tolist(), [[0] * w for _ in range(h)], []
    for sy, sx in zip(*np.nonzero(m)):
        if lab[sy][sx]:
            continue
        n = len(boxes) + 1
        lab[sy][sx] = n
        stack, x0, y0, x1, y1, area = [(sy, sx)], sx, sy, sx, sy, 0
        while stack:
            y, x = stack.pop()
            area += 1
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < h and 0 <= nx < w and ml[ny][nx] and not lab[ny][nx]:
                        lab[ny][nx] = n
                        stack.append((ny, nx))
        boxes.append([x0, y0, x1 + 1, y1 + 1, area, {n}])
    boxes = [b for b in boxes if b[4] >= h * w * 0.0005]

    def union(p, q):
        return [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]), p[4] + q[4], p[5] | q[5]]

    def nested(p, q):
        ix, iy = min(p[2], q[2]) - max(p[0], q[0]), min(p[3], q[3]) - max(p[1], q[1])
        return ix > 0 and iy > 0 and ix * iy >= 0.5 * min((b[2] - b[0]) * (b[3] - b[1]) for b in (p, q))
    merged = True
    while merged:
        merged = False
        for i, j in ((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            if nested(boxes[i], boxes[j]):
                boxes[i] = union(boxes[i], boxes.pop(j))
                merged = True
                break
    if not boxes:
        return []
    floor = 0.15 * float(np.median(sorted(b[4] for b in boxes)[-count:]))
    main = [b for b in boxes if b[4] >= floor]
    for b in boxes:
        if b[4] < floor:
            k = min(range(len(main)), key=lambda k: abs(main[k][0] + main[k][2] - b[0] - b[2]) +
                    abs(main[k][1] + main[k][3] - b[1] - b[3]))
            main[k] = union(main[k], b)
    H, W = A.shape
    g = max(1, int(max(h, w) * 0.02))
    labels = np.array(lab)
    cells = []
    for b in _reading_order(main):
        own = Image.fromarray(np.isin(labels, list(b[5])).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * g + 1))
        cells.append(((max(0, (b[0] - g) * f), max(0, (b[1] - g) * f), min(W, (b[2] + g) * f), min(H, (b[3] + g) * f)),
                      own.resize((w * f, h * f), Image.NEAREST)))
    return cells


def _robust_box(alpha, thr=24, frac=0.004):
    """Content bbox that ignores stray specks (rows/columns with only a few pixels over thr)."""
    m = alpha > thr
    ys = np.nonzero(m.sum(1) > max(1, frac * m.shape[1]))[0]
    xs = np.nonzero(m.sum(0) > max(1, frac * m.shape[0]))[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(xs) and len(ys) else None


def _slice_sheet(img, count, cols):
    """Frames of a sprite sheet in reading order: gutter cuts, else a clean equal grid, else connected blobs."""
    A = np.asarray(img.getchannel("A"))
    H, W = A.shape
    rows_n = math.ceil(count / cols)
    grid = [((round(i % cols * W / cols), round(i // cols * H / rows_n), round((i % cols + 1) * W / cols),
              round((i // cols + 1) * H / rows_n)), None) for i in range(count)]
    m = A > 64
    xs, ys = [round(c * W / cols) for c in range(1, cols)], [round(r * H / rows_n) for r in range(1, rows_n)]
    cuts = [m[:, max(0, x - 1):x + 2].mean() for x in xs] + [m[max(0, y - 1):y + 2].mean() for y in ys]
    how, cells = "gutters", [(c, None) for c in _gutter_cells(A)]
    if len(cells) != count and max(cuts, default=0) < 0.01:
        how, cells = "grid", grid
    elif len(cells) != count:
        blobs = _blob_cells(A, count)
        if abs(len(blobs) - count) < abs(len(cells) - count):
            how, cells = "blobs", blobs
        if abs(len(cells) - count) > count // 4:
            how, cells = "grid", grid
    crops = []
    for i, (box, mask) in enumerate(cells):
        cell = img.crop(box)
        if mask is not None:
            cell.putalpha(ImageChops.multiply(cell.getchannel("A"), mask.crop(box)))
        bb = _robust_box(np.asarray(cell.getchannel("A")))
        if not bb:
            print(f"  ! frame {i + 1} empty")
            continue
        if bb[0] <= 1 or bb[1] <= 1 or bb[2] >= cell.width - 1 or bb[3] >= cell.height - 1:
            print(f"  ! frame {i + 1} touches its cell edge (may be cut off)")
        crops.append(cell.crop(bb))
    print(f"  sliced {len(crops)} frames by {how}" + ("" if len(crops) == count else f" (asked for {count})"))
    return crops


def _anchor_point(img, anchor):
    """Base centre (bottom anchor) or centroid (center anchor) of the solid part."""
    s = np.asarray(img.getchannel("A")) > 128
    if not s.any():
        return img.width / 2, img.height if anchor == "bottom" else img.height / 2
    ys, xs = np.nonzero(s)
    if anchor == "bottom":
        band = ys >= ys.max() - 0.15 * (ys.max() - ys.min() + 1)
        return float(xs[band].mean()), float(ys.max() + 1)
    return float(xs.mean()), float(ys.mean())


def _align_frames(crops, anchor):
    """Place crops on one shared canvas with their anchor points coincident."""
    pts = [_anchor_point(c, anchor) for c in crops]
    left, up = max(p[0] for p in pts), max(p[1] for p in pts)
    right = max(c.width - p[0] for c, p in zip(crops, pts))
    down = max(c.height - p[1] for c, p in zip(crops, pts))
    pad = int(max(left + right, up + down) * 0.06)
    size = (math.ceil(left + right) + 2 * pad, math.ceil(up + down) + 2 * pad)
    out = []
    for c, (px, py) in zip(crops, pts):
        f = Image.new("RGBA", size, (0, 0, 0, 0))
        f.alpha_composite(c, (round(pad + left - px), round(pad + up - py)))
        out.append(f)
    return out


def _square(img):
    side = max(img.size)
    c = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    off = ((side - img.width) // 2, (side - img.height) // 2)
    c.alpha_composite(img, off)
    return c, off


def _register(layer, target, grid=160):
    """Best scale/offset placing `layer` (a part of `target`) onto target's canvas; returns the placed layer.
    Score = overlap - 2 x layer pixels outside the target, over 11 scales and a +-10% grid of offsets."""
    tm = np.asarray(target.getchannel("A").resize((grid, grid))) > 100
    lm_full = layer.getchannel("A")
    r = grid // 10
    best = (-1, 1.0, 0, 0)
    for s10 in range(85, 116, 3):
        s = s10 / 100
        size = max(1, round(grid * s))
        lm = np.asarray(lm_full.resize((size, size))) > 100
        o = (grid - size) // 2
        for dx in range(-r, r + 1, 2):
            for dy in range(-r, r + 1, 2):
                x, y = o + dx, o + dy                       # the layer's top-left on the canvas; clipped like a paste
                lx0, ly0, lx1, ly1 = max(0, -x), max(0, -y), min(size, grid - x), min(size, grid - y)
                if lx1 <= lx0 or ly1 <= ly0:
                    continue
                sub = lm[ly0:ly1, lx0:lx1]
                inter = int((sub & tm[y + ly0:y + ly1, x + lx0:x + lx1]).sum())
                sc = inter - 2 * (int(sub.sum()) - inter)
                if sc > best[0]:
                    best = (sc, s, dx, dy)
    _, s, dx, dy = best
    W, H = target.size
    placed = layer.resize((round(W * s), round(H * s)), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(placed, ((W - placed.width) // 2 + round(dx * W / grid), (H - placed.height) // 2 + round(dy * H / grid)))
    return out


def _xcorr(a, k):
    """Integer cross-correlation of boolean a with kernel k over every offset (FFT): the value at index
    (y + kh - 1, x + kw - 1) is the count of kernel pixels that land on true pixels of a when k sits at (x, y)."""
    H, W = a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1
    c = np.fft.irfft2(np.fft.rfft2(a.astype(np.float64), (H, W)) * np.fft.rfft2(k[::-1, ::-1].astype(np.float64), (H, W)),
                      (H, W))
    return np.rint(c).astype(np.int64)


def _fit_moving(src, static, frame, grid=128):
    """Scale + bottom-centre anchor that best overlays `frame` (moving part) on `src`, given its static layer.
    Score per placement: pixels on the visible moving part - 2 x pixels outside the asset (rows above the canvas
    count) - 0.7 x visible pixels missed; every offset of every scale is scored at once by cross-correlation."""
    k = grid / max(src.size)
    tw, th = max(1, round(src.width * k)), max(1, round(src.height * k))
    sa = np.asarray(src.getchannel("A").resize((tw, th))) > 100
    st = np.asarray(static.getchannel("A").resize((tw, th))) > 30
    visible = sa & ~st
    allowed = (np.asarray(src.getchannel("A").resize((tw, th))) > 40) | st
    fb = content_box(frame, 100)
    ys, xs = np.nonzero(visible)
    if not fb or not len(xs):
        return None
    fcrop = frame.crop(fb)
    s0 = (xs.max() - xs.min() + 1) / (fcrop.width * k)
    vis_total = int(visible.sum())
    best = None
    for f in np.linspace(0.55, 1.25, 15):
        sc = s0 * f
        fw, fh = max(1, round(fcrop.width * sc * k)), max(1, round(fcrop.height * sc * k))
        if fw > tw or fh > th + th // 2:
            continue
        fm = np.asarray(fcrop.getchannel("A").resize((fw, fh))) > 100
        y0, y1 = -(fh // 2), th - fh // 2
        outside = np.pad(~allowed, ((fh, 0), (0, 0)), constant_values=True)
        hit = _xcorr(visible, fm)[y0 + fh - 1:y1 + fh - 1, fw - 1:tw]
        spill = _xcorr(outside, fm)[y0 + 2 * fh - 1:y1 + 2 * fh - 1, fw - 1:tw]
        score = hit - 2 * spill - 0.7 * (vis_total - hit)
        idx = int(np.argmax(score))
        yy, xx = divmod(idx, score.shape[1])
        if best is None or score.flat[idx] > best[0]:
            best = (float(score.flat[idx]), sc, xx, y0 + yy, fw, fh)
    if best is None:
        return None
    _, sc, x, y, fw, fh = best
    return sc, (x + fw / 2) / k, (y + fh) / k, fb


def _placed_alpha(size, frame, s, org):
    m = Image.new("L", size, 0)
    m.paste(frame.getchannel("A").resize((max(1, round(frame.width * s)), max(1, round(frame.height * s)))),
            (round(org[0]), round(org[1])))
    return np.asarray(m)


def _box_gap(a, b):
    return max(0, a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])


def _target_clusters(target, size):
    """Spatial clusters of the target's components (e.g. coin + rays vs slivers the static redraw left behind)."""
    lab, n = _label(target)
    if n < 2:
        return []
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = [i + 1 for i in np.argsort(-areas)[:40] if areas[i] >= 0.01 * target.sum()]
    boxes = {}
    for i in keep:
        ys, xs = np.nonzero(lab == i)
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    clusters = [[i] for i in keep]
    merged = True
    while merged:
        merged = False
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                if min(_box_gap(boxes[i], boxes[j]) for i in clusters[x] for j in clusters[y]) < 0.06 * max(size):
                    clusters[x] += clusters.pop(y)
                    merged = True
                    break
            if merged:
                break
    if len(clusters) < 2:
        return []
    return [m for m in (np.isin(lab, c) for c in clusters) if m.sum() >= 0.03 * target.sum()]


def _placement(src, static, frames):
    """One scale + origin for every frame: silhouette fit of the middle frame or extent fit of any frame, by IoU."""
    raw = (np.asarray(src.getchannel("A")) > 60) & _moving_mask(static)
    # an opening drops slivers the static redraw leaves along its edges; thin moving parts keep the raw mask
    k = max(3, round(max(src.size) * 0.015)) | 1
    opened = np.asarray(Image.fromarray((raw * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(k))
                        .filter(ImageFilter.MaxFilter(k))) > 127
    target = opened if opened.sum() >= 0.3 * raw.sum() else raw
    if not target.any():
        return None
    mid = len(frames) // 2
    fit = _fit_moving(src, static, frames[mid])

    def best_for(t, silhouette):
        ys, xs = np.nonzero(t)
        tb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        cands = []
        if silhouette and fit:
            s, cx, by, fb = fit
            cands.append(("silhouette", mid, s, (cx - (fb[0] + fb[2]) / 2 * s, by - fb[3] * s)))
        for i, f in enumerate(frames):
            b = content_box(f, 60)
            if b:
                s = math.sqrt((tb[2] - tb[0]) / (b[2] - b[0]) * (tb[3] - tb[1]) / (b[3] - b[1]))
                cands.append(("extent", i, s, ((tb[0] + tb[2] - (b[0] + b[2]) * s) / 2,
                                               (tb[1] + tb[3] - (b[1] + b[3]) * s) / 2)))

        def iou(c):
            m = _placed_alpha(src.size, frames[c[1]], c[2], c[3]) > 60
            return (m & t).sum() / max(1, (m | t).sum())
        return max(((iou(c), c) for c in cands), key=lambda x: x[0], default=None)
    best = best_for(target, True)
    if best is None:
        return None
    for cl in _target_clusters(target, src.size):
        alt = best_for(cl, False)
        if alt and alt[0] > best[0] + 0.15:
            best = (alt[0], (alt[1][0] + " (cluster)",) + alt[1][1:])
    v, (how, i, s, org) = best
    print(f"  registered by {how} fit of f{i + 1} (IoU {v:.2f} with the asset's moving part)")
    return s, org, v


def _composite(src, static, frames, order):
    """Place every aligned moving frame with one shared transform onto the asset and stack the static layer."""
    fit = _placement(src, static, frames)
    if not fit:
        print("  ! could not locate the moving part in the asset; composites skipped")
        return [], 0.0
    s, org, v = fit
    P = max(src.size)
    size = (src.width + 2 * P, src.height + 2 * P)
    pos = (P + round(org[0]), P + round(org[1]))
    st = Image.new("RGBA", size, (0, 0, 0, 0))
    st.alpha_composite(static, (P, P))
    comps = []
    for f in frames:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        layer.paste(f.resize((max(1, round(f.width * s)), max(1, round(f.height * s))), Image.LANCZOS), pos)
        comp = Image.new("RGBA", size, (0, 0, 0, 0))
        for lyr in ((layer, st) if order == "front" else (st, layer)):
            comp.alpha_composite(lyr)
        comps.append(comp)
    # one shared crop box keeps every frame registered
    boxes = [content_box(c, 8) for c in comps]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    return [c.crop(box) for c in comps], v


# -- frame QA

_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])




def _lab_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    c = (np.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * _D65) @ np.linalg.inv(_XYZ).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def _solid_lab(img, mask=None, thr=160):
    arr = np.asarray(img.convert("RGBA"))
    sel = arr[..., 3] > thr
    if mask is not None:
        sel &= mask
    return _lab(arr[..., :3][sel]) if sel.any() else np.zeros((1, 3))


def _moving_mask(static):
    """Pixels of the approved asset that belong to the moving part (the asset minus a dilated static layer)."""
    return np.asarray(static.getchannel("A").filter(ImageFilter.MaxFilter(9))) <= 30


def _premul_thumb(img, side=128):
    t = img.copy()
    t.thumbnail((side, side), Image.LANCZOS)
    a = np.asarray(t, np.float64) / 255
    return np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1) * 255


def _alpha_stats(img, mask=None):
    """Alpha-weighted mean Lab colour and alpha mass (robust for glows and fading frames)."""
    arr = np.asarray(img.convert("RGBA"))
    w = arr[..., 3] / 255 * (1 if mask is None else mask)
    sel = w > 0.1
    if not sel.any():
        return np.zeros(3), 1.0
    return (_lab(arr[..., :3][sel]) * w[sel, None]).sum(0) / w[sel].sum(), float(w.sum())


def _outliers(X, thr, spike):
    """Frames off the median by > thr in an otherwise steady series, or sticking out from both neighbours by > spike."""
    X = np.asarray(X, np.float64).reshape(len(X), -1)
    dev = np.linalg.norm(X - np.median(X, 0), axis=1)
    out = set(np.nonzero((dev > thr) & (dev > 6 * np.median(dev)))[0].tolist())
    for i in range(len(X) if len(X) >= 4 else 0):
        p, q = X[i] - X[i - 1], X[i] - X[(i + 1) % len(X)]
        if min(np.linalg.norm(p), np.linalg.norm(q)) > spike and np.dot(p, q) > 0:
            out.add(i)
    return out


def _frame_qa(frames, ref, ref_mask=None):
    """Colour (Lab) and size outliers, centre drift and loop continuity of an aligned frame sequence."""
    n = len(frames)
    ref_lab = _alpha_stats(ref, ref_mask)[0]
    stats = [_alpha_stats(f) for f in frames]
    labs, mass = np.array([x[0] for x in stats]), np.array([max(1.0, x[1]) for x in stats])
    side = max(frames[0].size)
    pts = np.array([_anchor_point(f, "center") for f in frames]) / side
    med_lab, med_mass, med_pt = np.median(labs, 0), float(np.median(mass)), np.median(pts, 0)
    flags = {"colour": _outliers(labs, 8, 10), "size": _outliers(np.log(mass) / 2, 0.14, 0.26),
             "drift": _outliers(pts, 0.06, 0.08)}
    thumbs = [_premul_thumb(f) for f in frames]
    adj = [float(np.abs(thumbs[i] - thumbs[(i + 1) % n]).mean()) for i in range(n)]
    typical = max(1e-6, float(np.median(adj[:-1])) if n > 2 else adj[0])

    def near(i):  # steps just outside frame i's in/out steps, floored at the median step
        return max(typical, (adj[(i - 2) % n] + adj[(i + 1) % n]) / 2) if n > 4 else typical
    rows = []
    for i in range(n):
        r = {"frame": i + 1, "dE_seq": round(float(np.linalg.norm(labs[i] - med_lab)), 1),
             "dE_ref": round(float(np.linalg.norm(labs[i] - ref_lab)), 1),
             "size": round(math.sqrt(mass[i] / med_mass), 3),
             "drift": round(float(np.linalg.norm(pts[i] - med_pt)), 3),
             "step_in": round(adj[i - 1] / near(i), 2), "step_out": round(adj[i] / near(i), 2)}
        r["flags"] = [k for k, v in flags.items() if i in v]
        if n > 3 and min(r["step_in"], r["step_out"]) > 2:
            r["flags"].append("pop")
        rows.append(r)
    seq = {"frames": n, "dE_ref": round(float(np.linalg.norm(med_lab - ref_lab)), 1),
           "loop_ratio": round(adj[-1] / near(n - 1), 2), "step_median": round(typical, 2), "flags": []}
    if seq["dE_ref"] > 10:
        seq["flags"].append("style-drift")
    if seq["loop_ratio"] > 2:
        seq["flags"].append("loop-seam")
    return {"sequence": seq, "frames": rows}


def _qa_report(qa, label=""):
    s = qa["sequence"]
    bad = [r for r in qa["frames"] if r["flags"]]
    print(f"  QA{label}: dE vs asset {s['dE_ref']}, loop seam {s['loop_ratio']}x a typical step"
          f"{' ' + ','.join(s['flags']) if s['flags'] else ''}; flagged frames: "
          + (", ".join(f"f{r['frame']}({'+'.join(r['flags'])})" for r in bad) or "none")
          + (" (try --match-colors)" if "style-drift" in s["flags"] else ""))
    return [r["frame"] - 1 for r in bad]


def _badness(r):
    return max(r["dE_seq"] / 8, abs(math.log(max(r["size"], 1e-3))) / 0.14, r["drift"] / 0.06,
               min(r["step_in"], r["step_out"]) / 2)


def _match_colors(frames, ref, ref_mask=None):
    """One LCh transform for all frames: lightness mean/std, chroma ratio and hue rotation onto the approved asset."""
    r, p = _solid_lab(ref, ref_mask), np.concatenate([_solid_lab(f) for f in frames])
    gain = np.clip(r[:, 0].std() / (p[:, 0].std() + 1e-6), 0.6, 1.6)
    chroma = np.hypot(r[:, 1], r[:, 2]).mean() / max(1e-6, np.hypot(p[:, 1], p[:, 2]).mean())
    turn = np.arctan2(r[:, 2].mean(), r[:, 1].mean()) - np.arctan2(p[:, 2].mean(), p[:, 1].mean())
    out = []
    for f in frames:
        arr = np.array(f)
        vis = arr[..., 3] > 0
        lab = _lab(arr[..., :3][vis])
        c, h = np.hypot(lab[:, 1], lab[:, 2]) * chroma, np.arctan2(lab[:, 2], lab[:, 1]) + turn
        arr[..., :3][vis] = _lab_rgb(np.stack([(lab[:, 0] - p[:, 0].mean()) * gain + r[:, 0].mean(),
                                               c * np.cos(h), c * np.sin(h)], -1))
        out.append(Image.fromarray(arr))
    return out


# -- procedural motion

MOTIONS = {"bounce": 0.9, "pulse": 1.0, "float": 2.0, "wiggle": 1.0, "spin": 1.2, "breathe": 4.0}


def _motion(kind, t, amt):
    """Transform at loop phase t: (scale x, scale y, degrees clockwise, dy in content heights, opacity, pivot)."""
    wave = 0.5 - 0.5 * math.cos(2 * math.pi * t)
    if kind == "bounce":
        land = 0.14
        if t < land:
            sy = 1 - 0.16 * amt * math.sin(math.pi * t / land)
            return 1 / sy, sy, 0, 0, 1, "bottom"
        u = (t - land) / (1 - land)
        sy = 1 + 0.08 * amt * abs(1 - 2 * u)
        return 1 / sy, sy, 0, -0.22 * amt * 4 * u * (1 - u), 1, "bottom"
    if kind == "pulse":
        return 1 + 0.08 * amt * wave, 1 + 0.08 * amt * wave, 0, 0, 1, "center"
    if kind == "breathe":
        e = wave * wave * (3 - 2 * wave)
        return 1 + 0.06 * amt * e, 1 + 0.06 * amt * e, 0, 0, 0.8 + 0.2 * e, "center"
    if kind == "float":
        return 1, 1, 2 * amt * math.cos(2 * math.pi * t), -0.035 * amt * math.sin(2 * math.pi * t), 1, "center"
    if kind == "wiggle":
        return 1, 1, 8 * amt * math.sin(2 * math.pi * t), 0, 1, "bottom"
    return 1, 1, 360 * t, 0, 1, "center"


def _affine(p, w, h):
    sx, sy, deg, dy, _, pivot = p
    px, py = w / 2, h if pivot == "bottom" else h / 2
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]])
    return A, np.array([px, py + dy * h]) - A @ [px, py]


def _procedural(src, kind, count, amt):
    """Loop frames of `src` under an eased affine motion; returns frames, per-frame params, canvas origin, art."""
    art = src.crop(content_box(src, 8) or (0, 0, src.width, src.height))
    w, h = art.size
    params = [_motion(kind, i / count, amt) for i in range(count)]
    mats = [_affine(p, w, h) for p in params]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64)
    pts = np.concatenate([corners @ A.T + t for A, t in mats])
    lo, hi = np.floor(pts.min(0)) - 2, np.ceil(pts.max(0)) + 2
    size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    pm = art.convert("RGBa")
    frames = []
    for (A, t), p in zip(mats, params):
        inv = np.linalg.inv(A)
        b = -inv @ (t - lo)
        f = pm.transform(size, Image.AFFINE, (inv[0, 0], inv[0, 1], b[0], inv[1, 0], inv[1, 1], b[1]),
                         Image.BICUBIC).convert("RGBA")
        if p[4] < 1:
            f.putalpha(f.getchannel("A").point(lambda v, o=p[4]: round(v * o)))
        frames.append(f)
    return frames, params, lo, art


# -- outputs

def _lottie_ks_image(w, h):
    return {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [w / 2, h / 2, 0]},
            "a": {"a": 0, "k": [w / 2, h / 2, 0]}, "s": {"a": 0, "k": [100, 100, 100]}}


def _lottie_image(img, i):
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return {"id": f"img_{i}", "w": img.width, "h": img.height, "u": "", "e": 1,
            "p": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def lottie_image_sequence(frames, fps, name="gptkit"):
    """Lottie image sequence: each frame is an embedded PNG shown for exactly one Lottie frame."""
    w, h = frames[0].size
    layers = [{"ddd": 0, "ind": i + 1, "ty": 2, "nm": f"frame {i + 1}", "refId": f"img_{i}", "sr": 1, "ip": i,
               "op": i + 1, "st": 0, "bm": 0, "ks": _lottie_ks_image(w, h)} for i in range(len(frames))]
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": len(frames), "w": w, "h": h, "nm": name, "ddd": 0,
            "assets": [_lottie_image(f, i) for i, f in enumerate(frames)], "layers": layers}


def lottie_motion(art, kind, count, amt, lo, size, fps, px, name="gptkit"):
    """Lottie of one embedded image driven by transform keyframes (smooth at any frame rate)."""
    k = min(1.0, px / max(size))
    w, h = art.size
    img = art.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    params = [_motion(kind, i / count, amt) for i in range(count + 1)]
    pivot = (w / 2, h if params[0][5] == "bottom" else h / 2)

    def track(vals):
        if all(v == vals[0] for v in vals):
            return {"a": 0, "k": vals[0] if len(vals[0]) > 1 else vals[0][0]}
        return {"a": 1, "k": [{"t": i, "s": v, "o": {"x": [0], "y": [0]}, "i": {"x": [1], "y": [1]}}
                              for i, v in enumerate(vals[:-1])] + [{"t": count, "s": vals[-1]}]}
    ks = {"a": {"a": 0, "k": [pivot[0] * k, pivot[1] * k, 0]},
          "p": {"s": True, "x": {"a": 0, "k": (pivot[0] - lo[0]) * k},
                "y": track([[(pivot[1] + p[3] * h - lo[1]) * k] for p in params])},
          "s": track([[p[0] * 100, p[1] * 100, 100] for p in params]),
          "r": track([[p[2]] for p in params]), "o": track([[p[4] * 100] for p in params])}
    layer = {"ddd": 0, "ind": 1, "ty": 2, "nm": name, "refId": "img_0", "sr": 1, "ip": 0, "op": count, "st": 0, "bm": 0,
             "ks": ks}
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": count, "w": round(size[0] * k), "h": round(size[1] * k), "nm": name,
            "ddd": 0, "assets": [_lottie_image(img, 0)], "layers": [layer]}


def _lottie_html(lot, bg, size=512):
    return ("<!doctype html><meta charset=utf-8><body style='margin:0;background:" + bg + "'>"
            f"<div id=a style='width:{size}px;height:{size}px'></div>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js'></script>"
            "<script>const f=parseFloat(location.hash.slice(1));const an=lottie.loadAnimation({container:a,renderer:'svg',"
            f"loop:true,autoplay:isNaN(f),animationData:{json.dumps(lot, separators=(',', ':'))}}});"
            "if(!isNaN(f))an.addEventListener('DOMLoaded',()=>an.goToAndStop(f,true))</script>")


def _fit_px(img, px):
    if max(img.size) <= px:
        return img
    k = px / max(img.size)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def _durations(n, fps, unit=1):
    """Per-frame ms (multiples of `unit`) whose sum is the exact loop length: one rounded value per frame drifts
    0.8% at 24 fps. GIF stores centiseconds, so its list uses unit=10."""
    return [(round((i + 1) * 1000 / fps / unit) - round(i * 1000 / fps / unit)) * unit for i in range(n)]


def _write_anim(out, anim, seq, fps, a, labels=None, lottie=None):
    """preview.gif, animated WebP/APNG, Lottie, the Android animation-list (frames + XML under android/) and the
    review sheet for a frame sequence."""
    formats = {f.strip() for f in a.formats.split(",") if f.strip()}
    durs = _durations(len(seq), fps)
    small = [_fit_px(f, a.px) for f in seq]
    wrote = []
    if "gif" in formats:
        bg = Image.new("RGBA", small[0].size, a.preview_bg)
        flat = [Image.alpha_composite(bg, f).convert("RGB") for f in small]
        # one palette for the whole loop (per-frame adaptive palettes flicker between frames)
        contact = Image.new("RGB", (flat[0].width * len(flat), flat[0].height))
        for i, f in enumerate(flat):
            contact.paste(f, (i * f.width, 0))
        pal = contact.quantize(256)
        gif = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in flat]
        gif[0].save(out / "preview.gif", save_all=True, append_images=gif[1:], duration=_durations(len(seq), fps, 10),
                    loop=0, disposal=2)
        wrote.append("preview.gif")
    if "webp" in formats:
        # Pillow's animated writer defaults to method=0 (fastest, largest); exact keeps RGB under transparent pixels
        small[0].save(out / f"{anim}.webp", save_all=True, append_images=small[1:], duration=durs, loop=0, quality=90,
                      method=6, exact=True)
        wrote.append(f"{anim}.webp")
    if "apng" in formats:
        small[0].save(out / f"{anim}.apng", format="PNG", save_all=True, append_images=small[1:], duration=durs, loop=0)
        wrote.append(f"{anim}.apng")
    if "lottie" in formats:
        lot = lottie or lottie_image_sequence(small, fps, anim)
        (out / f"{anim}.json").write_text(json.dumps(lot, separators=(",", ":")))
        (out / "preview.html").write_text(_lottie_html(lot, a.preview_bg))
        wrote += [f"{anim}.json", "preview.html"]
    # Android: pixel-sized frames belong in drawable-nodpi, and the animation-list must name files that exist
    res = _res_name(anim)
    if res != anim:
        print(f"  (android resource name: {res}, from {anim!r})")
    nodpi, drw = out / "android" / "drawable-nodpi", out / "android" / "drawable"
    for sub, pat in ((nodpi, f"{res}_frame_*.png"), (drw, f"{res}_anim.xml")):  # only this animation's files:
        sub.mkdir(parents=True, exist_ok=True)                                  # `rain_*` would also delete rain_pulse_*
        for old in sub.glob(pat):
            old.unlink()
    for i, f in enumerate(small):
        f.save(nodpi / f"{res}_frame_{i + 1:02d}.png")
    items = "\n".join(f'    <item android:drawable="@drawable/{res}_frame_{i + 1:02d}" android:duration="{durs[i]}" />'
                      for i in range(len(seq)))
    (drw / f"{res}_anim.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<animation-list xmlns:android="http://schemas.android.com/apk/res/android"\n'
        f'    android:oneshot="false">\n{items}\n</animation-list>\n')
    labels = labels or [f"f{i + 1}" for i in range(len(seq))]
    sheet([list(zip(labels, seq))[j:j + 8] for j in range(0, len(seq), 8)], out / "frames_review.png", T=200)
    print(f"{len(seq)} frames @ {fps:g} fps ({sum(durs)} ms loop) -> {out}  "
          f"({', '.join(wrote + ['frames_review.png', f'android/drawable/{res}_anim.xml'])})")
    verdict = _aapt2_tree(out / "android")  # the animation-list must compile next to its PNGs, hence the whole tree
    print(f"  {'! ' if 'FAILED' in verdict else ''}{verdict}")


def _frames_procedural(a, src, out):
    period = MOTIONS[a.procedural]
    if max(src.size) > 2048:  # every frame is a full-size RGBA copy: an 8000 px source ran out of memory
        k = 2048 / max(src.size)
        print(f"  ~ assets/{a.name}.png is {src.width}x{src.height}: animating a {round(src.width * k)}x{round(src.height * k)} copy")
        src = src.resize((max(1, round(src.width * k)), max(1, round(src.height * k))), Image.LANCZOS)
    fps = a.fps or (24 if period * 24 <= 48 else 12)
    count = a.count or max(8, round(period * fps))
    frames, params, lo, art = _procedural(src, a.procedural, count, a.amount)
    for i, f in enumerate(frames):
        f.save(out / f"frame_{i + 1:02d}.png")
    (out / "motion.json").write_text(json.dumps({
        "motion": a.procedural, "amount": a.amount, "seconds": round(count / fps, 3), "fps": round(fps, 3),
        "pivot": params[0][5],
        "frames": [{"scale_x": round(p[0], 4), "scale_y": round(p[1], 4), "rotation_deg": round(p[2], 3),
                    "translate_y_of_height": round(p[3], 4), "opacity": round(p[4], 3)} for p in params]},
        indent=1))
    lot = lottie_motion(art, a.procedural, count, a.amount, lo, frames[0].size, fps, a.px, f"{a.name}_{a.procedural}")
    _write_anim(out, f"{a.name}_{a.procedural}", frames, fps, a, lottie=lot)


def _opaque_share(img):
    """(share of pixels at alpha 255, whether all four corners are opaque): the signature of a render that ignored
    background=transparent (a gateway that rejects the parameter, or a model that painted a backdrop)."""
    a = np.asarray(img.getchannel("A"))
    return float((a == 255).mean()), bool(min(int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])) == 255)


def _decode_result(raw, what):
    """RGBA image of the bytes an image call returned; RuntimeError naming `what` when they are not an image
    (a gateway can answer with HTML or a truncated file)."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im.convert("RGBA")
    except (OSError, ValueError):
        raise RuntimeError(f"{what}: the response is not a decodable image ({len(raw)} bytes starting {raw[:24]!r})")


def _opaque_error(img, what, model=None):
    """One-line reason when an image asked for with a transparent background came back opaque, else None."""
    share, corners = _opaque_share(img)
    if share > 0.95 or corners:
        c = _config()
        return (f"{what}: the image came back with an opaque background ({share:.0%} of it is opaque; "
                f"background=transparent was ignored by {pick_image_model(model, 'precise', 'transparent')} at {c['base']})")
    return None


def _retryable(e):
    """Whether a caller should try an image call again after the API layer gave up: transport failures, 408/429 and
    5xx only. A 4xx (moderation, bad request) or an opaque/undecodable result would come back the same."""
    st, code = getattr(e, "status", None), str(getattr(e, "code", None))
    if code == "insufficient_quota" or code.startswith("billing"):  # _call never retries these either: no credit left
        return False
    return isinstance(e, ApiError) and (st in (0, 408, 429) or (st or 0) >= 500)


def _fail_note(what, e, printed, prompt=None):
    """Print the whole error the first time it is seen (code, server message, the prompt), one line afterwards."""
    msg = str(e).replace("\n", " ")
    code = getattr(e, "code", None)  # ApiError.code (C2); a local RuntimeError (opaque/undecodable result) has none
    if msg[:160] in printed:
        print(f"  ! {what} failed with the same error{f' ({code})' if code else ''}", flush=True)
        return
    printed.add(msg[:160])
    hint = "; rephrase --moving/--locked/--look" if code == "moderation_blocked" or "moderation" in msg else ""
    print(f"  ! {what} failed: {msg}{hint}" + (f"\n    prompt: {prompt[:200]!r}" if prompt and "prompt:" not in msg else ""),
          flush=True)


def _pool(jobs, fn, items):
    """Run fn over items in a thread pool; results as they finish (None dropped). Ctrl-C cancels the queued jobs
    instead of waiting for every submitted call to time out."""
    ex = cf.ThreadPoolExecutor(max(1, jobs))
    futs = [ex.submit(fn, it) for it in items]
    try:
        out = [r for r in (f.result() for f in cf.as_completed(futs)) if r is not None]
    except BaseException:  # Ctrl-C or a job's own error: the queued calls must not start before the exit
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return out


FRAME_REROLL = ("The attached image is frame {j} of {n} of a seamless looping animation of {moving} ({motion}). Draw "
                "frame {i}: the SAME element in exactly the same style, colours, gradients, glow, size and anchor, with "
                "only the small natural change of shape between neighbouring frames. Only this one element, centred, "
                "transparent background, nothing else.")


def _reroll(a, crops, qa, bad, ref, ref_mask, motion, out):
    """Redraw flagged frames from their nearest good neighbour; keep a redraw only if its QA badness drops."""
    good = [i for i in range(len(crops)) if i not in bad]
    if not good:
        print("  ! every frame is flagged; re-roll skipped (re-run the sheet instead)")
        return crops, qa

    def job(i):
        j = min(good, key=lambda g: min(abs(g - i), len(crops) - abs(g - i)))
        prompt = FRAME_REROLL.format(i=i + 1, j=j + 1, n=len(crops), moving=a.moving, motion=motion)
        raw = edit(_png_bytes(_square(crops[j])[0]), prompt, "1024x1024", "transparent", a.model)[0]
        name = f"reroll_f{i + 1:02d}.png"
        img = _decode_result(raw, name)
        err = _opaque_error(img, name, a.model)
        if err:
            raise RuntimeError(err)
        (out / name).write_bytes(raw)                    # each redraw is kept as soon as it arrives
        return i, j, img
    ex = cf.ThreadPoolExecutor(2)
    futs = {ex.submit(job, i): i for i in bad}
    redraws = []
    try:
        for f in cf.as_completed(futs):
            try:
                redraws.append(f.result())
            except (ApiError, RuntimeError) as e:
                print(f"  ! frames: the re-roll f{futs[f] + 1} call failed: {e}")
    except BaseException:  # Ctrl-C or an unexpected error: the queued calls must not start before the exit
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    for i, j, img in sorted(redraws, key=lambda r: r[0]):
        bb = _robust_box(np.asarray(img.getchannel("A")))
        if not bb:
            print(f"  re-roll f{i + 1}: came back empty, discarded")
            continue
        img = img.crop(bb)
        k = math.sqrt(_alpha_stats(crops[j])[1] / _alpha_stats(img)[1])
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
        trial = crops[:i] + [img] + crops[i + 1:]
        tq = _frame_qa(_align_frames(trial, a.anchor), ref, ref_mask)
        old, new = _badness(qa["frames"][i]), _badness(tq["frames"][i])
        keep = new < old
        print(f"  re-roll f{i + 1}: badness {old:.2f} -> {new:.2f} {'KEPT' if keep else 'discarded'}")
        if keep:
            crops, qa = trial, tq
    return crops, qa


FRAME_FORMATS = ("gif", "webp", "apng", "lottie")


def cmd_frames(a):
    d = Path(a.dir)
    src = _open_image(d / "assets" / f"{a.name}.png", "asset", "run forge first, or check the name")
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    if not formats or set(formats) - set(FRAME_FORMATS):
        sys.exit(f"frames: --formats takes a comma list of {','.join(FRAME_FORMATS)}, got {a.formats!r}")
    if a.count is not None and a.count > 240:
        sys.exit(f"frames: --count {a.count} is more than 240 frames (10 s at 24 fps); a loop needs far fewer")
    if a.fps is not None and not 1 <= a.fps <= 120:
        sys.exit(f"frames: --fps {a.fps:g} must be between 1 and 120")
    if a.amount > 10:
        sys.exit(f"frames: --amount {a.amount:g} is a strength multiplier around 1 (at most 10)")
    if not np.asarray(src.getchannel("A")).any():
        sys.exit(f"frames: assets/{a.name}.png is fully transparent: there is nothing to animate")
    share, corners = _opaque_share(src)
    if share > 0.9 and corners:
        print(f"  ! assets/{a.name}.png has no transparent background ({share:.0%} opaque): the whole rectangle will "
              "move; forge it with transparent=true or cut the backdrop out first")
    out = d / "frames" / (f"{a.name}_{a.procedural}" if a.procedural else a.name)
    out.mkdir(parents=True, exist_ok=True)
    stale = ["frame_*.png", "moving_f*.png"] + ([] if a.reuse else ["reroll_f*.png", "static.png", "qa.json"])
    for pat in stale:
        for old in out.glob(pat):
            old.unlink()
    if a.count is not None and a.count < 2:
        sys.exit(f"frames: --count must be at least 2, got {a.count}")
    if a.procedural:
        return _frames_procedural(a, src, out)
    if not a.moving:
        sys.exit("frames: --moving is required (or use --procedural)")
    count, fps = a.count or 8, a.fps or 12
    cols = a.cols
    if cols > count:
        print(f"  (--cols {cols} > --count {count}: using {count} columns)")
        cols = count
    rows_n = math.ceil(count / cols)
    sheet_size = _size_for(cols * src.width, rows_n * src.height)
    canvas = {"1536x1024": "wide landscape", "1024x1536": "tall portrait"}.get(sheet_size, "square")
    motion = a.motion or "a gentle, natural looping motion"
    plural = lambda n, w: f"{n} {w}{'s' if n != 1 else ''}"  # noqa: E731
    moving_prompt = (f"The attached image is an APPROVED asset. Produce ONLY its moving part — {a.moving}"
                     f"{' with NO ' + a.static + ' at all' if a.static else ''} — as {count} animation frames of {motion}, "
                     f"laid out as a sprite sheet of {plural(cols, 'column')} x {plural(rows_n, 'row')} of equal cells on a "
                     f"{canvas} canvas (left-to-right, top-to-bottom). "
                     "Every frame: same style, colours, gradients and glow as the attached asset; same overall size and "
                     f"the SAME anchor point at the {'bottom centre' if a.anchor == 'bottom' else 'centre'} of its cell; "
                     "only the natural motion changes, and the "
                     "last frame flows back into the first. Transparent background everywhere. Absolutely no grid lines, "
                     "dividers, borders, numbers or labels. Leave empty transparent space between cells.")
    jobs = {}
    if a.reuse != "all":
        jobs["sheet"] = (moving_prompt, sheet_size)
    if a.static and not a.reuse:
        jobs["static"] = (f"The attached image is an APPROVED asset. Keep ONLY {a.static}, exactly as drawn: same size, "
                          f"position, angle and colours on the same canvas. Remove {a.moving} completely and repair "
                          "what it covered. Transparent background.", "1024x1024")
    for k in {"sheet", "static"} - set(jobs):
        if (k == "sheet" or a.static) and not (out / f"{k}_raw.png").exists():
            sys.exit(f"frames: --reuse needs {out / (k + '_raw.png')} from a previous run")
    _plan(len(jobs), 0, image_model=a.model, task="precise", background="transparent", edits=True)
    sq_src, sq_off = _square(src)
    ex = cf.ThreadPoolExecutor(2)
    futs = {ex.submit(edit, _png_bytes(sq_src if k == "static" else src), p, sz, "transparent", a.model): k
            for k, (p, sz) in jobs.items()}
    failed = []
    try:
        for f in cf.as_completed(futs):               # each result is written as it arrives, whatever the other does
            k = futs[f]
            try:
                raw = f.result()[0]
                img = _decode_result(raw, f"{k}_raw.png")
                err = _opaque_error(img, f"{k}_raw.png", a.model)
                if err:
                    (out / f"{k}_raw.opaque.png").write_bytes(raw)
                    raise RuntimeError(f"{err}; saved as {k}_raw.opaque.png")
                (out / f"{k}_raw.png").write_bytes(raw)
            except (ApiError, RuntimeError) as e:
                failed.append(k)
                print(f"  ! frames: the {k} call failed: {e}")
    except BaseException:  # Ctrl-C or an unexpected error: the queued call must not start before the exit
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    if failed:
        kept = [k for k in jobs if k not in failed]
        sys.exit(f"frames: the {' and '.join(failed)} call{'s' if len(failed) > 1 else ''} failed (see above)"
                 + (f"; {kept[0]}_raw.png was kept: re-run with --reuse {kept[0]}" if kept == ["static"] else ""))
    crops = _slice_sheet(_remove_dividers(_rgba(out / "sheet_raw.png")), count, cols)
    if not crops:
        sys.exit("frames: no frames found in the sheet")
    static = ref_mask = None
    if a.static:
        sq_static = _register(_rgba(out / "static_raw.png").resize(sq_src.size, Image.LANCZOS), sq_src)
        static = sq_static.crop((sq_off[0], sq_off[1], sq_off[0] + src.width, sq_off[1] + src.height))
        static.save(out / "static.png")
        ref_mask = _moving_mask(static)
    qa = _frame_qa(_align_frames(crops, a.anchor), src, ref_mask)
    if len(crops) != count:
        qa["sequence"]["flags"].append("frame-count")
        print(f"  ! only {len(crops)} of {count} frames: the loop timing changes and the seam may jump; re-run the sheet "
              "(no --reuse) or accept it")
    bad = _qa_report(qa)
    if a.reroll and bad:
        crops, qa = _reroll(a, crops, qa, bad, src, ref_mask, motion, out)
        _qa_report(qa, " after re-roll")
    frames = _align_frames(crops, a.anchor)
    if a.match_colors:
        frames = _match_colors(frames, src, ref_mask)
        flags = qa["sequence"]["flags"]
        qa = _frame_qa(frames, src, ref_mask)
        qa["sequence"]["flags"] += [f for f in flags if f == "frame-count"]
        _qa_report(qa, " after colour match")
    for i, f in enumerate(frames):
        f.save(out / f"moving_f{i + 1:02d}.png")
    comps, reg = _composite(src, static, frames, a.static_order) if static is not None else ([], None)
    if reg is not None:
        qa["sequence"]["registration_iou"] = round(float(reg), 3)
        if reg < 0.3:
            qa["sequence"]["flags"].append("misregistered")
            print(f"  ! MISREGISTERED: the moving frames match the asset's moving part poorly (IoU {reg:.2f}). Check "
                  "static.png (it must contain only the static part), describe --moving/--static more precisely, or "
                  "re-run with --reuse static; moving_fNN.png are still usable on their own")
    (out / "qa.json").write_text(json.dumps(qa, indent=1))
    seq = comps or frames
    for i, f in enumerate(seq):
        f.save(out / f"frame_{i + 1:02d}.png")
    labels = [f"f{r['frame']} {' '.join(r['flags'])}".strip() for r in qa["frames"]]
    _write_anim(out, a.name, seq, fps, a, labels)


# ---- layout-driven sequences: code lays the motion out, GPT renders every frame

# An image model has no notion of time. Asked for a sprite sheet it keeps the style but not the trajectory: drops
# jump between cells, the floor moves from row to row, cells come back empty. So the motion is decided in code
# (a periodic layout, one guide per frame) and GPT only renders each frame's finish, as a rendering artist working
# over an animator's layout. Nothing is interpolated: in-betweens synthesised from independently rendered keys
# stall and surge (see references/api-findings.md, "Layout-driven sequences").
SEQUENCE_BRIEF = (
    "You are the lighting and rendering artist on a premium animated app-icon production. The attached image is "
    "frame {n} of {total} of a seamless {seconds:g}-second looping animation at {fps:g} frames per second, as laid out "
    "by the animator. {locked} is final artwork and is locked. {moving} is the animator's rough layout: it shows "
    "exactly where every part of it is on this frame and how big it is.\n"
    "Render {moving_lc} as the finished, high-end frame: {look}\n"
    "This is one frame of a moving sequence, so the layout is binding: keep every part at exactly the position and "
    "size the layout gives it. Do not add, remove, merge away, move or resize anything, and do not tidy it into a "
    "different shape. Only the finish changes. {locked}, the camera, the framing, the scale and the lighting stay "
    "exactly as they are. Fully transparent background, no ground, no cast shadow, no text."
)
SEQUENCE_LOOK = ("with exactly the same finish as the locked artwork: the same materials, palette, lighting and level "
                 "of detail, with gentle self-shadowing where parts overlap (pass --look to describe it, e.g. "
                 "'stylised 3D, soft light from the upper left')")


def _seq_dir(a):
    d = Path(a.dir) / "sequence" / a.name
    if getattr(a, "stage", "layout") != "layout" and not (d / "sequence.json").exists():
        sys.exit(_seq_no_layout(a, d))  # before any mkdir: a typo in DIR or NAME must not leave empty folders behind
    for sub in ("guides", "raw", "frames", "inputs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _seq_no_layout(a, d):
    return (f"sequence {getattr(a, 'stage', '')}: no layout in {d / 'sequence.json'}: run `sequence {a.dir} {a.name} layout` first "
            "(--guides DIR, or --base/--sprites/--vent)")


def _seq_meta(d, **update):
    p = d / "sequence.json"
    meta = _read_json(p, "sequence.json") if p.exists() else {}
    if not isinstance(meta, dict):
        sys.exit(f"gptkit: sequence.json {p} is not a JSON object (hand-edited?): re-run `sequence ... layout`, which "
                 "starts a new one")
    if meta.get("frames") is not None:
        ok = {"frames": isinstance(meta["frames"], int) and not isinstance(meta["frames"], bool) and meta["frames"] >= 1,
              "fps": _num_ok(meta.get("fps")) and meta["fps"] > 0,
              "canvas": isinstance(meta.get("canvas"), int) and not isinstance(meta.get("canvas"), bool) and meta["canvas"] >= 1}
        bad = [f"{k}={meta.get(k)!r}" for k, v in ok.items() if not v]
        if bad:
            sys.exit(f"gptkit: sequence.json {p}: {', '.join(bad)} (frames and canvas must be positive integers, fps a "
                     "positive number; hand-edited or older file?): fix it or re-run `sequence ... layout`")
    if update:
        meta.update(update)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, p)
    return meta


def _seq_path(d, p):
    """A path stored in sequence.json (relative to the sequence dir since the inputs are copied there)."""
    p = Path(p)
    return p if p.is_absolute() else d / p


def _seq_sprites(d, meta):
    return [_rgba(_seq_path(d, s)) for s in meta["plume"]["sprites"]]


def _sha1(data):
    import hashlib
    return hashlib.sha1(data).hexdigest()


def _seq_raw_state(d, meta, i, name=None):
    """'ok' | 'missing' | 'stale' for raw/NNN.png: stale when its sidecar names a guide hash the layout no longer has
    (a raw without a sidecar is from an older kit and is trusted)."""
    p = d / "raw" / (name or f"{i:03d}.png")
    if not p.exists():
        return "missing"
    side = p.with_suffix(".json")
    if side.exists():
        try:
            want = (meta.get("guide_sha1") or [None] * meta["frames"])[i]
            if json.loads(side.read_text()).get("guide_sha1") != want:
                return "stale"
        except (OSError, ValueError, IndexError):
            return "stale"
    return "ok"


def _seq_stale(d, meta, idx):
    stale = [i for i in idx if _seq_raw_state(d, meta, i) == "stale"]
    if stale:
        names = ", ".join(f"raw/{i:03d}.png" for i in stale[:6]) + (" ..." if len(stale) > 6 else "")
        sys.exit(f"sequence: {names} {'was' if len(stale) == 1 else 'were'} rendered from an older layout: run "
                 "`sequence ... render` again (it re-renders stale frames)")


def _vec(text, n=2, flag="value"):
    try:
        vals = [float(v) for v in str(text).split(",")]
    except ValueError:
        vals = []
    if len(vals) != n or any(math.isnan(v) or math.isinf(v) for v in vals):
        raise SystemExit(f"sequence layout: {flag} expects x,y numbers (a negative x needs {flag}=-40,-3), got {text!r}")
    return vals


def _natural(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).stem)]


def plume_state(age, k, p):
    """Closed-form flight of emission k at `age`: linear drag toward the wind plus buoyancy, so the terminal
    velocity is wind + (0, buoyancy / drag). The per-emission jitter repeats every loop, which makes the layout
    periodic by construction: layout(t + loop) == layout(t)."""
    per_loop = round(p["loop"] / p["chuff"])
    rnd = np.random.default_rng(k % per_loop)
    jx, jy, js = rnd.uniform(-1, 1, 3)
    v0 = (p["velocity"][0] + p["spread"][0] * jx, p["velocity"][1] + p["spread"][1] * jy)
    w = (p["wind"][0], p["wind"][1] + p["buoyancy"] / p["drag"])
    e = (1 - math.exp(-p["drag"] * age)) / p["drag"]
    x = p["vent"][0] + w[0] * age + (v0[0] - w[0]) * e
    y = p["vent"][1] + w[1] * age + (v0[1] - w[1]) * e
    u = age / p["life"]
    scale = (p["grow"][0] + (p["grow"][1] - p["grow"][0]) * u) * (1 + 0.08 * js)
    alpha = min(1, age / 0.08) * min(1, (1 - u) / 0.4)
    return x, y, scale, max(0.0, alpha), k % max(1, len(p["sprites"]))


def plume_layout(t, p):
    """Every live emission at absolute time t, keyed by emission number (identity survives the loop wrap)."""
    out = {}
    for k in range(math.floor((t - p["life"]) / p["chuff"]), math.floor(t / p["chuff"]) + 1):
        age = t - k * p["chuff"]
        if 0 <= age < p["life"]:
            out[k] = plume_state(age, k, p)
    return out


def plume_guide(t, p, base=None, sprites=None, canvas=512):
    """One guide frame: the plume's sprites at time t, oldest first, behind (or in front of) the locked base art.
    `sprites` are the loaded sprite images (paths in p are relative to the sequence dir)."""
    sprites = sprites or [_rgba(s) for s in p["sprites"]]
    layer = Image.new("RGBA", (canvas, canvas))
    for _, (x, y, s, al, v) in sorted(plume_layout(t, p).items()):
        sp = sprites[v]
        w = p["size"] * s
        h = w * sp.height / sp.width
        im = sp.resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS)
        arr = np.array(im)
        arr[..., 3] = (arr[..., 3] * al).astype(np.uint8)
        layer.alpha_composite(Image.fromarray(arr), (round(x - w / 2), round(y - h / 2)))
    if base is None:
        return layer
    out = Image.new("RGBA", (canvas, canvas))
    for part in ((base, layer) if p.get("front") else (layer, base)):
        out.alpha_composite(part)
    return out


def _seq_save_guide(d, i, g):
    big = Image.new("RGBA", (1024, 1024))
    big.alpha_composite(g if g.size == (1024, 1024) else g.resize((1024, 1024), Image.LANCZOS))
    p = d / "guides" / f"{i:03d}.png"
    big.save(p, compress_level=3)                       # transient 1024 px files: half the encode time of level 6
    return _sha1(p.read_bytes())


def _seq_check_layout(d, meta):
    """The layout must have moving content, and a guide set must loop (the wrap step is a step like any other)."""
    if meta.get("plume"):
        p = meta["plume"]
        if not any(st[3] > 0 for i in range(meta["frames"]) for st in plume_layout(i / meta["fps"], p).values()):
            sys.exit("sequence layout: the plume has no live element on any frame (raise --life or --chuff)")
        return
    static = _seq_static_mask(d, meta)
    parts = [_seq_layout_part(d, meta, i, static).resize((256, 256), Image.LANCZOS) for i in range(meta["frames"])]
    if not any((np.asarray(p)[..., 3] > 3).any() for p in parts):
        sys.exit("sequence layout: the layout has no moving content: every guide is empty"
                 + (" after cutting --base" if meta.get("base") else "") + " (check the guides and --base)")
    if meta["frames"] < 4:
        return
    # the wrap step must look like a step the loop already contains (measured: non-looping sets score 4x+ the largest
    # in-loop motion-compensated residual, looping ones with in-loop jumps or splashes at most 1x)
    steps = [_seq_step(parts[i], parts[(i + 1) % len(parts)])[1] for i in range(len(parts))]
    ref = max(max(steps[:-1]), 0.5)
    if steps[-1] > 2 * ref:
        print(f"  ! the guides do not loop: the step from the last guide back to the first is {steps[-1] / ref:.1f}x "
              "the largest step inside the loop (frame 0 must follow the last frame as the next frame would)")


def _seq_fresh_meta(d, **fresh):
    """Layout starts a new sequence.json: only the render-side text (moving, locked, look) survives, never the old
    plume, route, qa, rect or propagated state."""
    try:
        old = _seq_meta(d)
    except SystemExit as e:  # layout is how a broken sequence.json gets replaced: never let it block that
        print(f"  ! {str(e.code)[:200]}; starting a new one")
        old = {}
    keep = {k: old[k] for k in ("moving", "locked", "look") if old.get(k)}
    p = d / "sequence.json"
    p.write_text(json.dumps({**keep, **fresh}, indent=1))
    return _seq_meta(d)


def cmd_sequence_layout(a):
    d = _seq_dir(a)
    for sub in ("guides",):
        for old in (d / sub).glob("*.png"):
            old.unlink()
    if a.guides:
        folder = _need(a.guides, "guide folder")
        paths = sorted(folder.glob("*.png"), key=_natural)
        if not paths:
            sys.exit(f"sequence layout: no PNG guides in {folder}")
        guides = [_open_image(p, "guide") for p in paths]
        sizes = sorted({g.size for g in guides})
        if len(sizes) != 1 or sizes[0][0] != sizes[0][1]:
            got = "; ".join(f"{w}x{h}: " + ", ".join(p.name for p, g in zip(paths, guides) if g.size == (w, h))[:60]
                            for w, h in sizes[:4])
            sys.exit(f"sequence layout: guides must be square and all one size, got {got}")
        base = None
        if a.base:
            base = _open_image(a.base, "--base")
            shutil.copyfile(a.base, d / "inputs" / "base.png")
        opaque = float(np.mean([(np.asarray(g.getchannel("A")) == 255).mean() for g in guides]))
        if opaque > 0.5 and not a.base:
            print(f"  ! the guides have an opaque backdrop ({opaque:.0%} of their pixels are opaque): pass it as --base "
                  "so build can cut it away, or supply transparent guides")
        hashes = [_seq_save_guide(d, i, g) for i, g in enumerate(guides)]
        names = [p.name for p in paths]
        print(f"  guides in order: {', '.join(names[:3])}"
              + (f" ... {', '.join(names[-3:])}" if len(names) > 6 else (", " + ", ".join(names[3:]) if len(names) > 3 else "")))
        meta = _seq_fresh_meta(d, frames=len(paths), fps=a.fps, seconds=len(paths) / a.fps, canvas=guides[0].width,
                               base="inputs/base.png" if base is not None else None, source=str(folder.resolve()),
                               guide_sha1=hashes)
    else:
        if not (a.base and a.sprites and a.vent):
            sys.exit("sequence layout: a plume layout needs --base, --sprites and --vent (or pass --guides DIR)")
        base = _open_image(a.base, "--base")
        sprite_paths = [_need(s, "sprite") for s in a.sprites.split(",") if s]
        sprites = [_open_image(s, "sprite") for s in sprite_paths]
        shutil.copyfile(a.base, d / "inputs" / "base.png")
        for i, sp in enumerate(sprite_paths):
            shutil.copyfile(sp, d / "inputs" / f"sprite_{i}.png")
        p = dict(loop=a.loop, chuff=a.chuff, life=a.life, vent=_vec(a.vent, flag="--vent"),
                 velocity=_vec(a.velocity, flag="--velocity"), spread=_vec(a.spread, flag="--spread"), drag=a.drag,
                 wind=_vec(a.wind, flag="--wind"), buoyancy=a.buoyancy, size=a.size, grow=_vec(a.grow, flag="--grow"),
                 sprites=[f"inputs/sprite_{i}.png" for i in range(len(sprites))], front=a.front)
        if not (math.isfinite(a.buoyancy) and abs(a.buoyancy) <= 10000):
            sys.exit(f"sequence layout: --buoyancy {a.buoyancy:g} must be a number between -10000 and 10000 (canvas units "
                     "per second squared; negative rises)")
        if min(p["grow"]) <= 0:
            sys.exit(f"sequence layout: --grow {a.grow} must be two scales > 0 (at birth, at death), e.g. 0.45,1.5")
        per_loop = p["loop"] / p["chuff"]
        if abs(per_loop - round(per_loop)) > 1e-6:
            sys.exit(f"sequence layout: --loop {a.loop} must be a whole number of --chuff {a.chuff} intervals, "
                     "or the loop cannot repeat exactly")
        n = round(a.loop * a.fps)
        if n < 2:
            sys.exit(f"sequence layout: --loop {a.loop} at {a.fps:g} fps is {n} frame(s): raise --loop or --fps")
        if n > 1000:
            sys.exit(f"sequence layout: --loop {a.loop:g} at {a.fps:g} fps is {n} frames (one image call each; the limit is "
                     "1000): lower --loop or --fps")
        if a.life / a.chuff > 2000:
            sys.exit(f"sequence layout: --life {a.life:g} / --chuff {a.chuff:g} keeps {a.life / a.chuff:.0f} elements alive "
                     "(limit 2000): use values on the scale of the defaults (--life 5.4 --chuff 0.75)")
        if a.size > 8 * base.width:
            sys.exit(f"sequence layout: --size {a.size:g} is over 8x the {base.width}px --base canvas (default 96)")
        hashes = [_seq_save_guide(d, i, plume_guide(i / a.fps, p, base, sprites, canvas=base.width)) for i in range(n)]
        meta = _seq_fresh_meta(d, frames=n, fps=a.fps, seconds=a.loop, canvas=base.width, base="inputs/base.png",
                               plume=p, guide_sha1=hashes)
    _seq_check_layout(d, meta)
    picks = list(range(0, meta["frames"], max(1, meta["frames"] // 12)))[:12]
    sheet([[(f"f{i}", _rgba(d / "guides" / f"{i:03d}.png")) for i in picks[r:r + 6]] for r in (0, 6)],
          d / "layout_review.png", T=200)
    print(f"{meta['frames']} guide frames ({meta['seconds']:g} s at {meta['fps']:g} fps) -> {d / 'guides'}\n"
          f"  review {d / 'layout_review.png'} before rendering: the motion is decided here, not by GPT")
    seq_plan(d, meta, a)


def _seq_prompt(meta, i):
    moving = meta.get("moving") or "The moving element"
    return SEQUENCE_BRIEF.format(n=i + 1, total=meta["frames"], seconds=meta["seconds"], fps=meta["fps"],
                                 locked=meta.get("locked") or "The rest of the artwork", moving=moving,
                                 moving_lc=moving[:1].lower() + moving[1:], look=meta.get("look") or SEQUENCE_LOOK)


def _seq_render_one(d, meta, i, out, model):
    """Render frame i from its guide into `out` (temp + rename, with a sidecar naming the guide it came from)."""
    raw = edit(_png_bytes(_rgba(d / "guides" / f"{i:03d}.png")), _seq_prompt(meta, i), "1024x1024", "transparent", model)[0]
    img = _decode_result(raw, f"raw/{out.name}")
    err = _opaque_error(img, f"raw/{out.name}", model)
    if err:
        out.with_name(out.stem + ".opaque.png").write_bytes(raw)
        raise RuntimeError(f"{err}; saved as raw/{out.stem}.opaque.png, not used")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, out)
    out.with_suffix(".json").write_text(json.dumps({"guide_sha1": (meta.get("guide_sha1") or [None] * (i + 1))[i]}))
    return out


def cmd_sequence_render(a):
    d = _seq_dir(a)
    meta = _seq_meta(d, moving=a.moving, locked=a.locked, look=a.look or SEQUENCE_LOOK)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    if a.every is None:
        a.every = _seq_route(d, meta, a)["every"]
    elif meta["frames"] % a.every:
        sys.exit(f"sequence render: --every {a.every} must divide the {meta['frames']} frames so the keys repeat with the "
                 "loop (propagate needs that)")
    todo = [i for i in range(0, meta["frames"], a.every) if a.force or _seq_raw_state(d, meta, i) != "ok"]
    _plan(len(todo), 0, image_model=a.model, task="precise", background="transparent", edits=True)
    if getattr(a, "dry_run", False):
        print("would render frames: " + (", ".join(str(i) for i in todo) or "none"))
        return
    print(f"rendering {len(todo)} of {meta['frames']} frames, {a.jobs} at a time (resumable: finished frames are kept)")
    printed = set()

    def job(i):
        for attempt in range(3):
            try:
                return _seq_render_one(d, meta, i, d / "raw" / f"{i:03d}.png", a.model) and i
            except (ApiError, RuntimeError) as e:
                again = _retryable(e) and attempt < 2
                _fail_note(f"frame {i}" + (f" (attempt {attempt + 1}, retrying)" if again else ""), e, printed,
                           _seq_prompt(meta, i))
                if not again:
                    return None
                time.sleep(4 * (attempt + 1))
        return None
    done = _pool(a.jobs, job, todo)
    # count only the frames asked for (keys with --every), by name, so reroll candidates never count as frames
    missing = sum(1 for i in range(0, meta["frames"], a.every) if _seq_raw_state(d, meta, i) != "ok")
    print(f"  {len(done)} rendered" + (f", {missing} still missing" if missing else ""))
    if missing:
        sys.exit(f"sequence render: {missing} of {len(range(0, meta['frames'], a.every))} frames still missing (the "
                 "errors are above): fix the cause and run render again; finished frames are kept")


def _seq_extract(d, meta, raw_path, i, static):
    """Register a render onto its guide and keep only the moving part (the locked art is cut away)."""
    canvas = meta["canvas"]
    guide = _rgba(d / "guides" / f"{i:03d}.png").resize((canvas, canvas), Image.LANCZOS)
    arr = np.array(_register(_rgba(raw_path).resize((canvas, canvas), Image.LANCZOS), guide))
    if static is not None:
        arr[static] = 0
    return Image.fromarray(arr)


def _seq_static_mask(d, meta, grow=2):
    if not meta.get("base"):
        return None
    base = _rgba(_seq_path(d, meta["base"])).resize((meta["canvas"], meta["canvas"]), Image.LANCZOS)
    m = base.getchannel("A").point(lambda v: 255 if v > 8 else 0).filter(ImageFilter.MaxFilter(grow * 2 + 1))
    return np.asarray(m) > 0


def _grey(img):
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    return arr[..., :3].mean(-1) * arr[..., 3] / 255


def _seq_step(a, b):
    """(motion vector, motion-compensated residual) from frame a to b; needs OpenCV, else a plain difference."""
    ga, gb = _grey(a), _grey(b)
    m = ga > 30
    if not m.any():
        return (0.0, 0.0), 0.0
    try:
        import cv2
    except ImportError:
        return (0.0, 0.0), float(np.abs(gb - ga)[m].mean())
    fl = cv2.calcOpticalFlowFarneback(ga.astype(np.uint8), gb.astype(np.uint8), None, 0.5, 4, 21, 4, 7, 1.5, 0)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(gb, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR)
    return (float(fl[..., 0][m].mean()), float(fl[..., 1][m].mean())), float(np.abs(warped - ga)[m].mean())


def _qa_size(frames, cap=384):
    """Copies with the long side at most `cap` px: flow QA cost grows with area and 384 px is plenty for it."""
    w, h = frames[0].size
    if max(w, h) <= cap:
        return frames
    k = cap / max(w, h)
    return [f.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS) for f in frames]


def _seq_qa(frames, layout=None):
    """Direction, seam and shimmer of a loop. A frame that disagrees with both neighbours much more than they agree
    with each other is an outlier (usually a render that ignored the layout): reroll it. With `layout` (the layout
    parts, cropped like the frames) every score is taken relative to what the layout itself does, so the birth of a
    puff or a splash the layout contains is not an outlier and a radial/oscillating layout gets no direction QA."""
    frames = _qa_size(frames)
    n = len(frames)
    steps = [_seq_step(frames[i], frames[(i + 1) % n]) for i in range(n)]
    vecs = np.array([s[0] for s in steps])
    heading = vecs.mean(0)
    norm = float(np.hypot(*heading)) or 1.0
    along = vecs @ (heading / norm)
    resid = np.array([s[1] for s in steps])
    excess = []
    for i in range(n):
        r_in, r_out = resid[(i - 1) % n], resid[i]
        _, r_skip = _seq_step(frames[(i - 1) % n], frames[(i + 1) % n])
        dev = abs(along[(i - 1) % n] - np.median(along)) + abs(along[i] - np.median(along))
        excess.append(r_in + r_out - r_skip + 4 * dev)
    excess = np.array(excess)
    base = _seq_qa(layout) if layout else None
    score = excess - np.array(base["excess"]) if base else excess
    order = sorted(range(n), key=lambda i: -score[i])
    med, mad = float(np.median(score)), float(np.median(np.abs(score - np.median(score))))
    # robust: well clear of the typical frame by both ratio and spread, so a clean loop reports none; against a
    # layout, a frame must also add more than 60% of a typical frame's own excess on top of what the layout does
    floor = 0.6 * float(np.median(excess)) if base else 0.0
    outliers = [i for i in order[:max(1, n // 12)]
                if score[i] > max(med * 1.6, floor) and score[i] > med + 4 * mad]
    backwards = int((along < 0).sum()) if norm > 0.05 else 0
    directional = bool(base is None or np.hypot(*base["heading_px"]) >= 0.3)
    if base:
        backwards = max(0, backwards - base["backwards"]) if directional else 0
    return {
        "frames": n,
        "heading": [round(float(v), 3) for v in heading / norm],
        "heading_px": [round(float(v), 3) for v in heading],
        "speed_along": [round(float(x), 2) for x in along],
        "backwards": backwards,
        "seam_step": round(float(along[-1]), 2),
        "median_step": round(float(np.median(along)), 2),
        "shimmer_mean": round(float(resid.mean()), 2),
        "shimmer_max": round(float(resid.max()), 2),
        "excess": [round(float(x), 2) for x in excess],
        "outliers": outliers,
        "outlier_scores": {i: round(float(score[i]), 1) for i in order[:8]},
        **({"vs_layout": True, "layout_backwards": base["backwards"], "directional": directional} if base else {}),
    }


def _seq_moving(d, meta, a, i, raw_path, static):
    """One render reduced to its moving part: registered, the locked art cut away, clipped below --below."""
    f = _seq_extract(d, meta, raw_path, i, static)
    if a.below is not None:
        arr = np.array(f)
        arr[int(a.below):] = 0
        f = Image.fromarray(arr)
    return f


def cmd_sequence_build(a):
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    n = meta["frames"]
    raws = [d / "raw" / f"{i:03d}.png" for i in range(n)]
    _seq_stale(d, meta, range(n))
    missing = [i for i, p in enumerate(raws) if not p.exists()]
    if missing:
        route = _seq_route(d, meta, a)
        every = a.every or route["every"]
        if every > 1 and route["engine"] == "ebsynth" \
                and all((d / "raw" / f"{k:03d}.png").exists() for k in range(0, n, every)):
            print(f"keys only (every {every} frames): propagating them with EbSynth, as planned")
            a.every = every
            return cmd_sequence_propagate(a)
    if missing:
        raise SystemExit(f"frames missing from raw/: {missing[:10]} — run `sequence ... render` again, "
                         "or `sequence ... propagate` if you rendered keys only (--every)")
    ref = _build_opts(d, meta, a)
    static = _seq_static_mask(d, meta)
    frames = [_seq_moving(d, meta, a, i, p, static) for i, p in enumerate(raws)]
    if ref is not None:
        # per frame, not one transform for the loop: every frame is an independent render with its own exposure, and
        # pulling each onto the approved part evens that out (measured shimmer 9.4 vs 10.5 for one shared transform)
        frames = [_match_colors([f], ref)[0] for f in frames]
    meta.pop("propagated", None)
    _seq_meta(d, propagated=None)
    _seq_finish(d, meta, a, frames)


def _build_opts(d, meta, a):
    """--ref/--below/--quality/--preview-bg for this build: what was passed, else what the last build or propagate
    used (kept in sequence.json, so a reroll + build does not silently change the output). Returns the --ref image."""
    last = meta.get("build_opts") or {}
    opts, reused = {}, []
    for k, default in (("ref", None), ("below", None), ("quality", 80), ("preview_bg", "#a0aabe")):
        v = getattr(a, k, None)
        if v is None and last.get(k) is not None:
            v = last[k]
            if v != default:
                reused.append(f"--{k.replace('_', '-')} {v}")
        opts[k] = default if v is None else v
        setattr(a, k, opts[k])
    if reused:
        print(f"  using {' '.join(reused)} from the last build")
    ref = None
    if opts["ref"]:
        src = _seq_path(d, opts["ref"])
        if getattr(a, "ref", None) and not str(opts["ref"]).startswith("inputs/"):
            ref = _open_image(src, "--ref")
            shutil.copyfile(src, d / "inputs" / "ref.png")
            opts["ref"] = "inputs/ref.png"
        else:
            ref = _open_image(src, "--ref", "the copy made at the last build is gone; pass --ref again")
        if _opaque_share(ref)[0] > 0.9:
            print("  ! --ref has no transparent background: its backdrop takes part in the colour statistics; crop or "
                  "cut it out for a truer match")
    if opts["below"] is not None and not 0 < opts["below"] <= meta["canvas"]:
        sys.exit(f"--below {opts['below']} must be a canvas row between 1 and {meta['canvas']}")
    meta["build_opts"] = opts
    _seq_meta(d, build_opts=opts)
    return ref


def _seq_finish(d, meta, a, frames):
    """Crop the loop to its union box, encode the WebP, measure it and write the preview and contact sheet."""
    n = len(frames)
    union = np.zeros(frames[0].size[::-1], bool)
    for f in frames:
        union |= np.asarray(f)[..., 3] > 3
    if not union.any():
        sys.exit("sequence build: every frame is empty after cutting the locked art: the renders in raw/ came back "
                 "blank or entirely inside --base; check raw/ and --below")
    for old in (d / "frames").glob("*.png"):
        old.unlink()
    for i, f in enumerate(frames):
        f.save(d / "frames" / f"{i:03d}.png")
    ys, xs = np.nonzero(union)
    box = (max(0, xs.min() - 4), max(0, ys.min() - 4), min(union.shape[1], xs.max() + 5), min(union.shape[0], ys.max() + 5))
    crops = [f.crop(box) for f in frames]
    webp = d / f"{a.name}.webp"
    crops[0].save(webp, save_all=True, append_images=crops[1:], duration=_durations(n, meta["fps"]), loop=0,
                  format="WEBP", lossless=False, quality=a.quality, method=6, exact=True)
    static = _seq_static_mask(d, meta)
    layout = []
    for i in range(n):
        part = _seq_layout_part(d, meta, i, static)
        if static is not None and meta.get("plume"):    # the renders lose what the locked art covers; so must the layout
            arr = np.array(part)
            arr[static] = 0
            part = Image.fromarray(arr)
        layout.append(part.crop(box))
    if a.below is not None and int(a.below) - box[1] < box[3] - box[1]:
        for f in layout:
            ImageDraw.Draw(f).rectangle([0, max(0, int(a.below) - box[1]), f.width, f.height], fill=(0, 0, 0, 0))
    qa = _seq_qa(crops, layout)
    if meta.get("plume"):
        try:
            qa["follows"] = _seq_follows(d, frames, meta, box)
        except ImportError:
            print("  (OpenCV is not installed: the `follows` check is skipped; pip install opencv-python-headless)")
    rect = [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
    _seq_meta(d, rect=rect, qa=qa)
    (d / "report.json").write_text(json.dumps({"rect": rect, **qa}, indent=1))
    base = _rgba(_seq_path(d, meta["base"])).resize((meta["canvas"], meta["canvas"])) if meta.get("base") else None
    comps = []
    for f in frames:
        c = Image.new("RGBA", f.size, a.preview_bg)
        c.alpha_composite(f)
        if base is not None:
            c.alpha_composite(base)
        comps.append(c.convert("RGB"))
    comps[0].save(d / "preview.gif", save_all=True, append_images=comps[1:], duration=_durations(n, meta["fps"], 10),
                  loop=0)
    picks = list(range(0, n, max(1, n // 12)))[:12]
    sheet([[(f"f{i}{' !' if i in qa['outliers'] else ''}", comps[i]) for i in picks[r:r + 6]] for r in (0, 6)],
          d / "contact.png", T=200)
    prop = meta.get("propagated")
    if prop and qa["outliers"]:
        near = sorted({(i // prop["every"]) * prop["every"] for i in qa["outliers"]}
                      | {((i // prop["every"]) * prop["every"] + prop["every"]) % n for i in qa["outliers"]})
        fix = (f"they were synthesised from keys {near}: delete the nearest key's raw/NNN.png, run `sequence ... "
               "render` (it renders only the missing keys), then `build`")
    else:
        fix = "run `sequence ... reroll` on them (2 candidates each, the best neighbour fit wins), then `build`"
    direction = (f"{qa['backwards']} steps backwards, seam {qa['seam_step']} vs median {qa['median_step']}"
                 if qa.get("directional", True) else "no direction QA (the layout has no dominant heading)")
    print(f"{n} frames @ {meta['fps']:g} fps -> {webp} ({webp.stat().st_size // 1024} KB), rect {rect} on the "
          f"{meta['canvas']} canvas\n"
          f"  QA: {direction}, shimmer {qa['shimmer_mean']} (max {qa['shimmer_max']})"
          + (f", follows the layout {qa['follows']}%" if "follows" in qa else "")
          + (f"\n  ! outliers {qa['outliers']}: {fix}" if qa["outliers"] else "")
          + ("\n  ! the texture barely moves with the layout: GPT likely painted the same plume in every key. Make the "
             "elements distinct (fewer, rounder, spaced by --chuff) and render the keys again"
             if qa.get("follows", 100) < 5 else ""))


def _seq_follows(d, frames, meta, box):
    """Share of the frame-to-frame change that disappears when each frame is carried along the layout's exact motion.
    A texture that travels with its plume scores about 15%+; one that boils in place scores about 0, which the other
    checks miss (a dense exhaust whose keys were all the same generic plume scored 0.8%)."""
    import cv2
    p = meta["plume"]
    sprites = _seq_sprites(d, meta)
    x0, y0, x1, y1 = box
    fps, n = meta["fps"], len(frames)
    k = min(1.0, 384 / max(x1 - x0, y1 - y0))          # flow-free, but the remaps scale with area: cap at 384 px
    size = (max(1, round((x1 - x0) * k)), max(1, round((y1 - y0) * k)))
    res, still = [], []
    for i in range(n):
        mx, my = _plume_source_coords(i / fps, (i + 1) / fps, p, sprites, meta["canvas"])
        mx, my = (mx[y0:y1, x0:x1] - x0) * k, (my[y0:y1, x0:x1] - y0) * k
        if k < 1:
            mx, my = cv2.resize(mx, size, interpolation=cv2.INTER_LINEAR), cv2.resize(my, size, interpolation=cv2.INTER_LINEAR)
        a = np.asarray(frames[i].crop(box).resize(size, Image.LANCZOS), dtype=np.float32)
        b = np.asarray(frames[(i + 1) % n].crop(box).resize(size, Image.LANCZOS), dtype=np.float32)
        pa, pb = a[..., :3] * a[..., 3:4] / 255, b[..., :3] * b[..., 3:4] / 255
        carried = cv2.remap(pa, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        m = b[..., 3] > 60
        if m.any():
            res.append(np.abs(carried - pb)[m].mean())
            still.append(np.abs(pa - pb)[m].mean())
    return round(float(100 * (1 - np.mean(res) / max(1e-6, np.mean(still)))), 1) if still else 0.0


# -- propagate: GPT paints keys, EbSynth carries their texture along the layout to every frame

def _find_ebsynth(a):
    """--ebsynth, $GPTKIT_EBSYNTH, PATH, then ~/.cache/gptkit; a path named explicitly must exist (no silent fallback)."""
    for c, what in ((getattr(a, "ebsynth", None), "--ebsynth"), (os.environ.get("GPTKIT_EBSYNTH"), "GPTKIT_EBSYNTH")):
        if c and not Path(c).exists():
            sys.exit(f"gptkit: {what} {c} not found")
        if c:
            return c
    for c in (shutil.which("ebsynth"), str(Path.home() / ".cache" / "gptkit" / "ebsynth" / "bin" / "ebsynth")):
        if c and Path(c).exists():
            return c
    return None


EBSYNTH_BUILD = ("  git clone https://github.com/jamriska/ebsynth ~/.cache/gptkit/ebsynth\n"
                 f"  cd ~/.cache/gptkit/ebsynth && sh build-{'macos' if sys.platform == 'darwin' else 'linux'}-cpu_only.sh"
                 f"   # build-{'linux' if sys.platform == 'darwin' else 'macos'}-cpu_only.sh on "
                 f"{'Linux' if sys.platform == 'darwin' else 'macOS'}\n"
                 "or pass --ebsynth PATH / set GPTKIT_EBSYNTH.")


def _ebsynth_bin(a):
    eb = _find_ebsynth(a)
    if not eb:
        raise SystemExit("EbSynth not found. Build it once (public-domain code; the PatchMatch algorithm is patented, "
                         "check that before redistributing commercially; CPU is enough):\n" + EBSYNTH_BUILD)
    return eb


# -- plan: decide whether EbSynth carries the keys (and how far apart they sit) or GPT renders every frame

SEQ_KEY_SECONDS = 0.25   # measured: keys every 0.25 s never stepped backwards, every 0.5 s stalled mid-interval
SEQ_MIN_REACH = 0.9      # share of each in-between's layout that a neighbouring key must show for EbSynth to carry it
SEQ_PATCH_MATCH = 0.08   # RMS patch distance below which a key shows the same thing (measured: shifts and growth match, a new shape or colour does not)
SEQ_DENSE = 3.0          # mean elements stacked on a plume pixel above which GPT paints one generic plume


def _seq_plume_reach(meta, every):
    """Worst share of an in-between's plume (by element area) that exists in one of its two keys. An element born and
    gone between two keys has nothing to carry it; that happens when the keys sit further apart than --life."""
    p, fps, n = meta["plume"], meta["fps"], meta["frames"]
    worst = 1.0
    for i in range(n):
        ka = (i // every) * every
        if i == ka:
            continue
        have = set(plume_layout(ka / fps, p)) | set(plume_layout((ka + every) / fps, p))
        live = plume_layout(i / fps, p)
        total = sum(st[2] ** 2 * st[3] for st in live.values())
        if total > 0:
            worst = min(worst, sum(st[2] ** 2 * st[3] for k, st in live.items() if k in have) / total)
    return worst


def _seq_patches(img, cell=8):
    """3x3-cell patches (premultiplied RGBA, lightly blurred) of a layout part, with each patch's coverage."""
    n = img.width // cell
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(3)).resize((n, n), Image.BOX), dtype=np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)))
    return np.stack([pad[y:y + n, x:x + n] for y in range(3) for x in range(3)], 2).reshape(n * n, -1), arr[..., 3].reshape(-1)


def _seq_key_patches(img, cell=8):
    """The patches a key offers EbSynth: at sub-cell offsets and a little smaller or larger, since elements travel and
    grow between keys."""
    out = []
    for s in (0.85, 1.0, 1.18):
        big = img.resize((round(img.width * s),) * 2, Image.LANCZOS)
        for dy in range(0, cell, 2):
            for dx in range(0, cell, 2):
                c = Image.new("RGBA", big.size)
                c.alpha_composite(big, (dx, dy))
                pt, w = _seq_patches(c, cell)
                out.append(pt[w > 0.02])
    return np.concatenate(out)


def _seq_guide_reach(d, meta, every, static, cache=None):
    """Worst share of an in-between's content that looks like something in one of its two keys. EbSynth matches
    patches by appearance wherever they sit, so travel, jumps and growth carry; a splash, a blink or a colour that
    neither key shows has no paint to carry and would be invented."""
    n = meta["frames"]
    cache = {} if cache is None else cache

    def part(i):
        if ("part", i) not in cache:
            cache[("part", i)] = _seq_layout_part(d, meta, i, static).resize((256, 256), Image.LANCZOS)
        return cache[("part", i)]

    def keyp(k):
        if ("key", k) not in cache:
            cache[("key", k)] = _seq_key_patches(part(k))
        return cache[("key", k)]
    worst = 1.0
    for i in range(n):
        ka = (i // every) * every
        if i == ka:
            continue
        pt, w = _seq_patches(part(i))
        side = math.isqrt(len(w))
        edge = np.ones((side, side), bool)
        edge[2:-2, 2:-2] = False                    # content sliced by the canvas edge is entering or leaving, not new
        m = (w > 0.02) & ~edge.reshape(-1)
        if not m.any():
            continue
        keys = np.concatenate([keyp(ka), keyp((ka + every) % n)])
        if not len(keys):
            return 0.0
        q = pt[m]
        k2 = (keys ** 2).sum(1)[None]
        best = np.concatenate([((q[j:j + 256] ** 2).sum(1)[:, None] + k2 - 2 * q[j:j + 256] @ keys.T).min(1)
                               for j in range(0, len(q), 256)])   # 256-row blocks: the full matrix is 1k x 100k
        dist = np.sqrt(np.maximum(best, 0) / pt.shape[1])
        worst = min(worst, float(w[m][dist < SEQ_PATCH_MATCH].sum() / w[m].sum()))
    return worst


def _seq_density(d, meta):
    """Mean number of plume elements stacked on a covered pixel, over a few frames of the loop."""
    p = meta["plume"]
    sprites = _seq_sprites(d, meta)
    canvas = meta["canvas"]
    depths = []
    for t in np.linspace(0, meta["seconds"], 4, endpoint=False):
        stack = np.zeros((canvas, canvas), np.float32)
        for x, y, s, al, v in plume_layout(t, p).values():
            sp = sprites[v]
            w = p["size"] * s
            h = w * sp.height / sp.width
            foot = Image.new("L", (canvas, canvas))
            foot.paste(sp.getchannel("A").resize((max(1, round(w)), max(1, round(h)))), (round(x - w / 2), round(y - h / 2)))
            stack += np.asarray(foot, dtype=np.float32) > 60
        if (stack > 0).any():
            depths.append(float(stack[stack > 0].mean()))
    return round(float(np.mean(depths)), 2) if depths else 0.0


def seq_plan(d, meta, a, announce=True):
    """Decide the engine for a laid-out sequence and record it in sequence.json as `route`.

    EbSynth (`propagate`) is the default: GPT paints keys a quarter-second apart and EbSynth carries their texture to
    every frame, which costs a fraction of the image calls and shimmers less. It is skipped, and GPT renders every
    frame (`build`), when the binary is missing, when the loop is too short or too slow for keys to save anything, or
    when the layout changes between keys in ways no key shows (EbSynth can only carry paint that a key has).
    `--engine` is remembered in sequence.json; the plan is cached there too, keyed by the guides and the EbSynth
    install, and only recomputed when they change (a 72-guide plan at 1024 px takes 40 s)."""
    n, fps = meta["frames"], meta["fps"]
    if getattr(a, "engine", None):
        meta["engine"] = a.engine
        _seq_meta(d, engine=a.engine)
    engine = meta.get("engine") or "auto"
    eb = _find_ebsynth(a)
    key = _sha1(json.dumps([meta.get("guide_sha1"), n, fps, engine, eb, meta.get("base"), meta.get("plume")],
                           sort_keys=True, default=str).encode())[:12]
    cached = meta.get("route")
    if cached and cached.get("key") == key:
        if announce:
            _seq_plan_print(cached, n)
        return cached
    reasons, warnings = [], []
    every = 1
    if engine == "gpt":
        reasons.append("--engine gpt: GPT renders every frame")
    elif engine == "ebsynth" and not eb:
        _ebsynth_bin(a)
    elif not eb:
        reasons.append("EbSynth is not installed, so GPT renders every frame; to save calls, build it once:\n"
                       + EBSYNTH_BUILD)
    else:
        target = int(round(fps * SEQ_KEY_SECONDS))
        options = [k for k in range(target, 1, -1) if n % k == 0]
        if target < 2:
            reasons.append(f"at {fps:g} fps a quarter-second key is every frame, so there is nothing to carry")
        elif not options:
            reasons.append(f"no key interval from 2 to {target} frames divides the {n} frames, so keys cannot repeat "
                           "with the loop")
        else:
            static = None if meta.get("plume") else _seq_static_mask(d, meta)
            cache = {}
            for k in options:
                reach = _seq_plume_reach(meta, k) if meta.get("plume") else _seq_guide_reach(d, meta, k, static, cache)
                if reach >= SEQ_MIN_REACH:
                    every = k
                    if k != target:
                        reasons.append(f"keys every {k} frames: at {target} something between the keys "
                                       "was in neither of them")
                    break
            else:
                reasons.append(f"the layout shows something between neighbouring frames that no key shows (at "
                               f"most {reach:.0%} of an in-between is carried at {options[-1]} frames apart): a splash, "
                               "a blink or a change of colour or shape, so GPT renders every frame")
            if every > 1:
                if engine == "ebsynth":
                    reasons.insert(0, "--engine ebsynth")
                reasons.insert(0, f"EbSynth carries keys every {every} frames ({every / fps:g} s) to all {n} frames")
    if engine == "ebsynth" and every == 1:
        raise SystemExit("--engine ebsynth, but " + "; ".join(reasons))
    if meta.get("plume"):
        dense = _seq_density(d, meta)
        if dense > SEQ_DENSE:
            warnings.append(f"about {dense:g} elements overlap on each plume pixel: GPT reads a dense stream as one "
                            "generic plume and paints it the same in every frame or key, so the texture will not "
                            "travel with either engine. Space the elements (a longer --chuff, a smaller --size)")
    keys = len(range(0, n, every))
    route = {"engine": "ebsynth" if every > 1 else "gpt", "every": every, "image_calls": keys,
             "ebsynth_runs": (2 * n - keys) if every > 1 else 0, "reasons": reasons, "warnings": warnings, "key": key}
    meta["route"] = route
    _seq_meta(d, route=route)
    _seq_plan_print(route, n)
    return route


def _seq_plan_print(route, n):
    every = route["every"]
    print(f"plan: {'propagate (EbSynth)' if every > 1 else 'build (GPT renders every frame)'}, {route['image_calls']} "
          f"image calls for {n} frames\n" + "".join(f"  - {r}\n" for r in route["reasons"])
          + "".join(f"  ! {w}\n" for w in route["warnings"]), end="")


def _seq_route(d, meta, a):
    """The plan this run follows: the recorded one while the guides, the engine choice and the EbSynth install are
    unchanged (silently), else a new one (printed)."""
    return seq_plan(d, meta, a, announce=False)


def cmd_sequence_plan(a):
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    seq_plan(d, meta, a)


def _seq_layout_part(d, meta, i, static):
    """The moving part of frame i's guide on the base canvas: the plume alone, or the guide with the locked art cut."""
    canvas = meta["canvas"]
    if meta.get("plume"):
        return plume_guide(i / meta["fps"], meta["plume"], None, _seq_sprites(d, meta), canvas)
    g = np.array(_rgba(d / "guides" / f"{i % meta['frames']:03d}.png").resize((canvas, canvas), Image.LANCZOS))
    if static is not None:
        g[static] = 0
    return Image.fromarray(g)


def _plume_source_coords(t_key, t, p, sprites, canvas):
    """For each pixel at time t, where that point of its element was at t_key (identity outside the plume).

    Positions are layered like the elements are drawn, oldest first, so the front-most element owns each pixel. Averaging
    over overlapping elements instead points into none of them: in a dense exhaust plume (about 9 puffs overlapping) it
    made the synthesised texture creep at half the layout's speed with backward jitter."""
    src, dst = plume_layout(t_key, p), plume_layout(t, p)
    gy, gx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    mx, my = gx.copy(), gy.copy()
    for k, (x1, y1, s1, al, v) in sorted(dst.items()):
        if k not in src:
            continue
        x0, y0, s0 = src[k][0], src[k][1], src[k][2]
        sp = sprites[v]
        w_ = p["size"] * s1
        h_ = w_ * sp.height / sp.width
        foot = Image.new("L", (canvas, canvas))
        foot.paste(sp.getchannel("A").resize((max(1, round(w_)), max(1, round(h_)))), (round(x1 - w_ / 2), round(y1 - h_ / 2)))
        a = np.clip(np.asarray(foot, dtype=np.float32) / 255 * 2.5, 0, 1)   # opaque inside, soft only at the rim
        mx = mx * (1 - a) + (x0 + (gx - x1) * s0 / s1) * a
        my = my * (1 - a) + (y0 + (gy - y1) * s0 / s1) * a
    return mx, my


def _position_image(mx, my, box):
    x = np.clip((mx - box[0]) / max(1, box[2] - box[0] - 1), 0, 1)
    y = np.clip((my - box[1]) / max(1, box[3] - box[1] - 1), 0, 1)
    b = (np.floor(x * 15) + 16 * np.floor(y * 15)) / 255
    return Image.fromarray((np.dstack([x, y, b]) * 255).astype(np.uint8)[box[1]:box[3], box[0]:box[2]])


def _premul_rgb(img):
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    return Image.fromarray((arr[..., :3] * arr[..., 3:4] / 255).astype(np.uint8))


def cmd_sequence_propagate(a):
    """Every frame is synthesised from the two nearest GPT keys by EbSynth: patches of the real renders, placed where
    the layout says each part has moved (appearance guide, plus a positional guide for plumes). Keys go through the
    same synthesis, so they never pop against their in-betweens; frames are cross-faded between the two keys.
    Resumable: each synthesis is named by a hash of its inputs and reused unless --force."""
    import shlex
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    if a.every is None:
        route = _seq_route(d, meta, a)
        if route["every"] < 2:
            raise SystemExit("the plan renders every frame with GPT (" + "; ".join(route["reasons"]) + ") — run "
                             "`sequence ... build`, or pass --every N / --engine ebsynth to propagate anyway")
        a.every = route["every"]
    n, every = meta["frames"], a.every
    if n % every:
        raise SystemExit(f"--every {every} must divide the {n} frames so the keys repeat with the loop")
    eb = _ebsynth_bin(a)
    timeout = a.ebsynth_timeout or 600
    keys = list(range(0, n, every))
    _seq_stale(d, meta, keys)
    missing = [k for k in keys if not (d / "raw" / f"{k:03d}.png").exists()]
    if missing:
        raise SystemExit(f"keys missing from raw/: {missing} — run `sequence ... render --every {every}`")
    ref = _build_opts(d, meta, a)
    static = _seq_static_mask(d, meta)
    work = d / "propagate"
    work.mkdir(exist_ok=True)
    if a.force:
        for old in work.glob("syn_*.png"):
            old.unlink()
    parts = [_seq_layout_part(d, meta, i, static) for i in range(n)]
    union = np.zeros(parts[0].size[::-1], bool)
    for pt in parts:
        union |= np.asarray(pt)[..., 3] > 3
    if not union.any():
        sys.exit("sequence propagate: the layout has no moving content (every guide is empty after cutting --base): "
                 "run `sequence ... layout` again")
    ys, xs = np.nonzero(union)
    pad = 24
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(union.shape[1], xs.max() + pad), min(union.shape[0], ys.max() + pad))
    raw_sha = {}
    for k in keys:
        raw_sha[k] = _sha1((d / "raw" / f"{k:03d}.png").read_bytes())
        key = _seq_moving(d, meta, a, k, d / "raw" / f"{k:03d}.png", static)
        if ref is not None:
            key = _match_colors([key], ref)[0]
        key.crop(box).save(work / f"key_{k:03d}.png")
    for i in range(n):
        _premul_rgb(parts[i]).crop(box).save(work / f"app_{i:03d}.png")
    plume = meta.get("plume")
    sprites = _seq_sprites(d, meta) if plume else None
    canvas = meta["canvas"]
    if plume:
        gy, gx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
        ident = _position_image(gx, gy, box)
        for k in keys:
            ident.save(work / f"pos_{k:03d}.png")
    guide_sha = meta.get("guide_sha1") or [""] * n
    ref_sha = _sha1(_png_bytes(ref)) if ref is not None else ""

    def stamp(kk, i):
        return _sha1(f"{raw_sha[kk]}|{guide_sha[kk]}|{guide_sha[i]}|{a.guide_weight}|{a.position_weight}|{a.below}|"
                     f"{ref_sha}|{box}".encode())[:10]

    def synth(job):
        k, i, t_key = job                     # key frame, target frame, key's absolute time (k == n at the wrap)
        kk = k % n
        out = work / f"syn_{i:03d}_from_{k:03d}_{stamp(kk, i)}.png"
        if out.exists():
            return out, False
        cmd = [eb, "-style", str(work / f"key_{kk:03d}.png"),
               "-guide", str(work / f"app_{kk:03d}.png"), str(work / f"app_{i:03d}.png"), "-weight", str(a.guide_weight)]
        if plume:
            pos = work / f"pos_{i:03d}_from_{k:03d}.png"
            _position_image(*_plume_source_coords(t_key, i / meta["fps"], plume, sprites, canvas), box).save(pos)
            cmd += ["-guide", str(work / f"pos_{kk:03d}.png"), str(pos), "-weight", str(a.position_weight)]
        cmd += ["-output", str(out)]
        where = f"frame {i} from key {kk}"
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"EbSynth timed out after {timeout} s on {where} (raise --ebsynth-timeout, or fewer --jobs)"
                             f"\n  command: {shlex.join(cmd)}")
        except OSError as e:
            raise SystemExit(f"EbSynth could not be run ({e}): pass --ebsynth PATH\n  command: {shlex.join(cmd)}")
        if r.returncode or not out.exists():
            log = (r.stdout + r.stderr).strip()[-300:] or "no output"
            raise SystemExit(f"EbSynth failed on {where} (exit {r.returncode}"
                             f"{', wrote no output file' if not r.returncode else ''}): {log}\n  command: {shlex.join(cmd)}")
        try:
            _rgba(out)
        except (OSError, ValueError) as e:
            out.unlink(missing_ok=True)
            raise SystemExit(f"EbSynth wrote an unreadable file for {where} ({e})\n  command: {shlex.join(cmd)}")
        return out, True
    jobs = []
    for i in range(n):
        ka = (i // every) * every
        jobs.append((ka, i, ka / meta["fps"]))
        if i != ka:
            jobs.append((ka + every, i, (ka + every) / meta["fps"]))
    print(f"propagating {len(keys)} keys to {n} frames: {len(jobs)} EbSynth runs, {a.jobs} at a time"
          + (" (finished syntheses from the last run are reused)" if list(work.glob("syn_*.png")) else ""))
    ex = cf.ThreadPoolExecutor(max(1, a.jobs))
    futs = [ex.submit(synth, j) for j in jobs]
    syn, runs = {}, 0
    try:
        for done, f in enumerate(cf.as_completed(futs), 1):
            out, ran = f.result()
            runs += ran
            if ran and runs % 10 == 0:
                print(f"  {done}/{len(jobs)} syntheses ({runs} run now)", flush=True)
    except KeyboardInterrupt:                   # stop queuing work; the finished syntheses stay for a resume
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:                       # a failed run: let the in-flight ones end (bounded by the timeout)
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown()
    print(f"  {runs} EbSynth runs, {len(jobs) - runs} reused")
    frames = []
    for i in range(n):
        ka = (i // every) * every
        pm = None
        for k, w in ((ka, 1.0),) if i == ka else ((ka, None), (ka + every, None)):
            arr = np.asarray(_rgba(work / f"syn_{i:03d}_from_{k:03d}_{stamp(k % n, i)}.png"), dtype=np.float32).copy()
            arr[..., :3] *= arr[..., 3:4] / 255
            if w is None:
                f = (i - ka) / every
                g = f * f * (3 - 2 * f)       # smoothstep: each key's texture eases in and out
                w = 1 - g if k == ka else g
            pm = arr * w if pm is None else pm + arr * w
        a_ = pm[..., 3:4]
        rgb = np.where(a_ > 0.5, pm[..., :3] * 255 / np.maximum(a_, 1e-3), 0)
        crop = np.dstack([rgb, a_])
        # keep content only where the layout has it (soft margin); patch synthesis fills its whole box otherwise
        m = parts[i].getchannel("A").point(lambda v: 255 if v > 10 else 0).filter(ImageFilter.MaxFilter(21))
        m = np.asarray(m.filter(ImageFilter.GaussianBlur(6)).crop(box), dtype=np.float32) / 255
        crop[..., 3] *= m
        full = Image.new("RGBA", (canvas, canvas))
        full.paste(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)), (box[0], box[1]))
        frames.append(full)
    meta["propagated"] = {"every": every, "keys": keys}
    _seq_meta(d, propagated=meta["propagated"])
    _seq_finish(d, meta, a, frames)
    if not a.keep_work:
        for old in list(work.glob("syn_*.png")) + list(work.glob("pos_*_from_*.png")):
            old.unlink()
    else:
        print(f"  work files kept in {work}")


def cmd_sequence_reroll(a):
    """Render candidates for the worst frames and keep whichever fits its neighbours best (the current one included)."""
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    n = meta["frames"]
    if a.frames:
        try:
            targets = sorted({int(x) for x in a.frames.split(",") if x.strip()})
        except ValueError:
            sys.exit(f"sequence reroll: --frames takes a comma list of frame numbers (0-{n - 1}), got {a.frames!r}")
        bad = [i for i in targets if not 0 <= i < n]
        if bad or not targets:
            sys.exit(f"sequence reroll: frames {bad} are outside 0-{n - 1} ({n} frames)")
    else:
        targets = (meta.get("qa") or {}).get("outliers", [])
    if not targets:
        sys.exit("no outliers recorded: run `sequence ... build` first, or pass --frames 0,6,30")
    prop = meta.get("propagated")
    if prop:
        every, keys = prop["every"], prop["keys"]
        for i in targets:
            if i not in keys:
                ka = (i // every) * every
                sys.exit(f"sequence reroll: frame {i} was synthesised from keys {ka} and {(ka + every) % n}: to change it "
                         f"delete raw/{ka:03d}.png or raw/{(ka + every) % n:03d}.png, run `sequence ... render` (it renders "
                         "only the missing keys), then `build`")
    _seq_stale(d, meta, [i for i in targets for i in ((i - 1) % n, i, (i + 1) % n)])
    for i in targets:
        for j in ((i - 1) % n, i, (i + 1) % n):
            if not (d / "raw" / f"{j:03d}.png").exists():
                sys.exit(f"sequence reroll: raw/{j:03d}.png is missing (frame {i} is judged against its neighbours): run "
                         "`sequence ... render` first")
    static = _seq_static_mask(d, meta)
    jobs = [(i, c) for i in targets for c in range(a.candidates)]
    printed = set()

    def job(ic):
        i, c = ic
        out = d / "raw" / f"{i:03d}_cand{c}.png"
        for attempt in range(3):
            try:
                return _seq_render_one(d, meta, i, out, a.model)
            except (ApiError, RuntimeError) as e:
                again = _retryable(e) and attempt < 2
                _fail_note(f"frame {i} candidate {c}" + (" (retrying)" if again else ""), e, printed, _seq_prompt(meta, i))
                if not again:
                    return None
                time.sleep(4 * (attempt + 1))
        return None
    rendered = _pool(a.jobs, job, jobs)
    if not rendered:
        sys.exit("sequence reroll: no candidate was rendered (the errors are above); the frames are unchanged")
    replaced = 0
    for i in targets:
        prev = _seq_extract(d, meta, d / "raw" / f"{(i - 1) % n:03d}.png", (i - 1) % n, static)
        nxt = _seq_extract(d, meta, d / "raw" / f"{(i + 1) % n:03d}.png", (i + 1) % n, static)
        qa = meta.get("qa") or {}
        speed, heading = qa.get("median_step", 0.0), np.array(qa.get("heading", [1.0, 0.0]))
        options = [d / "raw" / f"{i:03d}.png"] + sorted((d / "raw").glob(f"{i:03d}_cand*.png"))
        scored = []
        for path in options:
            f = _seq_extract(d, meta, path, i, static)
            (v1, r1), (v2, r2) = _seq_step(prev, f), _seq_step(f, nxt)
            drift = abs(float(np.dot(v1, heading)) - speed) + abs(float(np.dot(v2, heading)) - speed)
            scored.append((r1 + r2 + 4 * drift, path))
        scored.sort(key=lambda s: s[0])
        best = scored[0][1]
        print(f"  frame {i}: " + ", ".join(f"{p.stem.split('_')[-1] if '_' in p.stem else 'current'} {s:.1f}" for s, p in scored))
        if best != options[0]:
            shutil.copyfile(options[0], options[0].with_name(f"{i:03d}.prev.png"))   # the replaced render is kept
            shutil.copyfile(best, options[0])
            if best.with_suffix(".json").exists():
                shutil.copyfile(best.with_suffix(".json"), options[0].with_suffix(".json"))
            replaced += 1
    print(f"  {replaced} of {len(targets)} frames replaced (previous renders kept as raw/NNN.prev.png); now run "
          "`sequence ... build` again")


def cmd_sequence(a):
    {"layout": cmd_sequence_layout, "plan": cmd_sequence_plan, "render": cmd_sequence_render, "build": cmd_sequence_build,
     "propagate": cmd_sequence_propagate, "reroll": cmd_sequence_reroll}[a.stage](a)


# ---- vector keyframes + Lottie + AnimatedVectorDrawable

_TOK = re.compile(r"[MLCZmlczHhVvSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _arc(p0, rx, ry, phi, fa, fs, p1, n=4):
    """SVG endpoint arc -> n cubic segments (fixed n keeps morph structures aligned)."""
    (x1, y1), (x2, y2) = p0, p1
    if (x1, y1) == (x2, y2):
        return []
    rx, ry = abs(rx), abs(ry)
    if not rx or not ry:
        return [("L", p1)]
    cp, sp = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    xp, yp = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = xp * xp / (rx * rx) + yp * yp / (ry * ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp
    co = math.sqrt(max(0.0, num / (rx * rx * yp * yp + ry * ry * xp * xp))) * (-1 if fa == fs else 1)
    cxp, cyp = co * rx * yp / ry, -co * ry * xp / rx
    cx, cy = cp * cxp - sp * cyp + (x1 + x2) / 2, sp * cxp + cp * cyp + (y1 + y2) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    t1 = ang(1, 0, (xp - cxp) / rx, (yp - cyp) / ry)
    dt = ang((xp - cxp) / rx, (yp - cyp) / ry, (-xp - cxp) / rx, (-yp - cyp) / ry)
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    step = dt / n
    k = 4 / 3 * math.tan(step / 4)
    pt = lambda u, v: (cx + rx * u * cp - ry * v * sp, cy + rx * u * sp + ry * v * cp)
    segs = []
    for j in range(n):
        a1, a2 = t1 + j * step, t1 + (j + 1) * step
        c1, s1, c2, s2 = math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2)
        segs.append(("C", pt(c1 - k * s1, s1 + k * c1), pt(c2 + k * s2, s2 - k * c2), pt(c2, s2)))
    segs[-1] = segs[-1][:3] + (p1,)
    return segs


def _subpaths(d):
    """Any SVG path data -> subpaths of absolute L/C segments: [{pts: [start], segs, closed}]."""
    toks = _TOK.findall(d or "")
    subs, cur, i = [], None, 0
    x = y = sx = sy = 0.0
    cmd, prev, ctrl = None, None, None

    def take(n):
        nonlocal i
        if i + n > len(toks) or any(t.isalpha() for t in toks[i:i + n]):
            raise ValueError(f"truncated path data near token {i}")
        v = [float(t) for t in toks[i:i + n]]
        i += n
        return v

    def flag():
        nonlocal i
        t = toks[i]
        if len(t) > 1 and t[0] in "01" and not t.startswith(("0.", "1.")):
            toks[i] = t[1:]
            return int(t[0])
        i += 1
        return int(float(t))

    def add(s):
        nonlocal cur
        if cur is None or cur["closed"]:
            cur = {"pts": [(sx, sy)], "segs": [], "closed": False}
            subs.append(cur)
        cur["segs"].append(s)

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if cur:
                    cur["closed"] = True
                x, y, prev = sx, sy, cmd
                continue
        elif cmd is None or cmd in "Zz":
            raise ValueError("path data must start with a command")
        rel = cmd.islower()
        ox, oy = (x, y) if rel else (0.0, 0.0)
        c = cmd.upper()
        if c == "M":
            x, y = take(2)
            x, y = x + ox, y + oy
            sx, sy = x, y
            cur = {"pts": [(x, y)], "segs": [], "closed": False}
            subs.append(cur)
            cmd = "l" if rel else "L"
        elif c in "LHV":
            if c == "L":
                nx, ny = take(2)
                nx, ny = nx + ox, ny + oy
            elif c == "H":
                nx, ny = take(1)[0] + ox, y
            else:
                nx, ny = x, take(1)[0] + oy
            add(("L", (nx, ny)))
            x, y = nx, ny
        elif c in "CS":
            if c == "C":
                v = take(6)
                c1 = (v[0] + ox, v[1] + oy)
                v = v[2:]
            else:
                v = take(4)
                c1 = (2 * x - ctrl[0], 2 * y - ctrl[1]) if prev and prev.upper() in "CS" else (x, y)
            c2, p = (v[0] + ox, v[1] + oy), (v[2] + ox, v[3] + oy)
            add(("C", c1, c2, p))
            ctrl, (x, y) = c2, p
        elif c in "QT":
            if c == "Q":
                v = take(4)
                q, p = (v[0] + ox, v[1] + oy), (v[2] + ox, v[3] + oy)
            else:
                v = take(2)
                q = (2 * x - ctrl[0], 2 * y - ctrl[1]) if prev and prev.upper() in "QT" else (x, y)
                p = (v[0] + ox, v[1] + oy)
            add(("C", (x + 2 / 3 * (q[0] - x), y + 2 / 3 * (q[1] - y)),
                 (p[0] + 2 / 3 * (q[0] - p[0]), p[1] + 2 / 3 * (q[1] - p[1])), p))
            ctrl, (x, y) = q, p
        elif c == "A":
            rx, ry, phi = take(3)
            fa, fs = flag(), flag()
            p = tuple(v + o for v, o in zip(take(2), (ox, oy)))
            for s in _arc((x, y), rx, ry, phi, fa, fs, p):
                add(s)
            x, y = p
        prev = cmd
    return subs


def _fmtn(v, nd=None):
    """Compact number: 2 decimals, 3 below 1 (a viewBox of 0 0 1 1 must not quantise to 1 %)."""
    if nd is None:
        nd = 3 if abs(v) < 1 else 2
    return f"{round(v, nd) + 0:g}"


def _d(subs):
    out = []
    for sp in subs:
        out.append(f"M{_fmtn(sp['pts'][0][0])},{_fmtn(sp['pts'][0][1])}")
        out += [s[0] + " ".join(f"{_fmtn(p[0])},{_fmtn(p[1])}" for p in s[1:]) for s in sp["segs"]]
        if sp["closed"]:
            out.append("Z")
    return " ".join(out)


def _points(subs):
    return [p for sp in subs for p in sp["pts"][:1] + [q for s in sp["segs"] for q in s[1:]]]


def _as_cubic(p0, s):
    if s[0] == "C":
        return s
    (x0, y0), (x1, y1) = p0, s[1]
    return ("C", (x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3), (x0 + 2 * (x1 - x0) / 3, y0 + 2 * (y1 - y0) / 3), s[1])


def _align_keys(keys):
    """Make parsed keyframes morph-compatible: same subpath/segment counts required, L promoted to C where they differ."""
    shape = lambda subs: [(len(sp["segs"]), sp["closed"]) for sp in subs]
    if any(shape(k) != shape(keys[0]) for k in keys):
        raise ValueError("keyframes differ in command sequence (subpath or segment counts)")
    for si in range(len(keys[0])):
        for j in range(len(keys[0][si]["segs"])):
            if len({k[si]["segs"][j][0] for k in keys}) > 1:
                for k in keys:
                    sp = k[si]
                    p0 = sp["segs"][j - 1][-1] if j else sp["pts"][0]
                    sp["segs"][j] = _as_cubic(p0, sp["segs"][j])
    return keys


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _ellipse(cx, cy, rx, ry):
    k = 0.5522847498
    return {"pts": [(cx + rx, cy)], "closed": True, "segs": [
        ("C", (cx + rx, cy + k * ry), (cx + k * rx, cy + ry), (cx, cy + ry)),
        ("C", (cx - k * rx, cy + ry), (cx - rx, cy + k * ry), (cx - rx, cy)),
        ("C", (cx - rx, cy - k * ry), (cx - k * rx, cy - ry), (cx, cy - ry)),
        ("C", (cx + k * rx, cy - ry), (cx + rx, cy - k * ry), (cx + rx, cy))]}


def _ease(name, p):
    x1, y1, x2, y2 = EASES.get(name, EASES["ease_in_out"])
    bz = lambda u, a, b: 3 * (1 - u) ** 2 * u * a + 3 * (1 - u) * u * u * b + u ** 3
    lo, hi = 0.0, 1.0
    for _ in range(40):
        u = (lo + hi) / 2
        lo, hi = (u, hi) if bz(u, x1, x2) < p else (lo, u)
    return bz((lo + hi) / 2, y1, y2)


CHANNELS = ("dx", "dy", "sx", "sy", "rotate", "opacity")
REST = {"dx": 0.0, "dy": 0.0, "sx": 1.0, "sy": 1.0, "rotate": 0.0, "opacity": 1.0}


def _keys(motion):
    """Motion keys sorted into [0, 1] and closed into a loop (back to the first key unless a t=1 key exists)."""
    ks = sorted(({**REST, "ease": "ease_in_out", **k} for k in (motion or {}).get("keys") or []), key=lambda k: k["t"])
    if not ks:
        return []
    for k in ks:
        k["t"] = min(1.0, max(0.0, float(k["t"])))
        k["opacity"] = min(1.0, max(0.0, float(k["opacity"])))
    if ks[0]["t"] > 0:
        ks.insert(0, {**(ks[-1] if ks[-1]["t"] == 1 else ks[0]), "t": 0.0})
    if ks[-1]["t"] < 1:
        ks.append({**ks[0], "t": 1.0})
    return ks


def _motion_at(ks, t):
    if not ks:
        return dict(REST)
    for a, b in zip(ks, ks[1:]):
        if a["t"] <= t <= b["t"]:
            f = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            m = {c: a[c] + (b[c] - a[c]) * _ease(a["ease"], f) for c in CHANNELS}
            m["opacity"] = a["opacity"] + (b["opacity"] - a["opacity"]) * _ease(_fade_keys([a])[0]["ease"], f)
            return m
    return {c: ks[-1][c] for c in CHANNELS}


def _xf(pts, m, ax, ay):
    c, s = math.cos(math.radians(m["rotate"])), math.sin(math.radians(m["rotate"]))
    out = []
    for x, y in pts:
        u, v = (x - ax) * m["sx"], (y - ay) * m["sy"]
        out.append((ax + m["dx"] + u * c - v * s, ay + m["dy"] + u * s + v * c))
    return out


def _morph_at(keys, t):
    """Parsed keyframes at loop time t (ease_in_out between evenly spaced keys, wrapping to the first)."""
    n = len(keys)
    f = (t % 1.0) * n
    j = int(f) % n
    p = _ease("ease_in_out", f - int(f))
    a, b = keys[j], keys[(j + 1) % n]
    mix = lambda P, Q: (P[0] + (Q[0] - P[0]) * p, P[1] + (Q[1] - P[1]) * p)
    return [{"pts": [mix(sa["pts"][0], sb["pts"][0])], "closed": sa["closed"],
             "segs": [(x[0],) + tuple(mix(P, Q) for P, Q in zip(x[1:], y[1:])) for x, y in zip(sa["segs"], sb["segs"])]}
            for sa, sb in zip(a, b)]


def _layers(vec):
    """Normalised layers bottom to top: {name, kind, paths: [(subs, paint)] or keys, motion keys, anchor}."""
    out = {}
    for sl in vec.get("static_layers", []):
        out[sl["name"]] = {"name": sl["name"], "kind": "static", "motion": sl.get("motion"),
                           "paths": [(_subpaths(p["d"]), p) for p in sl["paths"]]}
    for al in vec.get("animated_layers", []):
        out[al["name"]] = {"name": al["name"], "kind": "morph", "motion": al.get("motion"), "paint": al,
                           "keys": _align_keys([_subpaths(k) for k in al["keyframes"]])}
    for g in vec.get("glows", []):
        out[g["name"]] = {"name": g["name"], "kind": "glow", "motion": g.get("motion"), "glow": g,
                          "paths": [([_ellipse(g["cx"], g["cy"], g["rx"], g["ry"])], {"fill": "glow"})]}
    order = [n for n in vec.get("layer_order", []) if n in out]
    rest = [n for n in out if n not in order]
    order = [n for n in rest if out[n]["kind"] == "glow"] + order + [n for n in rest if out[n]["kind"] != "glow"]
    res = []
    for n in order:
        L = out[n]
        L["ks"] = _keys(L["motion"])
        L["anchor"] = (float((L["motion"] or {}).get("anchor_x", 0)), float((L["motion"] or {}).get("anchor_y", 0)))
        res.append(L)
    return res


def _curve_pts(subs, n=6):
    out = []
    for sp in subs:
        p0 = sp["pts"][0]
        out.append(p0)
        for s in sp["segs"]:
            if s[0] == "C":
                c1, c2, p1 = s[1:]
                out += [tuple((1 - u) ** 3 * p0[k] + 3 * (1 - u) ** 2 * u * c1[k] + 3 * (1 - u) * u * u * c2[k] + u ** 3 * p1[k]
                              for k in (0, 1)) for u in (i / n for i in range(1, n + 1))]
            else:
                out.append(s[1])
            p0 = s[-1]
    return out


def _layer_pts(L, t, curve=False, moving=True):
    """The layer's points at loop time t (control points, or points on the curves with curve=True)."""
    pick = _curve_pts if curve else _points
    if L["kind"] == "morph":
        pts = pick(_morph_at(L["keys"], t) if moving else L["keys"][0])
    elif L["kind"] == "glow":
        g = L["glow"]
        pts = pick([_ellipse(g["cx"], g["cy"], g["rx"], g["ry"])])
    else:
        pts = [p for subs, _ in L["paths"] for p in pick(subs)]
    return _xf(pts, _motion_at(L["ks"], t), *L["anchor"]) if L["ks"] and moving else pts


def _times(L, n=48):
    ts = {i / n for i in range(n)} | {k["t"] for k in L["ks"] if k["t"] < 1}
    if L["kind"] == "morph":
        ts |= {j / len(L["keys"]) for j in range(len(L["keys"]))}
    return sorted(ts)


def motion_report(vec):
    """Per layer: 'move' = max mean point displacement from the t=0 pose / the layer's size; 'fade' = opacity range;
    'score' = max(move, fade / 4) capped at 1. 'motion' = area-weighted mean score of the non-glow layers (a tiny spark
    travelling five times its size must not carry a static flame; a pure fade counts as 25 %)."""
    rep = {}
    for L in _layers(vec):
        if L["kind"] != "morph" and not L["ks"]:
            continue
        p0 = _layer_pts(L, 0.0)
        x0, y0, x1, y1 = _bbox(p0)
        size = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
        move = max(sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(_layer_pts(L, t), p0)) / len(p0)
                   for t in _times(L)) / size
        ops = [_motion_at(L["ks"], t)["opacity"] for t in _times(L)] if L["ks"] else [1.0]
        fade = max(ops) - min(ops)
        rep[L["name"]] = {"kind": L["kind"], "move": round(move, 4), "fade": round(fade, 3), "size": round(size, 3),
                          "score": round(min(1.0, max(move, fade / 4)), 4)}
    moving = [(r["score"], r["size"] ** 2) for r in rep.values() if r["kind"] != "glow"]
    wsum = sum(w for _, w in moving)
    return {"layers": rep, "motion": round(sum(sc * w for sc, w in moving) / wsum, 4) if wsum else None}


def _normalize_vec(vec, notes=None):
    """Drop a closing keyframe that repeats the first (it stalls the loop) and store canonical path data.
    notes (a list) collects what changed, for the caller to print."""
    for al in vec.get("animated_layers", []):
        keys = _align_keys([_subpaths(k) for k in al["keyframes"]])
        if len(keys) > 2:
            a, b = _points(keys[0]), _points(keys[-1])
            x0, y0, x1, y1 = _bbox(a)
            if max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b)) < 0.005 * math.hypot(x1 - x0, y1 - y0):
                keys = keys[:-1]
                if notes is not None:
                    notes.append(f"dropped the closing keyframe of {al['name']} (it repeated the first one)")
        al["keyframes"] = [_d(k) for k in keys]
    return vec


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX6_RE = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
_CAPS = {"butt": "butt", "round": "round", "square": "square"}
_JOINS = {"miter": "miter", "round": "round", "bevel": "bevel", "miter-clip": "miter", "arcs": "miter"}


def _cap(v, dflt="round"):
    return _CAPS.get(str(v or dflt).strip().lower(), dflt)


def _join(v, dflt="round"):
    return _JOINS.get(str(v or dflt).strip().lower(), dflt)


def _finite(v):
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    except TypeError:
        return False


def _validate_vec(vec):
    """Problems with a vec (the layers.json shape) as one-line strings; [] when it builds. Pure: never modifies vec.
    Colours must already be '#RRGGBB' (see _canon_vec); the schema enforces this for model answers."""
    if not isinstance(vec, dict):
        return ["layers.json must be a JSON object"]
    errs = []
    try:
        vb = _vec_vb(vec)
        if len(vb) != 4 or not all(math.isfinite(v) for v in vb) or vb[2] <= 0 or vb[3] <= 0:
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError):
        errs.append(f"viewBox {vec.get('viewBox')!r} must be 4 numbers \"x y w h\" with w and h > 0")
    try:
        defs_root = _defs_root(vec)
    except ET.ParseError as e:
        errs.append(f"defs is not valid SVG XML ({e})")
        defs_root = None
    names = []

    def num(where, p, k, lo=None, hi=None):
        if k not in p:
            return
        v = p[k]
        if not _finite(v):
            errs.append(f"{where}: {k} {v!r} must be a number")
        elif (lo is not None and v < lo) or (hi is not None and v > hi):
            errs.append(f"{where}: {k} {v} must be between {lo} and {hi}" if hi is not None else f"{where}: {k} {v} must be >= {lo}")

    def paint(where, p):
        for k in ("fill", "stroke"):
            v = str(p.get(k) or "none")
            ref = re.fullmatch(r"url\(#([^)]+)\)", v)
            if ref:
                if defs_root is not None:
                    try:
                        _gradient(defs_root, ref.group(1))
                    except ValueError as e:
                        errs.append(f"{where}: {k} {e}")
            elif v != "none" and not _HEX6_RE.fullmatch(v):
                errs.append(f"{where}: {k} {v!r} is not a hex colour (#RRGGBB), url(#id) or none")
        for k in ("opacity", "fill_opacity", "stroke_opacity"):
            num(where, p, k, 0, 1)
        num(where, p, "stroke_width", 0)
        if p.get("fill_rule") not in (None, "nonzero", "evenodd"):
            errs.append(f"{where}: fill_rule {p.get('fill_rule')!r} must be nonzero or evenodd")
        for k, table in (("cap", _CAPS), ("join", _JOINS)):
            if p.get(k) is not None and str(p[k]).lower() not in table:
                errs.append(f"{where}: {k} {p[k]!r} must be one of {', '.join(sorted(set(table.values())))}")

    def pathdata(where, d):
        if not isinstance(d, str) or not d.strip():
            errs.append(f"{where}: empty path data")
            return None
        junk = _TOK.sub("", d).strip(" ,\t\r\n")
        if junk:
            errs.append(f"{where}: path data has unexpected characters {junk[:20]!r}")
            return None
        try:
            subs = _subpaths(d)
        except (ValueError, IndexError, TypeError) as e:
            errs.append(f"{where}: {e}")
            return None
        if not subs or not any(sp["segs"] for sp in subs):
            errs.append(f"{where}: path data draws nothing")
            return None
        return subs

    def motion(where, mo):
        if mo is None:
            return
        if not isinstance(mo, dict):
            errs.append(f"{where}: motion must be an object with anchor_x, anchor_y, keys")
            return
        for k in ("anchor_x", "anchor_y"):
            if k in mo and not _finite(mo[k]):
                errs.append(f"{where}: motion {k} {mo[k]!r} must be a number")
        keys = mo.get("keys") or []
        if not isinstance(keys, list):
            errs.append(f"{where}: motion keys must be a list")
            return
        for j, k in enumerate(keys, 1):
            if not isinstance(k, dict) or "t" not in k:
                errs.append(f"{where}: motion key {j} has no t")
                continue
            num(f"{where} key {j}", k, "t", 0, 1)
            num(f"{where} key {j}", k, "opacity", 0, 1)
            for c in ("dx", "dy", "sx", "sy", "rotate"):
                num(f"{where} key {j}", k, c)
            if "ease" in k and k["ease"] not in EASES:
                errs.append(f"{where} key {j}: ease {k['ease']!r} must be one of {', '.join(EASES)}")

    def name_of(where, L):
        n = L.get("name") if isinstance(L, dict) else None
        if not isinstance(n, str) or not n.strip():
            errs.append(f"{where}: missing name")
            return f"?{len(names)}"
        names.append(n)
        return n

    drawn = 0
    for i, sl in enumerate(vec.get("static_layers") or [], 1):
        n = name_of(f"static layer {i}", sl)
        where = f"static {n}"
        paths = sl.get("paths") if isinstance(sl, dict) else None
        if not isinstance(paths, list) or not paths:
            errs.append(f"{where}: no paths")
        else:
            drawn += 1
            for p in paths:
                if not isinstance(p, dict):
                    errs.append(f"{where}: each path must be an object with d, fill, stroke")
                    continue
                pathdata(where, p.get("d"))
                paint(where, p)
        motion(where, sl.get("motion") if isinstance(sl, dict) else None)
    for i, al in enumerate(vec.get("animated_layers") or [], 1):
        n = name_of(f"animated layer {i}", al)
        where = f"animated {n}"
        kf = al.get("keyframes") if isinstance(al, dict) else None
        if not isinstance(kf, list) or len(kf) < 2:
            errs.append(f"{where}: needs at least 2 keyframes")
        else:
            drawn += 1
            keys = [pathdata(f"{where} keyframe {j}", k) for j, k in enumerate(kf, 1)]
            if all(k is not None for k in keys):
                try:
                    keys = _align_keys(keys)
                    a = _points(keys[0])
                    x0, y0, x1, y1 = _bbox(a)
                    size = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
                    if all(max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, _points(k))) < 0.005 * size for k in keys[1:]):
                        errs.append(f"{where}: all keyframes are identical (nothing morphs): vary them or make it a static layer")
                except (ValueError, IndexError, TypeError) as e:
                    errs.append(f"{where}: {e}")
        if isinstance(al, dict):
            paint(where, al)
            motion(where, al.get("motion"))
    for i, g in enumerate(vec.get("glows") or [], 1):
        n = name_of(f"glow {i}", g)
        where = f"glow {n}"
        if not isinstance(g, dict):
            continue
        for k in ("cx", "cy", "rx", "ry"):
            if not _finite(g.get(k)):
                errs.append(f"{where}: {k} {g.get(k)!r} must be a number")
        if _finite(g.get("rx")) and _finite(g.get("ry")) and (g["rx"] <= 0 or g["ry"] <= 0):
            errs.append(f"{where}: rx/ry must be > 0")
        if not _HEX6_RE.fullmatch(str(g.get("color") or "")):
            errs.append(f"{where}: color {g.get('color')!r} must be a hex colour (#RRGGBB)")
        num(where, g, "opacity", 0, 1)
        motion(where, g.get("motion"))
    if not drawn and not any(e.endswith(": no paths") or "keyframes" in e for e in errs):
        errs.append("no static or animated layers: nothing to draw")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        errs.append(f"layer names must be unique (repeated: {', '.join(dup)})")
    safe = {}
    for n in names:
        safe.setdefault(_res_name(n, "layer_"), []).append(n)
    clash = [v for v in safe.values() if len(set(v)) > 1]
    if clash:
        errs.append("layer names collide after sanitising: " + "; ".join(" and ".join(repr(x) for x in v) for v in clash))
    unknown = [str(n) for n in (vec.get("layer_order") or []) if n not in names]
    if unknown:
        errs.append(f"layer_order names unknown layer(s): {', '.join(unknown)} (known: {', '.join(names)})")
    return errs


def _paint_color(v, current=None):
    """('#RRGGBB', alpha) for any colour the kit reads: hex 3/4/6/8 digits, CSS names, rgb()/rgba(), currentColor."""
    c = _color(v, 1.0, current)
    return (f"#{c[3:]}", int(c[1:3], 16) / 255) if len(c) == 9 else (c, 1.0)


def _layer_name(raw, used):
    """snake_case layer name unique within `used` (Android resource rules: [a-z0-9_], starts with a letter)."""
    base = _res_name(str(raw or "layer"), "layer_")
    n, k = base, 2
    while n in used:
        n, k = f"{base}_{k}", k + 1
    used.add(n)
    return n


def _canon_vec(vec, current=None):
    """Canonicalise a vec in place before validation: snake_case unique layer names (layer_order follows; renames are
    printed) and every colour as '#RRGGBB' with its alpha folded into fill_opacity / stroke_opacity (glows: opacity).
    Returns the colours that could not be read, as error strings."""
    if not isinstance(vec, dict):
        return []
    used, ren = set(), {}
    for key in ("static_layers", "animated_layers", "glows"):
        for L in vec.get(key) or []:
            if not isinstance(L, dict):
                continue
            raw = str(L.get("name") or "")
            L["name"] = _layer_name(raw, used)
            ren.setdefault(raw, L["name"])
    if isinstance(vec.get("layer_order"), list):
        vec["layer_order"] = [ren.get(str(n), str(n)) for n in vec["layer_order"]]
    renamed = [f"{a!r} -> {b}" for a, b in ren.items() if a != b]
    if renamed:
        print(f"  (layer names sanitised: {', '.join(renamed[:6])})")
    errs, notes = [], []

    def fix(where, p, key, okey):
        v = str(p.get(key) or "none").strip()
        if v == "none" or v.startswith("url("):
            p[key] = v
            return
        if v.lower() == "currentcolor" and not current:
            notes.append(where)
        try:
            c, al = _paint_color(v, current)
        except ValueError:
            errs.append(f"{where}: {key} {v!r} is not a colour (use #RRGGBB, url(#id) or none)")
            return
        p[key] = c
        if al < 1:
            p[okey] = round(min(1.0, max(0.0, _num(p.get(okey), 1.0))) * al, 4)
    for sl in vec.get("static_layers") or []:
        for p in (sl.get("paths") or []) if isinstance(sl, dict) else []:
            if isinstance(p, dict):
                fix(f"static {sl['name']}", p, "fill", "fill_opacity")
                fix(f"static {sl['name']}", p, "stroke", "stroke_opacity")
    for al in vec.get("animated_layers") or []:
        if isinstance(al, dict):
            fix(f"animated {al['name']}", al, "fill", "fill_opacity")
            fix(f"animated {al['name']}", al, "stroke", "stroke_opacity")
    for g in vec.get("glows") or []:
        if isinstance(g, dict):
            fix(f"glow {g['name']}", g, "color", "opacity")
    if notes:
        print(f"  ~ currentColor with no color attribute ({', '.join(notes[:4])}): using #000000")
    return errs


def _vec_vb(vec):
    return [float(x) for x in re.split(r"[\s,]+", vec["viewBox"].strip())]


def _fit_viewbox(vec, margin=0.02):
    """viewBox grown symmetrically so every pose of the animation (plus stroke) stays inside; None if it already fits."""
    x, y, w, h = _vec_vb(vec)
    rest, pts = [(x, y), (x + w, y + h)], []
    for L in _layers(vec):
        if L["kind"] == "glow":  # the halo is visible out to about 70 % of its radius (GLOW_STOPS)
            g = L["glow"]
            L = dict(L, glow={**g, "rx": g["rx"] * 0.7, "ry": g["ry"] * 0.7})
        sw = max([float(p.get("stroke_width") or 0) for _, p in L.get("paths", [])] +
                 [float(L.get("paint", {}).get("stroke_width") or 0)]) / 2
        bx0, by0, bx1, by1 = _bbox(_layer_pts(L, 0.0, True, False))
        rest += [(bx0 - sw, by0 - sw), (bx1 + sw, by1 + sw)]
        for t in _times(L, 24) if (L["ks"] or L["kind"] == "morph") else []:
            bx0, by0, bx1, by1 = _bbox(_layer_pts(L, t, True))
            pts += [(bx0 - sw, by0 - sw), (bx1 + sw, by1 + sw)]
    if not pts:
        return None
    rx0, ry0, rx1, ry1 = _bbox(rest)
    bx0, by0, bx1, by1 = _bbox(pts)
    if bx0 >= rx0 and by0 >= ry0 and bx1 <= rx1 and by1 <= ry1:
        return None
    cx, cy = x + w / 2, y + h / 2
    hw = max(w / 2, cx - bx0, bx1 - cx)
    hh = max(h / 2, cy - by0, by1 - cy)
    hw, hh = hw * (1 + margin), hh * (1 + margin)
    return f"{_fmtn(cx - hw)} {_fmtn(cy - hh)} {_fmtn(2 * hw)} {_fmtn(2 * hh)}"


def _xform_subs(subs, m):
    """Subpaths with the affine matrix applied to every anchor and control point (cubics are affine-invariant)."""
    if tuple(m) == _IDENTITY:
        return subs
    return [{"pts": [_mapply(m, *sp["pts"][0])], "closed": sp["closed"],
             "segs": [(s[0],) + tuple(_mapply(m, *p) for p in s[1:]) for s in sp["segs"]]} for sp in subs]


def _matrix(tf):
    """SVG transform list -> (a, b, c, d, e, f). ValueError names a function with the wrong argument count."""
    m = _IDENTITY
    for fn, args in re.findall(r"(\w+)\s*\(([^)]*)\)", tf or ""):
        v = [float(n) for n in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        if fn == "matrix" and len(v) == 6:
            q = tuple(v)
        elif fn == "translate" and len(v) in (1, 2):
            q = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif fn == "scale" and len(v) in (1, 2):
            q = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif fn == "rotate" and len(v) in (1, 3):
            r = math.radians(v[0])
            cx, cy = (v[1], v[2]) if len(v) > 2 else (0, 0)
            q = _mmul(_mmul((1, 0, 0, 1, cx, cy), (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)), (1, 0, 0, 1, -cx, -cy))
        elif fn == "skewX" and len(v) == 1:
            q = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif fn == "skewY" and len(v) == 1:
            q = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        elif fn in ("matrix", "translate", "scale", "rotate", "skewX", "skewY"):
            raise ValueError(f"transform {fn}({args.strip()}): wrong number of arguments")
        else:
            raise ValueError(f"transform {fn}(...) is not supported")
        m = _mmul(m, q)
    return m


def _defs_root(vec):
    defs = vec.get("defs", "") or ""
    vb = str(vec.get("viewBox") or "").strip()
    return ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
                         f'{f" viewBox={vb!r}" if vb else ""}>{defs if "<defs" in defs else f"<defs>{defs}</defs>"}</svg>')


def _grad_chain(defs_root, gid):
    """The gradient element with id gid followed by its href chain (nearest first)."""
    chain, seen = [], set()
    while gid and gid not in seen:
        seen.add(gid)
        gel = next((e for e in defs_root.iter() if e.get("id") == gid), None)
        if gel is None:
            if chain:
                break
            raise ValueError(f"gradient #{gid} not found in defs")
        tag = gel.tag.replace(SVG_NS, "") if isinstance(gel.tag, str) else "?"
        if tag not in ("linearGradient", "radialGradient"):
            raise ValueError(f"#{gid} is a <{tag}>, not a gradient (patterns, masks and filters are not supported)")
        chain.append(gel)
        gid = (gel.get("href") or gel.get("{http://www.w3.org/1999/xlink}href") or "").strip().lstrip("#")
    return chain


def _gradient(defs_root, gid, bbox=None, vb=None):
    """(linear, start, end, [(offset, '#RRGGBB', alpha)]) in user units with any gradientTransform baked in (radius
    scaled by sqrt|det|). href chains supply missing attributes and stops; SVG defaults apply (x1=y1=y2=0% x2=100%,
    cx=cy=r=50%). Percentages resolve against the viewBox (vb, or defs_root's viewBox); the default objectBoundingBox
    units need the referencing shape's bbox (x0, y0, x1, y1), else ValueError tells the author what to set."""
    chain = _grad_chain(defs_root, gid)
    gel = chain[0]
    attr = lambda k: next((e.get(k) for e in chain if e.get(k) is not None), None)
    stops_el = next((e for e in chain if any(isinstance(c.tag, str) and c.tag.replace(SVG_NS, "") == "stop" for c in e)), None)
    stops = []
    for st in (stops_el if stops_el is not None else []):
        if not isinstance(st.tag, str) or st.tag.replace(SVG_NS, "") != "stop":
            continue
        style = _style_dict(st)
        off = str(st.get("offset", "0"))
        off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
        try:
            col, al = _paint_color(style.get("stop-color", st.get("stop-color")) or "#000000")
        except ValueError as e:
            raise ValueError(f"gradient #{gid}: {e}")
        so = str(style.get("stop-opacity", st.get("stop-opacity")) or "1")
        so = _num(so) / 100 if so.strip().endswith("%") else _num(so, 1.0)
        off = min(1.0, max(0.0, off))
        if stops and off < stops[-1][0]:
            off = stops[-1][0]
        stops.append((off, col, min(1.0, max(0.0, al * so))))
    if len(stops) < 2:
        raise ValueError(f"gradient #{gid}: needs at least 2 stops (has {len(stops)})")
    linear = gel.tag.endswith("linearGradient")
    obb = (attr("gradientUnits") or "objectBoundingBox").strip() != "userSpaceOnUse"
    if obb and bbox is None:
        raise ValueError(f'gradient #{gid} uses objectBoundingBox units (the SVG default): set gradientUnits="userSpaceOnUse" '
                         f"and coordinates in viewBox units")
    if vb is None and defs_root.get("viewBox"):
        vb = [float(x) for x in re.split(r"[\s,]+", defs_root.get("viewBox").strip())]

    def coord(k, dflt, axis):
        v = attr(k)
        sv = str(v).strip() if v is not None else dflt
        pct = sv.endswith("%")
        n = _num(sv) / 100 if pct else _num(sv)
        if obb:
            x0, y0, x1, y1 = bbox
            return (x0 + n * (x1 - x0) if axis == "x" else y0 + n * (y1 - y0) if axis == "y"
                    else n * math.sqrt(((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2))
        if pct:
            if vb is None or len(vb) != 4:
                raise ValueError(f"gradient #{gid}: {k}={sv} needs a viewBox to resolve the percentage")
            vx, vy, vw, vh = vb
            return vx + n * vw if axis == "x" else vy + n * vh if axis == "y" else n * math.sqrt((vw ** 2 + vh ** 2) / 2)
        return n
    a, b, c, d, e, f = _matrix(attr("gradientTransform"))
    ap = lambda x, y: [a * x + c * y + e, b * x + d * y + f]
    if linear:
        return True, ap(coord("x1", "0%", "x"), coord("y1", "0%", "y")), ap(coord("x2", "100%", "x"), coord("y2", "0%", "y")), stops
    s = ap(coord("cx", "50%", "x"), coord("cy", "50%", "y"))
    r = coord("r", "50%", "r") * math.sqrt(abs(a * d - b * c))
    if not r > 0:
        raise ValueError(f"gradient #{gid}: radius must be > 0")
    return False, s, [s[0] + r, s[1]], stops


GLOW_STOPS = ((0, 1.0), (0.25, 0.8), (0.5, 0.42), (0.75, 0.14), (1, 0.0))


def _glow_grad(g):
    return False, [g["cx"], g["cy"]], [g["cx"] + g["rx"], g["cy"]], [(o, g["color"], al) for o, al in GLOW_STOPS]


def _lottie_shape(sp, drop_last=None):
    v = [sp["pts"][0]]
    ins, outs = [[0, 0]], [[0, 0]]
    for s in sp["segs"]:
        p0 = v[-1]
        if s[0] == "L":
            v.append(s[1])
            ins.append([0, 0])
            outs.append([0, 0])
        else:
            _, c1, c2, p1 = s
            outs[-1] = [c1[0] - p0[0], c1[1] - p0[1]]
            v.append(p1)
            ins.append([c2[0] - p1[0], c2[1] - p1[1]])
            outs.append([0, 0])
    if drop_last is None:
        drop_last = sp["closed"] and len(v) > 1 and abs(v[-1][0] - v[0][0]) < 1e-3 and abs(v[-1][1] - v[0][1]) < 1e-3
    if drop_last:
        ins[0] = ins[-1]
        v, ins, outs = v[:-1], ins[:-1], outs[:-1]
    return {"i": ins, "o": outs, "v": [list(p) for p in v], "c": sp["closed"]}


def _hex_rgb(h):
    h = h.lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h[:6]
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


LOTTIE_CAP = {"butt": 1, "round": 2, "square": 3}
LOTTIE_JOIN = {"miter": 1, "round": 2, "bevel": 3}


def _lottie_fill(fill, opacity, defs_root, width=None, cap="round", join="round", grad=None):
    """Lottie fill (or stroke when width is given) for a '#RRGGBB' colour or url(#gradient)."""
    ref = re.match(r"url\(#([^)]+)\)", fill or "")
    stroke = width is not None
    if grad is None and not ref:
        item = {"ty": "st" if stroke else "fl", "c": {"a": 0, "k": _hex_rgb(fill) + [1]}, "o": {"a": 0, "k": round(opacity * 100, 3)}}
    else:
        linear, s, e, stops = grad or _gradient(defs_root, ref.group(1))
        colors = [x for o, c, _ in stops for x in [o] + _hex_rgb(c)]
        alphas = [x for o, _, al in stops for x in (o, al)] if any(al < 1 for *_, al in stops) else []
        item = {"ty": "gs" if stroke else "gf", "t": 1 if linear else 2, "s": {"a": 0, "k": s}, "e": {"a": 0, "k": e},
                "g": {"p": len(stops), "k": {"a": 0, "k": colors + alphas}}, "o": {"a": 0, "k": round(opacity * 100, 3)}}
    if stroke:
        item.update({"w": {"a": 0, "k": width}, "lc": LOTTIE_CAP[_cap(cap)], "lj": LOTTIE_JOIN[_join(join)], "ml": 4})
    else:
        item["r"] = 1
    return item


def _paint_alphas(p):
    """(fill alpha, stroke alpha) of a paint dict: opacity times fill_opacity / stroke_opacity, clamped to 0..1."""
    op = min(1.0, max(0.0, _num(p.get("opacity"), 1.0)))
    return (min(1.0, max(0.0, op * _num(p.get("fill_opacity"), 1.0))), min(1.0, max(0.0, op * _num(p.get("stroke_opacity"), 1.0))))


def _lottie_paints(p, defs_root):
    """SVG paints stroke over fill; Lottie draws earlier items on top, so the stroke goes first."""
    items = []
    fa, sa = _paint_alphas(p)
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        items.append(_lottie_fill(p["stroke"], sa, defs_root, float(p["stroke_width"]), p.get("cap", "round"), p.get("join", "round")))
    if (p.get("fill") or "none") != "none":
        fl = _lottie_fill(p["fill"], fa, defs_root)
        fl["r"] = 2 if p.get("fill_rule") == "evenodd" else 1
        items.append(fl)
    return items


def _tr():
    return {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]},
            "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}, "sk": {"a": 0, "k": 0}, "sa": {"a": 0, "k": 0}}


def _lottie_prop(ks, fn, op):
    vals = [fn(k) for k in ks]
    if all(v == vals[0] for v in vals):
        return {"a": 0, "k": vals[0]}
    frames = []
    for j, k in enumerate(ks):
        fr = {"t": round(k["t"] * op, 3), "s": vals[j] if isinstance(vals[j], list) else [vals[j]]}
        if j < len(ks) - 1:
            x1, y1, x2, y2 = EASES.get(k["ease"], EASES["ease_in_out"])
            fr.update({"o": {"x": [x1], "y": [y1]}, "i": {"x": [x2], "y": [y2]}})
        frames.append(fr)
    return {"a": 1, "k": frames}


def _lottie_ks(L, vb, k, op):
    ax, ay = L["anchor"] if L["ks"] else (vb[0], vb[1])
    ks = L["ks"] or [{**REST, "t": 0.0, "ease": "linear"}]
    return {"a": {"a": 0, "k": [ax, ay, 0]},
            "p": _lottie_prop(ks, lambda q: [round((ax - vb[0] + q["dx"]) * k, 3), round((ay - vb[1] + q["dy"]) * k, 3), 0], op),
            "s": _lottie_prop(ks, lambda q: [round(q["sx"] * 100 * k, 3), round(q["sy"] * 100 * k, 3), 100], op),
            "r": _lottie_prop(ks, lambda q: round(q["rotate"], 3), op),
            "o": _lottie_prop(_fade_keys(ks), lambda q: round(q["opacity"] * 100, 2), op)}


def build_lottie(vec, size=512, seconds=1.2, fps=30):
    vb = _vec_vb(vec)
    k = size / max(vb[2], vb[3])
    op = max(1, round(seconds * fps))
    rate = round(op / seconds, 4)  # a fractional frame rate keeps the loop exactly `seconds` long (the AVD uses ms)
    defs_root = _defs_root(vec)
    layers = []
    for L in _layers(vec):
        if L["kind"] == "static":
            items = []
            for subs, p in L["paths"]:
                shapes = [{"ty": "sh", "ks": {"a": 0, "k": _lottie_shape(sp)}} for sp in subs]
                items.append({"ty": "gr", "it": shapes + _lottie_paints(p, defs_root) + [_tr()]})
            shapes = items[::-1]
        elif L["kind"] == "morph":
            keys, n = L["keys"], len(L["keys"])
            sh = []
            for si in range(len(keys[0])):
                end = lambda sp: sp["segs"][-1][-1] if sp["segs"] else sp["pts"][0]
                drop = keys[0][si]["closed"] and all(math.dist(kf[si]["pts"][0], end(kf[si])) < 1e-3 for kf in keys)
                fr = [{"t": round(op * j / n, 3), "s": [_lottie_shape(keys[j][si], drop)], "i": {"x": [0.58], "y": [1]},
                       "o": {"x": [0.42], "y": [0]}} for j in range(n)] + [{"t": op, "s": [_lottie_shape(keys[0][si], drop)]}]
                sh.append({"ty": "sh", "ks": {"a": 1, "k": fr}})
            shapes = [{"ty": "gr", "it": sh + _lottie_paints(L["paint"], defs_root) + [_tr()]}]
        else:
            g = L["glow"]
            circle = {"ty": "sh", "ks": {"a": 0, "k": _lottie_shape(_ellipse(g["cx"], g["cy"], g["rx"], g["rx"]))}}
            tr = _tr()
            tr.update({"p": {"a": 0, "k": [g["cx"], g["cy"]]}, "a": {"a": 0, "k": [g["cx"], g["cy"]]},
                       "s": {"a": 0, "k": [100, 100 * g["ry"] / g["rx"]]}})
            shapes = [{"ty": "gr", "it": [circle, _lottie_fill(None, g["opacity"], defs_root, grad=_glow_grad(g)), tr]}]
        layers.append({"ddd": 0, "ty": 4, "nm": L["name"], "sr": 1, "ip": 0, "op": op, "st": 0, "bm": 0,
                       "ks": _lottie_ks(L, vb, k, op), "shapes": shapes})
    layers = layers[::-1]
    for i, lyr in enumerate(layers):
        lyr["ind"] = i + 1
    return {"v": "5.7.0", "fr": rate, "ip": 0, "op": op, "w": round(vb[2] * k), "h": round(vb[3] * k), "nm": "gptkit",
            "ddd": 0, "assets": [], "layers": layers}


# ---- AnimatedVectorDrawable

def _avd_interp(ease, ind, path=None):
    x1, y1, x2, y2 = EASES.get(ease, EASES["ease_in_out"])
    attrs = (f'android:pathData="{path}"' if path else
             f'android:controlX1="{x1:g}" android:controlY1="{y1:g}" android:controlX2="{x2:g}" android:controlY2="{y2:g}"')
    return (f'{ind}<aapt:attr name="android:interpolator">\n{ind}    <pathInterpolator {attrs} />\n{ind}</aapt:attr>')


def _avd_pvh(prop, frames, ind):
    """frames: [(fraction, value, ease into this frame)]."""
    out = [f'{ind}<propertyValuesHolder android:propertyName="{prop}">']
    for f, v, e in frames:
        head = f'{ind}    <keyframe android:fraction="{_fmtn(f, 4)}" android:value="{_fmtn(v, 4)}"'
        out.append(head + (f">\n{_avd_interp(e, ind + '        ')}\n{ind}    </keyframe>" if e and e != "linear" else " />"))
    return "\n".join(out + [f"{ind}</propertyValuesHolder>"])


def _avd_animator(pvhs, ms, ind, attrs=""):
    return (f'{ind}<objectAnimator android:duration="{ms}" android:repeatCount="infinite"{attrs}\n'
            f'{ind}    android:interpolator="@android:interpolator/linear">\n' + "\n".join(pvhs) + f"\n{ind}</objectAnimator>")


def _avd_target(name, animators, ind="    "):
    body = animators[0] if len(animators) == 1 else (f"{ind}        <set>\n" + "\n".join(
        "    " + ln for x in animators for ln in x.split("\n")) + f"\n{ind}        </set>")
    return (f'{ind}<target android:name="{name}">\n{ind}    <aapt:attr name="android:animation">\n{body}\n'
            f'{ind}    </aapt:attr>\n{ind}</target>')


def _fade_keys(ks):
    """Opacity must not overshoot: AVD alpha above 1 wraps to near-transparent (seen on API 36)."""
    return [dict(k, ease="ease_out") if k["ease"] == "overshoot" else k for k in ks]


def _frames(ks, fn):
    return [(k["t"], fn(k), ks[j - 1]["ease"] if j else None) for j, k in enumerate(ks)]


def _alpha_frames(ks, base, window=None, eps=0.0005):
    """fillAlpha keyframes: base x motion opacity, optionally visible only inside window=(a, b)."""
    ks = _fade_keys(ks)
    base = min(1.0, max(0.0, base))
    cl = lambda v: min(1.0, max(0.0, v))
    fr = _frames(ks, lambda k: cl(base * k["opacity"])) if ks else [(0.0, base, None), (1.0, base, None)]
    if not window:
        return fr
    a, b = window
    at = lambda t: cl(base * _motion_at(ks, t)["opacity"]) if ks else base
    b_in = b if b == 1 else b - eps  # the outgoing copy fades over [b - eps, b] while the next one fades in: no double alpha
    inner = [(a, at(a), None)] + [f for f in fr if a < f[0] < b_in] + [(b_in, at(b), None)]
    out = [] if a == 0 else [(0.0, 0.0, None), (a - eps, 0.0, None)]
    out += inner
    out += [] if b == 1 else [(b, 0.0, None), (1.0, 0.0, None)]
    return out


def _avd_paint(p, defs_root, ind):
    attrs, kids = [], []

    def color(attr, val):
        ref = re.match(r"url\(#([^)]+)\)", val or "")
        if not ref:
            attrs.append(f'android:{attr}="{_color(val)}"')
            return
        linear, s, e, stops = p["_grad"] if "_grad" in p else _gradient(defs_root, ref.group(1))
        head = (f'android:type="linear" android:startX="{_fmtn(s[0])}" android:startY="{_fmtn(s[1])}" '
                f'android:endX="{_fmtn(e[0])}" android:endY="{_fmtn(e[1])}"' if linear else
                f'android:type="radial" android:centerX="{_fmtn(s[0])}" android:centerY="{_fmtn(s[1])}" '
                f'android:gradientRadius="{_fmtn(max(math.dist(s, e), 0.01))}"')
        items = "".join(f'\n{ind}        <item android:offset="{o:g}" android:color="{_color(c, al)}" />' for o, c, al in stops)
        kids.append(f'{ind}<aapt:attr name="android:{attr}">\n{ind}    <gradient {head}>{items}\n{ind}    </gradient>\n{ind}</aapt:attr>')
    if p.get("fill") == "glow":
        color("fillColor", "url(#glow)")
    elif (p.get("fill") or "none") != "none":
        color("fillColor", p["fill"])
        if p.get("fill_rule") == "evenodd":
            attrs.append('android:fillType="evenOdd"')
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        color("strokeColor", p["stroke"])
        attrs.append(f'android:strokeWidth="{_fmtn(float(p["stroke_width"]))}"')
        attrs.append(f'android:strokeLineCap="{_cap(p.get("cap"))}" android:strokeLineJoin="{_join(p.get("join"))}"')
    return attrs, kids


def build_avd(vec, seconds=1.2, width_dp=48):
    """<animated-vector> with the vector inlined: group transforms, fillAlpha fades and pathData morphs, all looping."""
    vb = _vec_vb(vec)
    ms = round(seconds * 1000)
    defs_root = _defs_root(vec)
    used = set()  # every android:name (groups and paths) is unique: targets resolve by name
    body, targets = [], []
    for L in _layers(vec):
        gname = _layer_name(L["name"], used)
        ks, (ax, ay) = L["ks"], L["anchor"]
        m0 = _motion_at(ks, 0.0)
        paths = []
        if L["kind"] == "morph":
            keys, n = L["keys"], len(L["keys"])
            p = L["paint"]
            if n == 2:
                paths.append((_layer_name(f"{gname}_p0", used), _d(keys[0]), p, None, [
                    f'            <objectAnimator android:propertyName="pathData" android:valueType="pathType"\n'
                    f'                android:valueFrom="{_d(keys[0])}"\n                android:valueTo="{_d(keys[1])}"\n'
                    f'                android:duration="{ms // 2}" android:repeatCount="infinite" android:repeatMode="reverse">\n'
                    f'{_avd_interp("ease_in_out", "                ")}\n            </objectAnimator>']))
            else:
                # frameworks can't keyframe pathData, so each segment morphs on its own copy, visible only in its window
                x1, y1, x2, y2 = EASES["ease_in_out"]
                for j in range(n):
                    a, b = j / n, (j + 1) / n
                    w = b - a
                    ip = ("M0,0" + (f" L{a:.4g},0" if a > 0 else "") + f" C{a + x1 * w:.4g},{y1:g} {a + x2 * w:.4g},{y2:g} {b:.4g},1" +
                          (" L1,1" if b < 1 else ""))
                    paths.append((_layer_name(f"{gname}_p{j}", used), _d(keys[j]), p, (a, b), [
                        f'            <objectAnimator android:propertyName="pathData" android:valueType="pathType"\n'
                        f'                android:valueFrom="{_d(keys[j])}"\n                android:valueTo="{_d(keys[(j + 1) % n])}"\n'
                        f'                android:duration="{ms}" android:repeatCount="infinite">\n'
                        f'{_avd_interp(None, "                ", ip)}\n            </objectAnimator>']))
        elif L["kind"] == "glow":
            g = L["glow"]
            p = {"fill": "glow", "_grad": _glow_grad(g), "opacity": g["opacity"]}
            paths.append((_layer_name(f"{gname}_p0", used), _d([_ellipse(g["cx"], g["cy"], g["rx"], g["rx"])]), p, None, []))
        else:
            for i, (subs, p) in enumerate(L["paths"]):
                paths.append((_layer_name(f"{gname}_p{i}", used), _d(subs), p, None, []))
        xml = []
        for pname, d, p, window, anims in paths:
            fa, sa = _paint_alphas(p)
            attrs, kids = _avd_paint(p, defs_root, "            ")
            fades = []
            for prop, on, base in (("fillAlpha", (p.get("fill") or "none") != "none", fa),
                                   ("strokeAlpha", (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0, sa)):
                if not on:
                    continue
                fr = _alpha_frames(ks, base, window)
                v0 = fr[0][1]
                if v0 < 1:
                    attrs.append(f'android:{prop}="{_fmtn(v0, 4)}"')
                if len({round(f[1], 4) for f in fr}) > 1:
                    fades.append(_avd_pvh(prop, fr, "                "))
            if fades:
                anims = anims + [_avd_animator(fades, ms, "            ")]
            if anims:
                targets.append(_avd_target(pname, anims))
            head = f'        <path\n            android:name="{pname}"\n            android:pathData="{d}"'
            tail = "".join(f"\n            {x}" for x in attrs)
            xml.append(head + tail + (">\n" + "\n".join(kids) + "\n        </path>" if kids else " />"))
        if L["kind"] == "glow" and L["glow"]["ry"] != L["glow"]["rx"]:
            g = L["glow"]
            xml = [f'        <group android:pivotX="{_fmtn(g["cx"])}" android:pivotY="{_fmtn(g["cy"])}" '
                   f'android:scaleY="{g["ry"] / g["rx"]:.4g}">'] + xml + ["        </group>"]
        if ks:
            chans = {"translateX": ("dx", 0), "translateY": ("dy", 0), "scaleX": ("sx", 1), "scaleY": ("sy", 1), "rotation": ("rotate", 0)}
            pvhs, gattrs = [], [f'android:name="{gname}"', f'android:pivotX="{_fmtn(ax)}"', f'android:pivotY="{_fmtn(ay)}"']
            for prop, (c, rest) in chans.items():
                if abs(m0[c] - rest) > 1e-6:
                    gattrs.append(f'android:{prop}="{_fmtn(m0[c], 4)}"')
                if len({round(k[c], 5) for k in ks}) > 1:
                    pvhs.append(_avd_pvh(prop, _frames(ks, lambda k, c=c: k[c]), "                "))
            if pvhs:
                targets.append(_avd_target(gname, [_avd_animator(pvhs, ms, "            ")]))
            xml = [f'        <group {" ".join(gattrs)}>'] + ["    " + ln for x in xml for ln in x.split("\n")] + ["        </group>"]
        body += xml
    if vb[0] or vb[1]:
        body = [f'        <group android:translateX="{_fmtn(-vb[0])}" android:translateY="{_fmtn(-vb[1])}">'] + \
               ["    " + ln for x in body for ln in x.split("\n")] + ["        </group>"]
    assert len(used) == len(set(used)) and all(_NAME_RE.match(n) for n in used), used
    hdp = width_dp * vb[3] / vb[2]
    vector = (f'    <aapt:attr name="android:drawable">\n        <vector android:width="{_fmtn(width_dp)}dp" android:height="{_fmtn(hdp)}dp"\n'
              f'            android:viewportWidth="{_fmtn(vb[2])}" android:viewportHeight="{_fmtn(vb[3])}">\n' +
              "\n".join("    " + ln for x in body for ln in x.split("\n")) + "\n        </vector>\n    </aapt:attr>")
    return ('<?xml version="1.0" encoding="utf-8"?>\n<animated-vector xmlns:android="http://schemas.android.com/apk/res/android"\n'
            '    xmlns:aapt="http://schemas.android.com/aapt">\n' + vector + "\n" + "\n".join(targets) + "\n</animated-vector>\n")


# ---- previews: lottie-web in headless Chrome, SVG fallback

LOTTIE_JS = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"


@functools.cache
def _chrome():
    if os.environ.get("GPTKIT_CHROME") == "none":
        return None
    for c in (os.environ.get("GPTKIT_CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
              *(shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"))):
        if c and Path(c).exists():
            return c
    return None


_LOTTIE_JS_REASON = ""  # why _lottie_js() returned None (set on every call)
_LOTTIE_REASON = ""  # why the last render_lottie() returned None
_SHOT_REASON = ""  # why the last _chrome_shot() returned None
_warned_shot = False


def _lottie_js_ok(txt):
    return len(txt) > 100_000 and (txt.startswith("/*!") or "lottie" in txt)


def _lottie_js_cache():
    return _cache_dir() / "lottie-5.12.2.min.js"


@functools.cache
def _lottie_js():
    """lottie-web source (cached under ~/.cache/gptkit) for inlining; None with _LOTTIE_JS_REASON set when unavailable."""
    global _LOTTIE_JS_REASON
    p = _lottie_js_cache()
    try:
        txt = p.read_text() if p.exists() else ""
    except OSError as e:
        txt, _LOTTIE_JS_REASON = "", f"cannot read {p}: {e}"
    if txt and _lottie_js_ok(txt):
        _LOTTIE_JS_REASON = ""
        return txt
    if txt:
        print(f"  ! cached {p} is not lottie-web ({len(txt)} bytes); fetching again")
    try:
        r = requests.get(LOTTIE_JS, timeout=60)
        r.raise_for_status()
        txt = r.text
    except Exception as e:  # requests.RequestException and friends; the preview is optional
        _LOTTIE_JS_REASON = f"could not fetch {LOTTIE_JS} ({type(e).__name__}: {str(e)[:120]})"
        return None
    if not _lottie_js_ok(txt):
        _LOTTIE_JS_REASON = f"{LOTTIE_JS} returned {len(txt)} bytes that are not lottie-web"
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(txt)
        os.replace(tmp, p)
    except OSError as e:
        print(f"  ! could not cache lottie-web at {p}: {e}")
    _LOTTIE_JS_REASON = ""
    return txt


def _chrome_shot(html, out, w, h, budget=8000, timeout=120):
    """Screenshot an HTML string at w x h with a transparent background; the RGBA image, or None (_SHOT_REASON says why)."""
    global _SHOT_REASON, _warned_shot
    exe = _chrome()
    if not exe:
        _SHOT_REASON = "no Chrome/Chromium found (set GPTKIT_CHROME)"
        return None
    out = Path(out)
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page, errf = Path(tmp) / "page.html", Path(tmp) / "stderr.txt"
        page.write_text(html)
        try:
            with errf.open("w") as ef:
                proc = subprocess.Popen([exe, *_chrome_flags(w, h), f"--user-data-dir={tmp}/p", f"--virtual-time-budget={budget}",
                                         f"--screenshot={out}", page.as_uri()], stdout=subprocess.DEVNULL, stderr=ef)
        except OSError as e:
            _SHOT_REASON = f"{exe}: {e}"
            return None
        # Chrome can linger after writing the screenshot (lottie's web worker), so stop once the file is complete
        size, end = -1, time.time() + timeout
        while proc.poll() is None and time.time() < end:
            cur = out.stat().st_size if out.exists() else -1
            if cur > 0 and cur == size:
                break
            size = cur
            time.sleep(0.5)
        proc.kill()
        proc.wait()
        if not out.exists():
            r = subprocess.CompletedProcess(exe, proc.returncode, "", errf.read_text(errors="replace"))
            _SHOT_REASON = f"{exe}: {_chrome_err(r)}"
            if not _warned_shot:
                print(f"  ! headless Chrome produced no screenshot: {_SHOT_REASON}")
                _warned_shot = True
            return None
    try:
        img = _rgba(out)
    except (OSError, ValueError) as e:
        _SHOT_REASON = f"{exe} wrote an unreadable screenshot: {e}"
        return None
    _SHOT_REASON = ""
    return img.crop((0, 0, w, h))


def render_lottie(lot, times, out_png, T=256):
    """Render the Lottie at loop fractions `times` with lottie-web; RGBA frames (transparent), or None (_LOTTIE_REASON)."""
    global _LOTTIE_REASON
    js = _lottie_js()
    if not js:
        _LOTTIE_REASON = f"lottie-web unavailable: {_LOTTIE_JS_REASON}"
        return None
    frames = [round(t * lot["op"], 3) for t in times]
    html = ("<!doctype html><meta charset=utf-8><style>body{margin:0;display:flex}div{width:%dpx;height:%dpx;flex:none}</style>"
            "<body>%s<script>%s</script><script>var D=%s;%s.forEach(function(f,i){var a=lottie.loadAnimation({container:"
            "document.getElementById('f'+i),renderer:'svg',loop:false,autoplay:false,animationData:JSON.parse(JSON.stringify(D))});"
            "a.addEventListener('DOMLoaded',function(){a.goToAndStop(f,true)});});</script>") % (
        T, T, "".join(f"<div id=f{i}></div>" for i in range(len(frames))), js, _js_json(lot), json.dumps(frames))
    strip = _chrome_shot(html, out_png, T * len(frames), T)
    if strip is None:
        _LOTTIE_REASON = f"lottie-web unavailable: headless Chrome failed ({_SHOT_REASON})"
        return None
    if strip.width < T * len(frames):
        _LOTTIE_REASON = f"lottie-web unavailable: screenshot is {strip.width}x{strip.height}, expected {T * len(frames)}x{T}"
        return None
    _LOTTIE_REASON = ""
    return [strip.crop((i * T, 0, (i + 1) * T, T)) for i in range(len(frames))]


def _js_json(obj):
    """json.dumps for inlining inside <script>: a layer name containing '</script>' must not end the element."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _pixel_motion(frames):
    """Largest share of the drawn area (t=0 union frame i) whose colour visibly changes from t=0."""
    if len(frames) < 2:
        return 0.0
    pm = lambda im: (lambda a: np.concatenate([a[..., :3] * a[..., 3:] / 255, a[..., 3:]], -1))(np.asarray(im, dtype=np.float32))
    f0 = pm(frames[0])
    best = 0.0
    for f in map(pm, frames[1:]):
        area = ((f0[..., 3] > 64) | (f[..., 3] > 64)).sum()
        best = max(best, float((np.abs(f - f0).max(-1) > 40).sum()) / max(1, area))
    return round(best, 3)


def _vec_svg(vec, t, size=256, bg=None):
    """Static SVG of the animation at loop time t (morphs, motion and glows applied)."""
    defs = vec.get("defs", "")
    defs = defs if "<defs" in defs else f"<defs>{defs}</defs>"
    glow_defs, body = [], []

    def paint(p):
        fa, sa = _paint_alphas(p)
        s = f' fill="{p.get("fill") or "none"}" fill-opacity="{fa:g}"'
        if p.get("fill_rule") == "evenodd":
            s += ' fill-rule="evenodd"'
        if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
            s += (f' stroke="{p["stroke"]}" stroke-opacity="{sa:g}" stroke-width="{p["stroke_width"]}" '
                  f'stroke-linecap="{_cap(p.get("cap"))}" stroke-linejoin="{_join(p.get("join"))}"')
        return s
    for L in _layers(vec):
        m = _motion_at(L["ks"], t)
        ax, ay = L["anchor"]
        tf = (f' transform="translate({ax + m["dx"]:.3f} {ay + m["dy"]:.3f}) rotate({m["rotate"]:.3f}) scale({m["sx"]:.4f} '
              f'{m["sy"]:.4f}) translate({-ax:.3f} {-ay:.3f})" opacity="{m["opacity"]:.3f}"') if L["ks"] else ""
        if L["kind"] == "static":
            inner = "".join(f'<path d="{_d(subs)}"{paint(p)}/>' for subs, p in L["paths"])
        elif L["kind"] == "morph":
            inner = f'<path d="{_d(_morph_at(L["keys"], t))}"{paint(L["paint"])}/>'
        else:
            g = L["glow"]
            gid = f"glow_{len(glow_defs)}"
            glow_defs.append(f'<radialGradient id="{gid}">' + "".join(
                f'<stop offset="{o}" stop-color="{g["color"]}" stop-opacity="{al}"/>' for o, al in GLOW_STOPS) + "</radialGradient>")
            inner = f'<ellipse cx="{g["cx"]}" cy="{g["cy"]}" rx="{g["rx"]}" ry="{g["ry"]}" fill="url(#{gid})" opacity="{g["opacity"]}"/>'
        body.append(f"<g{tf}>{inner}</g>")
    rect = f'<rect x="-100000" y="-100000" width="200000" height="200000" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="{vec["viewBox"]}">'
            f'{defs}<defs>{"".join(glow_defs)}</defs>{rect}{"".join(body)}</svg>')


_SVG_SHAPES = ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon")
_SVG_SKIP = ("defs", "title", "desc", "metadata", "linearGradient", "radialGradient", "stop", "clipPath", "mask", "filter",
             "pattern", "marker", "script")
_SVG_REJECT = ("use", "symbol", "style", "text", "tspan", "textPath", "image", "foreignObject", "switch")


def _svg_layers(svg_text, stem, file="svg"):
    """Import an SVG as static layers, one per top-level element (a <g> keeps its children together).
    Paint is inherited (style beats attributes, opacity multiplies), transforms are baked into the coordinates
    (stroke width scaled by sqrt|det|), fill-rule/fill-opacity/stroke-opacity/caps/joins are kept, display:none and
    visibility:hidden elements are skipped, every colour becomes '#RRGGBB' and gradients become explicit
    userSpaceOnUse defs. Raises ValueError (one line, naming the file) for <use>/<symbol>/CSS/clip-path/mask/filter/
    text/image and for malformed transforms, path data or colours."""
    root = ET.fromstring(svg_text)
    tag = lambda el: el.tag.replace(SVG_NS, "") if isinstance(el.tag, str) else ""
    if tag(root) != "svg":
        raise ValueError(f"{file}: root element is <{tag(root)}>, not <svg>")
    hint = f"flatten it first (svgo: inlineStyles, convertShapeToPath, removeUselessDefs) or re-run svg --only {stem}"
    for el in root.iter():
        if tag(el) in _SVG_REJECT:
            raise ValueError(f"{file}: <{tag(el)}> is not supported; {hint}")
        for k in ("clip-path", "mask", "filter"):
            if (el.get(k) or _style_dict(el).get(k) or "none") != "none" and tag(el) not in _SVG_SKIP:
                raise ValueError(f"{file}: {k} on <{tag(el)}> is not supported; {hint}")
    vb_s = root.get("viewBox")
    if vb_s:
        try:
            vbl = [float(x) for x in re.split(r"[\s,]+", vb_s.strip())]
            if len(vbl) != 4 or vbl[2] <= 0 or vbl[3] <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"{file}: viewBox {vb_s!r} must be 4 numbers \"x y w h\" with w and h > 0")
    else:
        vbl = [0.0, 0.0, _num(root.get("width"), 24), _num(root.get("height"), 24)]
        if vbl[2] <= 0 or vbl[3] <= 0:
            raise ValueError(f"{file}: no viewBox and no positive width/height")
    vb = " ".join(_fmtn(v, 4) for v in vbl)
    defs_out, grad_ids, notes, used = [], {}, [], set()

    def grad_ref(v, m, subs, what):
        """url(#id) -> the id of an explicit userSpaceOnUse copy resolved for this shape (bbox units, baked transform)."""
        ref = re.fullmatch(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)(\s+\S+)?", v.strip())
        if not ref:
            return None
        gid = ref.group(1)
        bbox = _bbox(_curve_pts(subs))
        try:
            linear, s0, e0, stops = _gradient(root, gid, bbox, vbl)
        except ValueError as e:
            raise ValueError(f"{file}: {what}: {e}")
        if not linear and abs((bbox[2] - bbox[0]) - (bbox[3] - bbox[1])) > 0.05 * max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1e-9) \
                and (next((el.get("gradientUnits") for el in _grad_chain(root, gid) if el.get("gradientUnits")), None) or "objectBoundingBox") != "userSpaceOnUse":
            notes.append(f"gradient #{gid} uses objectBoundingBox units on a non-square shape; approximated as a circle")
        s1, e1 = _mapply(m, *s0), _mapply(m, *e0)
        if not linear:
            e1 = (s1[0] + math.dist(s0, e0) * math.sqrt(abs(m[0] * m[3] - m[1] * m[2])), s1[1])
        key = (linear, tuple(round(x, 3) for x in s1 + e1), tuple(stops))
        if key not in grad_ids:
            base = re.sub(r"[^\w-]", "_", gid) or "g"
            nid = base if all(k[0] != base for k in grad_ids.values()) else f"{base}_{len(grad_ids) + 1}"
            grad_ids[key] = (nid,)
            geo = (f'x1="{_fmtn(s1[0], 3)}" y1="{_fmtn(s1[1], 3)}" x2="{_fmtn(e1[0], 3)}" y2="{_fmtn(e1[1], 3)}"' if linear else
                   f'cx="{_fmtn(s1[0], 3)}" cy="{_fmtn(s1[1], 3)}" r="{_fmtn(e1[0] - s1[0], 3)}"')
            kind = "linearGradient" if linear else "radialGradient"
            defs_out.append(f'<{kind} id="{nid}" gradientUnits="userSpaceOnUse" {geo}>' + "".join(
                f'<stop offset="{o:g}" stop-color="{c}"' + (f' stop-opacity="{al:g}"' if al < 1 else "") + "/>"
                for o, c, al in stops) + f"</{kind}>")
        return f"url(#{grad_ids[key][0]})"

    def pct(v, dflt=1.0):
        sv = str(v).strip() if v is not None else ""
        if not sv:
            return dflt
        n = _num(sv) / 100 if sv.endswith("%") else _num(sv, dflt)
        return min(1.0, max(0.0, n))

    def walk(el, inh, m):
        t = tag(el)
        if not t or t in _SVG_SKIP:
            return []
        a = _inherit(el, inh)
        if a.get("display") == "none":
            return []
        try:
            mm = _mmul(m, _matrix(el.get("transform")))
        except ValueError as e:
            raise ValueError(f"{file}: <{t}>: {e}")
        if t in ("g", "a"):
            return [p for ch in el for p in walk(ch, a, mm)]
        if t == "svg":
            raise ValueError(f"{file}: nested <svg> is not supported; {hint}")
        if t not in _SVG_SHAPES:
            notes.append(f"<{t}> ignored")
            return []
        if a.get("visibility") in ("hidden", "collapse"):
            return []
        d = _shape_path(el)
        if not d:
            return []
        what = f"<{t}{' id=' + el.get('id') if el.get('id') else ''}>"
        try:
            subs = _subpaths(d)
        except (ValueError, IndexError) as e:
            raise ValueError(f"{file}: {what}: {e}")
        if not subs or not any(sp["segs"] for sp in subs):
            return []
        cur = a.get("color")
        fill, stroke = (a.get("fill") or "").strip() or "#000000", (a.get("stroke") or "none").strip() or "none"
        fo, so = pct(a.get("fill-opacity")), pct(a.get("stroke-opacity"))
        try:
            if fill != "none":
                g = grad_ref(fill, mm, subs, what)
                if g:
                    fill = g
                else:
                    if fill.lower() == "currentcolor" and not cur:
                        notes.append(f"{what}: currentColor with no color attribute; using #000000")
                    fill, al = _paint_color(fill, cur)
                    fo *= al
            if stroke != "none":
                g = grad_ref(stroke, mm, subs, what)
                if g:
                    stroke = g
                else:
                    if stroke.lower() == "currentcolor" and not cur:
                        notes.append(f"{what}: currentColor with no color attribute; using #000000")
                    stroke, al = _paint_color(stroke, cur)
                    so *= al
        except ValueError as e:
            raise ValueError(f"{file}: {what}: {e}" if not str(e).startswith(file) else str(e))
        det = math.sqrt(abs(mm[0] * mm[3] - mm[1] * mm[2]))
        sw = _num(a.get("stroke-width"), 1.0) * det if stroke != "none" else 0.0
        if sw <= 0:
            stroke, sw = "none", 0.0
        if fill == "none" and stroke == "none":
            return []
        if stroke != "none" and (a.get("stroke-dasharray") or "none") != "none":
            notes.append(f"{what}: dashes are drawn solid")
        p = {"d": _d(_xform_subs(subs, mm)), "fill": fill, "opacity": round(min(1.0, max(0.0, _num(a.get("opacity"), 1.0))), 4),
             "stroke": stroke, "stroke_width": round(sw, 4), "cap": _cap(a.get("stroke-linecap"), "butt"),
             "join": _join(a.get("stroke-linejoin"), "miter")}
        if fo < 1:
            p["fill_opacity"] = round(fo, 4)
        if so < 1:
            p["stroke_opacity"] = round(so, 4)
        if fill != "none" and (a.get("fill-rule") or "nonzero") == "evenodd":
            p["fill_rule"] = "evenodd"
        return [p]

    layers = []
    base = _inherit(root, {})
    for el in (ch for ch in root if tag(ch) and tag(ch) not in _SVG_SKIP):
        ps = walk(el, base, _IDENTITY)
        if ps:
            layers.append({"name": _layer_name(el.get("id") or (stem if not layers else f"{stem}_{len(layers) + 1}"), used),
                           "paths": ps, "motion": {"anchor_x": 0, "anchor_y": 0, "keys": []}})
    for n in dict.fromkeys(notes):
        print(f"  ~ {file}: {n}")
    return vb, "".join(defs_out), layers


def _merge_imported(vec, imported):
    """Imported SVG layers are the source of truth for geometry; the model only adds motion and extra layers."""
    vb, defs, layers = imported
    vec["viewBox"] = vb
    if defs and defs not in (vec.get("defs") or ""):
        vec["defs"] = defs + (vec.get("defs") or "")
    got = {}
    for sl in vec.get("static_layers") or []:
        if isinstance(sl, dict):
            got.setdefault(_res_name(str(sl.get("name") or ""), "layer_"), sl)
    for L in layers:
        if L["name"] in got:
            got[L["name"]]["paths"] = json.loads(json.dumps(L["paths"]))
            got[L["name"]]["name"] = L["name"]
        else:
            vec.setdefault("static_layers", []).insert(0, json.loads(json.dumps(L)))
    return vec


def _review_sheet(vec, lot, src, out, bg, n=8):
    """keys_review.png: the source, then n evenly spaced frames of the real Lottie (lottie-web) or an SVG fallback.
    Returns (how, pixel_motion): how is 'lottie-web', 'svg fallback (...)' or 'NOT RENDERED (...)' with the reason."""
    out = Path(out)
    for stale in out.glob("key_*.svg"):  # written by older versions; the fallback now renders from a temp dir
        stale.unlink()
    times = [i / n for i in range(n)]
    frames = render_lottie(lot, times, out / ".strip.png")
    (out / ".strip.png").unlink(missing_ok=True)
    how, px = "lottie-web", None
    if frames is not None:
        px = _pixel_motion(frames)
    else:
        reason, frames = _LOTTIE_REASON, []
        with tempfile.TemporaryDirectory() as td:
            for i, t in enumerate(times):
                p = Path(td) / f"key_{i + 1}.svg"
                p.write_text(_vec_svg(vec, t, bg=bg))
                prev = _svg_image(p, Path(td) / f"key_{i + 1}.png", 256)
                if prev is not None:
                    frames.append(prev)
        how = f"svg fallback ({reason})" if frames else (f"NOT RENDERED ({reason}): fix Chrome (GPTKIT_CHROME, "
                                                         f"GPTKIT_CHROME_NO_SANDBOX=1) or pip install cairosvg")
    tiles = [("asset", src)] if src is not None else []
    for t, f in zip(times, frames):
        base = Image.new("RGBA", f.size, bg)
        base.alpha_composite(f)
        tiles.append((f"t={t:.2f}", base))
    if frames:
        sheet([tiles[i:i + 5] for i in range(0, len(tiles), 5)], out / "keys_review.png", T=220)
    else:
        (out / "keys_review.png").unlink(missing_ok=True)
    return how, px


def _svg_image(svg_path, out, size=512):
    """Rasterise an SVG with transparency (headless Chrome), else with the preview renderer."""
    b64 = base64.b64encode(Path(svg_path).read_bytes()).decode()
    img = _chrome_shot(f"<body style='margin:0'><img src='data:image/svg+xml;base64,{b64}' style='width:{size}px;"
                       f"height:{size}px'>", out, size, size)
    if img is None:
        p = _render_svg(svg_path, Path(out).parent, size)
        img = _rgba(p) if p else None
    return img


def _import_vec(imported):
    vb, defs, layers = imported
    return {"viewBox": vb, "defs": defs, "static_layers": layers, "animated_layers": [], "glows": [],
            "layer_order": [L["name"] for L in layers]}


def _contrast_bg(img, dark="#1b1d33"):
    """White when the drawn pixels are dark (alpha-weighted mean luminance < 0.5), else `dark`: a black outline icon
    must not be shown to the model on navy at 1.3:1 contrast."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    w = arr[..., 3] / 255
    if w.sum() < 1:
        return dark
    lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) / 255
    return "#FFFFFF" if float((lum * w).sum() / w.sum()) < 0.5 else dark


def _check_vec(vec):
    """_canon_vec + _validate_vec without repeating a colour error in two wordings."""
    cerrs = _canon_vec(vec)
    heads = {e.split(" is not")[0] for e in cerrs}
    return cerrs + [e for e in _validate_vec(vec) if e.split(" is not")[0] not in heads]


def _ask_vector(a, mock, asset, src, imported, bg):
    thr = a.min_motion if a.min_motion is not None else MOTION_MIN[a.strength]
    parts = [VECTOR_PROMPT.format(moving=a.moving, static=a.static or "nothing", k=a.keyframes, seconds=a.seconds,
                                  motion=a.motion or "a gentle natural loop", strength=MOTION_STRENGTH[a.strength])]
    if imported:
        parts.append(f"The asset already exists as vector static_layers (viewBox \"{imported[0]}\"). Return each with exactly "
                     "its name and paths: [] (the geometry is restored from the source) and add the motion, glows and any "
                     "extra layers the animation needs:\n" + json.dumps(imported[2]))
    if src is not None:
        flat = Image.new("RGBA", src.size, _contrast_bg(src, bg))
        flat.alpha_composite(src)
        parts.append(flat)
    if asset and mock is not None:
        parts += ["For reference, how it appears in the approved design:", ref_crop(mock, asset["bbox"], 0.3, 512)]
    kit_errs = set(_validate_vec(_import_vec(imported))) if imported else set()  # never blame the model for the import
    imported_names = {L["name"] for L in imported[2]} if imported else set()
    best, vec = None, None
    for attempt in range(2):
        vec = ask(parts, VECTOR_SCHEMA, "vector_layers")
        raw = json.loads(json.dumps(vec))
        if imported:
            _merge_imported(vec, imported)
        errs = [e for e in _check_vec(vec) if e not in kit_errs]
        rep = None if errs else motion_report(_normalize_vec(vec))
        weak = rep is not None and (rep["motion"] is None or rep["motion"] < thr)
        if rep and (best is None or (rep["motion"] or 0) > (best[1]["motion"] or 0)):
            best = (vec, rep)
        if rep and not weak or attempt:
            break
        per = ", ".join(f"{n} {r['move']:.1%} move / {r['fade']:.0%} fade" for n, r in rep["layers"].items() if r["kind"] != "glow") if rep else ""
        fb = (f"Your previous answer was invalid: {'; '.join(errs[:6])}. Fix it and follow every rule." if errs else
              "Your previous answer animates nothing: no layer has motion keys and there are no animated_layers. Add the motion "
              f"that was asked for ({a.moving})." if rep["motion"] is None else
              f"Your previous answer moves too little: its animated layers move {rep['motion']:.1%} of their size on "
              f"average ({per}), at least {thr:.0%} is needed. Keep the geometry but exaggerate the keyframes and motion keys.")
        print("  retrying:", fb, flush=True)
        for sl in raw.get("static_layers") or []:  # the model's own answer, imported geometry elided as it was asked
            if isinstance(sl, dict) and _res_name(str(sl.get("name") or ""), "layer_") in imported_names:
                sl["paths"] = []
        parts = parts + [fb + "\nPrevious answer:\n" + json.dumps(raw)]
    return best[0] if best else vec


def _lottie_lint(lot):
    """Portability checks on a built Lottie (a failure is a kit bug): no merge paths (Android/Windows only), no
    expressions (web only), every group ends with its transform, shape layers span the whole composition (image layers may take a window of it)."""
    errs = []

    def walk(items, where):
        for it in items:
            if it.get("ty") == "mm":
                errs.append(f"{where}: merge path (mm) is not rendered by lottie-ios or web players")
            if it.get("ty") == "gr":
                if not it.get("it") or it["it"][-1].get("ty") != "tr":
                    errs.append(f"{where}: group does not end with a transform")
                walk(it["it"], where)
            for k, v in it.items():
                if isinstance(v, dict) and isinstance(v.get("x"), str):
                    errs.append(f"{where}: expression in {k} (web only)")
    if not (isinstance(lot.get("op"), (int, float)) and lot["op"] > lot.get("ip", 0)):
        errs.append("op must be after ip")
    for lyr in lot.get("layers", []):
        walk(lyr.get("shapes", []), lyr.get("nm", "?"))
        for k, v in (lyr.get("ks") or {}).items():
            if isinstance(v, dict) and isinstance(v.get("x"), str):
                errs.append(f"{lyr.get('nm')}: expression in ks.{k} (web only)")
        if lyr.get("ty") == 2 and 0 <= lyr.get("ip", -1) < (lyr.get("op") or 0) <= lot["op"]:
            continue  # an image layer may show for a window of the composition (frames' image-sequence Lottie)
        if lyr.get("ip", 0) != 0 or lyr.get("op") != lot["op"]:
            errs.append(f"{lyr.get('nm')}: layer ip/op {lyr.get('ip')}/{lyr.get('op')} differ from the composition 0/{lot['op']}")
    return errs


def _aapt2_report(xml_path, what="AVD"):
    """Run aapt2 on one drawable XML and print the verdict per contract C4 (never 'valid' for compile-only)."""
    res = aapt2_check([xml_path])
    if res:
        for f, e in res.items():
            print(f"  ! aapt2: {f}: {e}")
    else:  # None (no aapt2) and {} (every available step passed) are both one honest line
        print(f"  {_aapt2_line()}")
    return res


def cmd_vectorize(a):
    d = Path(a.dir)
    if (d / "assets.json").exists():
        d, m, mock = _load(a.dir)
    elif a.svg or a.rebuild:
        m, mock = {"assets": []}, None
    else:
        sys.exit(f"vectorize: {d / 'assets.json'} not found: run manifest first, or pass --svg FILE (no design dir needed)")
    if not 2 <= a.keyframes <= 24:
        sys.exit(f"vectorize: --keyframes {a.keyframes} must be between 2 and 24 (the model writes every keyframe)")
    out = d / "vector" / a.name
    asset = next((x for x in m.get("assets", []) if isinstance(x, dict) and x.get("name") == a.name), None)
    png = d / "assets" / f"{a.name}.png"
    svg = _need(a.svg, "SVG", "pass --svg FILE") if a.svg else d / "svg" / f"{a.name}.svg"
    src, imported = None, None
    if png.exists() and not a.svg:
        src = _open_image(png, "asset")
    elif svg.exists():
        try:
            imported = _svg_layers(svg.read_text(), a.name, svg.name)
        except ET.ParseError as e:
            sys.exit(f"vectorize: {svg}: not valid SVG XML ({e})")
        except UnicodeDecodeError as e:  # a PNG passed as --svg
            sys.exit(f"vectorize: {svg} is not a text SVG file ({e.reason}); pass the .svg, or the PNG as DIR/assets/NAME.png")
        except (ValueError, IndexError) as e:
            sys.exit(f"vectorize: {e}" + ("" if a.rebuild else " (no API call was made)"))
        if not a.rebuild:
            if not imported[2]:
                sys.exit(f"vectorize: {svg}: no drawable shapes (path, circle, rect, ...) found; nothing to animate (no API call was made)")
            errs = _validate_vec(_import_vec(imported))
            if errs:
                sys.exit(f"vectorize: {svg}: {errs[0]} (fix the SVG; no API call was made)")
        out.mkdir(parents=True, exist_ok=True)
        src = _svg_image(svg, out / "source.png")
    elif not a.rebuild:
        sys.exit(f"vectorize: no source: neither {png} nor {svg} exists (pass --svg FILE)")
    out.mkdir(parents=True, exist_ok=True)
    lp = out / "layers.json"
    if a.rebuild:
        if not lp.exists():
            sys.exit(f"vectorize: {lp} not found: run without --rebuild first")
        try:
            vec = json.loads(lp.read_text())
        except json.JSONDecodeError as e:
            sys.exit(f"vectorize: {lp}: invalid JSON at line {e.lineno}: {e.msg}")
        errs = _check_vec(vec)
        if errs:
            sys.exit(f"vectorize: {lp} invalid: {errs[0]}; edit it and run --rebuild again")
    elif not a.moving:
        sys.exit("vectorize: --moving is required (or --rebuild to reuse an edited layers.json)")
    else:
        vec = _ask_vector(a, mock, asset, src, imported, a.preview_bg)
        errs = _check_vec(vec)
        lp.write_text(json.dumps(vec, indent=2))  # keep the answer for hand editing either way
        if errs:
            sys.exit(f"vectorize: the model's answer is still invalid after a retry: {errs[0]}; edit {lp} and run --rebuild")
    notes = []
    before = json.dumps(vec, sort_keys=True)
    _normalize_vec(vec, notes)
    if a.rebuild and json.dumps(vec, sort_keys=True) != before and not (out / "layers.orig.json").exists():
        (out / "layers.orig.json").write_text(lp.read_text())
        notes.append(f"layers.json normalised (paths as absolute M/L/C/Z, rounded); the file you edited is kept as layers.orig.json")
    for n in notes:
        print(f"  ({n})")
    lp.write_text(json.dumps(vec, indent=2))
    rep = motion_report(vec)
    if rep["motion"] is None and not rep["layers"]:
        if a.rebuild:
            print("  ! nothing moves: no layer has motion keys and there are no animated layers (the outputs are static)")
        else:
            sys.exit(f"vectorize: the model animated nothing (no motion keys, no animated layers) even after a retry; edit {lp} and "
                     "run --rebuild, or describe --moving/--motion more concretely")
    thr = a.min_motion if a.min_motion is not None else MOTION_MIN[a.strength]
    fit = None if a.no_fit else _fit_viewbox(vec)
    built = dict(vec, viewBox=fit) if fit else vec
    lot = build_lottie(built, a.size, a.seconds, a.fps)
    lint = _lottie_lint(lot)
    if lint:
        sys.exit(f"vectorize: internal error, the Lottie failed its portability check: {lint[0]} (please report)")
    (out / f"{a.name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    dp = a.dp or ((asset or {}).get("dp") or {}).get("w") or 48
    dp = round(dp * _vec_vb(built)[2] / _vec_vb(vec)[2], 1)
    avd = out / f"{_res_name(a.name)}_avd.xml"
    if avd.stem != f"{a.name}_avd":
        print(f"  (Android resource name: {avd.name})")
    avd.write_text(build_avd(built, a.seconds, dp))
    _aapt2_report(avd)
    js = _lottie_js()  # inline the cached player so the preview works offline; the CDN is the fallback
    (out / "preview.html").write_text(
        "<!doctype html><meta charset=utf-8><body style='margin:0;background:" + a.preview_bg + "'>"
        "<div id=a style='width:512px;height:512px'></div>"
        + (f"<script>{js}</script>" if js else f"<script src='{LOTTIE_JS}'></script>")
        + f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{_js_json(lot)}}})</script>")
    how, px = _review_sheet(built, lot, src, out, a.preview_bg, a.review_frames)
    for n, r in rep["layers"].items():
        print(f"  {n:24s} {r['kind']:6s} move={r['move']:.1%} fade={r['fade']:.2f} score={r['score']:.1%}")
    mm = rep["motion"]
    print(f"  motion {'-' if mm is None else f'{mm:.1%}'} (min {thr:.0%} for --strength {a.strength}; area-weighted, a full fade counts 25%)" +
          (" TOO SUBTLE: re-run with --strength strong or a stronger --motion" if mm is not None and mm < thr else "") +
          (f"; rendered frames change up to {px:.0%} of the drawn area" if px is not None else ""))
    for L in _layers(vec):
        if L["kind"] == "static" and len(L["paths"]) > 1 and rep["layers"].get(L["name"], {}).get("fade", 0) > 0:
            print(f"  ~ {L['name']} fades with {len(L['paths'])} overlapping paths: the AVD fades each path separately (no group alpha)")
    if abs(lot["op"] / a.fps - a.seconds) > 1e-6:
        print(f"  (Lottie: {lot['op']} frames at {lot['fr']:g} fps so the loop is exactly {a.seconds:g} s like the AVD's {round(a.seconds * 1000)} ms)")
    if a.seconds > 5:
        print("  ! loop longer than 5 s: the RenderThread AVD subsamples keyframed animators past ~300 frames "
              "(\"Duration for the animation is too long\"); prefer the Lottie or a shorter loop")
    if fit:
        print(f"  motion leaves the viewBox: canvas grown to \"{fit}\" (AVD {dp:g}dp wide; show the Lottie at {dp:g}dp too, "
              f"the {a.size}px canvas now holds the grown viewBox) so nothing clips")
    print(f"-> {out}/layers.json, {a.name}.json (Lottie), {avd.name} (AnimatedVectorDrawable), keys_review.png ({how}), preview.html")


# ---- export

DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}


def _fit(img, w, h):
    im = img.copy()
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def _px(v):
    """Android rounds density sizes as (int)(px + 0.5), not banker's rounding (round(23 * 1.5) is 34, Android says 35)."""
    return max(1, int(v + 0.5))


def _flat_art(img, max_colors=256):
    """True when the image has at most max_colors distinct RGBA values: lossless WebP is then small and exact."""
    return img.getcolors(max_colors) is not None


def _save_webp(im, path, lossless, quality):
    if lossless:
        im.save(path, "WEBP", lossless=True, method=6)
    else:
        im.save(path, "WEBP", quality=quality, method=6)


def cmd_export(a):
    img = _open_image(a.file, "source image")
    name = a.name or Path(a.file).stem
    fmt = a.format or "webp"
    if a.platform == "ios" and (a.format or a.quality is not None):
        sys.exit("export: --format/--quality do not apply to --platform ios: imagesets are PNG")
    if a.quality is not None and not 0 <= a.quality <= 100:
        sys.exit(f"export: --quality {a.quality} must be between 0 and 100 (100 = lossless)")
    if a.platform == "android":
        rn = _res_name(name)
        if rn != name:
            print(f'  (renamed "{name}" -> {rn} for Android: resource names are [a-z0-9_] and start with a letter)')
        name = rn
    elif a.platform == "web":  # a URL in src/srcset: no spaces (srcset splits on them), slashes or other URL-hostile characters
        wn = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "img"
        if wn != name:
            print(f'  (renamed "{name}" -> {wn} for the web: file names go into src/srcset URLs, [A-Za-z0-9._-] only)')
        name = wn
    elif "/" in name or "\\" in name or ":" in name:  # an imageset is a folder: path separators are not asset names
        nn = re.sub(r"[/\\:]+", "-", name).strip("-") or "img"
        print(f'  (renamed "{name}" -> {nn} for Xcode: asset names cannot contain path separators)')
        name = nn
    wdp = a.dp
    hdp = a.hdp or round(wdp * img.height / img.width, 2)
    res = Path(a.res)
    lossless = a.quality == 100 or (a.quality is None and _flat_art(img))
    quality = 90 if a.quality is None else a.quality
    biggest = {"android": max(DENSITIES.values()), "ios": 3, "web": 2}[a.platform]
    tw, th = _px(wdp * biggest), _px(hdp * biggest)
    if max(tw, th) > 8192:
        sys.exit(f"export: --dp {wdp:g}" + (f" --hdp {hdp:g}" if a.hdp else "") + f" makes a {tw}x{th} px file at "
                 f"{biggest}x (the limit is 8192 px): --dp/--hdp are dp/pt/css px, not device pixels")
    if res.exists() and not res.is_dir():
        sys.exit(f"export: --res {res} is a file; pass the res/ directory (android), the .xcassets folder (ios) or an "
                 "output directory (web)")
    if a.platform != "android" and (not name or name in (".", "..") or re.search(r"[/\\]", name)):
        sys.exit(f"export: --name {name!r} must be a plain file name (no / or \\): the files go inside --res")
    if img.width < tw or img.height < th:
        print(f"  ! source {img.width}x{img.height} px is smaller than the largest target {tw}x{th} px; it is upscaled and will "
              f"look soft (use a source at least {tw} px wide)")
    if img.getchannel("A").getextrema()[1] == 0:
        print("  ! source is fully transparent: every exported file is an empty image")
    if a.platform == "android":
        if res.name.lower().startswith(("drawable", "mipmap")):
            print(f"  ! --res {res} is itself a resource folder; density folders go under res/ (did you mean --res {res.parent}?)")
        for dens, sc in DENSITIES.items():
            out = res / f"drawable-{dens}"
            out.mkdir(parents=True, exist_ok=True)
            im = _fit(img, _px(wdp * sc), _px(hdp * sc))
            p = out / f"{name}.{fmt}"
            _save_webp(im, p, lossless, quality) if fmt == "webp" else im.save(p)
            print(f"  {p} {im.size} {p.stat().st_size} B")
        if fmt == "webp":
            print(f"  WebP {'lossless' if lossless else f'q{quality}'} (transparent WebP needs API 18+; --quality 100 forces lossless)")
    elif a.platform == "ios":
        if res.suffix == ".xcassets":
            res.mkdir(parents=True, exist_ok=True)
            if not (res / "Contents.json").exists():  # a new catalog: Xcode expects its root Contents.json
                (res / "Contents.json").write_text(json.dumps({"info": {"version": 1, "author": "xcode"}}, indent=2))
        out = res / f"{name}.imageset"
        out.mkdir(parents=True, exist_ok=True)
        images = []
        for sc in (1, 2, 3):
            fn = f"{name}@{sc}x.png"
            _fit(img, _px(wdp * sc), _px(hdp * sc)).save(out / fn)
            images.append({"idiom": "universal", "filename": fn, "scale": f"{sc}x"})
        (out / "Contents.json").write_text(json.dumps({"images": images, "info": {"version": 1, "author": "xcode"}}, indent=2))
        print(f"  {out} ({_px(wdp)}x{_px(hdp)} pt @1x/@2x/@3x)")
    else:
        res.mkdir(parents=True, exist_ok=True)
        for sc, suffix in ((1, ""), (2, "@2x")):
            p = res / f"{name}{suffix}.{fmt}"
            im = _fit(img, _px(wdp * sc), _px(hdp * sc))
            _save_webp(im, p, lossless, quality) if fmt == "webp" else im.save(p)
            print(f"  {p} {im.size} {p.stat().st_size} B")
        print(f'  <img src="{name}.{fmt}" srcset="{name}.{fmt} 1x, {name}@2x.{fmt} 2x" width="{_px(wdp)}" height="{_px(hdp)}" alt="">')
# ---- app launcher icons

ICON_RULES = ("Square 1:1 canvas. FULL-BLEED: the background fills the entire square edge to edge; no rounded corners, "
              "no border or frame, no drop shadow, no device, no mockup, no UI elements. One bold, simple central subject "
              "with a strong silhouette that still reads at 48 px, centred and about 60% of the canvas wide so circular "
              "masks never cut it, on a solid or smooth-gradient background with little or no texture. {text}")
ICON_REF_PROMPT = ("The attached image is an APPROVED app design, given only as a style reference. Design a NEW launcher "
                   "icon for this app in the SAME visual language: its palette ({colors}), rendering style, lighting and "
                   "mood. Do NOT reproduce the screenshot or any illustration or scene in it (no landscape, horizon, "
                   "water or card); invent a simple icon composition. Brief: {brief}. Direction: {direction}. ")
ICON_TXT_PROMPT = "App launcher icon. Brief: {brief}. Direction: {direction}. "
ICON_DIRECTIONS = ["one bold emblem of the core motif", "a minimal geometric mark with generous negative space",
                   "the core motif with one small supporting element", "the core motif inside a simple badge shape"]
ICON_FG_PROMPT = ("The attached image is an APPROVED app icon. Keep ONLY its foreground subject ({subject}) exactly as "
                  "drawn: same shapes, colours, highlights and glow, same size and position on the same square canvas. "
                  "Remove the background completely: transparent background (real alpha), no backdrop, no tile, no "
                  "shadow plate, no text.")
ICON_BG_PROMPT = ("The attached image is an APPROVED app icon. Keep ONLY its background, exactly as drawn: same colours, "
                  "gradient, texture and lighting on the same square canvas. Remove {subject} completely and continue the "
                  "background naturally where it was. Nothing else in the image.")
ICON_GLYPH_PROMPT = """The image is the foreground of an APPROVED app icon. Write it as a single-colour glyph for Android
13 themed icons (the system tints it; only the shape matters): one simplified, bold silhouette of the subject that keeps
its recognisable outline and one or two defining inner cut-outs, readable at 24 px. Rules:
- viewBox "0 0 108 108"; every shape inside the circle centred at (54,54) with radius 30.
- Only path, circle, ellipse, rect, polygon elements; all filled with #000000 (fill-rule evenodd allowed for
  cut-outs); no stroke, no gradient, no opacity, no transform, no text, no background shape.
Return JSON {"svg": "<svg ...>...</svg>"}."""


def _hex(c):
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in c[:3])


def _palette(img, k=6, min_de=20):
    """Main colours by area, skipping near-duplicates so accents are not crowded out by shades of the backdrop."""
    q = img.convert("RGB").resize((256, 256)).quantize(32, method=Image.Quantize.FASTOCTREE)
    pal = q.getpalette()
    picked = []
    for _, i in sorted(q.getcolors(), reverse=True):
        c = pal[i * 3:i * 3 + 3]
        if all(np.linalg.norm(_lab(c) - _lab(p)) >= min_de for p in picked):
            picked.append(c)
    return [_hex(c) for c in picked[:k]]




def _palette_de(img, ref, k=12):
    """Mean CIELAB distance from each icon pixel to the nearest of the reference's k main colours."""
    q = ref.convert("RGB").resize((160, 240)).quantize(k)
    pal = _lab(np.array(q.getpalette()[:k * 3], float).reshape(-1, 3))
    px = _lab(np.asarray(img.convert("RGB").resize((96, 96)), float).reshape(-1, 3))
    return float(np.sqrt(((px[:, None] - pal[None]) ** 2).sum(-1)).min(1).mean())


def _bg_feats(size, span=1.0):
    yy, xx = np.clip(np.mgrid[0:size, 0:size] / (size - 1) * 2 * span - span, -1, 1)
    return np.stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy, np.sqrt(xx ** 2 + yy ** 2)], -1)


def _bg_fit(img, S=192, ring=0.07, thr=26):
    """Smooth (quadratic + radial) background model of an opaque icon: returns (subject mask, bg residual rms, coef)."""
    a = np.asarray(img.convert("RGB").resize((S, S), Image.LANCZOS), float).reshape(-1, 3)
    F = _bg_feats(S)
    use = (np.maximum(abs(F[..., 1]), abs(F[..., 2])) > 1 - 2 * ring).reshape(-1)
    F = F.reshape(-1, 7)
    for _ in range(4):
        coef = np.linalg.lstsq(F[use], a[use], rcond=None)[0]
        dist = np.linalg.norm(a - F @ coef, axis=1)
        use = dist < thr
    resid = float(np.sqrt((dist[use] ** 2).mean())) if use.any() else 99.0
    return (~use).reshape(S, S), resid, coef


def _bg_render(coef, size, span=1.0):
    """The fitted background at `size` px; span > 1 extends it past the concept by clamping (edge continuation).
    Same values as _bg_feats(size, span) @ coef, evaluated separably (no size x size x 7 feature array)."""
    t = np.clip(np.arange(size) / (size - 1) * 2 * span - span, -1, 1)
    x, y = t[None, :], t[:, None]
    terms = (1.0, x, y, x * x, y * y, x * y, np.sqrt(x * x + y * y))
    rgb = np.stack([sum(c[k] * terms[k] for k in range(7)) for c in np.asarray(coef).T], -1)
    rgb = np.clip(np.broadcast_to(rgb, (size, size, 3)), 0, 255).round().astype(np.uint8)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def _radius(mask, q=99.5):
    """Extent of a centred subject: q-th percentile distance of its pixels from the centre, as a fraction of half-side."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    h, w = mask.shape
    d = np.sqrt(((xs + 0.5) / w * 2 - 1) ** 2 + ((ys + 0.5) / h * 2 - 1) ** 2)
    return float(np.percentile(d, q))


def _mask(shape, T, ss=4):
    S = T * ss
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    if shape == "circle":
        d.ellipse([0, 0, S - 1, S - 1], fill=255)
    elif shape == "squircle":
        t = np.linspace(-1, 1, S)
        m = Image.fromarray(((abs(t[None]) ** 5 + abs(t[:, None]) ** 5) <= 1).astype(np.uint8) * 255)
    elif shape == "rounded":
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
    elif shape == "teardrop":
        d.ellipse([0, 0, S - 1, S - 1], fill=255)
        d.rounded_rectangle([S // 2, S // 2, S - 1, S - 1], radius=int(S * 0.1), fill=255)
    else:
        d.rectangle([0, 0, S - 1, S - 1], fill=255)
    return m.resize((T, T), Image.LANCZOS)


MASKS = ("circle", "squircle", "rounded", "teardrop")
BUSY_RESID = 9.5


def _masked(img, shape, T):
    im = img.convert("RGBA").resize((T, T), Image.LANCZOS)
    im.putalpha(ImageChops.multiply(im.getchannel("A"), _mask(shape, T)))
    return im


def _icon_rows(label, img, T=160, small=48, wall=("#1c2340", "#e9e6f2")):
    """Review row: full-bleed art, each mask large, then each mask at `small` px (+2x zoom) on dark/light wallpaper."""
    n = len(MASKS) + 1
    pw = len(MASKS) * (small + 8) + 8
    row = Image.new("RGB", (T * n + 2 * (pw + 10) + 10, T + 18), "white")
    ImageDraw.Draw(row).text((4, 2), label[:140], fill="black")
    row.paste(img.convert("RGB").resize((T, T), Image.LANCZOS), (0, 18))
    for i, s in enumerate(MASKS):
        tile = checker(T, T, 10)
        m = _masked(img, s, T - 8)
        tile.paste(m, (4, 4), m)
        row.paste(tile, ((i + 1) * T, 18))
    for j, col in enumerate(wall):
        panel = Image.new("RGB", (pw, T), col)
        for i, s in enumerate(MASKS):
            ic = _masked(img, s, small)
            panel.paste(ic, (8 + i * (small + 8), 6), ic)
        zoom = _masked(img, "circle", small).resize((small * 2, small * 2), Image.NEAREST)
        panel.paste(zoom, (8, small + 12), zoom)
        row.paste(panel, (T * n + 10 + j * (pw + 10), 18))
    return row


def _stack(rows, out):
    W = max(r.width for r in rows)
    im = Image.new("RGB", (W, sum(r.height + 6 for r in rows)), "white")
    y = 0
    for r in rows:
        im.paste(r, (0, y))
        y += r.height + 6
    im.save(out)
    return out


def _square_ref(img, long_side=1024):
    im = img.convert("RGB")
    side = max(im.size)
    bg = tuple(int(v) for v in np.median(np.asarray(im).reshape(-1, 3), 0))
    c = Image.new("RGB", (side, side), bg)
    c.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return c.resize((long_side, long_side), Image.LANCZOS)


def _icon_metrics(img, ref=None):
    fg, resid, _ = _bg_fit(img)
    edge = np.concatenate([fg[:2].ravel(), fg[-2:].ravel(), fg[:, :2].ravel(), fg[:, -2:].ravel()]).mean()
    m = {"subject_r": round(_radius(fg), 3), "bg_resid": round(resid, 1), "edge_touch": round(float(edge), 3),
         "flags": []}
    if ref is not None:
        m["palette_de"] = round(_palette_de(img, ref), 1)
        if _mean_diff(img.convert("RGB").resize((64, 64)), _square_ref(ref, 64)) < 10:
            m["flags"].append("COPY:returned-reference")
    if resid > BUSY_RESID:
        m["flags"].append("BUSY-BACKGROUND")
    elif m["subject_r"] > 66 / 72:
        m["flags"].append("SUBJECT-TOO-BIG")
    if edge > 0.25:
        m["flags"].append("BLEEDS-OFF-EDGE")
    return m


def _copied_illustration(img, mockup_path):
    """(name, fid) of the design illustration a concept's subject reproduces, or None (needs assets.json by the mockup)."""
    mp = Path(mockup_path)
    man = next((d / "assets.json" for d in (mp.parent, mp.parent.parent) if (d / "assets.json").exists()), None)
    if man is None:
        return None
    try:  # best-effort check: a hand-edited or older assets.json skips it instead of aborting every variant
        m = json.loads(man.read_text())
        mock = _rgba(man.parent / m["mockup"])
        arts = [(x["name"], b) for x in m["assets"] if x.get("format", "png") == "png"
                and x.get("kind", "illustration") not in ("background", "photo")
                and (b := _clip_bbox(x.get("bbox") or {}, *mock.size))]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"  ! {man} does not match the assets.json schema ({type(e).__name__}: {str(e)[:120]}); "
              "skipping the copied-illustration check")
        return None
    if not arts:
        return None
    fg, _, _ = _bg_fit(img)
    sub = img.convert("RGBA")
    sub.putalpha(Image.fromarray((np.clip(np.asarray(fg, float).squeeze(), 0, 1) * 255).astype(np.uint8)).resize(img.size))
    sub = trim(sub, pad=0.02)
    name, fid = min(((n, score(mock, b, sub, None, True)["fid"]) for n, b in arts), key=lambda t: t[1])
    return (name, round(fid, 1)) if fid < 35 else None


def cmd_appicon_concepts(a):
    ref = _open_image(a.mockup, "mockup", "--mockup") if a.mockup else None
    if a.crop:
        if ref is None:
            sys.exit("appicon concepts: --crop needs --mockup (it selects a region of the mockup)")
        try:
            x, y, w, h = (int(round(float(v))) for v in a.crop.split(","))
        except ValueError:
            sys.exit(f"appicon concepts: --crop must be x,y,w,h in mockup pixels, e.g. 40,120,320,320 (got {a.crop!r})")
        b = _clip_bbox({"x": x, "y": y, "w": w, "h": h}, *ref.size)
        if b is None or w < 1 or h < 1:
            sys.exit(f"appicon concepts: --crop {a.crop} has no area inside the {ref.width}x{ref.height} mockup")
        if (b["x"], b["y"], b["w"], b["h"]) != (x, y, w, h):
            print(f"  ! --crop {a.crop} clamped to the mockup: {b['x']},{b['y']},{b['w']},{b['h']}")
        ref = ref.crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
    out = Path(a.out)
    (out / "review").mkdir(parents=True, exist_ok=True)
    text = f'The only text is "{a.text}", large and legible.' if a.text else "No text, letters or numbers anywhere."
    dirs = [s.strip() for s in a.directions.split("|")] if a.directions else ICON_DIRECTIONS
    colors = a.colors or (", ".join(_palette(ref)) if ref is not None else "")
    _plan(a.variants, 0, image_model=a.model, edits=ref is not None)

    def one(i):
        direction = dirs[i % len(dirs)]
        head = ICON_REF_PROMPT.format(colors=colors, brief=a.brief, direction=direction) if ref is not None else \
            ICON_TXT_PROMPT.format(brief=a.brief, direction=direction)
        prompt = head + ICON_RULES.format(text=text)
        for _ in range(2):
            if ref is not None:
                img, meta = edit(_png_bytes(_square_ref(ref)), prompt, "1024x1024", "opaque", a.model, a.fidelity, task="concept")
            else:
                img, meta = generate(prompt, "1024x1024", "opaque", a.model)
            if _aspect_ok(img, "1024x1024"):
                break
        p = out / f"{a.name}_v{i + 1}.png"
        im = Image.open(io.BytesIO(img)).convert("RGB")
        side = min(im.size)
        im.crop(((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2)).save(p)
        return p, direction, prompt, meta
    report_path = out / "concepts.json"
    try:
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
    except ValueError as e:
        sys.exit(f"appicon concepts: {report_path} is not valid JSON ({e}); move it aside or fix it")
    rows, failed = {}, []
    ex = cf.ThreadPoolExecutor(min(a.jobs, a.variants))
    futs = {ex.submit(one, i): i for i in range(a.variants)}
    try:
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                p, direction, prompt, meta = f.result()
            except (ApiError, OSError) as e:  # one blocked/failed prompt must not cost the other variants
                failed.append(i + 1)
                print(f"  ! variant {i + 1} ({dirs[i % len(dirs)]}) failed: {str(e)[:400]}")
                continue
            img = _rgba(p)
            m = _icon_metrics(img, ref)
            copied = _copied_illustration(img.convert("RGB"), a.mockup) if a.mockup else None
            if copied:
                m["flags"].append(f"COPIED-ILLUSTRATION:{copied[0]}")
                m["copy_fid"] = copied[1]
            report[p.name] = {"path": str(p), "direction": direction, "prompt": prompt, "metrics": m,
                              "reference": a.mockup, "quality": meta.get("quality")}
            report_path.write_text(json.dumps(report, indent=2))
            print(f"{p}  {direction}\n    {m}")
            rows[i] = _icon_rows(f"{p.stem}: {direction}  r={m['subject_r']} bg_resid={m['bg_resid']} "
                                 f"palette_de={m.get('palette_de', '-')} {' '.join(m['flags'])}", img)
    except BaseException:  # Ctrl-C, no API key (SystemExit from a worker), ...: stop queued variants, keep the saved ones
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    if not rows:
        sys.exit(f"appicon concepts: all {a.variants} variants failed (see the messages above)")
    print("review sheet ->", _stack([rows[i] for i in sorted(rows)], out / "review" / f"concepts_{time.strftime('%H%M%S')}.png"))
    if failed:
        print(f"  ! {len(failed)} of {a.variants} variants failed ({', '.join(map(str, sorted(failed)))}); "
              "the sheet shows the rest")


def _min_alpha(C, B, floor=0.04):
    al = np.where(C > B, (C - B) / np.maximum(255 - B, 1), (B - C) / np.maximum(B, 1)).max(-1)
    return np.clip((al - floor) / (1 - floor), 0, 1)[..., None]


def _unmix(C, B, floor=0.04, zone=0.03, Bz=None):
    """Min-alpha matte of C over known background B (exact recomposite), limited to a zone around the subject vs Bz."""
    al = _min_alpha(C, B, floor)
    F = np.clip((C - (1 - al) * B) / np.maximum(al, 1e-3), 0, 255)
    fg = Image.fromarray(np.dstack([F, al * 255]).round().astype(np.uint8), "RGBA")
    if zone:
        A = fg.getchannel("A")
        n = 256
        az = al if Bz is None else _min_alpha(C, Bz, floor)
        core = np.asarray(Image.fromarray((az[..., 0] * 255).astype(np.uint8)).resize((n, n), Image.BOX)
                          .point(lambda v: 255 if v > 77 else 0)).copy()
        e = max(1, n // 50)
        core[:e], core[-e:], core[:, :e], core[:, -e:] = 0, 0, 0, 0
        z = Image.fromarray(core).filter(ImageFilter.MaxFilter(int(n * zone) * 2 + 1))
        z = z.resize(A.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(A.width * zone / 2))
        fg.putalpha(ImageChops.multiply(A, z))
    return fg


def _sep_metrics(fg, C, B):
    """Separation cleanliness: recomposite error on the subject zone, kept-backdrop and semi-transparent fractions."""
    al = np.asarray(fg.getchannel("A"), float) / 255
    F = np.asarray(fg.convert("RGB"), float)
    comp = F * al[..., None] + B * (1 - al[..., None])
    zone = np.asarray(Image.fromarray((al > 0.1).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(9))) > 0
    solid = al > 0.9
    kept = (np.abs(F - B).sum(-1) < 30) & solid
    return {"recomposite_diff": round(float(np.abs(comp - C).mean(-1)[zone].mean()), 2) if zone.any() else None,
            "kept_backdrop": round(float(kept.sum() / max(solid.sum(), 1)), 3),
            "semi_transparent": round(float(((al > 0.1) & (al < 0.9)).sum() / max((al > 0.1).sum(), 1)), 3)}


def _fit_vd_gradient(Bv):
    """Best 3-stop linear/radial VectorDrawable gradient (108 canvas) for a background sampled on the 72dp viewport."""
    n = Bv.shape[0]
    g = (np.arange(n) + 0.5) / n * 72 + 18
    X, Y = np.meshgrid(g, g)
    cols = Bv.reshape(-1, 3)
    stops = np.array([0, 0.5, 1])

    def fit(t):
        H = np.maximum(0, 1 - np.abs(np.clip(t, 0, 1).reshape(-1, 1) - stops) / 0.5)
        coef = np.clip(np.linalg.lstsq(H, cols, rcond=None)[0], 0, 255)
        return float(np.sqrt(((H @ coef - cols) ** 2).mean())), coef
    best = (1e9, None)
    corners = np.array([[0, 0], [108, 0], [0, 108], [108, 108]], float)
    for ang in range(0, 360, 10):
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        lo, hi = (corners @ u).min(), (corners @ u).max()
        err, coef = fit(((X * u[0] + Y * u[1]) - lo) / (hi - lo))
        if err < best[0]:
            c = 54 * u.sum()
            best = (err, {"type": "linear", "start": (54 + (lo - c) * u[0], 54 + (lo - c) * u[1]),
                          "end": (54 + (hi - c) * u[0], 54 + (hi - c) * u[1]), "colors": coef})
    for cx in range(18, 91, 8):
        for cy in range(18, 91, 8):
            for r in (40, 56, 72, 90, 110):
                err, coef = fit(np.hypot(X - cx, Y - cy) / r)
                if err < best[0]:
                    best = (err, {"type": "radial", "center": (cx, cy), "radius": r, "colors": coef})
    return best


def _vd_gradient_xml(spec):
    if spec["type"] == "linear":
        head = (f'android:type="linear" android:startX="{spec["start"][0]:.2f}" android:startY="{spec["start"][1]:.2f}" '
                f'android:endX="{spec["end"][0]:.2f}" android:endY="{spec["end"][1]:.2f}"')
    else:
        head = (f'android:type="radial" android:centerX="{spec["center"][0]}" android:centerY="{spec["center"][1]}" '
                f'android:gradientRadius="{spec["radius"]}"')
    items = "".join(f'\n                <item android:offset="{o:g}" android:color="{_hex(c)}" />'
                    for o, c in zip((0, 0.5, 1), spec["colors"]))
    return ('<?xml version="1.0" encoding="utf-8"?>\n<vector xmlns:android="http://schemas.android.com/apk/res/android"\n'
            '    xmlns:aapt="http://schemas.android.com/aapt"\n    android:width="108dp"\n    android:height="108dp"\n'
            '    android:viewportWidth="108"\n    android:viewportHeight="108">\n'
            '    <path android:pathData="M0,0h108v108h-108z">\n        <aapt:attr name="android:fillColor">\n'
            f'            <gradient {head}>{items}\n            </gradient>\n        </aapt:attr>\n    </path>\n</vector>\n')


def _vd_gradient_render(spec, size):
    g = (np.arange(size) + 0.5) / size * 108
    X, Y = np.meshgrid(g, g)
    if spec["type"] == "linear":
        (x0, y0), (x1, y1) = spec["start"], spec["end"]
        t = ((X - x0) * (x1 - x0) + (Y - y0) * (y1 - y0)) / max((x1 - x0) ** 2 + (y1 - y0) ** 2, 1e-6)
    else:
        t = np.hypot(X - spec["center"][0], Y - spec["center"][1]) / spec["radius"]
    t = np.clip(t, 0, 1)[..., None]
    c = spec["colors"]
    img = np.where(t < 0.5, c[0] + (c[1] - c[0]) * t * 2, c[1] + (c[2] - c[1]) * (t - 0.5) * 2)
    return Image.fromarray(img.round().astype(np.uint8), "RGB").convert("RGBA")


def _svg_raster(svg_text, work, size=432):
    """Alpha mask of a single-colour SVG (forced black, rendered on white by whatever renderer exists) or None."""
    black = re.sub(r'(fill|stroke)="(?!none)[^"]*"', r'\1="#000000"', svg_text)
    black = re.sub(r'\b(fill|stroke)\s*:\s*(?!none)[^;"}<]+', r'\1:#000000', black)  # style="" and <style> fills too
    p = Path(work) / "_glyph_probe.svg"
    p.write_text(black)
    png = _render_svg(p, work, size)
    p.unlink(missing_ok=True)
    if not png:
        return None
    im = Image.open(png).convert("RGBA")
    png.unlink(missing_ok=True)
    L = np.asarray(Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("L").resize((size, size)), float)
    return Image.fromarray((255 - L).astype(np.uint8), "L")


def _vd_group(xml, s, tx, ty):
    decl, tag, body = xml.split(">", 2)
    return (f'{decl}>{tag}>\n    <group android:scaleX="{s:.4f}" android:scaleY="{s:.4f}" android:translateX="{tx:.3f}" '
            f'android:translateY="{ty:.3f}">' + body.replace("</vector>", "").rstrip() + "\n    </group>\n</vector>\n")




def _aapt2_check(res, name, work):
    """aapt2 over the icon's res/ tree, linked against a manifest that references the launcher mipmaps (contract C4)."""
    manifest = ('<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.gptkit.iconcheck">\n'
                '  <uses-sdk android:minSdkVersion="21" android:targetSdkVersion="34" />\n'
                f'  <application android:icon="@mipmap/{name}" android:roundIcon="@mipmap/{name}_round" />\n</manifest>\n')
    return _aapt2_tree(res, manifest, work)


def _actool_check(iconset, work):
    if not shutil.which("xcrun") or subprocess.run(["xcrun", "--find", "actool"], capture_output=True).returncode:
        return "skipped (no Xcode actool)"
    cat = work / "Assets.xcassets"
    shutil.rmtree(cat, ignore_errors=True)
    shutil.copytree(iconset, cat / iconset.name)
    (cat / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}}))
    (work / "out").mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["xcrun", "actool", str(cat), "--compile", str(work / "out"), "--platform", "iphoneos",
                        "--minimum-deployment-target", "17.0", "--app-icon", iconset.stem, "--output-partial-info-plist",
                        str(work / "out" / "partial.plist"), "--warnings", "--errors", "--output-format",
                        "human-readable-text"], capture_output=True, text=True)
    msgs = list(dict.fromkeys(ln.split(": ", 1)[-1] for ln in (r.stdout + r.stderr).splitlines()
                              if ": warning:" in ln or ": error:" in ln))
    if r.returncode == 0 and not msgs:
        return "ok (Assets.car compiled)"
    return "; ".join(msgs)[:800] or f"failed (exit {r.returncode})"


def _save_densities(img, res, folder, name, dp):
    for dens, s in DENSITIES.items():
        out = res / f"{folder}-{dens}"
        out.mkdir(parents=True, exist_ok=True)
        img.resize((round(dp * s), round(dp * s)), Image.LANCZOS).save(out / f"{name}.webp", "WEBP", lossless=True)


def _tiles(items, T=160):
    row = Image.new("RGB", (sum(max(T, im.width) + 8 for _, im in items), T + 18), "white")
    d = ImageDraw.Draw(row)
    x = 0
    for label, im in items:
        w = max(T, im.width)
        tile = checker(w, T, 10)
        im = im.convert("RGBA")
        tile.paste(im, ((w - im.width) // 2, (T - im.height) // 2), im)
        row.paste(tile, (x, 18))
        d.text((x + 2, 2), label[:max(8, w // 6)], fill="black")
        x += w + 8
    return row


def _themed(mono, T, bg, tint):
    lay = mono.resize((T * 3 // 2, T * 3 // 2), Image.LANCZOS)
    c = Image.new("RGBA", lay.size, bg)
    c.paste(Image.new("RGBA", lay.size, tint), (0, 0), lay.getchannel("A"))
    v = c.crop((T // 4, T // 4, T // 4 + T, T // 4 + T))
    return _masked(v, "circle", T)


def _icon_background(a, concept, coef, resid, res, work, N, M):
    """Background layer on the 108dp canvas (M px) plus its resource reference and fit report."""
    vp = (M // 6, M // 6, M // 6 + N, M // 6 + N)
    model = _bg_render(coef, M, 1.5)
    grad_err, grad = _fit_vd_gradient(np.asarray(model.crop(vp).resize((64, 64)).convert("RGB"), float))
    spread = float(np.ptp(np.asarray(model.crop(vp).convert("RGB"), float).reshape(-1, 3), 0).max())
    kind = a.bg
    if kind == "auto":
        kind = "image" if resid > BUSY_RESID else "color" if spread < 6 else "gradient" if grad_err <= 3.5 else "model"
    info = {"kind": kind, "fit_resid": round(resid, 1), "vd_gradient_rms": round(grad_err, 2), "spread": round(spread, 1)}
    name = f"{a.name}_background"
    if kind == "color":
        color = a.bg_color or _hex(np.median(np.asarray(model.crop(vp).convert("RGB")).reshape(-1, 3), 0))
        (res / "values").mkdir(exist_ok=True)
        (res / "values" / f"{name}.xml").write_text(
            f'<?xml version="1.0" encoding="utf-8"?>\n<resources>\n    <color name="{name}">{color}</color>\n</resources>\n')
        return Image.new("RGBA", (M, M), color), f"@color/{name}", info
    if kind == "gradient":
        (res / "drawable").mkdir(exist_ok=True)
        xml = _vd_gradient_xml(grad)
        (res / "drawable" / f"{name}.xml").write_text(xml)
        lint = _vd_lint(xml)  # what aapt2 link would reject, checked offline (compile alone does not see it)
        if lint:
            info["note"] = f"{name}.xml: VectorDrawable lint: " + "; ".join(lint)[:300]
            for e in lint:
                print(f"  ! {name}.xml lint: {e}")
        return _vd_gradient_render(grad, M), f"@drawable/{name}", info
    bgm = model
    if kind == "image":
        try:
            raw, _ = edit(_png_bytes(concept), ICON_BG_PROMPT.format(subject=a.subject), "1024x1024", "opaque", a.model)
            (work / "bg_api_raw.png").write_bytes(raw)
            plate = np.asarray(_rgba(work / "bg_api_raw.png").convert("RGB").resize((N, N), Image.LANCZOS))
            p = (M - N) // 2
            bgm = Image.fromarray(np.pad(plate, ((p, M - N - p), (p, M - N - p), (0, 0)), mode="reflect")).convert("RGBA")
        except (ApiError, OSError) as e:  # the fitted bitmap is a usable background: keep building
            info.update(kind="model", note=f"background plate edit failed: {str(e)[:300]}; using the fitted bitmap")
            print(f"  ! {info['note']}")
    _save_densities(bgm, res, "mipmap", name, 108)
    return bgm, f"@mipmap/{name}", info


def _icon_glyph(a, concept, sil_mask, limit, res, work, M):
    """gpt-6 single-colour glyph, measured and fitted onto the silhouette's box inside the safe zone -> VectorDrawable."""
    svg = ask([ICON_GLYPH_PROMPT + f"\nThe subject is: {a.subject}.", concept, "Its foreground silhouette:",
               Image.merge("RGB", [sil_mask.point(lambda v: 255 - v)] * 3).resize((512, 512))], "json")["svg"]
    (work / "glyph_raw.svg").write_text(svg)
    ET.fromstring(svg)
    g_mask = _svg_raster(svg, work, 432)
    s, tx, ty, glyph = 1.0, 0.0, 0.0, None
    # the glyph is only placed (and later referenced) when its render is measurable: no renderer, an empty render
    # (e.g. every shape outside the 108 viewBox) or only faint anti-aliasing leave glyph=None -> the silhouette ships
    thr = g_mask.point(lambda v: 255 if v > 128 else 0).getbbox() if g_mask is not None else None
    info = {"note": ("no SVG renderer: placement not verified" if not _renderer_name() else
                     "glyph did not render (empty or unsupported by the renderer): placement not verified") if g_mask is None else
            "glyph renders empty inside the 108 viewBox" if not g_mask.getbbox() else
            "glyph too faint/thin to place reliably" if not thr else
            "empty foreground silhouette" if not sil_mask.getbbox() else ""}
    if not info["note"]:
        place = lambda s, tx, ty: g_mask.transform((432, 432), Image.AFFINE, (1 / s, 0, -tx * 4 / s, 0, 1 / s, -ty * 4 / s),
                                                   Image.BILINEAR)
        gb = [v / 4 for v in thr]
        fb = [v * 108 / M for v in sil_mask.getbbox()]
        s = min((fb[2] - fb[0]) / (gb[2] - gb[0]), (fb[3] - fb[1]) / (gb[3] - gb[1]))
        tx, ty = (fb[0] + fb[2] - s * (gb[0] + gb[2])) / 2, (fb[1] + fb[3] - s * (gb[1] + gb[3])) / 2
        r = _radius(np.asarray(place(s, tx, ty)) > 128, 99.9) * 54
        if r > limit:
            k = limit / r
            s, tx, ty = s * k, 54 + (tx - 54) * k, 54 + (ty - 54) * k
        placed = place(s, tx, ty)
        glyph = Image.new("RGBA", (432, 432), (255, 255, 255, 0))
        glyph.putalpha(placed)
        sm, pm = np.asarray(sil_mask.resize((432, 432))) > 128, np.asarray(placed) > 128
        info = {"iou_vs_silhouette": round(float((sm & pm).sum() / max((sm | pm).sum(), 1)), 3),
                "radius_dp": round(_radius(pm, 99.9) * 54, 2), "scale": round(s, 3)}
    vd = _vd_group(svg_to_vd(svg, 108), s, tx, ty)
    # the system uses only the layer's alpha: several colours or a gradient mean the glyph is not the single-colour mark
    cols = sorted(set(re.findall(r'android:(?:fill|stroke)Color="([^"]+)"', vd)) - {"#00000000"})
    info["colors"] = cols
    if len(cols) > 1 or "<gradient" in vd or "fillAlpha" in vd or "strokeAlpha" in vd:
        info["single_colour"] = False
    info["lint"] = _vd_lint(vd)
    for e in info["lint"]:
        print(f"  ! {a.name}_monochrome.xml lint: {e}")
    if glyph is not None:  # an unverified glyph never lands in res/ (nothing would reference it)
        (res / "drawable").mkdir(exist_ok=True)
        (res / "drawable" / f"{a.name}_monochrome.xml").write_text(vd)
    else:
        (work / "glyph_unverified_monochrome.xml").write_text(vd)
    (work / "glyph.svg").write_text(re.sub(r"(<svg\b[^>]*>)", rf'\1<g transform="translate({tx:.3f} {ty:.3f}) '
                                           rf'scale({s:.4f})">', svg, count=1).replace("</svg>", "</g></svg>"))
    return glyph, info


def cmd_appicon_build(a):
    if a.name != _res_name(a.name):
        sys.exit(f"appicon build: --name {a.name!r} is not a valid Android resource name ([a-z0-9_], starting with a "
                 f"letter, not a Java keyword); use --name {_res_name(a.name, 'ic_')}")
    if a.bg_color is not None:
        if not re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", a.bg_color):
            sys.exit(f"appicon build: --bg-color must be #RRGGBB or #RGB (got {a.bg_color!r})")
        a.bg_color = "#" + a.bg_color.lstrip("#").upper()
    src = _open_image(a.concept, "concept", "the square concept PNG to build from")
    out = Path(a.out)
    res, work = out / "android" / "res", out / "work"
    for p in (res, work, out / "review"):
        p.mkdir(parents=True, exist_ok=True)
    N, M = 1024, 1536
    vp = (M // 6, M // 6, M // 6 + N, M // 6 + N)
    alpha = ImageStat.Stat(src.getchannel("A"))
    concept = src.convert("RGB")
    side = min(concept.size)
    concept = concept.crop(((concept.width - side) // 2, (concept.height - side) // 2,
                            (concept.width + side) // 2, (concept.height + side) // 2)).resize((N, N), Image.LANCZOS)
    C = np.asarray(concept, float)
    _, resid, coef = _bg_fit(concept)
    report = {"concept": str(a.concept)}
    if alpha.mean[0] < 250 or alpha.stddev[0] > 8:
        msg = (f"concept has transparency (alpha mean {alpha.mean[0]:.0f}, std {alpha.stddev[0]:.0f}): it is flattened onto "
               "black; appicon build expects the opaque full-bleed concept, not a foreground or background layer")
        print(f"  ! {msg}")
        report["warnings"] = [msg]

    # background first: the foreground is matted against exactly the background layer that ships
    bgm, bg_ref, report["background"] = _icon_background(a, concept, coef, resid, res, work, N, M)
    report["background"]["resource"] = bg_ref
    if report["background"].get("note"):
        report.setdefault("warnings", []).append(report["background"]["note"])
    B = np.asarray(bgm.crop(vp).convert("RGB"), float)
    fg = _unmix(C, B, Bz=np.asarray(_bg_render(coef, N).convert("RGB"), float))
    if resid > BUSY_RESID and report["background"]["kind"] != "image" and not report["background"].get("note"):
        report.setdefault("warnings", []).append(f"busy background (fit residual {resid:.1f} > {BUSY_RESID}): scenery ends up in the "
                              "foreground; use --bg image")
    report["separation"] = {"local": _sep_metrics(fg, C, B), "used": a.separate}
    semi = report["separation"]["local"]["semi_transparent"]
    if a.separate == "auto":
        a.separate = "api" if semi > 0.5 and _config()["key"] else "local"
        report["separation"]["used"] = a.separate
        if semi > 0.5:
            msg = (f"local matte left {semi:.0%} of the subject semi-transparent (subject close in colour to its "
                   "background): iOS dark/tinted, the silhouette and light backgrounds would look ghosted")
            print(f"  ! {msg}; " + ("separating with an image edit instead (1 call)" if a.separate == "api"
                                     else "set an API key or pass --separate api"))
            report.setdefault("warnings", []).append(msg)
    elif a.separate == "local" and semi > 0.5:
        report.setdefault("warnings", []).append(f"local matte {semi:.0%} semi-transparent; --separate api is cleaner here")
        print(f"  ! local matte left {semi:.0%} of the subject semi-transparent; --separate api (1 call) is cleaner here")
    if a.separate == "api":
        try:
            raw, _ = edit(_png_bytes(concept), ICON_FG_PROMPT.format(subject=a.subject), "1024x1024", "transparent", a.model)
            (work / "fg_api_raw.png").write_bytes(raw)
            api_fg = _rgba(work / "fg_api_raw.png").resize((N, N), Image.LANCZOS)
        except (ApiError, OSError) as e:  # the local matte is a usable foreground: keep building
            api_fg, msg = None, f"foreground edit failed: {str(e)[:300]}; using the local matte"
            print(f"  ! {msg}")
            report.setdefault("warnings", []).append(msg)
            report["separation"]["used"] = a.separate = "local"
        if api_fg is not None:
            target = Image.new("RGBA", (N, N), (0, 0, 0, 0))
            target.putalpha(fg.getchannel("A").point(lambda v: 255 if v > 128 else 0))
            fg = _register(api_fg, target)
            report["separation"]["api"] = _sep_metrics(fg, C, B)
    fg.save(work / "foreground_concept_aligned.png")

    # safe zone: the concept maps onto the 72dp viewport; shrink the subject only if it leaves the 66dp circle
    limit, shrink = 33 * a.fg_scale, 1.0
    for _ in range(3):
        fgm = Image.new("RGBA", (M, M), (0, 0, 0, 0))
        fs = fg.resize((round(N * shrink), round(N * shrink)), Image.LANCZOS)
        fgm.alpha_composite(fs, ((M - fs.width) // 2, (M - fs.height) // 2))
        Am = np.asarray(fgm.getchannel("A"))
        r = _radius(Am > 128, 99.9) * 54
        if r <= limit:
            break
        shrink *= limit / r * 0.995
    sz = {"shrink": round(shrink, 3), "subject_radius_dp": round(r, 2),
          "faint_radius_dp": round(_radius(Am > 20, 99.9) * 54, 2), "limit_dp": 33}
    report["safe_zone"] = {**sz, "ok": sz["subject_radius_dp"] <= 33.0}
    _save_densities(fgm, res, "mipmap", f"{a.name}_foreground", 108)

    # monochrome layer (Android 13 themed icons)
    sil_mask = fgm.getchannel("A").point(lambda v: 255 if v > 77 else 0).filter(ImageFilter.GaussianBlur(M / 400))
    sil_mask = sil_mask.point(lambda v: 255 if v > 127 else 0)
    sil = Image.new("RGBA", (M, M), (255, 255, 255, 0))
    sil.putalpha(sil_mask)
    monos, mono_ref = {"silhouette": sil}, None
    if a.mono in ("glyph", "both") and not _config()["key"]:
        report["glyph"] = {"ok": False, "note": "no API key: glyph skipped"}
        print("  ! no API key: monochrome glyph skipped, using the silhouette")
    elif a.mono in ("glyph", "both"):
        try:
            glyph, report["glyph"] = _icon_glyph(a, concept, sil_mask, limit, res, work, M)
            report["glyph"]["ok"] = glyph is not None
            if report["glyph"].get("lint"):
                report.setdefault("warnings", []).append(f"{a.name}_monochrome.xml: VectorDrawable lint: "
                                                         + "; ".join(report["glyph"]["lint"])[:300])
            if glyph is not None:
                monos["glyph"] = glyph
                mono_ref = f"@drawable/{a.name}_monochrome"
                if report["glyph"].get("single_colour") is False:
                    msg = (f"monochrome glyph is not single-colour ({', '.join(report['glyph']['colors'])}, or gradient/"
                           "alpha): the themed icon uses only its alpha; check it on the review sheet")
                    print(f"  ! {msg}")
                    report.setdefault("warnings", []).append(msg)
            else:
                print(f"  ! glyph placement not verified/empty ({report['glyph']['note']}); using the silhouette")
        except (ApiError, ValueError, KeyError, TypeError, ET.ParseError) as e:  # the answer's SVG/JSON can be unusable
            report["glyph"] = {"ok": False, "note": f"failed: {str(e)[:200]}"}
            print(f"  ! glyph failed ({str(e)[:200]}); using the silhouette")
    if a.mono in ("silhouette", "both") or (a.mono == "glyph" and not mono_ref):
        _save_densities(sil, res, "mipmap", f"{a.name}_monochrome", 108)
        mono_ref = mono_ref or f"@mipmap/{a.name}_monochrome"
    report["monochrome"] = mono_ref
    (res / "mipmap-anydpi-v26").mkdir(exist_ok=True)
    mono_line = f'\n    <monochrome android:drawable="{mono_ref}" />' if mono_ref else ""
    xml = ('<?xml version="1.0" encoding="utf-8"?>\n<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
           f'    <background android:drawable="{bg_ref}" />\n    <foreground android:drawable="@mipmap/{a.name}_foreground" />'
           f'{mono_line}\n</adaptive-icon>\n')
    for n in (a.name, f"{a.name}_round"):
        (res / "mipmap-anydpi-v26" / f"{n}.xml").write_text(xml)

    # legacy 48dp launcher icons (2dp inset) and the Play Store 512
    comp = bgm.copy()
    comp.alpha_composite(fgm)
    view = comp.crop(vp)
    report["composite_diff_vs_concept"] = round(_mean_diff(view, concept), 2)
    for suffix, shape in (("", "rounded"), ("_round", "circle")):
        legacy = _fit(_masked(view, shape, 176), 192, 192)
        for dens, s in DENSITIES.items():
            legacy.resize((round(48 * s),) * 2, Image.LANCZOS).save(res / f"mipmap-{dens}" / f"{a.name}{suffix}.webp",
                                                                    "WEBP", lossless=True)
    play = out / "android" / "play_store_512.png"
    view.convert("RGB").resize((512, 512), Image.LANCZOS).save(play)  # Play: 512 px, opaque 32-bit PNG, <= 1024 KB
    report["play_store"] = {"size": [512, 512], "opaque": True, "kb": round(play.stat().st_size / 1024, 1)}
    if play.stat().st_size > 1024 * 1024:
        report.setdefault("warnings", []).append(f"{play.name} is {report['play_store']['kb']} KB > Play's 1024 KB limit")
        print(f"  ! {report['warnings'][-1]}")

    # iOS: one opaque 1024 (the concept itself unless --ios-from composite) + dark / tinted appearances.
    # Dark: the foreground on transparency (the system draws the dark backing). Tinted: the foreground's luminance as
    # grayscale WITH its alpha (the system tints it and supplies the backing), no baked black square. Per Apple's asset
    # catalog format and HIG (iOS 18); NOT verified with actool here, and actool does not check alpha anyway.
    ios = out / "ios" / "AppIcon.appiconset"
    ios.mkdir(parents=True, exist_ok=True)
    ios_src = concept if a.ios_from == "concept" else view
    discarded = alpha.extrema[0][0] < 255 if a.ios_from == "concept" else view.getchannel("A").getextrema()[0] < 255
    ios_img = ios_src.convert("RGB")
    ios_img.save(ios / "AppIcon-1024.png")
    fg.save(ios / "AppIcon-1024-dark.png")
    tinted = Image.merge("LA", (fg.convert("L"), fg.getchannel("A")))
    tinted.save(ios / "AppIcon-1024-tinted.png")
    entry = lambda fn, **ap: {**({"appearances": [ap]} if ap else {}), "filename": fn, "idiom": "universal",
                              "platform": "ios", "size": "1024x1024"}
    (ios / "Contents.json").write_text(json.dumps({"images": [
        entry("AppIcon-1024.png"), entry("AppIcon-1024-dark.png", appearance="luminosity", value="dark"),
        entry("AppIcon-1024-tinted.png", appearance="luminosity", value="tinted")],
        "info": {"author": "xcode", "version": 1}}, indent=2))
    with Image.open(ios / "AppIcon-1024.png") as chk, Image.open(ios / "AppIcon-1024-tinted.png") as tc, \
            Image.open(ios / "AppIcon-1024-dark.png") as dc:
        ta, da = tc.getchannel("A").getextrema(), dc.getchannel("A").getextrema()
        report["ios"] = {"size": chk.size, "opaque": chk.mode == "RGB" and "transparency" not in chk.info,
                         "had_alpha_discarded": bool(discarded), "tinted_mode": tc.mode,
                         "tinted_has_alpha": ta[0] < 255 and ta[1] > 0, "dark_has_alpha": da[0] < 255,
                         "actool": _actool_check(ios, work / "actool") if a.validate else "skipped"}

    # web: favicon.ico, apple-touch-icon, PWA any + maskable (subject inside the central 80% circle)
    web = out / "web"
    web.mkdir(exist_ok=True)
    rounded = _masked(view, "rounded", 512)
    ico = [rounded.resize((s, s), Image.LANCZOS) for s in (16, 32, 48)]
    ico[2].save(web / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)], append_images=ico[:2])
    ios_img.resize((180, 180), Image.LANCZOS).save(web / "apple-touch-icon.png")
    for s in (192, 512):
        rounded.resize((s, s), Image.LANCZOS).save(web / f"icon-{s}.png")
    k = round(M * 66 / 0.8 / 108 / 2)
    mbox = (M // 2 - k, M // 2 - k, M // 2 + k, M // 2 + k)
    maskable = comp.crop(mbox).convert("RGB").resize((512, 512), Image.LANCZOS)
    maskable.save(web / "icon-maskable-512.png")
    (web / "manifest-icons.json").write_text(json.dumps({"icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}]}, indent=2))
    mr = _radius(np.asarray(fgm.crop(mbox).getchannel("A")) > 128, 99.9)
    report["web_maskable"] = {"subject_r": round(mr, 3), "ok": mr <= 0.8}
    report["aapt2"] = _aapt2_check(res, a.name, work / "aapt2") if a.validate else "skipped"

    guide = fgm.resize((200, 200), Image.LANCZOS)
    for dp, col in ((33, "#34c759"), (36, "#ff9500")):
        ImageDraw.Draw(guide).ellipse([100 - dp * 100 / 54, 100 - dp * 100 / 54, 100 + dp * 100 / 54, 100 + dp * 100 / 54],
                                      outline=col, width=2)
    on = lambda im, col, T: Image.alpha_composite(Image.new("RGBA", im.size, col), im.convert("RGBA")).resize((T, T), Image.LANCZOS)
    mono_tiles = [(f"{k_} {t}", _themed(v, T, bgc, tint)) for t, T, bgc, tint in
                  (("light", 140, "#d9e2ff", "#1b2c5b"), ("dark", 140, "#2d3142", "#c4d0ff"), ("48px", 48, "#d9e2ff", "#1b2c5b"))
                  for k_, v in monos.items()]
    maskable_prev = maskable.resize((180, 180), Image.LANCZOS)
    ImageDraw.Draw(maskable_prev).ellipse([18, 18, 162, 162], outline="#34c759", width=2)
    favs = Image.new("RGBA", (200, 130), "white")
    for im, xy in ((ico[0], (4, 4)), (ico[1], (28, 4)), (ico[2], (68, 4)),
                   (ico[0].resize((64, 64), Image.NEAREST), (4, 60)), (ico[1].resize((64, 64), Image.NEAREST), (76, 60))):
        favs.alpha_composite(im, xy)
    rows = [_icon_rows(f"adaptive icon (72dp viewport)  bg={report['background']['kind']} sep={a.separate} "
                       f"shrink={shrink:.2f} subject r={sz['subject_radius_dp']}dp (<=33)  "
                       f"composite diff={report['composite_diff_vs_concept']}", view),
            _tiles([("fg, 66dp safe + 72dp", guide), ("fg on black", on(fgm, "black", 200)),
                    ("background", bgm.resize((200, 200), Image.LANCZOS))] + mono_tiles, T=200),
            _tiles([(f"legacy {s}", Image.open(res / f"mipmap-{d}" / f"{a.name}.webp")) for d, s in
                    (("mdpi", 48), ("xhdpi", 96), ("xxxhdpi", 192))] +
                   [("round 192", Image.open(res / "mipmap-xxxhdpi" / f"{a.name}_round.webp")),
                    ("iOS (squircle)", _masked(ios_img, "squircle", 180)),
                    ("iOS dark", _masked(on(fg, "#1c1c1e", 1024), "squircle", 180)),
                    ("iOS tinted", _masked(on(Image.open(ios / "AppIcon-1024-tinted.png"), "#1c1c1e", 1024), "squircle",
                                           180)),
                    ("maskable, 80% circle", maskable_prev), ("favicon 16/32/48", favs)], T=200)]
    sheet_path = _stack(rows, out / "review" / f"appicon_{time.strftime('%H%M%S')}.png")
    (out / "appicon.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {res}\n-> {ios}\n-> {web}\nreview sheet -> {sheet_path}")


def cmd_pick(a):
    d, m, mock = _load(a.dir)
    asset = _asset(m, a.name)
    report = _read_report(d)
    entry = _entry(d, report, a.name)
    cands = entry.get("candidates") or []
    if Path(a.candidate).is_file():
        src, cand_id = Path(a.candidate), None
    else:
        if not a.candidate.isdigit():
            sys.exit(f"pick: candidate must be the cN number from the review sheet or an image file path, got {a.candidate!r}")
        n = int(a.candidate)
        src = _rep_abs(d, next((c["path"] for c in cands if c.get("cand") == n), f"raw/{a.name}_c{n}.png"))
        if not src.exists():
            have = {c["cand"] for c in cands if isinstance(c.get("cand"), int)}
            have |= {int(q.stem.rsplit("_c", 1)[1]) for q in (d / "raw").glob(f"{a.name}_c*.png") if q.stem.rsplit("_c", 1)[1].isdigit()}
            sys.exit(f"pick: candidate {n} not found: {src} (have: {', '.join(f'c{i}' for i in sorted(have)) or 'none'})")
        cand_id = n
    img = _open_image(src, "candidate")
    sc = score(mock, asset["bbox"], img, None, asset["transparent"])
    bad = _unusable(sc)
    if bad == "EMPTY":
        sys.exit(f"pick: {src} has no content (EMPTY): not written")
    if bad:
        print(f"  ! {src} is flagged {bad}; kept because you picked it")
    chosen = {"cand": cand_id if cand_id else _rep_rel(d, src), "path": _rep_rel(d, src), "meta": {}, "score": sc}
    if cand_id is None or not any(c.get("cand") == cand_id for c in cands):
        cands.append(chosen)
    entry.update(chosen=chosen, candidates=cands, picked_manually=True)
    entry["color_match"] = cm = _color_final(mock, asset, src, sc, a.color_match, d)
    if cm:
        print(f"colour match {cm['strength']}: fid {cm['fid_before']} -> {cm['fid_after']} ({'kept' if cm['applied'] else 'discarded'})")
    final = _finalize(asset, _rgba(_current(d, a.name, entry)))
    out = d / "assets" / f"{a.name}.png"
    final.save(out)
    entry.update(final=_rep_rel(d, out), final_size=list(final.size), needs_attention=bool(sc["flags"]), complete=True)
    entry.pop("unusable", None)
    _update_report(d, {a.name: entry})
    print(f"{a.name} <- {src} (fid={sc['fid']} flags={' '.join(sc['flags']) or '-'})")


# ---- raster -> vector tracing, part rigs, procedural keyframes

_CHROME_FAILED_REASON = None  # why the last headless capture produced nothing (printed once per process)


def _tr_chrome_reason():
    """One line saying why fidelity was not measured (no Chrome, or Chrome started and produced no screenshot)."""
    return _CHROME_FAILED_REASON or ("no Chrome/Chromium found (set GPTKIT_CHROME)" if not _chrome()
                                     else "headless Chrome produced no screenshot")


def chrome_shot(html, w, h, budget=3000):
    """Render an HTML string in headless Chrome at w x h CSS px on a transparent background -> RGBA image or None."""
    global _CHROME_FAILED_REASON
    import tempfile
    exe = _chrome()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        page, png = Path(td) / "page.html", Path(td) / "shot.png"
        page.write_text(html)
        try:
            r = subprocess.run([exe, *_chrome_flags(w, h), f"--user-data-dir={td}/prof", f"--virtual-time-budget={budget}",
                                f"--screenshot={png}", page.as_uri()], capture_output=True, text=True, timeout=180)
            reason = None if png.exists() else _chrome_err(r)
        except (subprocess.TimeoutExpired, OSError) as e:
            reason = f"{exe}: {type(e).__name__}"
        if reason is None:
            return Image.open(png).convert("RGBA").crop((0, 0, w, h))  # the window is padded (C1); the page is not
    if _CHROME_FAILED_REASON is None:
        print(f"  ! headless Chrome produced no screenshot: {reason}")
    _CHROME_FAILED_REASON = reason
    return None


def _tr_render_svgs(svgs, sizes):
    """Rasterise SVG strings at the given (w, h) with real alpha, batched into few Chrome calls. None if no Chrome."""
    out, batch, bw = [], [], 0
    for s, (w, h) in list(zip(svgs, sizes)) + [(None, (10 ** 9, 0))]:
        if batch and (s is None or bw + w > 8000):
            H = max(h for _, (_, h) in batch)
            x, tags = 0, []
            for bs, (w_, h_) in batch:
                tags.append(f'<img src="data:image/svg+xml;base64,{base64.b64encode(bs.encode()).decode()}" '
                            f'style="position:absolute;left:{x}px;top:0;width:{w_}px;height:{h_}px">')
                x += w_
            strip = chrome_shot(f'<!doctype html><body style="margin:0;background:transparent">{"".join(tags)}</body>', x, H)
            if strip is None:
                return None
            x = 0
            for _, (w_, h_) in batch:
                out.append(strip.crop((x, 0, x + w_, h_)))
                x += w_
            batch, bw = [], 0
        if s is not None:
            batch.append((s, (w, h)))
            bw += w
    return out


def trace_metrics(src, ren, display=None):
    """Fidelity of a rendered vector vs its raster source: grey-composited mean diff, alpha MAE, silhouette IoU."""
    def m(a, b):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        comp = lambda x: x[..., :3] * (x[..., 3:] / 255) + 128 * (1 - x[..., 3:] / 255)
        ma, mb = a[..., 3] > 127, b[..., 3] > 127
        ga, gb = (np.abs(np.diff(comp(x).mean(2), axis=0))[:, :-1] + np.abs(np.diff(comp(x).mean(2), axis=1))[:-1]
                  for x in (a, b))
        return {"diff": round(float(np.abs(comp(a) - comp(b)).mean()), 2),
                "edge_err": round(float(np.abs(ga - gb).mean()), 2),
                "alpha_mae": round(float(np.abs(a[..., 3] - b[..., 3]).mean()), 2),
                "iou": round(float((ma & mb).sum() / max(1, (ma | mb).sum())), 4)}
    res = m(src, ren.resize(src.size, Image.LANCZOS))
    if display:
        s = display / max(src.size)
        ds = (max(1, round(src.width * s)), max(1, round(src.height * s)))
        res["display"] = {"px": ds, **m(src.resize(ds, Image.LANCZOS), ren.resize(ds, Image.LANCZOS))}
    return res


def _label(mask):
    """4-connected components of a boolean mask -> (int32 labels, 0 = background, count)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab.astype(np.int32), int(n)
    except ImportError:
        pass
    H, W = mask.shape
    parent = [0]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    rows, prev = [], []
    for y in range(H):
        dd = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        cur, j = [], 0
        for s, e in zip(np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]):
            lid = None
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                r = find(prev[k][2])
                if lid is None:
                    lid = r
                elif r != lid:
                    parent[max(r, lid)] = min(r, lid)
                    lid = min(r, lid)
                k += 1
            if lid is None:
                lid = len(parent)
                parent.append(lid)
            cur.append((s, e, lid))
        rows.append(cur)
        prev = cur
    roots = {}
    lab = np.zeros((H, W), np.int32)
    for y, cur in enumerate(rows):
        for s, e, lid in cur:
            lab[y, s:e] = roots.setdefault(find(lid), len(roots) + 1)
    return lab, len(roots)


def _fill_holes(mask):
    lab, n = _label(~mask)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return mask | ((lab > 0) & ~np.isin(lab, list(edge)))


def _dilate(mask, r=1):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * r + 1))) > 127


def _blur(arr, sigma):
    out = np.asarray(arr, np.float32)
    if sigma <= 0:
        return out
    r = max(1, int(3 * sigma + 0.5))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for ax in (0, 1):
        p = np.pad(out, [(r, r) if i == ax else (0, 0) for i in range(out.ndim)], mode="edge")
        n = out.shape[ax]
        out = sum(k[i] * np.take(p, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1))
    return out.astype(np.float32)


def _to_lab(rgb):
    c = np.asarray(rgb, np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _kmeans(X, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(len(X), 20000), replace=False)] if len(X) > 20000 else X
    k = max(1, min(k, len(np.unique(S.round(1), axis=0))))
    C = [S[rng.integers(len(S))]]
    d2 = ((S - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        C.append(S[rng.choice(len(S), p=d2 / d2.sum())] if d2.sum() > 0 else S[rng.integers(len(S))])
        d2 = np.minimum(d2, ((S - C[-1]) ** 2).sum(1))
    C = np.array(C, np.float32)
    for _ in range(iters):
        lab = ((S[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = S[lab == j].mean(0)
    out = np.empty(len(X), np.int32)
    for i in range(0, len(X), 50000):
        out[i:i + 50000] = ((X[i:i + 50000, None] - C[None]) ** 2).sum(2).argmin(1)
    return C, out


# marching-squares edge pairs per case (corners tl=8 tr=4 br=2 bl=1; edges T R B L); saddles resolved by the centre
_MS = {1: ["LB"], 2: ["BR"], 3: ["LR"], 4: ["TR"], 6: ["TB"], 7: ["TL"], 8: ["TL"], 9: ["TB"], 11: ["TR"],
       12: ["LR"], 13: ["BR"], 14: ["LB"]}


def _contours(field, level=0.5):
    """Closed iso-contours of a 2-D field as (n, 2) float arrays in pixel-centre coordinates."""
    f = np.pad(np.asarray(field, np.float32), 1, constant_values=min(0.0, level - 1))
    b = f >= level
    case = b[:-1, :-1] * 8 + b[:-1, 1:] * 4 + b[1:, 1:] * 2 + b[1:, :-1] * 1
    ys, xs = np.nonzero((case > 0) & (case < 15))
    pts, adj = {}, {}

    def key(e, y, x):
        k = {"T": ("h", y, x), "B": ("h", y + 1, x), "L": ("v", y, x), "R": ("v", y, x + 1)}[e]
        if k not in pts:
            _, yy, xx = k
            a0 = f[yy, xx]
            a1 = f[yy, xx + 1] if k[0] == "h" else f[yy + 1, xx]
            t = 0.5 if a1 == a0 else min(1.0, max(0.0, (level - a0) / (a1 - a0)))
            pts[k] = (xx + t - 0.5, yy - 0.5) if k[0] == "h" else (xx - 0.5, yy + t - 0.5)
        return k
    for y, x, c in zip(ys.tolist(), xs.tolist(), case[ys, xs].tolist()):
        if c in (5, 10):
            centre = f[y:y + 2, x:x + 2].mean() >= level
            segs = ["TL", "BR"] if (c == 5) == centre else ["TR", "LB"]
        else:
            segs = _MS[c]
        for s in segs:
            k1, k2 = key(s[0], y, x), key(s[1], y, x)
            adj.setdefault(k1, []).append(k2)
            adj.setdefault(k2, []).append(k1)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = next((n for n in adj[cur] if n != prev and n not in seen), None)
            if nxt is None:
                break
            seen.add(nxt)
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.array([pts[k] for k in loop], np.float64))
    return loops


def _area(P):
    return 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


def _inside(p, P):
    x, y = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = (y > p[1]) != (y2 > p[1])
    xi = x + (p[1] - y) * (x2 - x) / np.where(y2 == y, 1e-12, y2 - y)
    return bool(np.count_nonzero(cross & (p[0] < xi)) % 2)


def _orient(loops):
    """Wind outer loops one way and holes the other so the nonzero fill rule (VD/Lottie default) is correct."""
    out = []
    for i, P in enumerate(loops):
        depth = sum(_inside(P[0], Q) for j, Q in enumerate(loops) if j != i and abs(_area(Q)) > abs(_area(P)))
        want_pos = depth % 2 == 0
        out.append(P if (_area(P) > 0) == want_pos else P[::-1].copy())
    return out


def _bez(b, t):
    mt = 1 - t
    return ((mt ** 3)[:, None] * b[0] + (3 * mt * mt * t)[:, None] * b[1] + (3 * mt * t * t)[:, None] * b[2]
            + (t ** 3)[:, None] * b[3])


def _gen_bezier(P, u, t1, t2):
    p0, p3 = P[0], P[-1]
    mt = 1 - u
    A1 = (3 * mt * mt * u)[:, None] * t1
    A2 = (3 * mt * u * u)[:, None] * t2
    c00, c01, c11 = (A1 * A1).sum(), (A1 * A2).sum(), (A2 * A2).sum()
    tmp = P - _bez(np.array([p0, p0, p3, p3]), u)
    x0, x1 = (A1 * tmp).sum(), (A2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = np.linalg.norm(p3 - p0)
    al, ar = ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det) if abs(det) > 1e-12 else (0, 0)
    if al < 1e-6 * seg or ar < 1e-6 * seg or al > 3 * seg or ar > 3 * seg:
        al = ar = seg / 3
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _fit_cubic(P, t1, t2, tol, depth=0):
    """Schneider's curve fitting: cubic Beziers through points P within `tol` px; t2 points back from the end."""
    n = len(P)
    seg = np.linalg.norm(P[-1] - P[0])
    if n <= 2 or depth > 14:
        return [np.array([P[0], P[0] + t1 * seg / 3, P[-1] + t2 * seg / 3, P[-1]])]
    u = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, n)
    b = _gen_bezier(P, u, t1, t2)
    for it in range(5):
        d = np.linalg.norm(_bez(b, u) - P, axis=1)
        split = int(d.argmax())
        if d[split] <= tol:
            return [b]
        if d[split] > 4 * tol or it == 4:
            break
        # Newton-Raphson reparameterisation
        q = _bez(b, u) - P
        d1 = 3 * ((1 - u) ** 2)[:, None] * (b[1] - b[0]) + (6 * (1 - u) * u)[:, None] * (b[2] - b[1]) + \
            (3 * u * u)[:, None] * (b[3] - b[2])
        d2 = (6 * (1 - u))[:, None] * (b[2] - 2 * b[1] + b[0]) + (6 * u)[:, None] * (b[3] - 2 * b[2] + b[1])
        den = (d1 * d1).sum(1) + (q * d2).sum(1)
        u = np.clip(u - np.where(np.abs(den) > 1e-12, (q * d1).sum(1) / np.where(den == 0, 1, den), 0), 0, 1)
        u[0], u[-1] = 0, 1
        u = np.maximum.accumulate(u)
        b = _gen_bezier(P, u, t1, t2)
    split = min(max(split, 1), n - 2)
    tc = P[split - 1] - P[split + 1]
    tc = tc / (np.linalg.norm(tc) or 1)
    return _fit_cubic(P[:split + 1], t1, tc, tol, depth + 1) + _fit_cubic(P[split:], -tc, t2, tol, depth + 1)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _fit_loop(P, tol=0.8, corner_deg=70, k=3):
    """Closed polyline -> list of cubic Beziers (4x2), keeping corners sharp and everything else smooth."""
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
    P = P[keep]
    n = len(P)
    if n < 4:
        return []
    k = min(k, max(1, n // 4))
    v1 = P - np.roll(P, k, axis=0)
    v2 = np.roll(P, -k, axis=0) - P
    cos = (v1 * v2).sum(1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    corners = [i for i in range(n) if ang[i] >= corner_deg and ang[i] == max(ang[(i + j) % n] for j in range(-k, k + 1))]
    if not corners:
        Q = np.vstack([P, P[:1]])
        t = _unit(P[k % n] - P[-k])
        return _fit_cubic(Q, t, -t, tol)
    out = []
    for ci, a in enumerate(corners):
        b_ = corners[(ci + 1) % len(corners)]
        idx = list(range(a, (b_ if b_ > a else b_ + n) + 1))
        seg = P[[i % n for i in idx]]
        if len(seg) < 2:
            continue
        kk = min(k, len(seg) - 1)
        out += _fit_cubic(seg, _unit(seg[kk] - seg[0]), _unit(seg[-1 - kk] - seg[-1]), tol)
    return out


def _tr_fmt(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else ("-0" if s == "-0.0" else s)


def _beziers_d(loops_bez, scale=1.0, dx=0.0, dy=0.0):
    """Beziers -> absolute M/L/C/Z path data (the command set build_lottie and svg_to_vd accept)."""
    out = []
    for bz in loops_bez:
        if not bz:
            continue
        P = lambda p: f"{_tr_fmt(p[0] * scale + dx)} {_tr_fmt(p[1] * scale + dy)}"
        s = [f"M{P(bz[0][0])}"]
        for b in bz:
            chord = b[3] - b[0]
            L = np.linalg.norm(chord)
            straight = L > 0 and all(abs(chord[0] * (c - b[0])[1] - chord[1] * (c - b[0])[0]) / L < 0.15
                                     for c in (b[1], b[2]))
            s.append(f"L{P(b[3])}" if straight else f"C{P(b[1])} {P(b[2])} {P(b[3])}")
        out.append(" ".join(s) + "Z")
    return " ".join(out)


def mask_to_path(mask, tol=0.8, corner_deg=70, sigma=0.7, min_area=4.0, field=None, dx=0.0, dy=0.0):
    """Binary mask (or a soft `field` in 0..1) -> smooth, correctly wound absolute path data, offset by (dx, dy)."""
    f = field if field is not None else _blur(mask.astype(np.float32), sigma)
    loops = [L for L in _contours(f, 0.5) if abs(_area(L)) >= min_area]
    return _beziers_d([_fit_loop(L, tol, corner_deg) for L in _orient(loops)], 1.0, dx, dy)


def _lmask(L, shape):
    """A traced layer's boolean mask on the full work canvas (stored cropped to its bbox: memory O(area), not O(canvas))."""
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = L["bbox"]
    m[y0:y1, x0:x1] = L["mask"]
    return m


def _lab_range(stops):
    """Largest colour distance (CIE76) between gradient stops: a gradient that spans < 6 is flat to the eye."""
    lab = _to_lab(np.array([c for _, c in stops], np.float32))
    return float(max(np.linalg.norm(lab[i] - lab[j]) for i in range(len(lab)) for j in range(i + 1, len(lab))))


def _profile_stops(s, C, s0, s1, nb=24, tol=3.0, max_stops=6):
    """1-D colour profile along s -> gradient stops [(offset, rgb)] simplified with RDP, plus per-pixel prediction."""
    span = max(s1 - s0, 1e-6)
    o = np.clip((s - s0) / span, 0, 1)
    bi = np.minimum((o * nb).astype(int), nb - 1)
    cnt = np.bincount(bi, minlength=nb)
    prof = np.stack([np.bincount(bi, C[:, c], nb) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    ok = cnt > 0
    xs = (np.arange(nb) + 0.5) / nb
    prof = np.stack([np.interp(xs, xs[ok], prof[ok, c]) for c in range(3)], 1)
    xs = np.r_[0, xs, 1]
    prof = np.vstack([prof[:1], prof, prof[-1:]])

    def rdp(i, j, t):
        if j <= i + 1:
            return []
        f = (xs[i + 1:j] - xs[i]) / (xs[j] - xs[i])
        interp = prof[i] + f[:, None] * (prof[j] - prof[i])
        e = np.linalg.norm(prof[i + 1:j] - interp, axis=1)
        m = int(e.argmax())
        return [] if e[m] <= t else rdp(i, i + 1 + m, t) + [i + 1 + m] + rdp(i + 1 + m, j, t)
    t = tol
    while True:
        keep = [0] + rdp(0, len(xs) - 1, t) + [len(xs) - 1]
        if len(keep) <= max_stops:
            break
        t *= 1.5
    stops = [(float(xs[i]), prof[i]) for i in keep]
    pred = np.stack([np.interp(o, [x for x, _ in stops], [c[ch] for _, c in stops]) for ch in range(3)], 1)
    return stops, pred


def fit_fill(P, C, kinds=("flat", "linear", "radial"), max_stops=6, tol=3.0):
    """Best SVG/VD-expressible fill for pixels at P (n,2) with colours C (n,3): flat, linear or circular radial."""
    rng = np.random.default_rng(0)
    if len(P) > 6000:
        idx = rng.choice(len(P), 6000, replace=False)
        P, C = P[idx], C[idx]
    C = C.astype(np.float64)
    mean = C.mean(0)
    rmse = lambda pred: float(np.sqrt(((pred - C) ** 2).sum(1).mean()))
    best = {"type": "flat", "color": mean, "rmse": rmse(mean[None])}
    if len(P) < 12 or best["rmse"] < 2.5:
        return best
    if "linear" in kinds:
        A = np.c_[np.ones(len(P)), P]
        coef = np.linalg.lstsq(A, C, rcond=None)[0]
        _, _, vt = np.linalg.svd(coef[1:].T)
        u = vt[0]
        s = P @ u
        s0, s1 = np.percentile(s, 1), np.percentile(s, 99)
        stops, pred = _profile_stops(s, C, s0, s1, tol=tol, max_stops=max_stops)
        r = rmse(pred)
        # a gradient must earn its API-24 dependency: a real colour span and a clear, not just relative, improvement
        if r < 0.8 * best["rmse"] and best["rmse"] - r > 2.0 and _lab_range(stops) > 6.0:
            best = {"type": "linear", "p0": u * s0, "p1": u * s1, "stops": stops, "rmse": r}
    if "radial" in kinds:
        lo, hi = P.min(0), P.max(0)
        ext = (hi - lo) * 0.5
        spanx, spany = (hi - lo + 2 * ext)
        c0 = lo - ext

        def ev(c):
            s = np.linalg.norm(P - c, axis=1)
            st, pred = _profile_stops(s, C, 0, np.percentile(s, 99), nb=16, tol=0.0, max_stops=18)
            return rmse(pred)
        grid = [c0 + np.array([spanx * i / 8, spany * j / 8]) for i in range(9) for j in range(9)]
        scores = [(ev(c), tuple(c)) for c in grid]
        r_best, c_best = min(scores)
        step = np.array([spanx, spany]) / 8
        for _ in range(3):
            step = step / 2
            cs = [np.array(c_best) + step * np.array([i, j]) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            r_best, c_best = min([(ev(c), tuple(c)) for c in cs] + [(r_best, c_best)])
        c = np.array(c_best)
        s = np.linalg.norm(P - c, axis=1)
        R = float(np.percentile(s, 99))
        stops, pred = _profile_stops(s, C, 0, R, tol=tol, max_stops=max_stops)
        r = rmse(pred)
        if r < 0.9 * best["rmse"] and best["rmse"] - r > 2.0 and R > 1 and _lab_range(stops) > 6.0:
            best = {"type": "radial", "c": c, "r": R, "stops": stops, "rmse": r}
    return best


def _tr_hex(c, alpha=None):
    c = np.clip(np.round(c), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c[:3])


def _fill_svg(fill, gid, scale=1.0):
    """fill spec -> (fill attribute, <defs> entry)."""
    if fill["type"] == "flat":
        return _tr_hex(fill["color"]), ""
    stops = "".join(f'<stop offset="{o:.3f}" stop-color="{_tr_hex(c)}"/>' for o, c in fill["stops"])
    if fill["type"] == "linear":
        (x1, y1), (x2, y2) = fill["p0"] * scale, fill["p1"] * scale
        return f"url(#{gid})", (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{_tr_fmt(x1)}" '
                                f'y1="{_tr_fmt(y1)}" x2="{_tr_fmt(x2)}" y2="{_tr_fmt(y2)}">{stops}</linearGradient>')
    cx, cy = fill["c"] * scale
    return f"url(#{gid})", (f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{_tr_fmt(cx)}" cy="{_tr_fmt(cy)}" '
                            f'r="{_tr_fmt(fill["r"] * scale)}">{stops}</radialGradient>')


def _quad_residual(xy, lab):
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[np.ones(len(x)), x, y, x * x, x * y, y * y]
    coef = np.linalg.lstsq(A, lab, rcond=None)[0]
    return ((A @ coef - lab) ** 2).sum(1)


def segment_regions(rgb, core, k=16, min_frac=0.0005, edge_tol=7.0, fit_tol=4.0, soft_tol=2.0):
    """Opaque core -> region map (-1 outside): k-means, components, speckle folding, smooth-gradient merging."""
    H, W = core.shape
    blur = np.stack([_blur(rgb[..., c], 0.8) for c in range(3)], -1)
    lab = _to_lab(blur)
    _, q = _kmeans(lab[core], k)
    Q = np.full((H, W), -1, np.int32)
    Q[core] = q
    R = np.full((H, W), -1, np.int32)
    n = 0
    for c in range(int(q.max()) + 1):
        lb, m = _label(Q == c)
        R[lb > 0] = lb[lb > 0] - 1 + n
        n += m
    if n <= 1:
        return R
    pairs = []
    for a_, b_, la, lb_ in ((R[:, :-1], R[:, 1:], lab[:, :-1], lab[:, 1:]), (R[:-1], R[1:], lab[:-1], lab[1:])):
        v = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
        lo, hi = np.minimum(a_[v], b_[v]), np.maximum(a_[v], b_[v])
        pairs.append((lo.astype(np.int64) * n + hi, np.linalg.norm(la[v] - lb_[v], axis=1)))
    keys = np.concatenate([p[0] for p in pairs])
    diffs = np.concatenate([p[1] for p in pairs])
    uk, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv)
    contrast = np.bincount(inv, diffs) / cnt
    edges = sorted(zip(contrast.tolist(), (uk // n).tolist(), (uk % n).tolist(), cnt.tolist()))
    flat = R.ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(n + 1))
    members = {i: order[starts[i]:starts[i + 1]] for i in range(n)}
    area = {i: len(v) for i, v in members.items()}
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if area[ra] < area[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        area[ra] += area[rb]
        members[ra] = np.concatenate([members[ra], members[rb]])
        return ra
    min_area = max(4, int(min_frac * core.sum()))
    shared = {}
    for c_, a, b, ct in edges:
        shared.setdefault(a, []).append((ct, b))
        shared.setdefault(b, []).append((ct, a))
    # boundary length per region (4-neighbour edges to any other label, the outside or the image border)
    perim = np.zeros(n)
    for a_, b_ in ((R[:, :-1], R[:, 1:]), (R[:-1], R[1:])):
        v = a_ != b_
        perim += np.bincount(a_[v & (a_ >= 0)], minlength=n) + np.bincount(b_[v & (b_ >= 0)], minlength=n)
    border = np.concatenate([R[0], R[-1], R[:, 0], R[:, -1]])
    perim += np.bincount(border[border >= 0], minlength=n)
    for i in sorted(range(n), key=lambda i: area[i]):
        # speckle (too small) and slivers (a 1-3 px resampling seam: area / boundary < 1.5 px) join their best neighbour
        if (area[find(i)] < min_area or len(members[i]) / max(1.0, perim[i]) < 1.5) and shared.get(i):
            best = max(shared[i], key=lambda t: t[0] if find(t[1]) != find(i) else -1)
            if find(best[1]) != find(i):
                union(i, best[1])
    rng = np.random.default_rng(0)
    yy, xx = np.divmod(np.arange(H * W), W)
    xy_all = np.c_[xx / W, yy / H]
    lab_flat = lab.reshape(-1, 3)
    for c_, a, b, ct in edges:
        if c_ > edge_tol:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        small, big = sorted((members[ra], members[rb]), key=len)
        ia = rng.choice(big, min(len(big), 3000), replace=False)
        ib = rng.choice(small, min(len(small), 2000), replace=False)
        res = _quad_residual(xy_all[np.r_[ia, ib]], lab_flat[np.r_[ia, ib]])
        # an invisible source boundary is worse as a seam than as a looser fit, unless the smaller side is a
        # distinct feature (a highlight) that the joint fit would erase
        if (np.sqrt(res.mean()) <= fit_tol * (2.0 if c_ < soft_tol else 1.0)
                and np.sqrt(res[len(ia):].mean()) <= 2 * fit_tol):
            union(ra, rb)
    roots = {}
    out = np.full(H * W, -1, np.int32)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
            out[members[r]] = roots[r]
    return out.reshape(H, W)


def _alpha_bands(alpha, rgb, core, n, tol=1.2, fill_kinds=("flat",), gamma=1.0):
    """Soft alpha (glow, shadow, feathering) as nested iso-alpha bands whose stacked opacities rebuild the falloff."""
    soft = (alpha > 0.02) & ~core
    if n <= 0 or soft.sum() < 0.03 * max(1, core.sum()):
        return []
    # the soft ring must be wider than resampling ringing (mean width over the core's boundary > 2 px)
    edge = core & ~(np.asarray(Image.fromarray(core.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 127)
    if core.any() and soft.sum() / max(1, int(edge.sum())) <= 2.0:
        return []
    # smooth contours, but never spill past a crisp edge (a halo there reads as a dark outline)
    a_s = np.minimum(_blur(alpha, 1.2), np.asarray(Image.fromarray(alpha.astype(np.float32), "F").filter(ImageFilter.MaxFilter(3))))
    thr = [round(0.5 * ((i + 1) / (n + 1)) ** gamma, 3) for i in range(n)]
    layers, prev_a, prev_c = [], 0.0, np.zeros(3)
    for i, t in enumerate(thr):
        inner = thr[i + 1] if i + 1 < len(thr) else 0.5
        region = a_s >= t
        ring = region & (a_s < inner) & ~core
        if ring.sum() < 12:
            continue
        a_i = float(alpha[ring].mean())
        w = alpha[ring][:, None]
        c_i = (rgb[ring] * w).sum(0) / max(float(w.sum()), 1e-6)
        o = float(np.clip(1 - (1 - a_i) / (1 - prev_a) if a_i > prev_a else 0.01, 0.01, 1))
        if o < 0.03:
            continue  # invisible band: only adds a VD path
        c_draw = np.clip((a_i * c_i - (1 - o) * prev_a * prev_c) / o, 0, 255)
        fill = {"type": "flat", "color": c_draw, "rmse": 0.0}
        if "linear" in fill_kinds or "radial" in fill_kinds:
            ys, xs = np.nonzero(ring)
            fill = fit_fill(np.c_[xs + 0.5, ys + 0.5], rgb[ring], fill_kinds, 3, 6.0)
            if fill["type"] != "flat":
                shift = c_draw - c_i
                fill["stops"] = [(o_, np.clip(c + shift, 0, 255)) for o_, c in fill["stops"]]
            else:
                fill["color"] = c_draw
        d = mask_to_path(region, tol=max(2.0, tol), sigma=1.5, min_area=12)
        if d:
            ys, xs = np.nonzero(region)
            layers.append({"name": f"glow_{len(layers) + 1}", "role": "glow", "d": d, "fill": fill, "opacity": round(o, 3),
                           "area": int(region.sum()), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                           "color": _tr_hex(c_i), "mask": region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]})
        prev_a, prev_c = a_i, c_i
    return layers


TRACE_PRESETS = {
    "flat": dict(work=512, k=8, edge_tol=8.0, fit_tol=3.0, min_frac=0.001, kinds=("flat", "linear"), stops=3, tol=0.8,
                 corner=60, bands=4, band_gamma=1.6, stop_tol=4.0),
    "illustration": dict(work=512, k=16, edge_tol=7.0, fit_tol=4.0, min_frac=0.0006, kinds=("flat", "linear", "radial"),
                         stops=6, tol=0.8, corner=75, bands=8, band_gamma=1.6, stop_tol=2.5),
    "gradient": dict(work=512, k=16, edge_tol=7.0, fit_tol=4.0, soft_tol=3.0, min_frac=0.0006,
                     kinds=("flat", "linear", "radial"), stops=8, tol=0.8, corner=75, bands=12, band_gamma=1.6, stop_tol=1.5),
}


def _vtracer():
    try:
        import vtracer
        return vtracer
    except ImportError:
        return None


def _bake_translate(d, dx, dy):
    """Fold a translate() into absolute M/L/C path data (VectorDrawable has no SVG transforms)."""
    out, k = [], 0
    for t in _TOK.findall(d):
        if t.isalpha():
            if t not in "MLCZ":
                raise ValueError(f"unsupported path command {t}")
            out.append(t)
            continue
        out.append(_tr_fmt(float(t) + (dx if k % 2 == 0 else dy)))
        k += 1
    return " ".join(out).replace("Z ", "Z")


VT_PATH = re.compile(r'<path d="([^"]*)" fill="(#[0-9A-Fa-f]{6})"(?: transform="translate\(([-\d.]+),([-\d.]+)\)")?')


def trace_vtracer(img, work=512, bands=8, band_gamma=1.6, **kw):
    """vtracer colour mode on the opaque core (soft alpha stripped first, then rebuilt as glow bands)."""
    s = min(1.0, work / max(img.size))
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    arr = np.asarray(img.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255
    core = alpha >= 0.5
    flat = np.zeros((h, w, 4), np.uint8)
    flat[core, :3] = rgb[core].astype(np.uint8)
    flat[core, 3] = 255
    opts = dict(colormode="color", hierarchical="stacked", mode="spline", filter_speckle=4, color_precision=6,
                layer_difference=16, corner_threshold=60, length_threshold=4.0, splice_threshold=45, path_precision=2)
    opts.update(kw)
    svg = _vtracer().convert_raw_image_to_svg(_png_bytes(Image.fromarray(flat)), img_format="png", **opts)
    layers = _alpha_bands(alpha, rgb, core, bands, 1.2, ("flat",), band_gamma)
    for m_ in VT_PATH.finditer(svg):
        d = _bake_translate(m_.group(1), float(m_.group(3) or 0), float(m_.group(4) or 0))
        layers.append({"name": f"region_{len(layers) + 1}", "role": "region", "d": d, "opacity": 1.0,
                       "fill": {"type": "flat", "color": np.array(_hex_rgb(m_.group(2))) * 255, "rmse": 0.0},
                       "color": m_.group(2).lower(), "area": 0, "bbox": [0, 0, w, h]})
    if core.any() and not any(L["role"] == "region" for L in layers):
        print("  ! vtracer returned no <path d fill> the kit recognises (a vtracer version with another attribute order?): "
              "the vtracer candidate has no regions")
    return {"preset": "vtracer", "params": {"work": work, "bands": bands, **opts}, "size": (w, h), "src_size": img.size,
            "layers": layers}


def trace_image(img, preset="illustration", max_regions=400, **over):
    """RGBA image -> layered vector: glow bands (bottom) + gradient-filled regions. Returns layers + svg text.
    Every per-region step runs on the region's bbox (+ 4 px), so cost is O(total area), not O(regions x canvas);
    more than `max_regions` regions is photographic and returns {"skipped": ...} with no layers."""
    P = dict(TRACE_PRESETS[preset])
    P.update({k: v for k, v in over.items() if v is not None})
    s = min(1.0, P["work"] / max(img.size))  # never upscale: that traces the source's own pixel staircase
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    arr = np.asarray(img.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255
    core = alpha >= 0.5
    out = {"preset": preset, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
           "size": (w, h), "src_size": img.size, "layers": []}
    layers = _alpha_bands(alpha, rgb, core, P["bands"], P["tol"] * 1.5, P.get("band_kinds", ("flat",)),
                          P.get("band_gamma", 1.0))
    R = segment_regions(rgb, core, P["k"], P["min_frac"], P["edge_tol"], P["fit_tol"], P.get("soft_tol", 2.0)) \
        if core.any() else np.full(core.shape, -1, np.int32)
    n = int(R.max()) + 1
    if n > max_regions:
        out["skipped"] = f"photographic: {n} colour regions at {w}x{h} (> {max_regions}); keep the raster (export)"
        return out
    flat = R.ravel()
    order_ = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order_], np.arange(n + 1))
    win, crops = {}, {}
    for r in range(n):
        idx = order_[starts[r]:starts[r + 1]]
        if len(idx) < 4:
            continue
        ys, xs = np.divmod(idx, w)
        x0, y0, x1, y1 = max(0, xs.min() - 4), max(0, ys.min() - 4), min(w, xs.max() + 5), min(h, ys.max() + 5)
        win[r] = (x0, y0, x1, y1)
        m = R[y0:y1, x0:x1] == r
        crops[r] = (m, (_fill_holes(m) & core[y0:y1, x0:x1]) | m, len(idx))
    order = sorted(win, key=lambda r: -int(crops[r][1].sum()))
    later = np.zeros_like(core)
    draw = {}
    for r in reversed(order):
        x0, y0, x1, y1 = win[r]
        m, filled, _ = crops[r]
        draw[r] = filled | (_dilate(m, 2) & later[y0:y1, x0:x1])
        later[y0:y1, x0:x1] |= m
    for r in order:
        x0, y0, x1, y1 = win[r]
        m = crops[r][0]
        ys, xs = np.nonzero(m)
        px = rgb[y0:y1, x0:x1][ys, xs]
        fill = fit_fill(np.c_[xs + x0 + 0.5, ys + y0 + 0.5], px, P["kinds"], P["stops"], P["stop_tol"])
        d = mask_to_path(draw[r], P["tol"], P["corner"], 0.7, 4.0, dx=x0, dy=y0)
        if d:
            bx0, by0, bx1, by1 = int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0 + 1, int(ys.max()) + y0 + 1
            layers.append({"name": f"region_{len(layers) + 1}", "role": "region", "d": d, "fill": fill, "opacity": 1.0,
                           "area": int(len(xs)), "bbox": [bx0, by0, bx1, by1], "color": _tr_hex(px.mean(0)),
                           "mask": m[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]})
    out["layers"] = layers
    return out


def _tr_gid(name):
    """Gradient id prefix per asset: two traced SVGs inlined in one HTML page must not share ids."""
    return (re.sub(r"[^A-Za-z0-9_-]", "_", str(name)) + "_") if name else ""


def layers_svg(layers, vb, size=None, bg=None, name=None):
    """Traced layers -> SVG text (absolute M/L/C/Z paths, userSpaceOnUse gradients, no transforms)."""
    w, h = vb
    W, H = size or vb
    defs, body = [], []
    for i, L in enumerate(layers):
        fill, gd = _fill_svg(L["fill"], f"{_tr_gid(name)}g{i}")
        defs.append(gd)
        op = f' opacity="{L["opacity"]:g}"' if L.get("opacity", 1) < 1 else ""
        body.append(f'<path d="{L["d"]}" fill="{fill}"{op}/>')
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {w} {h}">'
            f'<defs>{"".join(defs)}</defs>{rect}' + "\n".join(body) + "</svg>")


def layers_vec(res, asset=None):
    """A trace as a build_lottie/animate-ready layers.json: static layers 'glow' (bands) and 'art' (regions)."""
    defs, static = [], []
    for role, name in (("glow", "glow"), ("region", "art")):
        paths = []
        for L in (x for x in res["layers"] if x["role"] == role):
            f, gd = _fill_svg(L["fill"], f"{_tr_gid(asset)}{name}_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": L.get("opacity", 1)})
        if paths:
            static.append({"name": name, "paths": paths})
    w, h = res["size"]
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static]}


def _tr_path_error(layers):
    """One line naming the first layer whose pathData Android could not inflate (nan/inf or unparsable), else None."""
    for L in layers:
        d = L.get("d") or ""
        if re.search(r"nan|inf", d, re.I):
            return f"{L.get('name', '?')}: pathData contains nan/inf"
        try:
            _subpaths(d)
        except (ValueError, IndexError, TypeError) as e:
            return f"{L.get('name', '?')}: {e}"
    return None


def _tr_aapt2(vd_text, res_name):
    """aapt2 verdict for one VectorDrawable, written under its final resource name (contract C4 wording):
    {"aapt2": message, "aapt2_ok": True | False | None, "aapt2_level": "link" | "compile" | None}."""
    import tempfile
    level = aapt2_level()
    if level is None:
        return {"aapt2": _aapt2_line(), "aapt2_ok": None, "aapt2_level": None}
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / f"{res_name}.xml"
        f.write_text(vd_text)
        errs = aapt2_check([str(f)])
    if errs:
        return {"aapt2": "aapt2 error: " + "; ".join(f"{k}: {v}" if k != "*" else v for k, v in errs.items())[:300],
                "aapt2_ok": False, "aapt2_level": level}
    return {"aapt2": _aapt2_line(), "aapt2_ok": True, "aapt2_level": level}


def vector_advice(m, layers, dp=None):
    """Vector or raster? Fidelity bar plus Android guidance (VDs <= 200 dp, lint VectorPath > 800 chars, kit size heuristic)."""
    lens = [len(re.sub(r"\s+", " ", L["d"])) for L in layers]
    grads = sum(L["fill"]["type"] != "flat" for L in layers)
    heavy = (f"{len(layers)} paths / {sum(lens)} pathData chars (kit heuristic: > 64 paths or > 40000 chars is a heavy "
             f"first draw; prefer WebP for that slot)") if len(layers) > 64 or sum(lens) > 40000 else None
    if not m:
        # no render, no fidelity: say so instead of guessing (only the size hint can be given from the paths alone)
        return {"fidelity": "unknown", "vector_ok": None, "vd_ok": None, "prefer": "unknown", "paths": len(layers),
                "pathdata_chars": sum(lens), "max_path_chars": max(lens, default=0), "gradients": grads,
                "reasons": [f"fidelity not measured: {_tr_chrome_reason()}"] + ([heavy] if heavy else [])}
    disp = m.get("display", m)
    fid = ("faithful" if disp["diff"] <= 2 else "close" if disp["diff"] <= 3.5 else
           "stylised" if disp["diff"] <= 5 else "lossy")
    if disp["iou"] < 0.95:
        fid = "lossy"
    reasons = [f"display-size diff {disp['diff']} / silhouette IoU {disp['iou']} -> {fid}"]
    vec_ok = fid in ("faithful", "close")
    vd_ok, why_not_vd = vec_ok, (None if vec_ok else reasons[0])
    if dp and max(dp) > 200:
        vd_ok = False
        reasons.append(f"{max(dp):g}dp > 200dp: Android recommends raster above 200x200dp (draw time)")
        why_not_vd = why_not_vd or reasons[-1]
    if heavy:
        vd_ok = False
        reasons.append(heavy)
        why_not_vd = why_not_vd or heavy
    long_ = sum(n > 800 for n in lens)
    if long_:
        reasons.append(f"{long_} path(s) over 800 chars: lint VectorPath warns (MAX_PATH_DATA_LENGTH 800; it is documented as "
                       f"a draw-time hint, not an error)")
    if grads:
        reasons.append(f"{grads} gradient fill(s): VectorDrawable gradients need API 24+ or AndroidX VectorDrawableCompat")
    if fid == "stylised":
        reasons.append("visibly simplified vs the raster; acceptable only if a flatter look is fine")
    prefer = "vd" if vd_ok else ("svg/lottie" if vec_ok else "png/webp")
    return {"fidelity": fid, "vector_ok": vec_ok, "vd_ok": vd_ok, "prefer": prefer, "paths": len(layers),
            "pathdata_chars": sum(lens), "max_path_chars": max(lens, default=0), "gradients": grads, "reasons": reasons,
            "why_not_vd": why_not_vd}


def _asset_dp(png):
    """dp size of a forged asset, looked up from the build dir's assets.json when the PNG lives in <dir>/assets/."""
    p = Path(png).resolve()
    mf = p.parent.parent / "assets.json"
    if p.parent.name != "assets" or not mf.exists():
        return None
    try:
        a = next((x for x in json.loads(mf.read_text())["assets"] if x.get("name") == p.stem), None)
        dp = a.get("dp") if a else None
        if not dp:
            return None
        aspect = Image.open(p).height / max(1, Image.open(p).width)
        if isinstance(dp, dict):  # {"w": 48, "h": 48}; h optional
            w = float(dp["w"])
            return w, float(dp["h"]) if dp.get("h") else w * aspect
        if isinstance(dp, (list, tuple)):  # [w, h] or [w]
            w = float(dp[0])
            return w, float(dp[1]) if len(dp) > 1 else w * aspect
        return float(dp), float(dp) * aspect  # a bare number is the width
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
        print(f"  ! {mf}: dp of {p.stem!r} unreadable ({e}); pass --dp")
        return None


def _diff_heat(a, b):
    comp = lambda x: np.asarray(x, np.float32)[..., :3] * (np.asarray(x, np.float32)[..., 3:] / 255)
    d = np.abs(comp(a) - comp(b.resize(a.size))).mean(2)
    v = np.clip(d * 4, 0, 255).astype(np.uint8)
    return Image.merge("RGB", (Image.fromarray(v), Image.fromarray(v // 3), Image.fromarray(255 - v)))


def _on(img, bg):
    base = Image.new("RGBA", img.size, bg)
    base.alpha_composite(img)
    return base


def trace_asset(src, presets, dp=None, display=None, name=None, **over):
    """Trace with each preset, render all with Chrome, score, and pick the smallest near-best candidate."""
    cands = []
    for pr in presets:
        t = time.time()
        res = trace_vtracer(src, **{k: v for k, v in over.items() if k in ("work", "bands") and v is not None}) \
            if pr == "vtracer" else trace_image(src, pr, **over)
        svg = layers_svg(res["layers"], res["size"], src.size, name=name)
        cands.append({"preset": pr, "res": res, "svg": svg, "secs": round(time.time() - t, 2), "bytes": len(svg)})
        if res.get("skipped"):
            cands[-1]["render"] = None  # photographic under one preset is photographic under all: stop here
            break
        if len(cands) == 1 and len(presets) > 1:
            # without a renderer the presets cannot be compared: check on the first one and do not trace the rest
            first = _tr_render_svgs([svg], [src.size])
            if first is None:
                break
            cands[0]["render"] = first[0]
    # display-size fidelity: 3x the dp width, else at least 192 px on the long side and 96 px on the short side
    display = display or (round(max(dp) * 3) if dp else max(192, round(96 * max(src.size) / max(1, min(src.size)))))
    todo = [c for c in cands if "render" not in c]
    rens = _tr_render_svgs([c["svg"] for c in todo], [src.size] * len(todo)) if todo else []
    for c, r in zip(todo, rens or [None] * len(todo)):
        c["render"] = r
    for c in cands:
        c["metrics"] = trace_metrics(src, c["render"], display) if c["render"] is not None else None
    scored = [c for c in cands if c["metrics"]]
    if scored:
        best = min(c["metrics"]["display"]["diff"] for c in scored)
        near = [c for c in scored if c["metrics"]["display"]["diff"] <= best * 1.1 + 0.15]
        chosen = min(near, key=lambda c: c["bytes"])
    else:
        live = [c for c in cands if not c["res"].get("skipped")] or cands
        chosen = next((c for c in live if c["preset"] == "illustration"), live[0])
    return chosen, cands


def cmd_trace(a):
    out = Path(a.out)
    if a.name and len(a.files) > 1:
        sys.exit("trace: --name needs a single input file (the second file would overwrite the first)")
    if a.preset == "auto":
        presets = ["flat", "illustration", "gradient"] + (["vtracer"] if _vtracer() else [])
    else:
        presets = [a.preset]
        if a.preset == "vtracer" and not _vtracer():
            sys.exit("trace: vtracer not installed: pip install vtracer (or use --preset flat|illustration|gradient)")
    if a.preset == "vtracer" and (a.colors is not None or a.tol is not None):
        print("  ! --preset vtracer ignores --colors and --tol (vtracer has its own colour/curve settings)")
    if not _chrome():
        print("  (no Chrome/Chromium found: fidelity is not measured; set GPTKIT_CHROME to enable it)")
    (out / "review").mkdir(parents=True, exist_ok=True)
    failed = 0
    for f in a.files:
        try:
            if not Path(f).is_file():
                raise OSError("no such file")
            src = _rgba(f)
        except (OSError, ValueError) as e:  # UnidentifiedImageError is an OSError
            print(f"  ! {f}: cannot read image: {e}")
            failed += 1
            continue
        if not (np.asarray(src.getchannel("A")) >= 128).any():
            print(f"  ! {f}: nothing opaque to trace (fully transparent); skipped")
            failed += 1
            continue
        name = a.name or Path(f).stem
        vd_name = _res_name(a.prefix + name)
        if vd_name != a.prefix + name:
            print(f"  ({a.prefix + name!r} -> {vd_name}: Android resource names are [a-z0-9_] starting with a letter)")
        dp = (a.dp, a.dp * src.height / src.width) if a.dp else _asset_dp(f)
        chosen, cands = trace_asset(src, presets, dp, a.display, name, work=a.work, k=a.colors, bands=a.bands, tol=a.tol)
        res = chosen["res"]
        report = {"source": str(f), "src_size": src.size, "dp": dp, "chosen": chosen["preset"],
                  "candidates": [{k: c[k] for k in ("preset", "bytes", "secs", "metrics")} | {"paths": len(c["res"]["layers"])}
                                 for c in cands]}
        for c in cands:
            if c["res"].get("skipped"):
                report["candidates"][cands.index(c)]["skipped"] = c["res"]["skipped"]
        if res.get("skipped"):
            report["advice"] = {"fidelity": "not traced", "vector_ok": False, "vd_ok": False, "prefer": "png/webp",
                                "reasons": [res["skipped"]]}
            (out / f"{name}.trace.json").write_text(json.dumps(report, indent=2, default=str))
            print(f"{name}: not traced ({res['skipped']}) -> {out / f'{name}.trace.json'}")
            print("  advice: prefer png/webp (photographic input; export writes the density set)")
            continue
        svg_path = out / f"{name}.svg"
        svg_path.write_text(chosen["svg"])
        m = chosen["metrics"]
        report["advice"] = vector_advice(m, res["layers"], dp)
        adv = report["advice"]
        bad = _tr_path_error(res["layers"])
        if bad:
            report["vd"] = {"error": f"not written: {bad}"}
        else:
            try:
                vd_w = round(dp[0], 1) if dp else round(48 * src.width / max(src.size), 1)
                vd = svg_to_vd(chosen["svg"], vd_w)
                (out / f"{vd_name}.xml").write_text(vd)
                report["vd"] = {"file": str(out / f"{vd_name}.xml"), "res_name": vd_name, "bytes": len(vd), "width_dp": vd_w,
                                "width_from": "dp" if dp else "default 48dp long side (pass --dp)", **_tr_aapt2(vd, vd_name),
                                "lint": _vd_lint(vd)}
                for e in report["vd"]["lint"]:
                    print(f"  ! {vd_name}.xml lint: {e}")
                vr = _tr_render_svgs([vd_to_svg(vd)], [src.size])
                if vr and chosen.get("render") is not None:
                    report["vd"]["vs_svg_diff"] = trace_metrics(chosen["render"], vr[0])["diff"]
            except ValueError as e:
                report["vd"] = {"error": str(e)}
        (out / f"{name}.layers.json").write_text(json.dumps(layers_vec(res, name), indent=1))
        (out / f"{name}.trace.json").write_text(
            json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        rows = []
        if chosen.get("render") is not None:
            r = chosen["render"]
            rows.append([("source", _on(src, a.bg)), (f"{chosen['preset']} vector", _on(r, a.bg)), ("source", src),
                         ("vector", r), ("|diff| x4", _diff_heat(src, r))])
            rows.append([(f"{c['preset']} d={c['metrics']['display']['diff']} {c['bytes'] // 1024}KB p={len(c['res']['layers'])}",
                          _on(c["render"], a.bg)) for c in cands if c.get("render") is not None])
            sheet(rows, out / "review" / f"{name}_trace.png", T=300)
        print(f"{name}: {chosen['preset']} -> {svg_path} ({chosen['bytes'] // 1024} KB, {len(res['layers'])} paths)"
              + (f"  display diff={m['display']['diff']} iou={m['display']['iou']}" if m else ""))
        for c in cands:
            if c["res"].get("skipped"):
                print(f"    {c['preset']:13s} skipped: {c['res']['skipped']}")
            elif c["metrics"] and len(cands) > 1:
                print(f"    {c['preset']:13s} d={c['metrics']['display']['diff']:5.2f} iou={c['metrics']['display']['iou']:.3f} "
                      f"{c['bytes'] // 1024:4d} KB {len(c['res']['layers']):3d} paths")
        if m:
            print(f"  advice: prefer {adv['prefer']} ({adv['fidelity']})" + "".join(f"; {r}" for r in adv["reasons"][1:]))
        else:
            print(f"  advice: prefer unknown ({'; '.join(adv['reasons'])})")
        v = report["vd"]
        if "file" in v:
            note = f"  (not recommended: {adv['why_not_vd']})" if adv.get("why_not_vd") else ""
            print(f"  vd: {v['file']}  {v['aapt2']}" + (f"  VD->SVG round-trip diff={v['vs_svg_diff']}" if "vs_svg_diff" in v else "")
                  + note)
        else:
            print(f"  ! vd: {v['error']}")
    if failed:
        sys.exit(f"trace: {failed} of {len(a.files)} input(s) not traced (see the lines above)")


def map_path(d, fn):
    """Apply fn(x, y) -> (x', y') to every point of absolute M/L/C/Z path data; the command sequence never changes."""
    toks = _TOK.findall(d)
    out, nums = [], []
    for t in toks + ["Z"]:
        if t.isalpha():
            for i in range(0, len(nums) - 1, 2):
                x, y = fn(nums[i], nums[i + 1])
                out.append(f"{_tr_fmt(x)} {_tr_fmt(y)}")
            nums = []
            out.append(t)
        else:
            nums.append(float(t))
    return " ".join(out[:-1]).replace(" Z", "Z")


def _path_points(d):
    v = [float(t) for t in _TOK.findall(d) if not t.isalpha()]
    return np.array(v[:len(v) // 2 * 2]).reshape(-1, 2)


TR_MOTIONS = {  # default amplitude per motion (fraction of the part's width/height)
    "sway": 0.06, "flicker": 0.05, "breathe": 0.04, "squash": 0.08, "bob": 0.04, "wave": 0.04, "ripple": 0.06,
}


def motion_field(kind, box, t, amp=1.0, phase=0.0, seed=0):
    """Deterministic loop-periodic deformation at time t in [0, 1); t=0 is the rest pose (the approved design)."""
    ft, f0 = _field(kind, box, t, amp, phase, seed), _field(kind, box, 0.0, amp, phase, seed)

    def f(x, y):
        (xt, yt), (x0, y0) = ft(x, y), f0(x, y)
        return x + xt - x0, y + yt - y0
    return f


def _field(kind, box, t, amp, phase, seed):
    if kind not in TR_MOTIONS:
        raise ValueError(f"unknown motion {kind!r} (choose {', '.join(TR_MOTIONS)})")
    x0, y0, x1, y1 = box
    # a flat box (a horizontal stroke, a thin bar) still gets a usable height/width: 10 % of the other side, >= 1 unit
    W, H = max(x1 - x0, (y1 - y0) * 0.1, 1.0), max(y1 - y0, (x1 - x0) * 0.1, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    A = TR_MOTIONS[kind] * amp
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 6)
    tau = 2 * np.pi * (t + phase)

    def h(y):  # 0 at the base (bottom), 1 at the top
        return min(1.0, max(0.0, (y1 - y) / H))
    if kind == "sway":
        return lambda x, y: (x + A * W * np.sin(tau) * h(y) ** 1.5, y)
    if kind == "flicker":
        def f(x, y):
            k = h(y)
            lat = 0.6 * np.sin(tau + ph[0]) + 0.4 * np.sin(2 * tau + 3.0 * k + ph[1]) + 0.25 * np.sin(3 * tau + 5.0 * k + ph[2])
            st = 0.5 * np.sin(2 * tau + ph[3]) + 0.35 * np.sin(3 * tau + ph[4])
            tip = k ** 1.4
            return (x + A * W * lat * tip + 0.35 * A * (x - cx) * np.sin(2 * tau + ph[5]) * k,
                    y1 - (y1 - y) * (1 + 1.2 * A * st * tip))
        return f
    if kind == "breathe":
        s = 1 + A * np.sin(tau)
        return lambda x, y: (cx + (x - cx) * s, cy + (y - cy) * s)
    if kind == "squash":
        sy = 1 + A * np.sin(tau)
        sx = 1 / np.sqrt(sy)
        return lambda x, y: (cx + (x - cx) * sx, y1 - (y1 - y) * sy)
    if kind == "bob":
        return lambda x, y: (x, y - A * H * (0.5 - 0.5 * np.cos(tau)))
    if kind == "wave":
        return lambda x, y: (x + A * W * np.sin(tau - 2 * np.pi * (y - y0) / H), y)
    if kind == "ripple":
        s = 1 + A * (0.5 - 0.5 * np.cos(tau))
        return lambda x, y: (cx + (x - cx) * s, cy + (y - cy) * (1 + (s - 1) * 0.6))
    raise ValueError(f"unknown motion {kind!r} (choose {', '.join(TR_MOTIONS)})")


def procedural_keyframes(paths, kind, n=8, amp=1.0, phase=0.0, seed=0, box=None):
    """Paths -> n morph-compatible keyframes each, all deformed by one shared field so a part stays registered."""
    paths = [_d(_subpaths(d)) for d in paths]  # relative / H V S Q A commands -> absolute M/L/C/Z first
    if box is None:
        pts = np.vstack([_path_points(d) for d in paths])
        box = (*pts.min(0), *pts.max(0))
    return [[map_path(d, motion_field(kind, box, j / n, amp, phase, seed)) for j in range(n)] for d in paths]


def _grad_geom(defs, gid):
    """(kind, [(x, y), (x, y)]) of gradient #gid in the defs text: linear -> start/end, radial -> centre and a point at
    the radius. None for gradients the mover does not handle (gradientTransform, objectBoundingBox units)."""
    m = re.search(rf'<(linear|radial)Gradient\b[^>]*\bid="{re.escape(gid)}"[^>]*>', defs or "")
    if not m or "gradientTransform" in m.group(0) or 'gradientUnits="userSpaceOnUse"' not in m.group(0):
        return None
    at = dict(re.findall(r'([\w:-]+)="([^"]*)"', m.group(0)))
    try:
        if m.group(1) == "linear":
            return "linear", [(float(at["x1"]), float(at["y1"])), (float(at["x2"]), float(at["y2"]))]
        cx, cy, r = float(at["cx"]), float(at["cy"]), float(at["r"])
        return "radial", [(cx, cy), (cx + r, cy)]
    except (KeyError, ValueError):
        return None


def _grad_patch(defs, gid, kind, pts):
    """defs text with gradient #gid moved to pts (the shapes _grad_geom returns)."""
    (x1, y1), (x2, y2) = pts
    new = {"x1": x1, "y1": y1, "x2": x2, "y2": y2} if kind == "linear" else {"cx": x1, "cy": y1, "r": math.hypot(x2 - x1, y2 - y1)}

    def sub(m):
        tag = m.group(0)
        for k, v in new.items():
            tag = re.sub(rf'\b{k}="[^"]*"', f'{k}="{_tr_fmt(v)}"', tag)
        return tag
    return re.sub(rf'<(linear|radial)Gradient\b[^>]*\bid="{re.escape(gid)}"[^>]*>', sub, defs, count=1)


def animate_layers(vec, specs, n=8):
    """Static layers -> animated ones. specs: {layer: (motion, amp, phase[, anchor layer sharing its field])}."""
    vec = json.loads(json.dumps(vec))
    specs = dict(specs)
    names_ = {l["name"] for l in vec["static_layers"]}
    for name, spec in list(specs.items()):
        if f"glow_{name}" in names_ and f"glow_{name}" not in specs:
            specs[f"glow_{name}"] = (*spec[:3], name)
    order = vec.get("layer_order") or [l["name"] for l in vec["static_layers"]]
    boxes = dict((vec.get("rig") or {}).get("boxes", {}))
    for l in vec["static_layers"]:
        if l["name"] not in boxes:
            pts = np.vstack([_path_points(_d(_subpaths(p["d"]))) for p in l["paths"]])
            boxes[l["name"]] = [float(v) for v in (*pts.min(0), *pts.max(0))]
    for name, spec in specs.items():
        kind, amp, phase = spec[:3]
        anchor = (spec[3] if len(spec) > 3 else None) or name
        sl = next((l for l in vec["static_layers"] if l["name"] == name), None)
        if sl is None:
            raise ValueError(f"no static layer {name!r} (have: {', '.join(l['name'] for l in vec['static_layers'])})")
        if anchor not in boxes:
            raise ValueError(f"unknown anchor layer {anchor!r}")
        keys = procedural_keyframes([p["d"] for p in sl["paths"]], kind, n, amp, phase,
                                    seed=sum(map(ord, anchor)), box=boxes[anchor])
        names = [name if len(sl["paths"]) == 1 else f"{name}__{i + 1}" for i in range(len(sl["paths"]))]
        paint = ("fill", "opacity", "stroke", "stroke_width", "cap", "join")  # every paint key survives the morph
        for nm, p, k in zip(names, sl["paths"], keys):
            al = {"name": nm, "fill": p.get("fill", "none"), "opacity": p.get("opacity", 1)}
            al.update({key: p[key] for key in paint if key in p})
            if sl.get("motion"):
                al["motion"] = sl["motion"]  # the layer's rigid motion (vectorize schema) rides on top of the morph
            al["keyframes"] = k
            ref = re.match(r"url\(#([^)]+)\)", al["fill"] or "")
            geom = _grad_geom(vec.get("defs", ""), ref.group(1)) if ref else None
            if geom:
                # the gradient's control points ride the same field, so the shading stays on the part as it moves
                fields = [motion_field(kind, boxes[anchor], j / n, amp, phase, sum(map(ord, anchor))) for j in range(n)]
                al["grad_kind"] = geom[0]
                al["grad_keys"] = [[[round(float(c), 2) for c in f(*pt)] for pt in geom[1]] for f in fields]
            vec["animated_layers"].append(al)
        vec["static_layers"] = [l for l in vec["static_layers"] if l["name"] != name]
        i = order.index(name) if name in order else len(order)
        order = order[:i] + names + order[i + 1:]
    vec["layer_order"] = order
    return vec


def _tr_animate_gradients(lot, vec, linear=True):
    """Animate gf/gs start+end of morphing layers whose gradient rides the motion field (animate_layers' grad_keys)."""
    vb = [float(x) for x in vec["viewBox"].split()]
    k = lot["w"] / max(1e-9, vb[2])
    grads = {al["name"]: al for al in vec["animated_layers"] if al.get("grad_keys")}
    ease = ({"i": {"x": [1], "y": [1]}, "o": {"x": [0], "y": [0]}} if linear else
            {"i": {"x": [0.58], "y": [1]}, "o": {"x": [0.42], "y": [0]}})
    for layer in lot["layers"]:
        al = grads.get(layer["nm"])
        if not al:
            continue
        gk, n, op = al["grad_keys"], len(al["grad_keys"]), lot["op"]
        pt = lambda p: [round((p[0] - vb[0]) * k, 2), round((p[1] - vb[1]) * k, 2)]
        for gr in layer["shapes"]:
            for it in gr.get("it", []):
                if it.get("ty") in ("gf", "gs"):
                    for prop, idx in (("s", 0), ("e", 1)):
                        it[prop] = {"a": 1, "k": [{"t": round(op * j / n, 3), "s": pt(gk[j][idx]), **ease} for j in range(n)]
                                    + [{"t": op, "s": pt(gk[0][idx])}]}
    return lot


def _tidy_lottie(lot, linear=True):
    """Round shape vertices to 2 decimals and optionally interpolate keys linearly (no ease-to-stop at every key)."""
    def rnd(sh):
        for key in ("i", "o", "v"):
            sh[key] = [[round(c, 2) for c in pt] for pt in sh[key]]
    for layer in lot["layers"]:
        for gr in layer["shapes"]:
            for it in gr.get("it", []):
                ks = it.get("ks", {})
                if it.get("ty") != "sh":
                    continue
                if ks.get("a") == 1:
                    for k in ks["k"]:
                        for sh in k.get("s", []):
                            rnd(sh)
                        if linear and "i" in k:
                            k["i"], k["o"] = {"x": [1], "y": [1]}, {"x": [0], "y": [0]}
                else:
                    rnd(ks["k"])
    return lot


TR_LOTTIE_JS = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"


def lottie_filmstrip(lot, frames, size=256, bg="#1b1d33"):
    """Render Lottie frames with lottie-web in headless Chrome -> one strip image (None without Chrome)."""
    w = round(size * lot["w"] / max(lot["w"], lot["h"]))
    h = round(size * lot["h"] / max(lot["w"], lot["h"]))
    divs = "".join(f'<div id="f{i}" style="position:absolute;left:{i * w}px;top:0;width:{w}px;height:{h}px;background:{bg}"></div>'
                   for i in range(len(frames)))
    js = (f"var d={json.dumps(lot, separators=(',', ':'))};var fr={json.dumps(list(frames))};"
          "fr.forEach(function(f,i){var an=lottie.loadAnimation({container:document.getElementById('f'+i),renderer:'svg',"
          "loop:false,autoplay:false,animationData:JSON.parse(JSON.stringify(d))});an.addEventListener('DOMLoaded',"
          "function(){an.goToAndStop(f,true)});});")
    lib = _lottie_js()
    html = (f"<!doctype html><body style='margin:0;background:{bg}'>{divs}"
            + (f"<script>{lib}</script>" if lib else f"<script src='{TR_LOTTIE_JS}'></script>") + f"<script>{js}</script></body>")
    return chrome_shot(html, w * len(frames), h, budget=6000)


def region_components(res):
    """Group traced regions into connected pieces of the silhouette -> {region number: component number}."""
    regs = [dict(L, mask=_lmask(L, res["size"][::-1])) for L in res["layers"] if L["role"] == "region"]
    if not regs:
        return {}
    lab, _ = _label(np.logical_or.reduce([L["mask"] for L in regs]))
    comp = {}
    for i, L in enumerate(regs):
        ids = lab[L["mask"]]
        comp[i + 1] = int(np.bincount(ids[ids > 0]).argmax()) if (ids > 0).any() else 0
    order = sorted(set(comp.values()), key=lambda c: -sum(regs[i - 1]["area"] for i, v in comp.items() if v == c))
    rank = {c: k + 1 for k, c in enumerate(order)}
    return {i: rank[c] for i, c in comp.items()}


def region_overlay(src, res, bg="#1b1d33"):
    """Source | numbered regions | connected components c1..cN: what Claude or the vision model reads to name parts."""
    w, h = res["size"]
    regs = [dict(L, mask=_lmask(L, (h, w))) for L in res["layers"] if L["role"] == "region"]
    comps = region_components(res)
    rng = np.random.default_rng(7)
    S = max(1, 768 // max(w, h))
    panels = []
    for mode in ("regions", "components"):
        vis = np.zeros((h, w, 3), np.uint8)
        cols = {}
        for i, L in enumerate(regs):
            key = i + 1 if mode == "regions" else comps[i + 1]
            vis[L["mask"]] = cols.setdefault(key, rng.integers(60, 256, 3))
        im = Image.fromarray(vis).resize((w * S, h * S), Image.NEAREST)
        d = ImageDraw.Draw(im)
        seen = set()
        for i, L in enumerate(regs):
            label = str(i + 1) if mode == "regions" else f"c{comps[i + 1]}"
            if label in seen:
                continue
            seen.add(label)
            m_ = L["mask"] if mode == "regions" else np.logical_or.reduce(
                [R["mask"] for j, R in enumerate(regs) if comps[j + 1] == comps[i + 1]])
            ys, xs = np.nonzero(m_)
            o = np.lexsort((xs, ys))
            x, y = xs[o[len(o) // 2]] * S, ys[o[len(o) // 2]] * S
            d.rectangle([x - 2, y - 2, x + 8 * len(label) + 2, y + 12], fill="black")
            d.text((x, y), label, fill="white")
        panels.append(im)
    left = _on(src.resize((w * S, h * S), Image.LANCZOS), bg).convert("RGB")
    out = Image.new("RGB", (left.width * 3 + 32, left.height), "white")
    for k, im in enumerate([left] + panels):
        out.paste(im, (k * (left.width + 16), 0))
    return out


def name_regions(src, res, parts, model=None):
    """One vision call: assign every numbered region to a named part (or 'other')."""
    n = sum(L["role"] == "region" for L in res["layers"])
    names = list(parts) + ["other"]
    schema = {"type": "object", "additionalProperties": False, "required": ["assignments"], "properties": {
        "assignments": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                   "required": ["region", "part"],
                                                   "properties": {"region": {"type": "integer"},
                                                                  "part": {"type": "string", "enum": names}}}}}}
    desc = "; ".join(f"{k}: {v}" for k, v in parts.items())
    r = ask([f"LEFT is an illustration. MIDDLE is the same image split into {n} numbered colour regions (RIGHT only "
             "shows which regions touch, as c1..cN; ignore those labels). Assign EVERY "
             f"region number 1..{n} to exactly one part: {desc}; use 'other' only if it belongs to none. Judge by position "
             "and by what the region depicts in the LEFT image.", region_overlay(src, res)], schema, "region_parts", model)
    return {int(x["region"]): x["part"] for x in r["assignments"] if 1 <= int(x["region"]) <= n}


def _reg_search(S, part, grid, scales, shifts, cap=60.0, centre=(0.0, 0.0)):
    """Best (cost, scale, dx, dy) placing `part` on the square reference S (grid px) under a truncated colour cost."""
    Sa = S[..., 3] > 128
    best = None
    for s in scales:
        n = max(2, round(grid * s))
        Pm = np.asarray(part.resize((n, n), Image.LANCZOS), np.float32)
        ys, xs = np.nonzero(Pm[..., 3] > 128)
        if len(xs) < 8:
            continue
        cols = Pm[ys, xs, :3]
        c0 = (grid - n) / 2
        for dy in shifts:
            for dx in shifts:
                yy = np.round(ys + c0 + dy + centre[1]).astype(int)
                xx = np.round(xs + c0 + dx + centre[0]).astype(int)
                ok = (yy >= 0) & (yy < grid) & (xx >= 0) & (xx < grid)
                d = np.full(len(xs), cap, np.float32)
                v = np.nonzero(ok)[0]
                v = v[Sa[yy[v], xx[v]]]
                d[v] = np.minimum(np.abs(S[yy[v], xx[v], :3] - cols[v]).mean(1), cap)
                # summed, not averaged: a mean rewards shrinking the part into its best-matching patch
                c = float((d - cap / 2).sum()) / (grid * grid)
                if best is None or c < best[0]:
                    best = (c, s, dx + centre[0], dy + centre[1])
    return best


def register_part(part_sq, src):
    """Register a GPT-isolated part (square canvas) onto the source; GPT drifts up to ~25% in scale and position."""
    sq, off = _square(src)
    part_sq = part_sq.resize(sq.size, Image.LANCZOS)
    S96 = np.asarray(sq.resize((96, 96), Image.LANCZOS), np.float32)
    _, s, dx, dy = _reg_search(S96, part_sq, 96, np.arange(0.7, 1.31, 0.03), range(-20, 21))
    G = 256
    S2 = np.asarray(sq.resize((G, G), Image.LANCZOS), np.float32)
    f = G / 96
    _, s, dx, dy = _reg_search(S2, part_sq, G, np.arange(s - 0.03, s + 0.031, 0.01), range(-4, 5), centre=(dx * f, dy * f))
    n = round(sq.width * s)
    k = sq.width / G
    pad = sq.width
    canvas = Image.new("RGBA", (sq.width + 2 * pad, sq.height + 2 * pad), (0, 0, 0, 0))
    canvas.alpha_composite(part_sq.resize((n, n), Image.LANCZOS),
                           (pad + round((sq.width - n) / 2 + dx * k), pad + round((sq.width - n) / 2 + dy * k)))
    out = canvas.crop((pad + off[0], pad + off[1], pad + off[0] + src.width, pad + off[1] + src.height))
    return out, {"scale": round(float(s), 3), "dx": round(dx * k), "dy": round(dy * k)}


def isolate_parts(src, parts, model=None, jobs=2, cache=None):
    """Route (a): GPT redraws each named part alone on the asset's canvas (occluded areas completed).
    Returns (done {part: png bytes}, failed {part: (error, prompt)}); each result is written to cache[part] on arrival,
    so a later failure never loses a paid call."""
    sq, off = _square(src)
    names = list(parts)

    def one(name):
        others = ", ".join(parts[o] for o in names if o != name) or "everything else"
        prompt = (f"The attached image is an APPROVED illustration. Keep ONLY {parts[name]}, exactly as drawn: same "
                  f"size, position, angle, colours and gradients on the same canvas. Remove {others} completely. Where "
                  f"the removed parts covered {parts[name]}, complete its hidden shape naturally. Transparent background, "
                  "no glow, no shadow, nothing else.")
        raw, _ = edit(_png_bytes(sq), prompt, "1024x1024", "transparent", model)
        if cache and name in cache:
            Path(cache[name]).write_bytes(raw)
        return name, raw, prompt
    done, failed = {}, {}
    with cf.ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(one, n): n for n in names}
        for fut in cf.as_completed(futs):
            try:
                name, raw, _ = fut.result()
                done[name] = raw
            except ApiError as e:  # refusal, exhausted retries, no/undecodable image: all ApiError (C2)
                name = futs[fut]
                failed[name] = (str(e)[:300], f"Keep ONLY {parts[name]}, ...")
    return done, failed


def _hull_mask(mask, min_frac=0.02):
    """Filled convex hull of each sizeable connected component (a cheap amodal-completion prior)."""
    lab, n = _label(mask)
    out = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    d = ImageDraw.Draw(out)
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < max(8, min_frac * mask.sum()):
            continue
        pts = sorted(set(zip(xs.tolist(), ys.tolist())))
        cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, hi = [], []
        for p in pts:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
                lo.pop()
            lo.append(p)
        for p in reversed(pts):
            while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
                hi.pop()
            hi.append(p)
        d.polygon(lo[:-1] + hi[:-1], fill=255)
    return (np.asarray(out) > 0) | mask


def build_rig(src, res, assign, order, amodal=None, ext_frac=0.06, tol=0.8, corner=75, min_detail=0.0, glow_mode="part"):
    """Traced regions -> named, gap-free parts (extended under the parts above them) as a layers.json dict."""
    w, h = res["size"]
    sc = w / src.width
    regs = [dict(L, mask=_lmask(L, (h, w))) for L in res["layers"] if L["role"] == "region"]
    glow = [dict(L, mask=_lmask(L, (h, w))) for L in res["layers"] if L["role"] == "glow"]
    parts, merged = {}, {}
    for i, L in enumerate(regs):
        parts.setdefault(assign.get(i + 1, "other"), []).append(L)
    if "other" in parts and "other" not in order and len(parts) > 1:
        # unnamed slivers join the named part they share the most boundary with
        for L in parts.pop("other"):
            ring = _dilate(L["mask"], 2) & ~L["mask"]
            near = {p: sum(int((ring & R["mask"]).sum()) for R in rs) for p, rs in parts.items()}
            parts[max(near, key=near.get)].append(L)
            merged[regs.index(L) + 1] = max(near, key=near.get)
    order = [p for p in order if p in parts or (p == "glow" and glow)]
    order = order + [p for p in parts if p not in order]
    pmask = {p: np.logical_or.reduce([L["mask"] for L in parts[p]]) for p in parts}
    glows = {}
    if glow and glow_mode == "part":
        # each part owns the glow nearest to it (Voronoi cells), so the glow travels with whatever emits it
        cell = np.zeros((h, w), np.uint8)
        named = [p for p in order if p in pmask]
        for k_, p in enumerate(named):
            cell[pmask[p]] = k_ + 1
        arr = np.asarray(src.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
        alpha = arr[..., 3] / 255
        for _ in range(4 * max(w, h)):
            todo = (cell == 0) & (alpha > 0.01)
            if not todo.any():
                break
            grown = np.asarray(Image.fromarray(cell).filter(ImageFilter.MaxFilter(3)))
            cell = np.where(cell == 0, grown, cell)
        P_ = res["params"]
        for k_, p in enumerate(named):
            a_p = np.where(cell == k_ + 1, alpha, 0)
            gl = _alpha_bands(a_p, arr[..., :3], a_p >= 0.5, P_.get("bands", 8), P_.get("tol", 0.8) * 1.5,
                              tuple(P_.get("band_kinds", ("flat",))), P_.get("band_gamma", 1.6))
            if gl:
                glows[f"glow_{p}"] = gl
        order = list(glows) + order
    elif glow:
        glows = {"glow": [L for L in res["layers"] if L["role"] == "glow"]}  # bbox-cropped masks, as _alpha_bands stores them
        order = ["glow"] + [p for p in order if p != "glow"]
    am = {}
    if amodal:
        for p, img in amodal.items():
            am[p] = np.asarray(img.resize((w, h), Image.LANCZOS).getchannel("A")) > 128
    defs, static, boxes, info = [], [], {}, {}
    for zi, p in enumerate(order):
        if p in glows:
            paths = []
            for L in glows[p]:
                f, gd = _fill_svg(L["fill"], f"{p}_{len(defs)}")
                defs.append(gd)
                paths.append({"d": L["d"], "fill": f, "opacity": L["opacity"]})
            m = np.logical_or.reduce([_lmask(L, (h, w)) for L in glows[p]])
        else:
            above = np.zeros((h, w), bool)
            for q in order[zi + 1:]:
                if q in pmask:
                    above |= pmask[q]
            pm = pmask[p]
            r = max(2, int(ext_frac * max(np.ptp(np.nonzero(pm)[0]) + 1, np.ptp(np.nonzero(pm)[1]) + 1)))
            if p in am:
                m = pm | (am[p] & above)
            else:
                m = pm | ((_fill_holes(pm) | _hull_mask(pm) | _dilate(pm, r)) & above)
            rs = sorted(parts[p], key=lambda L: -int(_fill_holes(L["mask"]).sum()))
            later = np.zeros((h, w), bool)
            draws = {}
            for j in range(len(rs) - 1, -1, -1):
                L = rs[j]
                draws[j] = m if j == 0 else (_fill_holes(L["mask"]) & pm) | (_dilate(L["mask"], 2) & later)
                later |= L["mask"]
            paths = []
            for j, L in enumerate(rs):
                if j and L["area"] < min_detail * pm.sum():
                    continue
                d = mask_to_path(draws[j], tol, corner, 0.7, 4.0)
                if not d:
                    continue
                f, gd = _fill_svg(L["fill"], f"{p}_{len(defs)}")
                defs.append(gd)
                paths.append({"d": d, "fill": f, "opacity": 1})
            info[p] = {"regions": [regs.index(L) + 1 for L in rs], "amodal_px": int((m & ~pm).sum())}
        if not paths:
            continue
        ys, xs = np.nonzero(m)
        boxes[p] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        static.append({"name": p, "paths": paths})
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static],
            "rig": {"parts": info, "boxes": boxes, "src_size": list(src.size), "work_scale": round(sc, 5),
                    "merged": merged}}


def _vec_render(vec, size, k=0, only=None):
    """layers.json -> SVG text at keyframe k, optionally only some layers (exploded previews)."""
    v = json.loads(json.dumps(vec))
    if only is not None:
        v["static_layers"] = [l for l in v["static_layers"] if l["name"] in only]
        v["animated_layers"] = [l for l in v["animated_layers"] if l["name"].split("__")[0] in only]
    for al in v["animated_layers"]:
        al["keyframes"] = [al["keyframes"][k % len(al["keyframes"])]]
        if al.get("grad_keys"):
            gid = re.match(r"url\(#([^)]+)\)", al["fill"]).group(1)
            v["defs"] = _grad_patch(v["defs"], gid, al["grad_kind"], al["grad_keys"][k % len(al["grad_keys"])])
    return _vec_svg(v, 0, size=1).replace('width="1" height="1"', f'width="{size[0]}" height="{size[1]}"', 1)


def _parse_kv(s, sep=";"):
    out = {}
    for item in (s or "").split(sep):
        k, eq, v = item.partition("=")
        if eq and k.strip():
            out[k.strip()] = v.strip()
    return out


def _parse_motions(s):
    """'outer_flame=flicker:1.2:0.3; glow=flicker:0.8@outer_flame' -> {layer: (motion, amp, phase, anchor)}"""
    out = {}
    for k, v in _parse_kv(s).items():
        v, _, anchor = v.partition("@")
        bits = v.split(":")
        if bits[0].strip() not in TR_MOTIONS:
            raise ValueError(f"unknown motion {bits[0].strip()!r} in '{k}={v}' (choose {', '.join(TR_MOTIONS)})")
        try:
            out[k] = (bits[0].strip(), float(bits[1]) if len(bits) > 1 and bits[1] else 1.0,
                      float(bits[2]) if len(bits) > 2 and bits[2] else 0.0, anchor.strip() or None)
        except ValueError:
            raise ValueError(f"bad amp/phase in '{k}={v}' (use layer=motion[:amp[:phase]][@anchor], numbers only)")
        if not (abs(out[k][1]) <= 10 and abs(out[k][2]) <= 100):  # also nan/inf: they became 'nan' path data
            raise ValueError(f"amp/phase out of range in '{k}={v}' (amp is a multiplier around 1, at most 10)")
    return out


def _motion_stats(vec):
    """Mean / max point travel (viewBox units) of each animated layer across its keyframes."""
    st = {}
    for al in vec["animated_layers"]:
        P = [_path_points(k) for k in al["keyframes"]]
        dev = np.array([np.linalg.norm(p - P[0], axis=1) for p in P])
        st[al["name"]] = {"mean": round(float(dev.mean()), 2), "max": round(float(dev.max()), 2)}
    return st


def write_animation(vec, out, name, seconds=1.6, fps=30, size=512, ease="linear", bg="#1b1d33", strip=8):
    """layers.json with animated layers -> Lottie + preview.html + keyframe sheet + lottie-web filmstrip."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    errs = _validate_vec(vec)
    if errs:
        raise ValueError("; ".join(errs))
    (out / "layers.json").write_text(json.dumps(vec, indent=1))
    lot = build_lottie(vec, size, seconds, fps)
    lot = _tidy_lottie(lot, ease == "linear")
    _tr_animate_gradients(lot, vec, ease == "linear")
    (out / f"{name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    lib = _lottie_js()  # inline the cached lottie-web so the preview also opens offline; CDN only as a fallback
    (out / "preview.html").write_text(
        f"<!doctype html><meta charset=utf-8><body style='margin:0;background:{bg}'>"
        f"<div id=a style='width:{lot['w']}px;height:{lot['h']}px'></div>"
        + (f"<script>{lib}</script>" if lib else f"<script src='{TR_LOTTIE_JS}'></script>")
        + f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{json.dumps(lot)}}})</script>")
    vb = [float(x) for x in vec["viewBox"].split()]
    sz = (round(300 * vb[2] / max(vb[2:])), round(300 * vb[3] / max(vb[2:])))
    nk = max((len(al["keyframes"]) for al in vec["animated_layers"]), default=1)
    stats = _motion_stats(vec)
    report = {"lottie": str(out / f"{name}.json"), "bytes": len(json.dumps(lot, separators=(",", ":"))),
              "frames": lot["op"], "fps": fps, "motion": stats}
    for nm, v in stats.items():
        if v["max"] < 0.2:
            print(f"  ! {nm}: max travel {v['max']} viewBox units (< 0.2): the motion is invisible; raise the amplitude or "
                  f"check the layer's box")
    # the art without its glow bands: soft halos would hide a gap in the tear metric
    art = [n.split("__")[0] for n in vec["layer_order"] if not n.split("__")[0].startswith("glow")]
    art = list(dict.fromkeys(art)) or None
    svgs = [_vec_render(vec, sz, k) for k in range(nk)] + [_vec_render(vec, sz, k, only=art) for k in range(nk)]
    rens = _tr_render_svgs(svgs, [sz] * len(svgs))
    if rens:
        keys, bare = rens[:nk], rens[nk:]
        sheet([[(f"key {k + 1}", _on(im, bg)) for k, im in enumerate(keys)]], out / "keys_review.png", T=220)
        report["tear_px"], report["tear_frac_max"], report["tear_radius_px"] = _tear_px(bare, vec, stats, sz)
    else:
        report["keys_review"] = f"not rendered: {_tr_chrome_reason()}"
        print(f"  ! keys_review.png and tear_px NOT measured (Chrome failed: {_tr_chrome_reason()})")
    frames = [round(lot["op"] * i / strip) for i in range(strip)]
    film = lottie_filmstrip(lot, frames, 220, bg)
    if film is not None:
        film.save(out / "lottie_frames.png")
        tiles = [np.asarray(film.crop((i * film.width // strip, 0, (i + 1) * film.width // strip, film.height)), np.float32)
                 for i in range(strip)]
        report["lottie_frame_delta"] = [round(float(np.abs(tiles[i] - tiles[i - 1]).mean()), 2) for i in range(1, strip)]
        report["lottie_rendered"] = bool(np.asarray(film)[..., :3].reshape(-1, 3).std(0).max() > 3)
    else:
        report["lottie_rendered"] = f"not measured: {_tr_chrome_reason()}"
        print(f"  ! lottie_frames.png NOT rendered (Chrome failed: {_tr_chrome_reason()})")
    (out / "animation_report.json").write_text(json.dumps(report, indent=2))
    return report


def _tear_px(keys, vec, stats, sz):
    """Tears per key: rest-pose interior (eroded by the largest travel + 1 px) left uncovered by that key's art.
    Open and enclosed gaps both count; key 0 is the rest pose, so its value is 0 by construction."""
    vb = [float(x) for x in vec["viewBox"].split()]
    travel = max((v["max"] for v in stats.values()), default=0.0) * max(sz) / max(vb[2:])
    r = int(min(max(2, math.ceil(travel) + 1), max(sz) // 4))
    ops = [np.asarray(im.getchannel("A")) > 127 for im in keys]
    core = np.asarray(Image.fromarray(ops[0].astype(np.uint8) * 255).filter(ImageFilter.MinFilter(2 * r + 1))) > 127
    tears = [int((core & ~op).sum()) for op in ops]
    return tears, round(max(tears) / max(1, int(ops[0].sum())), 5), r


def cmd_animate(a):
    src = Path(a.layers)
    vec = _read_json(src, "layers.json", required=("viewBox", "static_layers"))
    errs = _check_vec(vec)
    if errs:
        sys.exit(f"animate: {src} invalid: {errs[0]}" + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else "") +
                 "; fix it or re-export it from rig, vectorize or trace")
    vec.setdefault("animated_layers", [])
    try:
        specs = _parse_motions(a.part)
        if not specs:
            raise ValueError("give --part 'layer=motion[:amp[:phase]]; ...' (motions: " + ", ".join(TR_MOTIONS) + ")")
        vec = animate_layers(vec, specs, a.keyframes)
    except ValueError as e:
        sys.exit(f"animate: {e}")
    stem = src.name[:-len(".layers.json")] if src.name.endswith(".layers.json") else ""
    out = Path(a.out) if a.out else src.parent / "anim" / stem
    try:
        rep = write_animation(vec, out, a.name or stem or src.parent.name, a.seconds, a.fps, a.size, a.ease, a.preview_bg)
    except ValueError as e:
        sys.exit(f"animate: {e}")
    made = [f for f in ("keys_review.png", "lottie_frames.png", "preview.html") if (out / f).exists()]
    print(f"-> {out}/layers.json, {rep['lottie']} ({rep['bytes'] // 1024} KB, {rep['frames']} frames), {', '.join(made)}")
    for k, v in rep["motion"].items():
        print(f"  {k:24s} travel mean={v['mean']} max={v['max']} (viewBox units)")
    if "lottie_frame_delta" in rep:
        print(f"  lottie-web frame-to-frame delta: {rep['lottie_frame_delta']}  rendered={rep['lottie_rendered']}")


def cmd_rig(a):
    d = Path(a.dir)
    src = _open_image(d / "assets" / f"{a.name}.png", "asset", "forge it first or copy the PNG to DIR/assets/")
    out = d / "rig" / a.name
    parts = _parse_kv(a.parts)
    if not parts:
        sys.exit("rig: give --parts 'name=description; ...' listed bottom to top")
    order = [p.strip() for p in (a.order.split(",") if a.order else parts) if p.strip()]
    unknown = [p for p in order if p not in parts]
    if unknown:
        sys.exit(f"rig: --order names {', '.join(unknown)} which are not in --parts ({', '.join(parts)})")
    if a.route == "gpt" and (a.assign or a.min_detail):
        print("  ! --route gpt traces GPT's redraws directly: --assign, --namer and --min-detail are not used")
    res = trace_image(src, a.preset)
    if not any(L["role"] == "region" for L in res["layers"]):
        sys.exit(f"rig: nothing opaque to rig in {d / 'assets' / f'{a.name}.png'} (no traced region)")
    out.mkdir(parents=True, exist_ok=True)
    region_overlay(src, res).save(out / "regions.png")
    amodal, gpt_imgs = None, {}
    if a.route in ("gpt", "hybrid"):
        (out / "parts_gpt").mkdir(exist_ok=True)
        cached = {p: out / "parts_gpt" / f"{p}_raw.png" for p in parts}
        todo = {p: v for p, v in parts.items() if a.force or not cached[p].exists()}
        if todo:
            print(f"isolating {len(todo)} part(s) with the image model ({len(todo)} image calls)...")
            done, failed = isolate_parts(src, todo, a.model, cache=cached)
            if failed:
                for p, (err, prompt) in failed.items():
                    print(f"  ! part {p!r}: {err} — prompt \"{prompt}\" — edit the part description or use --route local --assign")
                kept = len(parts) - len(todo)
                sys.exit(f"rig: {len(failed)} of {len(todo)} part isolation(s) failed; {len(done)} succeeded and are cached in "
                         f"{out / 'parts_gpt'}" + (f" next to {kept} from an earlier run" if kept else "") + " (re-run to resume)")
        reg_info = {}
        for p in parts:
            gpt_imgs[p], reg_info[p] = register_part(Image.open(cached[p]).convert("RGBA"), src)
            gpt_imgs[p].save(out / "parts_gpt" / f"{p}.png")
        print("  registered GPT parts (scale, shift px):", reg_info)
        amodal = gpt_imgs
    hint = f"regions -> {out / 'regions.png'}: Read it, then re-run with --assign 'part=1,2,3; other_part=4,5'"
    if a.route == "gpt":
        vec = _rig_from_parts(src, gpt_imgs, order, res, a.preset)
    else:
        if a.assign:
            assign, issues = parse_assign(a.assign, res)
            for line in issues:
                print(f"  ! {line}")
        elif a.route == "hybrid":
            assign = _assign_by_masks(res, gpt_imgs, order)
        elif a.namer == "gpt":
            print("naming regions with the vision model (1 call)...")
            try:
                assign = name_regions(src, res, parts, a.text_model)
            except (ApiError, ValueError, LookupError, TypeError) as e:  # ApiError, or an answer the int()/dict parse rejects
                print(f"  ! region naming failed: {str(e)[:300]}")
                sys.exit(hint)
        else:
            print(hint)
            return
        stray = sorted({p for p in assign.values() if p not in parts and p != "other"})
        for p in stray:
            print(f"  ! part {p!r} is not in --parts (it becomes an extra layer; a typo?)")
        empty = [p for p in parts if p not in assign.values()]
        if empty:
            print(f"  ! parts with no regions: {', '.join(empty)}")
            if not a.allow_empty:
                sys.exit(f"rig: {', '.join(empty)} got no region; fix --assign (regions.png) or pass --allow-empty")
        vec = build_rig(src, res, assign, order, amodal if a.route == "hybrid" else None, min_detail=a.min_detail,
                        glow_mode=a.glow)
        merged = vec["rig"].get("merged", {})
        if merged:
            regs = [L for L in res["layers"] if L["role"] == "region"]
            frac = sum(regs[i - 1]["area"] for i in merged) / max(1, sum(L["area"] for L in regs))
            print(f"  unassigned regions {','.join(str(i) for i in sorted(merged))} ({100 * frac:.1f}% of area) merged into "
                  f"their neighbours: {', '.join(f'{i}->{p}' for i, p in sorted(merged.items()))}")
        final = {str(i): {"part": assign.get(i, merged.get(i, "other")), **({"merged": True} if i in merged else {})}
                 for i in range(1, sum(L["role"] == "region" for L in res["layers"]) + 1)}
        (out / "assign.json").write_text(json.dumps(final, indent=1))
    vec["rig"]["route"] = a.route
    (out / "layers.json").write_text(json.dumps(vec, indent=1))
    names = vec["layer_order"]
    svgs = [_vec_render(vec, src.size)] + [_vec_render(vec, src.size, only=[p]) for p in names]
    rens = _tr_render_svgs(svgs, [src.size] * len(svgs))
    rep = {"route": a.route, "layers": names, "parts": vec["rig"]["parts"]}
    if rens:
        rep["recomposite_vs_source"] = trace_metrics(src, rens[0], 192)
        rows = [[("source", _on(src, a.preview_bg)), (f"rig ({a.route})", _on(rens[0], a.preview_bg))] +
                [(p, _on(r, a.preview_bg)) for p, r in zip(names, rens[1:])]]
        if gpt_imgs:
            rows.append([(f"gpt {p}", _on(im, a.preview_bg)) for p, im in gpt_imgs.items()])
        sheet(rows, out / "rig_preview.png", T=260)
    print(f"rig ({a.route}) -> {out / 'layers.json'}  layers bottom->top: {', '.join(names)}")
    if rens:
        r_ = rep["recomposite_vs_source"]["display"]
        print(f"  recomposite vs source (display): diff={r_['diff']} iou={r_['iou']}   preview -> {out / 'rig_preview.png'}")
    else:
        rep["recomposite_vs_source"] = f"not measured: {_tr_chrome_reason()}"
        print(f"  ! recomposite fidelity NOT measured, no rig_preview.png (Chrome failed: {_tr_chrome_reason()})")
    if a.animate:
        try:
            vec = animate_layers(vec, _parse_motions(a.animate), a.keyframes)
            rep["animation"] = write_animation(vec, out / "anim", a.name, a.seconds, a.fps, 512, "linear", a.preview_bg)
        except ValueError as e:
            (out / "rig_report.json").write_text(json.dumps(rep, indent=2, default=str))
            sys.exit(f"rig --animate: {e} (the rig itself was written)")
        made = [f for f in ("keys_review.png", "lottie_frames.png", "preview.html") if (out / "anim" / f).exists()]
        print(f"  animated -> {out / 'anim'} (Lottie {a.name}.json, {', '.join(made)})")
        for k, v in rep["animation"]["motion"].items():
            print(f"    {k:24s} travel mean={v['mean']} max={v['max']}")
    (out / "rig_report.json").write_text(json.dumps(rep, indent=2, default=str))


def parse_assign(spec, res):
    """'drop=c1; inner_ring=c2; outer_ring=rest; logs=3,4,7' -> {region number: part} (cN = a whole component)."""
    comps = region_components(res)
    n, nc = len(comps), max(comps.values(), default=0)
    assign, rest, bad = {}, None, []
    for part, ids in _parse_kv(spec).items():
        for tok in re.split(r"[,\s]+", ids.strip()):
            if tok == "rest":
                rest = part
            elif tok.startswith("c") and tok[1:].isdigit():
                hit = {i: part for i, c in comps.items() if c == int(tok[1:])}
                assign.update(hit) if hit else bad.append(tok)
            elif tok.isdigit() and 1 <= int(tok) <= n:
                assign[int(tok)] = part
            elif tok:
                bad.append(tok)
    if rest:
        assign.update({i: rest for i in comps if i not in assign})
    issues = [f"--assign: {', '.join(bad)} do not exist (regions 1..{n}, components c1..c{nc})"] if bad else []
    return assign, issues


def _assign_by_masks(res, masks, order):
    """Hybrid route: each traced region goes to the top-most GPT part that covers most of it."""
    w, h = res["size"]
    M = {p: np.asarray(im.resize((w, h), Image.LANCZOS).getchannel("A")) > 128 for p, im in masks.items()}
    assign = {}
    for i, L in enumerate(r for r in res["layers"] if r["role"] == "region"):
        x0, y0, x1, y1 = L["bbox"]
        cover = {p: (M[p][y0:y1, x0:x1] & L["mask"]).sum() / max(1, L["mask"].sum()) for p in M}
        top = [p for p in reversed(order) if p in cover and cover[p] >= 0.5]
        assign[i + 1] = top[0] if top else max(cover, key=cover.get)
    return assign


def _rig_from_parts(src, imgs, order, res, preset):
    """Route (a): trace every GPT-isolated part on its own and stack them (glow bands from the source trace)."""
    w, h = res["size"]
    defs, static, boxes, info = [], [], {}, {}
    glow = [L for L in res["layers"] if L["role"] == "glow"]
    if glow:
        paths = []
        for L in glow:
            f, gd = _fill_svg(L["fill"], f"glow_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": L["opacity"]})
        static.append({"name": "glow", "paths": paths})
    for p in order:
        r = trace_image(imgs[p], preset, bands=0, work=max(w, h))
        paths = []
        for L in r["layers"]:
            f, gd = _fill_svg(L["fill"], f"{p}_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": 1})
        if paths:
            pts = np.vstack([_path_points(x["d"]) for x in paths])
            boxes[p] = [float(v) for v in (*pts.min(0), *pts.max(0))]
            static.append({"name": p, "paths": paths})
            info[p] = {"paths": len(paths)}
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static], "rig": {"parts": info, "boxes": boxes, "src_size": list(src.size)}}


def _tr_int_min(lo, hi=None):
    """argparse type: integer >= lo (e.g. --work 16, --keyframes 2), and <= hi when given."""
    def t(s):
        v = _pos_int(s)
        if v < lo or (hi is not None and v > hi):
            raise argparse.ArgumentTypeError(f"expected an integer >= {lo}" + (f" and <= {hi}" if hi else "") + f", got {v}")
        return v
    return t


def _tr_nonneg_int(s):
    """argparse type: integer >= 0 (--bands 0 = no bands)."""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {s!r}")
    if v < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {v}")
    return v


def _add_trace_rig_parsers(sub):
    p = sub.add_parser("trace", help="raster asset -> multi-colour SVG + VectorDrawable with gradient fits, glow "
                                     "bands, measured fidelity and a vector-or-raster recommendation")
    p.add_argument("files", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=["auto", "flat", "illustration", "gradient", "vtracer"], default="auto")
    p.add_argument("--dp", type=_pos_float, help="display width in dp (default: from assets.json when tracing DIR/assets/*.png, "
                                                 "else the VD is 48dp on its long side)")
    p.add_argument("--display", type=_tr_int_min(16), help="px size (long side) the display-size fidelity is measured at "
                                                          "(default: 3x the dp width, else max(192, 96 x aspect))")
    p.add_argument("--work", type=_tr_int_min(16), help="tracing resolution, long side (default 512, never above the source size)")
    p.add_argument("--colors", type=_pos_int, help="k-means colours before region merging")
    p.add_argument("--bands", type=_tr_nonneg_int, help="alpha bands used to rebuild glow/soft edges (0 = none)")
    p.add_argument("--tol", type=_pos_float, help="curve-fit tolerance in work px")
    p.add_argument("--name", help="output stem (single input only; the VD is --prefix + name, sanitised to [a-z0-9_])")
    p.add_argument("--prefix", default="img_", help="VectorDrawable file prefix")
    p.add_argument("--bg", type=_color_arg, default="#1b1d33", help="preview backdrop"); p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("rig", help="split a forged illustration into named, registered, gap-free vector parts "
                                   "(layers.json for Lottie), optionally animated procedurally")
    p.add_argument("dir"); p.add_argument("name")
    p.add_argument("--parts", required=True, help="'name=description; ...' listed bottom to top")
    p.add_argument("--route", choices=["local", "hybrid", "gpt"], default="local",
                   help="local: traced regions named by --assign or vision; hybrid: GPT part masks assign the traced "
                        "regions and complete occluded shapes; gpt: trace GPT's isolated part redraws")
    p.add_argument("--assign", help="'part=1,2,3; other=4,5' region numbers from regions.png (skips the vision call)")
    p.add_argument("--namer", choices=["gpt", "none"], default="gpt", help="none: only write regions.png for Claude")
    p.add_argument("--order", help="comma list bottom->top (default: --parts order)")
    p.add_argument("--preset", choices=list(TRACE_PRESETS), default="illustration")
    p.add_argument("--min-detail", type=_unit_float, default=0.0, help="drop detail regions smaller than this fraction of "
                                                                       "their part (smaller Lottie; the part base covers them)")
    p.add_argument("--allow-empty", action="store_true", help="continue when a --parts part gets no region (default: exit)")
    p.add_argument("--animate", help="'part=motion[:amp[:phase]][@anchor_part]; ...' motions: " + ", ".join(TR_MOTIONS))
    p.add_argument("--keyframes", type=_tr_int_min(2, 120), default=8, help="procedural morph keys per loop (2-120)")
    p.add_argument("--seconds", type=_pos_float, default=1.6, help="loop length")
    p.add_argument("--fps", type=_pos_int, default=30)
    p.add_argument("--force", action="store_true", help="re-run the GPT part isolation even when parts_gpt/ is cached")
    p.add_argument("--model", default=None, help="image model for --route hybrid|gpt")
    p.add_argument("--text-model", default=None, help="vision model for --namer gpt")
    p.add_argument("--glow", choices=["part", "shared"], default="part",
                   help="part: each part owns (and moves with) the glow nearest to it; shared: one glow layer")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="backdrop of rig_preview.png / keys_review.png"); p.set_defaults(fn=cmd_rig)
    p = sub.add_parser("animate", help="procedural morph keyframes (sway/flicker/breathe/squash/bob/wave/ripple) "
                                       "for named layers of a layers.json -> Lottie; no API calls")
    p.add_argument("layers", help="layers.json from rig, vectorize or trace")
    p.add_argument("--part", required=True, help="'layer=motion[:amp[:phase]][@anchor_layer]; ...' (anchor: move "
                                                 "with that layer's field, e.g. glow=flicker@outer_flame)")
    p.add_argument("--keyframes", type=_tr_int_min(2, 120), default=8, help="procedural morph keys per loop (2-120)")
    p.add_argument("--seconds", type=_pos_float, default=1.6, help="loop length")
    p.add_argument("--fps", type=_pos_int, default=30); p.add_argument("--size", type=_tr_int_min(16), default=512,
                                                                     help="Lottie canvas long side in px")
    p.add_argument("--ease", choices=["linear", "keys"], default="linear",
                   help="linear between procedural keys (smooth); keys = build_lottie's ease at every key")
    p.add_argument("--out", help="output dir (default: <layers dir>/anim/<stem>)")
    p.add_argument("--name", help="Lottie file stem (default: the layers.json stem)")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="backdrop of keys_review.png / lottie_frames.png / preview.html")
    p.set_defaults(fn=cmd_animate)


# ---- design-to-code: colour science + text measurement (shared by tokens, fonts, compare)

_SRGB_LIN = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92, ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4)


def _lab_arr(rgb):
    """sRGB 0-255 array (...x3) -> CIELAB (D65). Integer-valued input goes through a 256-entry linearisation table."""
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        c = _SRGB_LIN[a]
    else:
        a = a.astype(np.float64, copy=False)
        r = np.rint(a)
        if a.size and np.array_equal(a, r) and 0 <= a.min() and a.max() <= 255:
            c = _SRGB_LIN[r.astype(np.uint8)]
        else:
            c = np.clip(a, 0, 255) / 255.0
            c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124564, 0.2126729, 0.0193339], [0.3575761, 0.7151522, 0.1191920],
                        [0.1804375, 0.0721750, 0.9503041]])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _de00(lab1, lab2):
    """CIEDE2000 colour difference (vectorised)."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=np.float64), -1, 0)
    cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cb ** 7 / (cb ** 7 + 25.0 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1, c2 = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    zero = c1 * c2 == 0
    dh = h2 - h1
    dh = np.where(zero, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dH = 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh / 2))
    lm, cm, hs = (L1 + L2) / 2, (c1 + c2) / 2, h1 + h2
    hm = np.where(zero, hs, np.where(np.abs(h1 - h2) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    t = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    rt = -np.sin(np.radians(60 * np.exp(-(((hm - 275) / 25) ** 2)))) * 2 * np.sqrt(cm ** 7 / (cm ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lm - 50) ** 2 / np.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + 0.045 * cm, 1 + 0.015 * cm * t
    dl, dc = (L2 - L1) / sl, (c2 - c1) / sc
    return np.sqrt(dl ** 2 + dc ** 2 + (dH / sh) ** 2 + rt * dc * (dH / sh))


def _norm_hex(s):
    """'#rgb', '#rgba', '#rrggbb', '#rrggbbaa', 'rgb(...)' or 'rgba(...)' -> ('#RRGGBB', alpha 0..1); ValueError otherwise."""
    t = str(s).strip()
    m_ = re.fullmatch(r"rgba?\(\s*([^)]*?)\s*\)", t, re.I)
    if m_:
        parts = [p for p in re.split(r"[,\s/]+", m_.group(1)) if p]
        if len(parts) not in (3, 4):
            raise ValueError(f"unreadable colour {s!r}; use #RRGGBB")
        try:
            chan = lambda v: float(v[:-1]) * 2.55 if v.endswith("%") else float(v)
            rgb = [int(round(min(255.0, max(0.0, chan(v))))) for v in parts[:3]]
            a = (float(parts[3][:-1]) / 100 if parts[3].endswith("%") else float(parts[3])) if len(parts) == 4 else 1.0
        except ValueError:
            raise ValueError(f"unreadable colour {s!r}; use #RRGGBB")
        return "#" + "".join(f"{v:02X}" for v in rgb), min(1.0, max(0.0, a))
    h = t[1:] if t.startswith("#") else t
    if len(h) not in (3, 4, 6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", h):
        raise ValueError(f"unreadable colour {s!r}; use #RRGGBB")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    return "#" + h[:6].upper(), (int(h[6:8], 16) / 255 if len(h) == 8 else 1.0)


def _hexi(h):
    return tuple(int(_norm_hex(h)[0][i:i + 2], 16) for i in (1, 3, 5))


def _to_hex(rgb):
    return "#" + "".join(f"{int(round(min(255, max(0, float(v))))):02X}" for v in rgb)


def _de_hex(a, b):
    return round(float(_de00(_lab_arr(_hexi(a)), _lab_arr(_hexi(b)))), 1)


def _dominant(rgb):
    px = rgb.reshape(-1, 3).astype(int)
    key = (px[:, 0] >> 4) * 256 + (px[:, 1] >> 4) * 16 + (px[:, 2] >> 4)
    return px[key == np.bincount(key).argmax()].mean(0)


def _d2c_runs(flags, merge=1):
    """[start, end) runs of True, joining runs separated by gaps shorter than `merge`."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(flags, dtype=np.int8), [0]])))
    out = []
    for s, e in zip(idx[::2], idx[1::2]):
        if out and s - out[-1][1] < merge:
            out[-1][1] = int(e)
        else:
            out.append([int(s), int(e)])
    return out


def _blobs(mask):
    """8-connected components of a small boolean mask -> list of (ys, xs)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, ys, xs = [(y0, x0)], [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        out.append((np.array(ys), np.array(xs)))
    return out


def _text_expect(text, size, tracking=0.0):
    """Rough ink height/width of one text line, only used to pick the right line/segment."""
    up = any(c.isupper() or c.isdigit() or c in "bdfhklt" for c in text)
    down = any(c in "gjpqy," for c in text)
    adv = lambda c: (0.28 if c in "iIl1jtfr.,:;'!|" else 0.85 if c in "mwMW" else 0.25 if c == " " else
                     0.64 if c.isupper() or c.isdigit() else 0.52)
    width = sum(adv(c) + tracking for c in text) - adv(text[-1]) / 2 if text else 0
    return size * ((0.71 if up else 0.52) + (0.21 if down else 0)), size * max(0.3, width)


def _ink_stats(region, bg):
    d = np.linalg.norm(_lab_arr(region) - _lab_arr(bg), axis=-1)
    m = d > 0.5 * np.percentile(d, 99.5)
    inner = m.copy()
    inner[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:] & m[:-2, :-2] & m[2:, 2:] & m[:-2, 2:] & m[2:, :-2]
    inner[[0, -1], :] = False
    inner[:, [0, -1]] = False
    core = inner if inner.sum() >= 40 else d >= 0.95 * np.percentile(d, 99.5)
    ink = np.median(region[core], 0) if core.any() else bg
    lin = lambda c: np.where(np.asarray(c) / 255 <= 0.04045, np.asarray(c) / 255 / 12.92, ((np.asarray(c) / 255 + 0.055) / 1.055) ** 2.4)
    v = lin(ink) - lin(bg)
    alpha = np.clip(((lin(region) - lin(bg)) @ v) / max(float(v @ v), 1e-9), 0, 1)
    return ink, alpha


def _baseline(keep, r):
    prof = np.append(keep[r[0]:r[1]].sum(1).astype(float), 0)
    half = (len(prof) - 1) // 2
    return r[0] + half + int(np.argmax(prof[half:-1] - prof[half + 1:])) + 1


def _locate_text(rgb, box, text, size, tracking=0.0, color=None, lines=1):
    """Tight ink box, background, ink colour and coverage of one text line searched inside `box`."""
    x, y, w, h = box
    H, W = rgb.shape[:2]
    pad = max(4, int(0.2 * size))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    crop = rgb[y0:y1, x0:x1].astype(np.float64)
    if crop.size == 0:
        return None
    d = np.linalg.norm(_lab_arr(crop) - _lab_arr(_dominant(crop)), axis=-1)
    mask = d > max(15.0, 0.3 * np.percentile(d, 99.5))
    keep = np.zeros_like(mask)
    ch, cw = mask.shape
    bg0 = _dominant(crop)
    hint = np.array(_hexi(color), dtype=np.float64) if color else _ink_stats(crop, bg0)[0]
    v = hint - bg0
    alpha0 = ((crop - bg0) @ v) / max(float(v @ v), 1e-6)
    if color:
        mask &= alpha0 > 0.3
    edge = lambda ys, xs: ys.min() == 0 or xs.min() == 0 or ys.max() == ch - 1 or xs.max() == cw - 1
    min_area = max(4, (0.06 * size) ** 2)
    for ys, xs in _blobs(mask):
        if edge(ys, xs):
            sub = np.zeros_like(mask)
            sub[ys, xs] = alpha0[ys, xs] >= 0.6
            for ys2, xs2 in _blobs(sub):
                if len(ys2) >= min_area and not edge(ys2, xs2):
                    keep[ys2, xs2] = True
        elif len(ys) >= min_area and alpha0[ys, xs].max() >= 0.55:
            keep[ys, xs] = True
    e_h, e_w = _text_expect(text, size, tracking)
    runs = _d2c_runs(keep.any(1), merge=max(2, int(0.12 * size)))
    if not runs:
        return None

    def run_cost(i, r):
        cost = abs(math.log(max(r[1] - r[0], 1) / e_h))
        if color:
            band = crop[r[0]:r[1]]
            ink, _ = _ink_stats(band, _dominant(band))
            cost += float(_de00(_lab_arr(ink), _lab_arr(_hexi(color)))) / 25
        return cost
    costs = [run_cost(i, r) for i, r in enumerate(runs)]
    ri = next(i for i, c in enumerate(costs) if c <= min(costs) + 0.3)
    ry0, ry1 = runs[ri]
    pitch = None
    if lines > 1 and ri + 1 < len(runs):
        nxt = runs[ri + 1]
        if 0.7 < (nxt[1] - nxt[0]) / (ry1 - ry0) < 1.4 and nxt[0] - ry1 < size:
            pitch = int(_baseline(keep, nxt) - _baseline(keep, runs[ri]))
    lh = ry1 - ry0
    segs = _d2c_runs(keep[ry0:ry1].any(0), merge=max(3, int(0.6 * lh)))
    if len(segs) > 2 and segs[-1][1] - segs[0][0] > 1.8 * e_w:
        groups = [(i, j) for i in range(len(segs)) for j in range(i, len(segs))]
        i, j = min(groups, key=lambda g: abs(segs[g[1]][1] - segs[g[0]][0] - e_w) / e_w + 0.25 * g[0])
        segs = segs[i:j + 1]
    hint_lab = _lab_arr(_hexi(color)) if color else None
    seg_de = lambda sg: float(_de00(_lab_arr(_ink_stats(crop[ry0:ry1, sg[0]:sg[1]], _dominant(crop[ry0:ry1]))[0]), hint_lab))
    while len(segs) > 1:
        full = segs[-1][1] - segs[0][0]
        gaps = [segs[n + 1][0] - segs[n][1] for n in range(len(segs) - 1)]
        drop = None
        for i, gap in ((0, gaps[0]), (-1, gaps[-1])):
            wseg = segs[i][1] - segs[i][0]
            inner_gaps = gaps[1:] if i == 0 else gaps[:-1]
            big_gap = gap >= 1.3 * max(inner_gaps) if inner_gaps else gap >= 0.5 * lh
            closer = abs(full - wseg - gap - e_w) < abs(full - e_w)
            if hint_lab is not None and seg_de(segs[i]) > 15 and min(seg_de(sg) for sg in segs) < 10:
                drop = i
            elif wseg <= 2.0 * lh and big_gap and closer:
                drop = i
            if drop is not None:
                break
        if drop is None:
            break
        segs.pop(drop)
    cx0, cx1 = segs[0][0], segs[-1][1]
    rows = np.flatnonzero(keep[ry0:ry1, cx0:cx1].any(1))
    ty0, ty1 = ry0 + rows[0], ry0 + rows[-1] + 1
    m = max(3, int(0.15 * size))
    bg = _dominant(crop[max(0, ty0 - m):ty1 + m, max(0, cx0 - m):cx1 + m])
    ink, alpha = _ink_stats(crop[ty0:ty1, cx0:cx1], bg)
    return {"box": [x0 + int(cx0), y0 + int(ty0), int(cx1 - cx0), int(ty1 - ty0)], "bg": _to_hex(bg), "ink": _to_hex(ink),
            "baseline": int(y0 + _baseline(keep, [ty0, ty1])), "line_pitch": pitch, "alpha": alpha}


def _style_samples(style):
    return [s.strip() for s in re.split(r"\s+/\s+|\n", style.get("sample", "")) if s.strip()]


def _style_alternatives(style):
    lines = [ln for ln in style.get("sample", "").split("\n") if ln.strip()]
    return [s.strip() for s in re.split(r"\s+/\s+", lines[0] if lines else "") if s.strip()][1:]


def _component_for_text(spec, sample, size, screen_area):
    hits = []
    for c in spec.get("components", []):
        b = c["bbox"]
        if b["w"] * b["h"] > 0.4 * screen_area or b["h"] < 0.5 * size:
            continue
        hay = f"{c['name']} {c.get('notes', '')}"
        pat = r"(?<![\w])" + re.escape(sample) + r"(?![\w])"
        exact = re.search(pat, hay) is not None
        if exact or re.search(pat, hay, re.I):
            hits.append((not exact, b["w"] * b["h"], c))
    return [c for *_, c in sorted(hits, key=lambda t: t[:2])]


def _locate_in(rgb, comps, sample, st):
    """First component whose located text looks like this style (ink colour and height plausible)."""
    e_h, _ = _text_expect(sample, st["size_px"], st.get("letter_spacing_em", 0))
    for c in comps:
        b = c["bbox"]
        loc = _locate_text(rgb, [b["x"], b["y"], b["w"], b["h"]], sample, st["size_px"], st.get("letter_spacing_em", 0),
                           st.get("color"))
        if loc and 0.6 < loc["box"][3] / e_h < 1.6 and (not st.get("color") or _de_hex(loc["ink"], st["color"]) <= 20):
            return c, loc
    return None, None


TEXT_BOX_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["boxes"], "properties": {"boxes": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["index", "text", "x", "y", "w", "h"],
                               "properties": {"index": {"type": "integer"}, "text": {"type": "string"},
                                              **{k: {"type": "integer"} for k in "xywh"}}}}}}


def _text_targets(d, spec, mock, vision=False):
    """Locate every typography sample in the mockup (spec components first, optional vision fallback)."""
    rgb = np.asarray(mock.convert("RGB"))
    area = mock.width * mock.height
    vpath = d / "text_boxes_vision.json"
    vboxes = {}
    if vpath.exists():
        try:
            for key, v in json.loads(vpath.read_text()).items():
                b = _clip_bbox(dict(zip("xywh", v)), mock.width, mock.height) if isinstance(v, (list, tuple)) and len(v) == 4 else None
                if b:
                    vboxes[key] = [b["x"], b["y"], b["w"], b["h"]]
        except (OSError, ValueError, AttributeError):
            print(f"  ! {vpath} is unreadable; ignoring the cached vision boxes")
    styles = spec.get("typography", [])
    plan = []
    for st in styles:
        samples = _style_samples(st)
        if not samples:
            continue
        comp, loc = _locate_in(rgb, _component_for_text(spec, samples[0], st["size_px"], area), samples[0], st)
        alts = [(smp, *_locate_in(rgb, _component_for_text(spec, smp, st["size_px"], area), smp, st))
                for smp in _style_alternatives(st)[:5]]
        plan.append((st, samples[0], comp, loc, [a_ for a_ in alts if a_[1]]))
    missing = [s for st, s, c, *_ in plan if not c and s not in vboxes]
    if missing and vision:
        print(f"  locating {len(missing)} text samples with a vision call")
        norm = lambda t: re.sub(r"\s+", " ", str(t)).strip().lower()
        by_text = {norm(s): s for s in missing}
        try:
            res = ask([f"This UI design is {mock.width}x{mock.height}px. For each numbered string below return its number as "
                       "`index` and the TIGHT pixel bbox of its first visible occurrence (one line; ignore other lines). Strings:\n" +
                       "\n".join(f"{i + 1}. {s}" for i, s in enumerate(missing)), mock.convert("RGB")], TEXT_BOX_SCHEMA, "text_boxes")
            boxes = res["boxes"] if isinstance(res, dict) and isinstance(res.get("boxes"), list) else None
            if boxes is None:
                raise ValueError("no 'boxes' in the response")
            dropped = []
            for b in boxes:
                idx = b.get("index") if isinstance(b, dict) else None
                key = missing[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(missing) else by_text.get(norm(b.get("text", "")))
                clipped = _clip_bbox(b, mock.width, mock.height) if isinstance(b, dict) else None
                if key is None or clipped is None or clipped["w"] * clipped["h"] < 0.25 * max(1, b.get("w", 0) * b.get("h", 0)):
                    dropped.append(str(b.get("text", "?"))[:30] if isinstance(b, dict) else "?")
                    continue
                vboxes[key] = [clipped["x"], clipped["y"], clipped["w"], clipped["h"]]
            if dropped:
                print(f"  ! vision returned {len(dropped)} box(es) outside the mockup or for unknown strings; ignored: {', '.join(dropped)}")
            vpath.write_text(json.dumps(vboxes, indent=2))
        except (ApiError, KeyError, ValueError, TypeError) as e:  # the call, or a box the parse above cannot read
            print(f"  ! vision text location failed ({e}); continuing without it")
    out = {}
    for st, sample, comp, loc, alts in plan:
        if comp:
            b = comp["bbox"]
            box, src = [b["x"], b["y"], b["w"], b["h"]], f"component: {comp['name']}"
        elif sample in vboxes:
            vx, vy, vw, vh = vboxes[sample]
            pad = int(0.4 * st["size_px"])
            box, src = [vx - pad, vy - pad, vw + 2 * pad, vh + 2 * pad], "vision"
        else:
            out[st["role"]] = {"sample": sample, "found": False}
            continue
        nlines = len([ln for ln in st.get("sample", "").split("\n") if ln.strip()])
        if not comp or nlines > 1:
            loc = _locate_text(rgb, box, sample, st["size_px"], st.get("letter_spacing_em", 0), st.get("color"), nlines)
        if not loc:
            out[st["role"]] = {"sample": sample, "found": False, "source": src}
            continue
        lh = None
        if comp:
            m_ = re.search(r"(\d+(?:\.\d+)?)\s*px\s+line[- ]height|line[- ]height\s+(?:of\s+)?(?:approximately\s+)?"
                           r"(\d+(?:\.\d+)?)\s*px", comp.get("notes", ""), re.I)
            lh = float(m_.group(1) or m_.group(2)) if m_ else None
        out[st["role"]] = {"sample": sample, "found": True, "source": src, "search_box": box,
                           "line_height_px": loc["line_pitch"] or lh, **loc}
        inst = [{"sample": sample, "component": comp["name"] if comp else None, "search_box": box, **loc}]
        for smp, c, li in alts:
            b = c["bbox"]
            inst.append({"sample": smp, "component": c["name"], "search_box": [b["x"], b["y"], b["w"], b["h"]], **li})
        pitch = loc.get("line_pitch")
        for n, line in enumerate([ln.strip() for ln in st.get("sample", "").split("\n") if ln.strip()][1:], 1):
            if not pitch:
                break
            bx, by, bw, bh = loc["box"]
            e_w = _text_expect(line, st["size_px"], st.get("letter_spacing_em", 0))[1]
            sx0 = max(0, min(box[0], bx - int(0.3 * st["size_px"])))
            sx1 = min(rgb.shape[1], max(box[0] + box[2], bx + int(max(bw, e_w * 1.4)) + int(0.3 * st["size_px"])))
            sb = [sx0, by + n * pitch - int(0.25 * st["size_px"]), sx1 - sx0, bh + int(0.5 * st["size_px"])]
            li = _locate_text(rgb, sb, line, st["size_px"], st.get("letter_spacing_em", 0), st.get("color"))
            if li:
                inst.append({"sample": line, "component": comp["name"] if comp else None, "search_box": sb, **li})
        out[st["role"]]["instances"] = inst
    return out


def _save_text_boxes(d, mock, texts, path):
    ov = mock.convert("RGB")
    dr = ImageDraw.Draw(ov)
    for role, t in texts.items():
        if t.get("found"):
            x, y, w, h = t["box"]
            sx, sy, sw, sh = t["search_box"]
            dr.rectangle([sx, sy, sx + sw, sy + sh], outline="#3a86ff", width=1)
            dr.rectangle([x, y, x + w, y + h], outline="#ff2d55", width=2)
            dr.text((x, max(0, y - 11)), role[:40], fill="#ffd60a")
    ov.save(path)
    strip = lambda t: {k: ([strip(i) for i in v] if k == "instances" else v) for k, v in t.items() if k != "alpha"}
    Path(path).with_suffix(".json").write_text(json.dumps({r: strip(t) for r, t in texts.items()}, indent=2))


# ---- tokens: design_spec.json -> Compose / SwiftUI / CSS / Tailwind

_TOKEN_STOP = {"the", "and", "with", "for", "from", "into", "text", "color", "colour", "subtle", "approximately", "main",
               "dominant", "variation", "local", "fill", "shade", "tone", "light", "dark", "very", "slightly", "around",
               "behind", "inside", "outside", "used", "uses", "use", "its", "their", "this", "that", "includes", "plus",
               "icon", "icons", "label", "labels", "background", "a", "an", "of", "on", "to", "in", "at", "is", "or", "by",
               "as", "it", "up", "px", "dp", "sp", "pt"}


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _TOKEN_STOP}


def _word_rel(w, words):
    return any(w == y or (min(len(w), len(y)) >= 5 and (w.startswith(y) or y.startswith(w))) for y in words)


def _link_regions(query, regions, scored=False):
    """Regions whose name (and, weaker, notes) share informative words with `query` (IDF-weighted; the threshold scales
    with the region count so a unique word still links among 2-3 regions). An exact, case-insensitive name match wins
    outright (score 10). `scored` returns every candidate as (score, region), best first, instead of the top group."""
    norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
    exact = [r for r in regions if norm(r["name"]) == norm(query)]
    if exact:
        return [(10.0, r) for r in exact] if scored else exact
    docs = [(_words(r["name"]), _words(r.get("notes", ""))) for r in regions]
    vocab = [nw | tw for nw, tw in docs]
    idf = lambda w: math.log((len(regions) + 1) / (sum(1 for v in vocab if _word_rel(w, v)) + 0.5))
    scores = []
    for (nw, tw), r in zip(docs, regions):
        sc = sum(max(idf(w) if _word_rel(w, nw) else 0, 0.5 * idf(w) if _word_rel(w, tw) else 0) for w in _words(query))
        scores.append((sc, r))
    floor = min(1.0, 0.9 * math.log((len(regions) + 1) / 1.5))
    best = max((sc for sc, _ in scores), default=0)
    if scored:
        return sorted(((sc, r) for sc, r in scores if sc >= floor and sc >= 0.8 * best), key=lambda t: -t[0])
    return [r for sc, r in scores if best >= floor and sc >= 0.8 * best]


def _mode_near(lab_px, rgb_px, start, need=None):
    """Mean-shift from a guessed colour to the nearest dense colour cluster; returns (rgb, share) or (None, 0)."""
    need = need or max(25, int(0.002 * len(lab_px)))
    for r in (6.0, 10.0, 16.0):
        sel = np.linalg.norm(lab_px - start, axis=1) < r
        if sel.sum() < need:
            continue
        cur = lab_px[sel].mean(0)
        for _ in range(20):
            nsel = np.linalg.norm(lab_px - cur, axis=1) < r
            new = lab_px[nsel].mean(0)
            if nsel.sum() < need or np.linalg.norm(new - start) > r + 2:
                break
            done = np.linalg.norm(new - cur) < 0.05
            cur, sel = new, nsel
            if done:
                break
        for rr_ in (2.0, 1.0):  # the flat colour itself: most frequent exact RGB (else the median) within 2, then 1 Lab units
            near = np.linalg.norm(lab_px - cur, axis=1) < rr_
            if near.sum() >= min(need, 25):
                vals, counts = np.unique(rgb_px[near].astype(np.uint8), axis=0, return_counts=True)
                top = int(counts.argmax())
                pick = vals[top].astype(float) if counts[top] >= 0.2 * near.sum() else np.median(rgb_px[near], 0)
                return pick, float(sel.mean())
        return rgb_px[sel].mean(0), float(sel.mean())
    return None, 0.0


def _edge_color(rgb, b):
    """Colour of a thin border line along a box (straight sides, or the inscribed circle for round elements)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1
    t = 7
    clip = lambda v, hi: int(min(max(v, 0), hi - 1))

    def line(xa, ya, xb, yb):
        n = 2 * t + 1
        return rgb[[clip(ya + (yb - ya) * i / (n - 1), H) for i in range(n)],
                   [clip(xa + (xb - xa) * i / (n - 1), W) for i in range(n)]].astype(np.float64)

    def pick(profiles):
        found = []
        for pr in profiles:
            lab = _lab_arr(pr)
            dev = np.minimum(np.linalg.norm(lab - lab[:2].mean(0), axis=1), np.linalg.norm(lab - lab[-2:].mean(0), axis=1))
            i = int(dev.argmax())
            if dev[i] > 4:
                found.append(pr[i])
        return found, len(profiles)
    rect = []
    for f in np.linspace(0.3, 0.7, 24):
        xs, ys = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        rect += [line(xs, y0 - t, xs, y0 + t), line(xs, y1 + t, xs, y1 - t), line(x0 - t, ys, x0 + t, ys),
                 line(x1 + t, ys, x1 - t, ys)]
    sets = [pick(rect)]
    if 0.8 < b["w"] / max(b["h"], 1) < 1.25:
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, min(b["w"], b["h"]) / 2
        sets.append(pick([line(cx + (r + t) * math.cos(g), cy + (r + t) * math.sin(g), cx + (r - t) * math.cos(g),
                               cy + (r - t) * math.sin(g)) for g in np.linspace(0, 2 * math.pi, 72, endpoint=False)]))
    found, n = max(sets, key=lambda s_: len(s_[0]) / s_[1])
    if len(found) < 8:
        return None, 0.0
    return np.median(np.array(found), 0), len(found) / n


_AREA_NAME = re.compile(r"\b(fill|button|surface|background|disc|card|panel|overlay|base|bar|sheet|chip|tile|border|"
                        r"outline|stroke)s?\b", re.I)


def _region_pixels(rgb, boxes):
    H, W = rgb.shape[:2]
    m = np.zeros((H, W), bool)
    for b in boxes:
        m[max(0, b["y"]):min(H, b["y"] + b["h"]), max(0, b["x"]):min(W, b["x"] + b["w"])] = True
    return rgb[m]


def _snap_palette(spec, mock, texts, assets=()):
    rgb = np.asarray(mock.convert("RGB"))
    regions = [c for c in spec.get("components", []) if c["bbox"]["w"] * c["bbox"]["h"] < 0.9 * mock.width * mock.height]
    regions += [{"name": x["name"].replace("_", " "), "notes": x.get("description", ""), "bbox": x["bbox"]} for x in assets]
    by_hex = {}
    for st in spec.get("typography", []):
        t = texts.get(st["role"], {})
        if t.get("found") and st.get("color"):
            by_hex.setdefault(st["color"], []).append((st["role"], t))
    rng = np.random.default_rng(0)
    out = []
    for p in spec.get("palette", []):
        guess = p["hex"].upper()
        text = p["name"] + " " + p.get("usage", "")
        entry = {"name": p["name"], "usage": p.get("usage", ""), "hex_guess": guess, "alpha": float(p.get("alpha", 1.0))}
        linked = by_hex.get(guess, [])
        snapped = None
        if linked and not _AREA_NAME.search(p["name"]):
            inks = np.array([_hexi(t["ink"]) for _, t in linked], dtype=float)
            snapped = np.median(inks, 0)
            entry.update(source="text ink: " + ", ".join(r for r, _ in linked),
                         spread_de=round(max(_de_hex(_to_hex(i), _to_hex(snapped)) for i in inks), 1))
        else:
            screen_bg = re.search(r"\b(screen|canvas|page|app|window)\b", p["name"] + " " + p.get("usage", "")[:60], re.I) and \
                re.search(r"background|canvas|\bbase\b", p["name"] + " " + p.get("usage", "")[:60], re.I)
            picks = [] if screen_bg else _link_regions(text, regions)
            if picks and re.search(r"border|outline|stroke|divider", p["name"], re.I):
                found = [(sh, c, r) for r in picks for c, sh in [_edge_color(rgb, r["bbox"])] if c is not None]
                if found:
                    sh, snapped, r = max(found, key=lambda f: f[0])
                    entry.update(source=f"border of {r['name']}", support=round(sh, 3))
            if snapped is None:
                if picks:
                    px, need = _region_pixels(rgb, [r["bbox"] for r in picks]), None
                else:
                    px, need = rgb.reshape(-1, 3), 25
                if len(px) > 250000:
                    px = px[rng.choice(len(px), 250000, replace=False)]
                snapped, share = _mode_near(_lab_arr(px), px.astype(float), _lab_arr(_hexi(guess)), need)
                entry.update(source=", ".join(r["name"] for r in picks) if picks else "whole screen", support=round(share, 4))
        if snapped is None:
            entry.update(hex=guess, de00=0.0, snapped=False)
        else:
            entry.update(hex=_to_hex(snapped), de00=_de_hex(guess, _to_hex(snapped)), snapped=True)
        out.append(entry)
    return out


def _ident(s, case="pascal", max_words=6):
    words = re.findall(r"[A-Za-z0-9]+", s.replace("'", ""))[:max_words] or ["token"]
    if case == "kebab":
        out = "-".join(w.lower() for w in words)
    else:
        out = "".join(w[:1].upper() + w[1:].lower() for w in words)
        if case == "camel":
            out = out[:1].lower() + out[1:]
    return ("n" if case != "pascal" else "N") + out if out[:1].isdigit() else out


def _uniq(names):
    seen, out = {}, []
    for n in names:
        k = n
        i = 2
        while k in seen:
            k = f"{n}{i}"
            i += 1
        seen[k] = 1
        out.append(k)
    return out


_SWIFT_KEYWORDS = frozenset("as associatedtype async await break case catch class continue default defer deinit do else enum "
                            "extension fallthrough false fileprivate for func guard if import in init inout internal is let "
                            "nil open operator precedencegroup private protocol public repeat rethrows return self static "
                            "struct subscript super switch throw throws true try typealias var where while some any".split())
_SWIFTUI_COLORS = frozenset("primary secondary accentColor clear black white gray red green blue orange yellow pink purple "
                            "brown cyan indigo mint teal".split())


def _names(labels, colors=()):
    """One identifier set per label, unique across the whole token namespace (colours, styles, families, radii, spacing):
    kt (PascalCase), sw (camelCase, Swift keywords backticked, SwiftUI colour names suffixed Token), css (kebab-case).
    `colors` holds the indices of the colour labels."""
    words = [re.findall(r"[A-Za-z0-9]+", str(s).replace("'", ""))[:6] or ["token"] for s in labels]
    pascal = ["".join(w[:1].upper() + w[1:].lower() for w in ws) for ws in words]
    pascal = [("N" + p) if p[:1].isdigit() else p for p in pascal]
    kt = _uniq(pascal)
    sw, css = [], []
    for i, (p, k, ws) in enumerate(zip(pascal, kt, words)):
        suffix = k[len(p):]
        camel = k[:1].lower() + k[1:]
        if i in colors and camel in _SWIFTUI_COLORS:
            camel += "Token"
        sw.append(camel)
        css.append("-".join(w.lower() for w in ws) + (f"-{suffix}" if suffix else ""))
    sw = [f"`{s}`" if s in _SWIFT_KEYWORDS else s for s in _uniq(sw)]
    return [{"kt": k, "sw": s, "css": c} for k, s, c in zip(kt, sw, css)]


def _r05(v):
    return round(v * 2) / 2


def _num_s(v):
    return f"{v:g}"


def _num_ok(v):
    """A finite int/float (not a bool): what a hand-edited JSON number must be before arithmetic touches it."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _json_lists(data, keys, path, rerun):
    """data (from _read_json) with each of `keys` checked to be a list of objects; else a one-line exit naming the file."""
    for k in keys:
        v = data.get(k) if isinstance(data, dict) else None
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            sys.exit(f"gptkit: {path} '{k}' must be a list of objects (hand-edited or older file? re-run `{rerun}`)")
    return data


def _d2c_manifest(d, size=None):
    """DIR/assets.json for tokens/fonts/compare/redline, or None when absent: a non-object exits naming the file; non-numeric
    screen_dp/width/height are dropped; with size=(W, H) assets without a name or a bbox inside the mockup are dropped."""
    p = Path(d) / "assets.json"
    if not p.exists():
        return None
    m = _read_json(p, "assets.json")
    if not isinstance(m, dict):
        sys.exit(f"gptkit: assets.json {p} is not a JSON object (hand-edited? re-run manifest)")
    for k in ("screen_dp", "width", "height"):
        v = m.get(k)
        if v is not None and not (_num_ok(v) and v > 0):
            print(f"  ! {p}: {k} {v!r} is not a positive number; ignored")
            m.pop(k)
    if m.get("mockup") is not None and not (isinstance(m["mockup"], str) and m["mockup"]):
        print(f"  ! {p}: mockup {m.pop('mockup')!r} is not a file name; ignored")
    keep = []
    for x in m.get("assets") if isinstance(m.get("assets"), list) else []:
        if not (isinstance(x, dict) and isinstance(x.get("name"), str) and x["name"]):
            continue
        b = _clip_bbox(x.get("bbox") or {}, *size) if size and isinstance(x.get("bbox"), dict) else x.get("bbox")
        if size and not b:
            continue
        keep.append({**x, "bbox": b, "kind": str(x.get("kind") or ""), "description": str(x.get("description") or "")})
    m["assets"] = keep
    return m


def _d2c_load_spec(dir_, mockup=None, required=True):
    """(dir, design_spec, mockup RGBA) with every palette/typography colour normalised to #RRGGBB (+ alpha) and one-line
    exits for missing or corrupt files. `required=False` (compare) tolerates a missing design_spec.json."""
    d = Path(dir_)
    if not (d / "design_spec.json").exists():
        if required:
            sys.exit(f"gptkit: no {d / 'design_spec.json'}; run `spec {d}` first")
        spec = {}
    else:
        spec = _read_json(d / "design_spec.json", "design_spec.json")
        if not isinstance(spec, dict):
            sys.exit(f"gptkit: design_spec.json {d / 'design_spec.json'} is not a JSON object")
    for key in ("palette", "typography", "radii", "spacing", "components"):
        if not isinstance(spec.get(key), list):
            spec[key] = []
        bad = next((i for i, x in enumerate(spec[key]) if not isinstance(x, dict)), None)
        if bad is not None:
            sys.exit(f"gptkit: design_spec.json {d / 'design_spec.json'}: {key} entry {bad + 1} is not an object "
                     f"({spec[key][bad]!r:.40}); fix it or re-run `spec {d}`")
    for key in ("radii", "spacing"):
        for x in spec[key]:
            if x.get("px") is not None and not _num_ok(x["px"]):
                sys.exit(f"gptkit: design_spec.json {key} '{x.get('element') or x.get('between') or '?'}': px {x['px']!r} "
                         "is not a number")
    if spec.get("px_to_dp") is not None and not (_num_ok(spec["px_to_dp"]) and spec["px_to_dp"] > 0):
        print(f"  ! design_spec.json px_to_dp {spec.pop('px_to_dp')!r} is not a positive number; ignored")
    for p in spec["palette"]:
        try:
            p["hex"], p["alpha"] = _norm_hex(p.get("hex", ""))
        except (ValueError, TypeError) as e:
            sys.exit(f"gptkit: design_spec.json palette '{p.get('name', '?')}': {e}")
    for i, st in enumerate(spec["typography"]):
        for key in ("role", "size_px"):
            if not st.get(key):
                sys.exit(f"gptkit: design_spec.json typography entry {i + 1}: missing '{key}'")
        try:
            st["size_px"] = float(st["size_px"])
            st["weight"] = int(st.get("weight") or 400)
        except (TypeError, ValueError):
            sys.exit(f"gptkit: design_spec.json typography '{st['role']}': size_px {st['size_px']!r} / weight "
                     f"{st.get('weight')!r} must be numbers")
        if st.get("color"):
            try:
                st["color"] = _norm_hex(st["color"])[0]
            except (ValueError, TypeError) as e:
                sys.exit(f"gptkit: design_spec.json typography '{st['role']}': {e}")
        else:
            st["color"] = None
    m = _d2c_manifest(d)
    mpath = Path(mockup) if mockup else d / (m or {}).get("mockup", "mockup.png")
    mock = _open_image(mpath, "mockup", "pass --mockup or fix assets.json 'mockup'")
    if m and m.get("screen_dp") and m.get("width"):
        spec.setdefault("px_to_dp", round(m["screen_dp"] / m["width"], 5))
    if not spec.get("px_to_dp"):
        print("  ! no px_to_dp in design_spec.json and no screen_dp/width in assets.json: using 1 px = 1 dp; "
              "run `spec DIR --recalibrate`")
        spec["px_to_dp"] = 1.0
    good = []
    for c in spec["components"]:
        b = _clip_bbox(c.get("bbox") or {}, mock.width, mock.height)
        if not c.get("name") or not b:
            print(f"  ! design_spec.json: component {c.get('name') or '?'} has no usable bbox inside the mockup; ignored")
            continue
        c["bbox"] = b
        good.append(c)
    spec["components"] = good
    return d, spec, mock


_M3_SCALE = [("displayLarge", 57, 64), ("displayMedium", 45, 52), ("displaySmall", 36, 44), ("headlineLarge", 32, 40),
             ("headlineMedium", 28, 36), ("headlineSmall", 24, 32), ("titleLarge", 22, 28), ("titleMedium", 16, 24),
             ("titleSmall", 14, 20), ("bodyLarge", 16, 24), ("bodyMedium", 14, 20), ("bodySmall", 12, 16), ("labelLarge", 14, 20),
             ("labelMedium", 12, 16), ("labelSmall", 11, 16)]  # M3 baseline type scale (sp size, sp line height); r5 (g)
_HIG_SCALE = [("largeTitle", 34), ("title", 28), ("title2", 22), ("title3", 20), ("headline", 17), ("body", 17), ("callout", 16),
              ("subheadline", 15), ("footnote", 13), ("caption", 12), ("caption2", 11)]  # iOS Dynamic Type defaults (pt); r2/r5


def _m3_slot(role, size_sp, platform_style=None):
    """(slot, source, off) for a text style: the spec's platform_style when it is an M3 slot, else the nearest slot by size
    (ties broken by a role keyword, then body > title > label); off = True when the size is > 15% from the slot's."""
    by_name = {n: (sz, lh) for n, sz, lh in _M3_SCALE}
    if platform_style and platform_style in by_name:
        sz = by_name[platform_style][0]
        return platform_style, "spec", abs(size_sp - sz) / sz > 0.15
    low = role.lower()
    kw = next((f for f in ("display", "headline", "title", "body", "label") if f in low), None)
    order = {"body": 0, "title": 1, "label": 2, "headline": 3, "display": 4}
    best = min(_M3_SCALE, key=lambda t: (abs(size_sp - t[1]), 0 if kw and t[0].startswith(kw) else 1,
                                         order[re.match(r"[a-z]+", t[0]).group(0)]))
    return best[0], "nearest by size", abs(size_sp - best[1]) / best[1] > 0.15


def _hig_style(size_pt, platform_style=None):
    """The spec's platform_style when it is a HIG text style, else the nearest by size."""
    if platform_style in {n for n, _ in _HIG_SCALE}:
        return platform_style
    return min(_HIG_SCALE, key=lambda t: abs(size_pt - t[1]))[0]


def _kt_weight(w):
    """Compose FontWeight literal: the W100..W900 constants exist only for hundreds; anything else is FontWeight(n)."""
    w = int(min(1000, max(1, round(float(w)))))
    return f"FontWeight.W{w}" if w % 100 == 0 and w <= 900 else f"FontWeight({w})"


def _token_model(spec, palette, texts, fonts, redline=None):
    k = spec.get("px_to_dp", 1.0)
    families = {f["family"]: f for f in (fonts or {}).get("families", [])}
    types_in = spec.get("typography", [])
    radii_in = spec.get("radii", [])
    spacing_in = [s for s in spec.get("spacing", []) if s.get("between") and isinstance(s.get("px"), (int, float))
                  and not re.search(r"pixels? per|\bpx ?per\b|per (compose )?dp|px_to_dp|ratio|density|scale factor",
                                    s["between"], re.I)]
    labels = ([p["name"] for p in palette] + [s["role"] for s in types_in] + [r.get("element", "") for r in radii_in]
              + [s["between"] for s in spacing_in] + list(families))
    names = _names(labels, colors=range(len(palette)))
    nc, nt, nr, ns = len(palette), len(types_in), len(radii_in), len(spacing_in)
    colors = [dict(p, alpha=float(p.get("alpha", 1.0)), **n) for p, n in zip(palette, names[:nc])]
    by_guess = {c["hex_guess"]: c for c in colors}
    fstyles = {s["role"]: s for s in (fonts or {}).get("styles", [])}
    fam_names = dict(zip(families, names[nc + nt + nr + ns:]))
    for fam, n in fam_names.items():
        families[fam] = dict(families[fam], kt=n["kt"], css=n["css"])
    types = []
    for st, n in zip(types_in, names[nc:nc + nt]):
        fs = fstyles.get(st["role"], {})
        size_px = fs.get("size_px") or st["size_px"]
        track = fs.get("tracking_em", st.get("letter_spacing_em") or 0)
        t = texts.get(st["role"], {})
        lh_px = t.get("line_height_px")
        slot, slot_src, slot_off = _m3_slot(st["role"], size_px * k, st.get("platform_style"))
        if lh_px:
            lh_src = "measured line pitch"
        elif slot_src == "spec":
            ratio = next(lh / sz for n, sz, lh in _M3_SCALE if n == slot)
            lh_px, lh_src = round(size_px * ratio), f"M3 {slot} ratio ({ratio:.2f} x size)"
        else:
            lh_px, lh_src = round(size_px * 1.25), "estimate: 1.25 x size"
        guess = st.get("family_guess", "")
        generic = "serif" if re.search(r"(?<!sans[- ])serif|baskerville|garamond|times|georgia|playfair|bodoni", guess, re.I) \
            and not re.search(r"sans", guess, re.I) else "sans"
        fam = families.get(fs.get("family"))
        col = by_guess.get(st["color"]) if st.get("color") else None
        ink = t.get("ink") if t.get("found") else None
        weight = int(min(1000, max(1, round(float(fs.get("weight") or st.get("weight") or 400)))))
        types.append({"role": st["role"], **n, "family": fam["family"] if fam else None,
                      "family_kt": fam and fam["kt"], "family_css": fam and fam["css"], "generic": generic,
                      "weight": weight, "italic": bool(fs.get("italic", False)),
                      "size": _r05(size_px * k), "line_height": _r05(lh_px * k), "line_height_source": lh_src,
                      "tracking": round(float(track), 3), "color": col,
                      "color_hex": col["hex"] if col else (ink or st.get("color") or "#000000"),
                      "platform_style": st.get("platform_style"), "m3_slot": slot, "m3_slot_source": slot_src, "m3_slot_off": slot_off,
                      "hig_style": _hig_style(_r05(size_px * k), st.get("platform_style")),
                      "postscript": (fam or {}).get("postscript", {}).get(str(fs.get("weight")) + ("i" if fs.get("italic") else "")),
                      "natural_lh": fs.get("natural_line_height_em"), "size_source": "fonts" if fs.get("size_px") else "spec"})
    measured = {r["element"]: r for r in (redline or {}).get("radii", []) if r.get("measured_px") is not None}
    radii = []
    for r, n in zip(radii_in, names[nc + nt:nc + nt + nr]):
        px = float(r.get("px") or 0)
        if px < 0:  # CSS drops a negative border-radius, Compose/SwiftUI shapes take none: no platform has one
            print(f"  ! radius {r.get('element', '')!r}: {px:g}px is negative; using 0")
            px = 0.0
        hit = measured.get(r.get("element"))
        radii.append({"name": r.get("element", ""), **n, "dp": _r05((hit["measured_px"] if hit else px) * k),
                      "source": f"redline: {hit['component']}" if hit else "spec (GPT estimate)",
                      "native": hit.get("native") if hit else None, "shape": hit.get("shape") if hit else None})
    spacing = [{"name": s["between"], **n, "dp": _r05(float(s["px"]) * k), "source": "spec (GPT estimate)"}
               for s, n in zip(spacing_in, names[nc + nt + nr:nc + nt + nr + ns])]
    for sp in spacing:
        got = _measured_spacing(sp["name"], (redline or {}).get("items", []), k)
        if got:
            sp["dp"], sp["source"] = _r05(got[0]), got[1]
    return {"colors": colors, "types": types, "radii": radii, "spacing": spacing, "families": list(families.values()),
            "px_to_dp": k}


def _measured_spacing(name, items, k):
    """(dp, source) for a spacing name from redline's measured items, or None: 'gap/between/below' -> the gap below the
    linked item, 'edge/screen/margin' -> the item's distance from the screen edge, 'padding/inset' -> the smallest inset
    of the item's children (its content padding) or, for a leaf, its own inset."""
    measured = [i for i in items if i.get("kind") in ("surface", "object", "text")]
    if not measured:
        return None
    hits = _link_regions(name, [{"name": i["name"], "notes": i.get("role", "")} for i in measured], scored=True)
    if not hits:
        return None
    it = next(i for i in measured if i["name"] == hits[0][1]["name"])
    low = name.lower()
    side = ("left" if re.search(r"left|start|horizontal", low) else "right" if re.search(r"right|end", low) else
            "bottom" if re.search(r"bottom|below", low) else "top" if re.search(r"top|above|vertical", low) else None)
    if re.search(r"gap|between|below|above|spacing|distance", low) and it.get("gap_below_dp"):
        return it["gap_below_dp"][1], f"redline: gap below {it['name']} (to {it['gap_below_dp'][0]})"
    if re.search(r"edge|screen|margin", low):
        sd = side or "left"
        v = {"left": it["px"][0], "top": it["px"][1], "right": it["px"][0], "bottom": it["px"][1]}[sd] * k
        return round(v, 1), f"redline: {it['name']} {sd} of screen edge"
    # padding: layout-box insets for text (ink insets overstate the top), the smaller of left/top when no side is named
    inset = lambda i, sd: (i.get("layout_top_inset_dp") if sd == "top" and i.get("layout_top_inset_dp") is not None else i["inset_dp"][sd])
    pick = lambda i: inset(i, side) if side else min(inset(i, "left"), inset(i, "top"))
    kids = [i for i in measured if i.get("parent") == it["name"] and i.get("inset_dp")]
    if kids:
        return min(pick(i) for i in kids), f"redline: {it['name']} {side or 'left/top'} inset of its content ({len(kids)} items)"
    if it.get("inset_dp"):
        return pick(it), f"redline: {it['name']} {side or 'left/top'} inset in {it.get('parent')}"
    return None


_SCHEME_RULES = [("background", r"screen|background|canvas"), ("surface", r"surface|panel|sheet"),
                 ("primary", r"primary.*(fill|button)|accent fill|button fill|\baccent\b"),
                 ("onPrimary", r"on[- ]accent|on[- ]primary|button label"), ("onBackground", r"primary text"),
                 ("onSurface", r"primary text"), ("onSurfaceVariant", r"secondary text"), ("outline", r"border|outline")]


def _kt_header(a, pkg):
    head = f"/**\n * @author {a.author}\n * @since {time.strftime('%d/%m/%y')}\n */\n" if a.author else ""
    return f"{head}package {pkg}\n\n"


def _kt_color(c):
    return f"Color(0x{round(c.get('alpha', 1.0) * 255):02X}{c['hex'][1:]})"


def _emit_compose(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    pkg, P = a.package, a.prefix
    lum = lambda h: sum(v * w for v, w in zip(_hexi(h), (0.2126, 0.7152, 0.0722))) / 255
    scheme = {}
    for role, pat in _SCHEME_RULES:
        c = next((c for c in tm["colors"] if re.search(pat, f"{c['name']} {c.get('usage', '')}", re.I)), None)
        if c and role not in scheme:
            scheme[role] = c["kt"]
    bg = next((c for c in tm["colors"] if c["kt"] == scheme.get("background")), None)
    dark = bg is not None and lum(bg["hex"]) < 0.5
    fn = "darkColorScheme" if dark else "lightColorScheme"
    lines = [f"    val {c['kt']} = {_kt_color(c)}" for c in tm["colors"]]
    args = "".join(f"    {r} = {P}Colors.{v},\n" for r, v in scheme.items())
    other = "lightColorScheme" if dark else "darkColorScheme"
    measured_name, other_name = (f"{P}DarkColorScheme", f"{P}LightColorScheme") if dark else (f"{P}LightColorScheme", f"{P}DarkColorScheme")
    tm["m3_roles_baseline"] = [r for r in _M3_COLOR_ROLES if r not in scheme]
    (out / "Color.kt").write_text(_kt_header(a, pkg) + f"import androidx.compose.material3.ColorScheme\n"
                                  f"import androidx.compose.material3.darkColorScheme\nimport androidx.compose.material3.lightColorScheme\n"
                                  "import androidx.compose.ui.graphics.Color\n\n"
                                  f"object {P}Colors {{\n" + "\n".join(lines) + "\n}\n\n"
                                  f"// Measured on the {'dark' if dark else 'light'} mockup. Roles not listed stay at the M3 baseline: "
                                  + (", ".join(tm["m3_roles_baseline"][:8]) + (" ..." if len(tm["m3_roles_baseline"]) > 8 else "")) + "\n"
                                  f"val {measured_name}: ColorScheme = {fn}(\n{args})\n\n"
                                  f"// TODO: verify against a {'light' if dark else 'dark'} mockup; scaffolded from the same values\n"
                                  f"val {other_name}: ColorScheme = {other}(\n{args})\n\n"
                                  f"val {P}ColorScheme: ColorScheme = {measured_name}\n")
    rpkg = a.r_package or re.sub(r"(\.ui)?(\.theme)?$", "", pkg)
    fam_lines = []
    for f in tm["families"]:
        fonts_ = ",\n".join(f"    Font(R.font.{r['android']}, {_kt_weight(r['weight'])}"
                            + (", FontStyle.Italic" if r.get("italic") else "") + ")" for r in f["files"])
        fam_lines.append(f"val {f['kt']} = FontFamily(\n{fonts_},\n)")
    uses_italic = any(t["italic"] for t in tm["types"]) or any(r.get("italic") for f in tm["families"] for r in f["files"])
    body = []
    for t in tm["types"]:
        fam = t["family_kt"] or ("FontFamily.Serif" if t["generic"] == "serif" else "FontFamily.SansSerif")
        col = f"{P}Colors.{t['color']['kt']}" if t["color"] else f"Color(0xFF{t['color_hex'][1:]})"
        ls = f"({t['tracking']:g}).em" if t["tracking"] < 0 else f"{t['tracking']:g}.em"
        body.append(f"    val {t['kt']} = TextStyle(\n        fontFamily = {fam},\n        fontWeight = {_kt_weight(t['weight'])},\n"
                    + ("        fontStyle = FontStyle.Italic,\n" if t["italic"] else "")
                    + f"        fontSize = {_num_s(t['size'])}.sp,\n        lineHeight = {_num_s(t['line_height'])}.sp,\n"
                    f"        letterSpacing = {ls},\n        color = {col},\n    )")
    imports = ["androidx.compose.material3.Typography", "androidx.compose.ui.graphics.Color", "androidx.compose.ui.text.TextStyle",
               "androidx.compose.ui.text.font.FontFamily", "androidx.compose.ui.text.font.FontWeight",
               "androidx.compose.ui.unit.em", "androidx.compose.ui.unit.sp"]
    if tm["families"]:
        imports += ["androidx.compose.ui.text.font.Font", f"{rpkg}.R"]
    if uses_italic:
        imports.append("androidx.compose.ui.text.font.FontStyle")
    slots, slot_notes = {}, []
    for t in tm["types"]:
        if t["m3_slot"] in slots:
            slot_notes.append(f"{t['role']}: {t['m3_slot']} already taken by {slots[t['m3_slot']]['role']}")
            continue
        slots[t["m3_slot"]] = t
        if t["m3_slot_off"]:
            slot_notes.append(f"{t['role']}: {_num_s(t['size'])}sp is >15% from the {t['m3_slot']} baseline size")
    tm["m3_slot_notes"] = slot_notes
    tm["m3_slots_baseline"] = [s_ for s_ in _M3_TYPE_SLOTS if s_ not in slots]
    m3 = "".join(f"    {s_} = {P}Typography.{slots[s_]['kt']},  // {slots[s_]['m3_slot_source']}\n" for s_ in _M3_TYPE_SLOTS if s_ in slots)
    (out / "Type.kt").write_text(_kt_header(a, pkg) + "".join(f"import {i}\n" for i in sorted(imports)) + "\n"
                                 + "".join(f"{f}\n\n" for f in fam_lines)
                                 + f"object {P}Typography {{\n" + "\n\n".join(body) + "\n}\n\n"
                                 f"// M3 slots not listed keep the Typography() defaults (nearest slot by size when the spec gave none).\n"
                                 f"val {P}M3Typography = Typography(\n{m3})\n")
    shapes = [f"val {r['kt']}Shape = RoundedCornerShape({_num_s(r['dp'])}.dp)  // {r['source']}" for r in tm["radii"]]
    m3_shapes = {}
    for slot, pat in (("small", r"chip"), ("medium", r"card|tile|list item"), ("large", r"fab|drawer|floating"),
                      ("extraLarge", r"sheet|dialog|modal")):
        hit = next((r for r in tm["radii"] if re.search(pat, f"{r['name']} {r.get('native') or ''}", re.I)), None)
        if hit:
            m3_shapes[slot] = hit
    tm["m3_shapes"] = {k_: v["kt"] for k_, v in m3_shapes.items()}
    (out / "Shape.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.foundation.shape.RoundedCornerShape\n"
                                  "import androidx.compose.material3.Shapes\nimport androidx.compose.ui.unit.dp\n\n" + "\n".join(shapes)
                                  + "\n\n// M3 shape slots from the measured radii (small: chips, medium: cards, large: FABs, extraLarge: sheets/dialogs)\n"
                                  f"val {P}Shapes = Shapes(\n" + "".join(f"    {k_} = {v['kt']}Shape,\n" for k_, v in m3_shapes.items()) + ")\n")
    dims = [f"    val {s['kt']} = {_num_s(s['dp'])}.dp" + (f"  // {s['source']}" if s["source"].startswith("redline") else "  // GPT estimate")
            for s in tm["spacing"]]
    (out / "Dimens.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.ui.unit.dp\n\n"
                                   f"object {P}Dimens {{\n" + "\n".join(dims) + "\n}\n")
    wdp = tm.get("width_dp")
    if wdp:
        (out / "Preview.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.runtime.Composable\n"
                                        "import androidx.compose.ui.tooling.preview.Preview\n\n"
                                        f"// Renders at the design's effective width: Roborazzi (ComposablePreviewScanner) or the Compose Preview\n"
                                        f"// Screenshot Testing plugin pick this up; then `gptkit compare DIR shot.png --shot-width-dp {wdp}`.\n"
                                        f"@Preview(name = \"{P} 1:1\", widthDp = {wdp}, showBackground = true)\n@Composable\n"
                                        f"fun {P}ThemePreview() {{\n    {P}Theme {{\n        // TODO: the screen composable built from this theme\n    }}\n}}\n")
    (out / "Theme.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.foundation.isSystemInDarkTheme\n"
                                  "import androidx.compose.material3.MaterialTheme\nimport androidx.compose.runtime.Composable\n\n"
                                  f"@Composable\nfun {P}Theme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {{\n"
                                  f"    MaterialTheme(\n        colorScheme = if (darkTheme) {P}DarkColorScheme else {P}LightColorScheme,\n"
                                  f"        typography = {P}M3Typography,\n        shapes = {P}Shapes,\n        content = content,\n    )\n}}\n")
    return [out / f for f in ("Color.kt", "Type.kt", "Shape.kt", "Dimens.kt", "Theme.kt") + (("Preview.kt",) if wdp else ())]


def _sw_color(hex_, alpha=1.0):
    r, g, b = _hexi(hex_)
    return f"Color(red: {r} / 255, green: {g} / 255, blue: {b} / 255" + (f", opacity: {alpha:.3g})" if alpha < 1 else ")")


def _emit_swift(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"    static let {c['sw']} = {_sw_color(c['hex'], c.get('alpha', 1.0))}" for c in tm["colors"]]
    wmap = {100: "ultraLight", 200: "thin", 300: "light", 400: "regular", 500: "medium", 600: "semibold", 700: "bold",
            800: "heavy", 900: "black"}
    styles = []
    for t in tm["types"]:
        if t["postscript"]:
            font = f'.custom("{t["postscript"]}", size: {_num_s(t["size"])}, relativeTo: .{t["hig_style"]})'
            natural = (t["natural_lh"] or 1.2) * t["size"]
        else:
            design = ".serif" if t["generic"] == "serif" else ".default"
            font = f".system(size: {_num_s(t['size'])}, weight: .{wmap.get(min(900, max(100, round(t['weight'] / 100) * 100)), 'regular')}, design: {design})"
            natural = 1.19 * t["size"]
            if t["italic"]:
                font += ".italic()"
        col = f".{t['color']['sw']}" if t["color"] else _sw_color(t["color_hex"])
        styles.append(f"    static let {t['sw']} = TextToken(font: {font}, tracking: {round(t['tracking'] * t['size'], 2):g}, "
                      f"lineSpacing: {max(0.0, round(t['line_height'] - natural, 1)):g}, color: {col})")
    radii = [f"    static let {r['sw']}: CGFloat = {_num_s(r['dp'])}  // {r['source']}" for r in tm["radii"]]
    space = [f"    static let {s['sw']}: CGFloat = {_num_s(s['dp'])}" + ("" if s["source"].startswith("redline") else "  // GPT estimate")
             for s in tm["spacing"]]
    p = a.prefix
    src = ("import SwiftUI\n\nextension Color {\n" + "\n".join(cols) + "\n}\n\n"
           f"struct {p}TextToken {{\n    let font: Font\n    let tracking: CGFloat\n    let lineSpacing: CGFloat\n    let color: Color\n}}\n\n"
           f"extension {p}TextToken {{\n" + "\n".join(styles).replace("TextToken(", f"{p}TextToken(") + "\n}\n\n"
           "@available(iOS 16.0, macOS 13.0, *)\nextension View {\n"
           f"    func textToken(_ t: {p}TextToken) -> some View {{\n"
           "        font(t.font).tracking(t.tracking).lineSpacing(t.lineSpacing).foregroundStyle(t.color)\n    }\n}\n\n"
           f"enum {p}Radius {{\n" + "\n".join(radii) + "\n}\n\n" f"enum {p}Spacing {{\n" + "\n".join(space) + "\n}\n")
    (out / f"{p}Tokens.swift").write_text(src)
    xc = out / "Colors.xcassets"
    xc.mkdir(exist_ok=True)
    (xc / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2))
    for c in tm["colors"]:
        r, g_, b = _hexi(c["hex"])
        cs = xc / f"{c['sw'].strip('`')}.colorset"
        cs.mkdir(exist_ok=True)
        (cs / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}, "colors": [
            {"idiom": "universal", "color": {"color-space": "srgb", "components": {"red": f"0x{r:02X}", "green": f"0x{g_:02X}",
                                                                                    "blue": f"0x{b:02X}", "alpha": f"{c.get('alpha', 1.0):.3f}"}}}]},
                                          indent=2))
    return [out / f"{p}Tokens.swift", xc]


def _css_family(t):
    fallback = "serif" if t["generic"] == "serif" else "sans-serif"
    return f'"{t["family"]}", {fallback}' if t["family"] else ("Georgia, serif" if t["generic"] == "serif" else
                                                              "system-ui, -apple-system, sans-serif")


def _css_hex(c):
    a = c.get("alpha", 1.0)
    return c["hex"] + (f"{round(a * 255):02X}" if a < 1 else "")


def _emit_web(tm, a, out, fonts_css=None):
    out.mkdir(parents=True, exist_ok=True)
    fams = {}
    for t in tm["types"]:
        fams.setdefault(t["family_css"] or t["generic"], _css_family(t))
    v = [f"  --color-{c['css']}: {_css_hex(c)};" for c in tm["colors"]]
    v += [f"  --font-{k}: {f};" for k, f in fams.items()]
    v += [f"  --radius-{r['css']}: {_num_s(r['dp'])}px;" for r in tm["radii"]]
    v += [f"  --space-{s['css']}: {_num_s(s['dp'])}px;" for s in tm["spacing"]]
    cls = []
    for t in tm["types"]:
        col = f"var(--color-{t['color']['css']})" if t["color"] else t["color_hex"]
        cls.append(f".text-{t['css']} {{\n  font-family: var(--font-{t['family_css'] or t['generic']});\n"
                   f"  font-weight: {t['weight']};\n" + ("  font-style: italic;\n" if t["italic"] else "")
                   + f"  font-size: {_num_s(t['size'])}px;\n  line-height: {_num_s(t['line_height'])}px;\n"
                   f"  letter-spacing: {t['tracking']:g}em;\n  color: {col};\n}}")
    head = f'@import "{fonts_css}";\n\n' if fonts_css else ""
    (out / "tokens.css").write_text(head + ":root {\n" + "\n".join(v) + "\n}\n\n" + "\n\n".join(cls) + "\n")
    js = lambda o: json.dumps(o, indent=6).replace("\n}", "\n    }")
    cfg = {"colors": {c["css"]: _css_hex(c) for c in tm["colors"]},
           "fontFamily": {k: [x.strip().strip('"') for x in f.split(",")] for k, f in fams.items()},
           "fontSize": {t["css"]: [f"{_num_s(t['size'])}px", {"lineHeight": f"{_num_s(t['line_height'])}px",
                                                               "letterSpacing": f"{t['tracking']:g}em",
                                                               "fontWeight": str(t["weight"])}] for t in tm["types"]},
           "borderRadius": {r["css"]: f"{_num_s(r['dp'])}px" for r in tm["radii"]},
           "spacing": {s["css"]: f"{_num_s(s['dp'])}px" for s in tm["spacing"]}}
    body = ",\n".join(f"    {k}: {js(val)}" for k, val in cfg.items())
    (out / "tailwind.tokens.cjs").write_text("// Tailwind v3 preset: in tailwind.config.js add  presets: [require('./tailwind.tokens.cjs')]\n"
                                             "/** @type {import('tailwindcss').Config} */\nmodule.exports = {\n"
                                             f"  theme: {{\n    extend: {{\n{body}\n    }},\n  }},\n}};\n")
    th = [f"  --color-{c['css']}: {_css_hex(c)};" for c in tm["colors"]]
    th += [f"  --font-{k}: {f};" for k, f in fams.items()]
    for t in tm["types"]:
        th += [f"  --text-{t['css']}: {_num_s(t['size'])}px;", f"  --text-{t['css']}--line-height: {_num_s(t['line_height'])}px;",
               f"  --text-{t['css']}--letter-spacing: {t['tracking']:g}em;", f"  --text-{t['css']}--font-weight: {t['weight']};"]
    th += [f"  --radius-{r['css']}: {_num_s(r['dp'])}px;" for r in tm["radii"]]
    th += [f"  --spacing-{s['css']}: {_num_s(s['dp'])}px;" for s in tm["spacing"]]
    (out / "tailwind.theme.css").write_text('/* Tailwind v4: in your CSS put  @import "tailwindcss"; @import "./tailwind.theme.css";  */\n\n'
                                            "@theme {\n" + "\n".join(th) + "\n}\n")
    return [out / f for f in ("tokens.css", "tailwind.tokens.cjs", "tailwind.theme.css")]


def _emit_dtcg(tm, out):
    """tokens.tokens.json: W3C Design Tokens (the Style Dictionary v4 subset: $type/$value/$description, dimensions as
    "16px" strings, typography composites) with the measured/spec provenance in $description."""
    tok = lambda typ, val, desc: {"$type": typ, "$value": val, "$description": desc}
    doc = {"$description": "Generated by gptkit tokens; DTCG (Style Dictionary v4 subset). Sizes are dp/pt, written as px.",
           "color": {c["css"]: tok("color", _css_hex(c), f"{c['name']}: " + ("snapped to the mockup (dE00 %s from the spec guess %s)"
                                                                            % (c.get("de00"), c.get("hex_guess")) if c.get("snapped")
                                                                            else f"spec guess {c.get('hex_guess', c['hex'])} (not found in the mockup)"))
                     for c in tm["colors"]},
           "typography": {t["css"]: tok("typography", {"fontFamily": t["family"] or ("serif" if t["generic"] == "serif" else "sans-serif"),
                                                       "fontWeight": t["weight"], "fontSize": f"{_num_s(t['size'])}px",
                                                       "lineHeight": f"{_num_s(t['line_height'])}px", "letterSpacing": f"{t['tracking']:g}em"},
                                        f"{t['role']}: size from {t['size_source']}, line height {t['line_height_source']}, "
                                        f"M3 {t['m3_slot']} ({t['m3_slot_source']}), HIG {t['hig_style']}") for t in tm["types"]},
           "dimension": {"radius": {r["css"]: tok("dimension", f"{_num_s(r['dp'])}px", f"{r['name']}: {r['source']}") for r in tm["radii"]},
                         "spacing": {s_["css"]: tok("dimension", f"{_num_s(s_['dp'])}px", f"{s_['name']}: {s_['source']}") for s_ in tm["spacing"]}}}
    (out / "tokens.tokens.json").write_text(json.dumps(doc, indent=2))
    return out / "tokens.tokens.json"


def _lint_tokens(files):
    """Pure-Python checks of the emitted theme code (no compiler needed): unique identifiers per scope, no Compose style
    whose colour names a sibling of the same object (recursive type-check), FontWeight.W only for hundreds, 8-digit
    Color(0x...) literals, Swift keywords backticked, no duplicate CSS custom properties. Returns a list of findings."""
    out = []
    for f in files:
        f = Path(f)
        text = f.read_text()
        if f.suffix == ".kt":
            scope, vals = "<top>", {}
            for n, line in enumerate(text.splitlines(), 1):
                m_ = re.match(r"^(object|class)\s+(\w+)", line)
                if m_:
                    scope = m_.group(2)
                elif line.startswith("}"):
                    scope = "<top>"
                m_ = re.match(r"^\s*(?:val|var|fun)\s+(\w+)", line)
                if m_:
                    key = (scope if line.startswith(" ") else "<top>", m_.group(1))
                    if key in vals:
                        out.append(f"{f.name}:{n}: duplicate identifier {m_.group(1)} in {key[0]} (first at line {vals[key]})")
                    vals[key] = n
                for w in re.findall(r"FontWeight\.W(\d+)", line):
                    if int(w) not in range(100, 1000, 100):
                        out.append(f"{f.name}:{n}: FontWeight.W{w} does not exist (use FontWeight({w}))")
                for w in re.findall(r"FontWeight\((\d+)\)", line):
                    if not 1 <= int(w) <= 1000:
                        out.append(f"{f.name}:{n}: FontWeight({w}) outside 1..1000")
                for lit in re.findall(r"Color\((0x[0-9A-Fa-f]+)\)", line):
                    if not re.fullmatch(r"0x[0-9A-F]{8}", lit):
                        out.append(f"{f.name}:{n}: {lit} is not an 8-digit ARGB literal")
            for obj, body in re.findall(r"^object (\w+) \{\n(.*?)^\}", text, re.S | re.M):
                members = set(re.findall(r"^\s+val (\w+) =", body, re.M))
                for ref in re.findall(r"^\s+color = (\w+),", body, re.M):
                    if ref in members:
                        out.append(f"{f.name}: object {obj}: `color = {ref}` names a sibling of the same object "
                                   f"(recursive type check); qualify it")
        elif f.suffix == ".swift":
            scope, seen = "<top>", {}
            for n, line in enumerate(text.splitlines(), 1):
                m_ = re.match(r"^(extension|enum|struct|class)\s+([\w.]+)", line)
                if m_:
                    scope = m_.group(2)
                m_ = re.match(r"^\s*(?:static\s+)?(?:let|var|func)\s+(`?)(\w+)(`?)", line)
                if m_:
                    name = m_.group(2)
                    if not m_.group(1) and name in _SWIFT_KEYWORDS:
                        out.append(f"{f.name}:{n}: `{name}` is a Swift keyword; backtick it")
                    if (scope, name) in seen:
                        out.append(f"{f.name}:{n}: duplicate declaration {name} in {scope} (first at line {seen[(scope, name)]})")
                    seen[(scope, name)] = n
        elif f.suffix == ".css":
            for block in re.findall(r"\{([^{}]*)\}", text):
                props = re.findall(r"^\s*(--[\w-]+)\s*:", block, re.M)
                for p in sorted({p for p in props if props.count(p) > 1}):
                    out.append(f"{f.name}: duplicate custom property {p}")
    return out


_KT_STUBS = {
    "AnimationCore.kt": "package androidx.compose.animation.core\ninterface Easing\nobject LinearEasing : Easing\n"
                      "class CubicBezierEasing(val a: Float, val b: Float, val c: Float, val d: Float) : Easing\n"
                      "class KeyframesSpec<T>\nclass KeyframeEntity<T>\nclass KeyframesSpecConfig<T> { var durationMillis: Int = 0; var delayMillis: Int = 0; "
                      "infix fun T.at(timeMillis: Int): KeyframeEntity<T> = KeyframeEntity(); infix fun KeyframeEntity<T>.using(easing: Easing): KeyframeEntity<T> = this }\n"
                      "fun <T> keyframes(init: KeyframesSpecConfig<T>.() -> Unit): KeyframesSpec<T> { KeyframesSpecConfig<T>().init(); return KeyframesSpec() }\n",
    "Color.kt": "package androidx.compose.ui.graphics\nclass Color(val value: Long) { companion object { val White = Color(0xFFFFFFFF); "
                "val Black = Color(0xFF000000); val Transparent = Color(0) } }\ninterface Shape\n",
    "Unit.kt": "package androidx.compose.ui.unit\nclass TextUnit(val v: Float)\nclass Dp(val v: Float)\n"
               "val Int.sp: TextUnit get() = TextUnit(toFloat()); val Double.sp: TextUnit get() = TextUnit(toFloat())\n"
               "val Int.em: TextUnit get() = TextUnit(toFloat()); val Double.em: TextUnit get() = TextUnit(toFloat())\n"
               "val Int.dp: Dp get() = Dp(toFloat()); val Double.dp: Dp get() = Dp(toFloat())\n",
    "Font.kt": "package androidx.compose.ui.text.font\nclass FontWeight(val weight: Int) { init { require(weight in 1..1000) }\n"
               "  companion object { val W100 = FontWeight(100); val W200 = FontWeight(200); val W300 = FontWeight(300); val W400 = FontWeight(400)\n"
               "    val W500 = FontWeight(500); val W600 = FontWeight(600); val W700 = FontWeight(700); val W800 = FontWeight(800); val W900 = FontWeight(900) } }\n"
               "class FontStyle(val v: Int) { companion object { val Normal = FontStyle(0); val Italic = FontStyle(1) } }\n"
               "interface Font\nclass ResFont(val res: Int, val weight: FontWeight, val style: FontStyle) : Font\n"
               "fun Font(resId: Int, weight: FontWeight = FontWeight.W400, style: FontStyle = FontStyle.Normal): Font = ResFont(resId, weight, style)\n"
               "open class FontFamily { companion object { val Serif = FontFamily(); val SansSerif = FontFamily(); val Default = FontFamily(); "
               "val Monospace = FontFamily(); val Cursive = FontFamily() } }\n"
               "class FontListFontFamily(val fonts: List<Font>) : FontFamily()\nfun FontFamily(vararg fonts: Font): FontFamily = FontListFontFamily(fonts.toList())\n",
    "Text.kt": "package androidx.compose.ui.text\nimport androidx.compose.ui.graphics.Color\nimport androidx.compose.ui.text.font.*\n"
               "import androidx.compose.ui.unit.TextUnit\ndata class TextStyle(val color: Color? = null, val fontSize: TextUnit? = null, "
               "val fontWeight: FontWeight? = null, val fontStyle: FontStyle? = null, val fontFamily: FontFamily? = null, "
               "val letterSpacing: TextUnit? = null, val lineHeight: TextUnit? = null)\n",
    "Shape.kt": "package androidx.compose.foundation.shape\nimport androidx.compose.ui.unit.Dp\nimport androidx.compose.ui.graphics.Shape\n"
                "class RoundedCornerShape(val r: Dp) : Shape\n",
    "Runtime.kt": "package androidx.compose.runtime\n@Target(AnnotationTarget.FUNCTION, AnnotationTarget.TYPE, AnnotationTarget.PROPERTY_GETTER)\n"
                  "annotation class Composable\n",
    "ComposeTest.kt": "package androidx.compose.ui.test\nimport androidx.compose.runtime.Composable\n"
                       "class MainTestClock { var autoAdvance: Boolean = true; fun advanceTimeBy(milliseconds: Long) {} }\n"
                       "class SemanticsNodeInteraction\nclass ComposeContentTestRule { val mainClock = MainTestClock(); "
                       "fun setContent(content: @Composable () -> Unit) { content() }; fun onRoot() = SemanticsNodeInteraction() }\n",
    "ComposeTestRule.kt": "package androidx.compose.ui.test.junit4\nimport androidx.compose.ui.test.ComposeContentTestRule\n"
                           "fun createComposeRule() = ComposeContentTestRule()\n",
    "Roborazzi.kt": "package com.github.takahirom.roborazzi\nimport androidx.compose.ui.test.SemanticsNodeInteraction\n"
                    "fun SemanticsNodeInteraction.captureRoboImage(filePath: String) {}\n",
    "JUnit.kt": "package org.junit\n@Target(AnnotationTarget.PROPERTY_GETTER, AnnotationTarget.FIELD)\nannotation class Rule\n"
                "@Target(AnnotationTarget.FUNCTION) annotation class Test\n",
    "Foundation.kt": "package androidx.compose.foundation\nimport androidx.compose.runtime.Composable\n@Composable fun isSystemInDarkTheme(): Boolean = false\n",
    "Preview.kt": "package androidx.compose.ui.tooling.preview\nannotation class Preview(val name: String = \"\", val widthDp: Int = -1, "
                  "val heightDp: Int = -1, val showBackground: Boolean = false, val backgroundColor: Long = 0)\n",
    "Material3.kt": "package androidx.compose.material3\nimport androidx.compose.runtime.Composable\nimport androidx.compose.ui.graphics.Color\n"
                    "import androidx.compose.ui.graphics.Shape\nimport androidx.compose.ui.text.TextStyle\nclass ColorScheme\n"
                    "fun darkColorScheme(ROLES): ColorScheme = ColorScheme()\nfun lightColorScheme(ROLES): ColorScheme = ColorScheme()\n"
                    "class Typography(SLOTS)\nclass Shapes(val extraSmall: Shape? = null, val small: Shape? = null, val medium: Shape? = null, "
                    "val large: Shape? = null, val extraLarge: Shape? = null)\n"
                    "object MaterialTheme { val colorScheme: ColorScheme = ColorScheme(); val typography: Typography = Typography(); val shapes: Shapes = Shapes() }\n"
                    "@Composable fun MaterialTheme(colorScheme: ColorScheme = ColorScheme(), shapes: Shapes = Shapes(), typography: Typography = Typography(), "
                    "content: @Composable () -> Unit) { content() }\n",
}
_M3_COLOR_ROLES = ("primary onPrimary primaryContainer onPrimaryContainer inversePrimary secondary onSecondary secondaryContainer "
                   "onSecondaryContainer tertiary onTertiary tertiaryContainer onTertiaryContainer background onBackground surface "
                   "onSurface surfaceVariant onSurfaceVariant surfaceTint inverseSurface inverseOnSurface error onError errorContainer "
                   "onErrorContainer outline outlineVariant scrim surfaceBright surfaceDim surfaceContainer surfaceContainerHigh "
                   "surfaceContainerHighest surfaceContainerLow surfaceContainerLowest").split()
_M3_TYPE_SLOTS = ("displayLarge displayMedium displaySmall headlineLarge headlineMedium headlineSmall titleLarge titleMedium "
                  "titleSmall bodyLarge bodyMedium bodySmall labelLarge labelMedium labelSmall").split()


def _kotlinc_check(kt_files, rpkg, res_fonts):
    """Type-check the generated Kotlin with kotlinc (PATH or GPTKIT_KOTLINC) against signature stubs of the Compose APIs
    the files use (no Compose jars needed). Returns a one-line verdict."""
    exe = os.environ.get("GPTKIT_KOTLINC") or shutil.which("kotlinc")
    if not exe:
        return "skipped (no kotlinc on PATH; the pure-Python lint ran)"
    with tempfile.TemporaryDirectory() as td:
        stubs = Path(td) / "stubs"
        stubs.mkdir()
        roles = ", ".join(f"{r}: Color? = null" for r in _M3_COLOR_ROLES)
        slots = ", ".join(f"val {s}: TextStyle? = null" for s in _M3_TYPE_SLOTS)
        for name, src in _KT_STUBS.items():
            (stubs / name).write_text(src.replace("ROLES", roles).replace("SLOTS", slots))
        consts = "; ".join(f"const val {n} = {i + 1}" for i, n in enumerate(res_fonts)) or "const val placeholder = 0"
        (stubs / "R.kt").write_text(f"package {rpkg}\nobject R {{ object font {{ {consts} }} }}\n")
        try:
            r = subprocess.run([exe, "-nowarn", "-d", str(Path(td) / "out"), *map(str, kt_files), *map(str, stubs.glob("*.kt"))],
                               capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"FAILED: kotlinc did not run ({type(e).__name__})"
    errs = [l for l in (r.stderr + r.stdout).splitlines() if "error:" in l]
    return "ok (kotlinc against Compose API stubs)" if r.returncode == 0 and not errs else "FAILED: " + "; ".join(errs[:4])[:600]


_CHROME_PROBE_WARNED = False


def _verify_tokens(tm, out, a, platforms=("compose", "swiftui", "web")):
    global _CHROME_PROBE_WARNED
    res = {}
    files = []
    if "compose" in platforms:
        files += sorted((out / "compose").glob("*.kt"))
    if "swiftui" in platforms:
        files += sorted((out / "swiftui").glob("*.swift"))
    if "web" in platforms:
        files += [out / "web" / "tokens.css"]
    lint = _lint_tokens([f for f in files if f.exists()])
    res["lint"] = (f"ok ({len(files)} files: unique identifiers, no self-references, valid weights/colour literals)" if not lint
                   else "FAILED: " + "; ".join(lint[:6]))
    if "compose" in platforms:
        rpkg = a.r_package or re.sub(r"(\.ui)?(\.theme)?$", "", a.package)
        res["compose"] = _kotlinc_check(sorted((out / "compose").glob("*.kt")), rpkg,
                                        [r["android"] for f in tm["families"] for r in f["files"]])
    if "swiftui" in platforms:
        sw = out / "swiftui" / f"{a.prefix}Tokens.swift"
        if sys.platform == "darwin" and shutil.which("xcrun"):
            r = subprocess.run(["xcrun", "--sdk", "iphonesimulator", "swiftc", "-typecheck", "-target",
                                "arm64-apple-ios16.0-simulator", str(sw)], capture_output=True, text=True)
            res["swiftui"] = "ok (swiftc -typecheck, iOS simulator SDK)" if r.returncode == 0 else "FAILED: " + r.stderr.strip()[:600]
        else:
            res["swiftui"] = "skipped (swiftc type-check needs macOS with Xcode; the lint ran)"
    if "web" in platforms:
        if shutil.which("node"):
            r = subprocess.run(["node", "-e", f"const c=require({json.dumps(str((out / 'web' / 'tailwind.tokens.cjs').resolve()))});"
                                              "if(!c.theme.extend.colors)process.exit(3)"], capture_output=True, text=True)
            res["tailwind"] = "ok (node require)" if r.returncode == 0 else "FAILED: " + r.stderr.strip()[:400]
        else:
            res["tailwind"] = "skipped (no node)"
        chrome = _chrome()
        if chrome:
            names = [f"--color-{c['css']}" for c in tm["colors"]] + [f"--radius-{r['css']}" for r in tm["radii"]]
            probe = out / "web" / ".css_probe.html"
            probe.write_text("<!doctype html><meta charset=utf-8><link rel=stylesheet href='tokens.css'><body>"
                             + "".join(f"<p class='text-{t['css']}'>x</p>" for t in tm["types"]) +
                             "<pre id=r></pre><script>const s=getComputedStyle(document.documentElement);const o={vars:{},text:{}};"
                             f"for(const n of {json.dumps(names)})o.vars[n]=s.getPropertyValue(n).trim();"
                             "for(const p of document.querySelectorAll('p')){const c=getComputedStyle(p);"
                             "o.text[p.className]=[c.fontSize,c.lineHeight,c.letterSpacing,c.fontWeight,c.color];}"
                             "document.getElementById('r').textContent=JSON.stringify(o);</script>")
            dom, why = "", ""
            with tempfile.TemporaryDirectory() as td:
                try:
                    r = subprocess.run([chrome, *_chrome_flags(800, 600), "--allow-file-access-from-files", f"--user-data-dir={td}/prof",
                                        "--virtual-time-budget=2000", "--dump-dom", probe.resolve().as_uri()],
                                       capture_output=True, text=True, timeout=60)
                    dom, why = r.stdout, _chrome_err(r)
                except (subprocess.TimeoutExpired, OSError) as e:
                    why = type(e).__name__
            m_ = re.search(r'<pre id="r">(.*?)</pre>', dom, re.S)
            if m_:
                got = json.loads(m_.group(1).replace("&quot;", '"'))
                empty = [n for n, v in got["vars"].items() if not v]
                bad = [t["css"] for t in tm["types"] if got["text"].get(f"text-{t['css']}", [""])[0] != f"{_num_s(t['size'])}px"]
                res["css"] = ("ok (Chrome parsed %d vars, %d text classes)" % (len(got["vars"]), len(got["text"]))
                              if not empty and not bad else f"FAILED: empty vars {empty[:5]}, wrong text sizes {bad[:5]}")
            else:
                res["css"] = f"FAILED: headless Chrome produced no probe output ({why})"
                if not _CHROME_PROBE_WARNED:
                    _CHROME_PROBE_WARNED = True
                    print(f"  ! headless Chrome produced no screenshot: {why}")
            probe.unlink(missing_ok=True)
        else:
            res["css"] = "skipped (no Chrome; set GPTKIT_CHROME)"
    return res


def _palette_sheet(palette, mock, texts, spec, path):
    rows = len(palette)
    im = Image.new("RGB", (560, 26 * rows + 24), "white")
    dr = ImageDraw.Draw(im)
    dr.text((6, 6), "name", fill="black")
    dr.text((250, 6), "spec guess | snapped   dE00  source", fill="black")
    for i, p in enumerate(palette):
        y = 24 + 26 * i
        dr.text((6, y + 6), p["name"][:38], fill="black")
        dr.rectangle([250, y, 300, y + 22], fill=p["hex_guess"])
        dr.rectangle([300, y, 350, y + 22], fill=p["hex"])
        dr.text((356, y + 6), f"{p['de00']:>5}  {p.get('source', '')[:28]}", fill="black" if p["snapped"] else "red")
    im.save(path)


def _platform_set(s):
    alias = {"android": "compose", "kotlin": "compose", "ios": "swiftui", "swift": "swiftui", "css": "web", "tailwind": "web"}
    got = []
    for p in re.split(r"[,\s]+", (s or "").strip()):
        if not p:
            continue
        q = alias.get(p.lower(), p.lower())
        if q not in ("compose", "swiftui", "web"):
            sys.exit(f"gptkit: --platforms: unknown '{p}' (use compose, swiftui, web; aliases android, ios)")
        if q not in got:
            got.append(q)
    if not got:
        sys.exit("gptkit: --platforms: nothing to emit (use compose, swiftui, web)")
    return [p for p in ("compose", "swiftui", "web") if p in got]


def cmd_tokens(a):
    platforms = _platform_set(a.platforms)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", a.prefix or ""):
        sys.exit(f"gptkit: --prefix {a.prefix!r} is not an identifier (letters, digits and _, starting with a letter)")
    pkg_re = r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
    kt_keywords = _JAVA_KEYWORDS | {"as", "fun", "in", "is", "object", "typealias", "typeof", "val", "var", "when"}
    for flag, val in (("--package", a.package), ("--r-package", a.r_package)):
        if val is not None and not re.fullmatch(pkg_re, val):
            sys.exit(f"gptkit: {flag} {val!r} is not a valid package name (dotted lowercase identifiers)")
        bad = [seg for seg in (val or "").split(".") if seg in kt_keywords]
        if bad:
            sys.exit(f"gptkit: {flag} {val!r}: {bad[0]!r} is a Kotlin/Java keyword, which no package segment may be")
    if "compose" in platforms and a.package == "com.example.app.ui.theme" and not a.r_package:
        print("  ! --package not given: Kotlin files use com.example.app.ui.theme and import com.example.app.R; "
              "pass --package (and --r-package if the module namespace differs)")
    d, spec, mock = _d2c_load_spec(a.dir)
    out = Path(a.out) if a.out else d / "tokens"
    out.mkdir(parents=True, exist_ok=True)
    texts = _text_targets(d, spec, mock, a.vision)
    _save_text_boxes(d, mock, texts, out / "text_boxes.png")
    assets = (_d2c_manifest(d, mock.size) or {}).get("assets", [])
    assets = [x for x in assets if isinstance(x, dict) and x.get("name") and _clip_bbox(x.get("bbox") or {}, mock.width, mock.height)]
    for x in assets:
        x["bbox"] = _clip_bbox(x["bbox"], mock.width, mock.height)
    palette = _snap_palette(spec, mock, texts, assets) if not a.no_snap else \
        [{"name": p["name"], "usage": p.get("usage", ""), "hex_guess": p["hex"], "hex": p["hex"], "alpha": p.get("alpha", 1.0),
          "de00": 0.0, "snapped": False} for p in spec.get("palette", [])]
    fonts_path = Path(a.fonts) if a.fonts else d / "fonts" / "fonts.json"
    if a.fonts:
        _need(fonts_path, "fonts.json (--fonts)", "run `fonts DIR` first")
    fonts = _json_lists(_read_json(fonts_path, "fonts.json", ("families", "styles")), ("families", "styles"), fonts_path,
                        f"fonts {d}") if fonts_path.exists() else None
    rl_path = d / "redline" / "redline.json"
    redline = _json_lists(_read_json(rl_path, "redline.json", ("items", "radii")), ("items", "radii"), rl_path,
                          f"redline {d}") if rl_path.exists() else None
    tm = _token_model(spec, palette, texts, fonts, redline)
    tm["width_dp"] = int(spec.get("effective_width_dp") or round(mock.width * tm["px_to_dp"]))
    files = []
    if "compose" in platforms:
        files += _emit_compose(tm, a, out / "compose")
    if "swiftui" in platforms:
        files += _emit_swift(tm, a, out / "swiftui")
    if "web" in platforms:
        fcss = fonts_path.parent / "web" / "fonts.css"
        files += _emit_web(tm, a, out / "web", os.path.relpath(fcss, out / "web") if fonts and fcss.exists() else None)
    if a.dtcg:
        files.append(_emit_dtcg(tm, out))
    _palette_sheet(palette, mock, texts, spec, out / "palette_check.png")
    report = {"palette": palette, "text": {r: {k: v for k, v in t.items() if k not in ("alpha", "instances")}
                                           for r, t in texts.items()},
              "typography": [{k: t[k] for k in ("role", "kt", "sw", "css", "family", "weight", "size", "line_height", "line_height_source",
                                                "tracking", "color_hex", "size_source", "platform_style", "m3_slot", "m3_slot_source",
                                                "hig_style")} for t in tm["types"]],
              "m3": {k_: tm.get(k_) for k_ in ("m3_roles_baseline", "m3_slots_baseline", "m3_slot_notes", "m3_shapes")},
              "prefix": a.prefix,
              "radii": [{k: r[k] for k in ("name", "dp", "source", "shape")} for r in tm["radii"]],
              "spacing": [{k: s[k] for k in ("name", "dp", "source")} for s in tm["spacing"]],
              "platforms": platforms, "px_to_dp": tm["px_to_dp"], "fonts": str(fonts_path) if fonts else None}
    print(f"palette ({sum(p['snapped'] for p in palette)}/{len(palette)} snapped to mockup pixels):")
    for p in palette:
        mark = f"{p['hex_guess']} -> {p['hex']}  dE00 {p['de00']:>4}" if p["snapped"] else f"{p['hex_guess']}  (kept: not found)"
        print(f"  {p['name'][:30]:30s} {mark}   [{p.get('source', '')[:60]}]")
    miss = [r for r, t in texts.items() if not t.get("found")]
    print(f"text samples located: {len(texts) - len(miss)}/{len(texts)}"
          + (f"  missing: {', '.join(miss)} " + ("(vision couldn't find it: check the sample text in design_spec.json)"
                                                 if a.vision else "(add --vision)") if miss else ""))
    if not fonts:
        print("  (no fonts.json: generic serif/sans families; run `fonts` first for matched Google Fonts)")
    if "compose" in platforms:
        print(f"M3 colour roles left at baseline: {', '.join(tm['m3_roles_baseline']) or 'none'}")
        print(f"M3 type slots left at baseline: {', '.join(tm['m3_slots_baseline']) or 'none'}"
              + (f"; shapes: {tm['m3_shapes']}" if tm["m3_shapes"] else "; shapes: none matched (chip/card/fab/sheet names)"))
        for n_ in tm["m3_slot_notes"]:
            print(f"  ! {n_}")
    for s_ in tm["spacing"]:
        if not s_["source"].startswith("redline"):
            print(f"  ~ spacing {s_['name']}: GPT estimate {s_['dp']:g}dp (no redline item matched; run `redline` or name the component)")
    if a.verify:
        report["verify"] = _verify_tokens(tm, out, a, platforms)
        for k, v in report["verify"].items():
            print(f"verify {k:8s}: {v}")
    (out / "tokens.json").write_text(json.dumps(report, indent=2))
    print(*[f"  {f}" for f in files], sep="\n")
    print(f"-> {out}/tokens.json, palette_check.png, text_boxes.png")


# ---- fonts: match typography to Google Fonts by rendering and measuring

GF_METADATA = "https://fonts.google.com/metadata/fonts"
GF_WEBFONTS = "https://www.googleapis.com/webfonts/v1/webfonts"
GF_HOSTS = ("fonts.google.com (family list; or the Developer API at www.googleapis.com with GOOGLE_FONTS_API_KEY), "
            "fonts.googleapis.com + fonts.gstatic.com (TTF downloads), raw.githubusercontent.com (licence texts)")
_GENERIC_WORDS = {"display", "text", "sans", "serif", "italic", "mono", "pro", "regular", "bold", "medium", "light", "condensed",
                  "similar", "alternatives", "transitional", "contrast", "high", "slanted", "lightly", "rounded", "humanist",
                  "geometric", "grotesque", "neo", "system", "font", "family", "style", "classic", "modern", "elegant"}
_GF_TTF_ERRORS = {}  # (family, weight, italic) -> why the last download attempt failed


def _cache_dir():
    return Path(os.environ.get("GPTKIT_CACHE", "~/.cache/gptkit")).expanduser()


def _redact_key(s):
    return re.sub(r"(key=)[^&\s]+", r"\1***", str(s))


def _gf_fetch_meta():
    """Fresh family metadata as (list, source): the fonts.google.com list, else the Developer API (GOOGLE_FONTS_API_KEY)
    mapped to the same shape (family, category, subsets, fonts {"400": {}, "700i": {}}, popularity)."""
    errors = []
    try:
        r = requests.get(GF_METADATA, timeout=60)
        r.raise_for_status()
        t = r.text[4:] if r.text.startswith(")]}'") else r.text
        fams = json.loads(t)["familyMetadataList"]
        if not isinstance(fams, list) or not fams or not all(isinstance(f, dict) and f.get("family") for f in fams[:20]):
            raise ValueError("familyMetadataList is not a family list")
        return fams, "fonts.google.com"
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        errors.append(f"{GF_METADATA}: {_redact_key(e)[:160]}")
    key = os.environ.get("GOOGLE_FONTS_API_KEY")
    if key:
        cat = {"sans-serif": "Sans Serif", "serif": "Serif", "display": "Display", "handwriting": "Handwriting", "monospace": "Monospace"}
        try:
            r = requests.get(GF_WEBFONTS, params={"key": key, "sort": "popularity"}, timeout=60)
            r.raise_for_status()
            fams = []
            for i, it in enumerate(r.json()["items"]):
                fonts = {}
                for v in it.get("variants", []):
                    m_ = re.fullmatch(r"(\d{3})?(italic)?", str(v).replace("regular", ""))
                    if m_:
                        fonts[(m_.group(1) or "400") + ("i" if m_.group(2) else "")] = {}
                fams.append({"family": it["family"], "category": cat.get(it.get("category"), it.get("category")),
                             "subsets": it.get("subsets", []), "fonts": fonts, "popularity": i + 1,
                             "isNoto": str(it["family"]).startswith("Noto"), "isOpenSource": True, "classifications": [], "stroke": None})
            if not fams:
                raise ValueError("no items")
            return fams, "webfonts/v1"
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            errors.append(f"{GF_WEBFONTS}: {_redact_key(e)[:160]}")
    else:
        errors.append("Developer API not tried (set GOOGLE_FONTS_API_KEY)")
    raise RuntimeError("could not load the Google Fonts family list (" + "; ".join(errors) + f"); needs network to {GF_HOSTS} once, "
                       "then it is cached for 7 days")


@functools.cache
def _gf_meta():
    """Google Fonts family metadata: the validated cache (refreshed after 7 days when reachable, kept when not), else a
    fresh fetch written atomically. A corrupt cache is deleted and refetched; offline, RuntimeError names the file."""
    p = _cache_dir() / "gf_metadata.json"
    cached, corrupt = None, None
    if p.exists():
        try:
            data = json.loads(p.read_text())
            cached = data["familyMetadataList"]
            if not isinstance(cached, list) or not cached or not all(isinstance(f, dict) and f.get("family") for f in cached[:20]):
                raise ValueError("familyMetadataList is not a family list")
        except (OSError, ValueError, KeyError, TypeError) as e:
            corrupt, cached = f"{type(e).__name__}: {str(e)[:80]}", None
    if cached and time.time() - p.stat().st_mtime <= 7 * 86400:
        return cached
    try:
        fams, src = _gf_fetch_meta()
    except RuntimeError as e:
        if cached:
            print("  (Google Fonts unreachable: using the cached metadata)")
            return cached
        if corrupt:
            p.unlink(missing_ok=True)
            raise RuntimeError(f"cached Google Fonts metadata at {p} was corrupt ({corrupt}; file deleted) and the network is "
                               f"unavailable ({e}); re-run online")
        raise
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps({"familyMetadataList": fams, "source": src, "fetched": time.strftime("%Y-%m-%d")}))
    tmp.replace(p)
    return fams


def _gf_ttf(family, weight, italic=False):
    """Static TTF of one Google Fonts family/weight via the CSS API (cached); None when it cannot be downloaded, with the
    reason kept in _GF_TTF_ERRORS[(family, weight, italic)]."""
    p = _cache_dir() / "fonts" / re.sub(r"\W+", "_", family) / f"{weight}{'i' if italic else ''}.ttf"
    if p.exists():
        return p
    key = (family, weight, italic)
    url = f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:ital,wght@{int(italic)},{weight}"
    try:
        css = requests.get(url, headers={"User-Agent": "gptkit"}, timeout=30)
        if not css.ok:
            _GF_TTF_ERRORS[key] = f"HTTP {css.status_code} from fonts.googleapis.com"
            return None
        m_ = re.search(r"url\((https://[^)]+\.ttf)\)", css.text)
        if not m_:
            _GF_TTF_ERRORS[key] = "no TTF url in the css2 response (weight not available?)"
            return None
        r = requests.get(m_.group(1), timeout=60)
        if not r.ok:
            _GF_TTF_ERRORS[key] = f"HTTP {r.status_code} from fonts.gstatic.com"
            return None
    except requests.RequestException as e:
        _GF_TTF_ERRORS[key] = f"{type(e).__name__}: {str(e)[:120]}"
        return None
    if not r.content[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
        _GF_TTF_ERRORS[key] = "downloaded file is not a TrueType/OpenType font"
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.{id(r)}.part")
    tmp.write_bytes(r.content)
    tmp.replace(p)
    _GF_TTF_ERRORS.pop(key, None)
    return p


def _gf_license(family):
    """{license, dir, text} from the google/fonts repository METADATA.pb (+ the licence file). Only complete lookups are
    cached, so an offline run never pins 'unknown' for a family."""
    slug = re.sub(r"[^a-z0-9]", "", family.lower())
    cache = _cache_dir() / "licenses" / f"{slug}.json"
    if cache.exists():
        try:
            got = json.loads(cache.read_text())
            if got.get("dir") and got.get("text"):
                return got
        except (OSError, ValueError):
            pass
    res = {"license": "unknown (see fonts.google.com)", "dir": None, "text": None}
    for lic_dir, fname in (("ofl", "OFL.txt"), ("apache", "LICENSE.txt"), ("ufl", "UFL.txt")):
        base = f"https://raw.githubusercontent.com/google/fonts/main/{lic_dir}/{slug}/"
        try:
            r = requests.get(base + "METADATA.pb", timeout=20)
            if not r.ok:
                continue
            m_ = re.search(r'license:\s*"([^"]+)"', r.text)
            t = requests.get(base + fname, timeout=20)
            res = {"license": m_.group(1) if m_ else lic_dir.upper(), "dir": f"{lic_dir}/{slug}", "text": t.text if t.ok else None}
        except requests.RequestException as e:
            res["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        break
    if res.get("dir") and res.get("text"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.part")
        tmp.write_text(json.dumps(res))
        tmp.replace(cache)
    return res


def _gf_family(name, what):
    """The metadata entry whose family name matches `name` case-insensitively; exits with close matches otherwise."""
    import difflib
    meta = _gf_meta()
    hit = next((f for f in meta if f["family"].lower() == name.strip().lower()), None)
    if hit:
        return hit
    close = difflib.get_close_matches(name.strip(), [f["family"] for f in meta], n=4, cutoff=0.6)
    sys.exit(f"fonts: {what} '{name}' is not a Google Fonts family" + (f" (close: {', '.join(close)})" if close else ""))


def _ttf_names(path):
    """(PostScript name, full name) from a TTF/OTF name table."""
    import struct
    b = Path(path).read_bytes()
    found = {}
    for i in range(struct.unpack(">H", b[4:6])[0]):
        tag, _, off, _ = struct.unpack(">4sIII", b[12 + 16 * i:28 + 16 * i])
        if tag != b"name":
            continue
        _, count, soff = struct.unpack(">HHH", b[off:off + 6])
        for j in range(count):
            pid, _, _, nid, ln, o = struct.unpack(">6H", b[off + 6 + 12 * j:off + 18 + 12 * j])
            if nid in (4, 6) and nid not in found:
                raw = b[off + soff + o:off + soff + o + ln]
                found[nid] = raw.decode("utf-16-be" if pid in (0, 3) else "latin-1", "replace")
    return found.get(6), found.get(4)


def _named_families(guess):
    meta = [f for f in _gf_meta() if "latin" in f.get("subsets", [])]
    low = guess.lower()
    named = [f for f in sorted(meta, key=lambda f: -len(f["family"]))
             if re.search(r"(?<![a-z])" + re.escape(f["family"].lower()) + r"(?![a-z])", low)]
    return [f for f in named if not any(f is not g and f["family"].lower() in g["family"].lower() for g in named)]


def _font_shortlist(guess, weight, italic, n_similar, display_ok=False):
    meta = [f for f in _gf_meta() if "latin" in f.get("subsets", [])]
    low = guess.lower()
    named = _named_families(guess)
    rest = low
    for f in named:
        rest = rest.replace(f["family"].lower(), " ")
    cats = [f["category"] for f in named]
    for w in re.findall(r"[a-z]{5,}", rest):
        if w not in _GENERIC_WORDS:
            named += [f for f in meta if w in f["family"].lower().split() and f not in named][:3]
    if cats:
        cat = cats[0]
    elif re.search(r"sans", low):
        cat = "Sans Serif"
    elif re.search(r"serif|baskerville|garamond|didone|slab|caslon|bodoni", low):
        cat = "Serif"
    elif re.search(r"mono|code", low):
        cat = "Monospace"
    elif re.search(r"script|hand", low):
        cat = "Handwriting"
    else:
        cat = "Sans Serif"
    slab = "slab" in low
    banned = {"Handwriting", "Symbols"} | (set() if display_ok else {"Display"})

    def usable(f):
        keys = [k for k in f["fonts"] if k.endswith("i") == italic]
        return keys and min(abs(int(k.rstrip("i")) - weight) for k in keys) <= 100
    similar = [f for f in sorted(meta, key=lambda f: f.get("popularity") or 9999)
               if f["category"] == cat and f not in named and usable(f) and not f.get("isNoto")
               and f.get("isOpenSource", True) and (f.get("stroke") != "Slab Serif" or slab)
               and not set(f.get("classifications") or []) & banned]
    return [f for f in named if usable(f)] + similar[:n_similar], cat


@functools.cache
def _gpos_pairs(path):
    """Pair-kerning lookups of a font (GPOS PairPos) when fontTools is installed; [] otherwise."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None, [], 1000
    f = TTFont(str(path), lazy=True)
    subs = []
    if "GPOS" in f:
        for li in f["GPOS"].table.LookupList.Lookup:
            for st in li.SubTable:
                lt = li.LookupType
                if lt == 9:
                    lt, st = st.ExtensionLookupType, st.ExtSubTable
                if lt != 2:
                    continue
                cov = {gname: i for i, gname in enumerate(st.Coverage.glyphs)}
                if st.Format == 1:
                    pairs = {}
                    for gname, i in cov.items():
                        pairs[gname] = {r.SecondGlyph: getattr(r.Value1, "XAdvance", 0) or 0
                                        for r in st.PairSet[i].PairValueRecord if r.Value1 is not None}
                    subs.append((1, cov, pairs))
                elif st.Format == 2:
                    subs.append((2, cov, (st.ClassDef1.classDefs, st.ClassDef2.classDefs, st.Class1Record)))
    return f.getBestCmap(), subs, f["head"].unitsPerEm


def _kern(path, text, size):
    """Pixel kerning after each character of `text` at `size`."""
    cmap, subs, upm = _gpos_pairs(str(path))
    if not subs:
        return [0.0] * len(text)
    out = []
    for a, b in zip(text, text[1:]):
        ga, gb = cmap.get(ord(a)), cmap.get(ord(b))
        k = 0
        for fmt, cov, data in subs if ga and gb else ():
            if ga not in cov:
                continue
            if fmt == 1:
                k = data.get(ga, {}).get(gb)
            else:
                cd1, cd2, rec = data
                v = rec[cd1.get(ga, 0)].Class2Record[cd2.get(gb, 0)].Value1
                k = (getattr(v, "XAdvance", 0) or 0) if v is not None else None
            if k is not None:
                break
        out.append((k or 0) * size / upm)
    return out + [0.0]


FONT_EMBOLDEN = 1  # quarter-pixel stem darkening at 4x supersampling; calibrated against Chrome/Skia text


def _render_line(font, text, track_px, kern=None, ss=4, stroke=None):
    """Coverage map of one text line (tight, supersampled) plus each character's [x0, x1) advance span in it.
    `stroke` is the stem darkening in quarter pixels (default FONT_EMBOLDEN, calibrated against Chrome/Skia text)."""
    from PIL import ImageFont
    big = ImageFont.truetype(font.path, font.size * ss)
    asc, desc = big.getmetrics()
    advs = [big.getlength(c) for c in text]
    kern = kern or [0.0] * len(text)
    pad = int(big.size) + 4
    im = Image.new("L", (int(sum(advs) + (abs(track_px) + max(map(abs, kern))) * ss * len(text)) + 2 * pad,
                        asc + desc + 2 * pad), 0)
    dr = ImageDraw.Draw(im)
    x, spans = pad, []
    for ch, adv, k in zip(text, advs, kern):
        dr.text((x, pad + asc), ch, font=big, fill=255, anchor="ls", stroke_width=FONT_EMBOLDEN if stroke is None else stroke,
                stroke_fill=255)
        spans.append((x, x + adv))
        x += adv + (track_px + k) * ss
    bb = im.getbbox()
    if not bb:
        return None, []
    x0 = bb[0] - bb[0] % ss
    y0 = bb[1] - bb[1] % ss
    im = im.crop((x0, y0, bb[2] + (-bb[2]) % ss, bb[3] + (-bb[3]) % ss))
    im = im.resize((im.width // ss, im.height // ss), Image.BOX)
    arr = np.asarray(im, dtype=np.float64) / 255
    rows, cols = np.flatnonzero(arr.max(1) > _TIGHT), np.flatnonzero(arr.max(0) > _TIGHT)
    if not len(rows):
        return None, []
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    off = x0 / ss + cols[0]
    return arr, [(a / ss - off, b / ss - off) for a, b in spans]


def _soft_iou(a, b, sigma):
    fa = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    fb = Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    x, y = np.asarray(fa, dtype=np.float64), np.asarray(fb, dtype=np.float64)
    return float(np.minimum(x, y).sum() / max(np.maximum(x, y).sum(), 1e-6))


def _edges(a):
    gy, gx = np.gradient(a)
    return np.clip(np.hypot(gx, gy) * 2, 0, 1)


def _font_metric(t, c):
    sigma = max(0.6, 0.012 * t.shape[0])
    return (_soft_iou(t, c, sigma) + _soft_iou(t, c, 2 * sigma) + _soft_iou(_edges(t), _edges(c), sigma)) / 3


def _word_cols(target, n):
    """Column spans of `n` words in a coverage map (None unless the gaps split it into exactly n)."""
    ink = (target > 0.3).any(0)
    for merge in range(1, max(2, target.shape[0])):
        segs = _d2c_runs(ink, merge)
        if len(segs) == n:
            return segs
        if len(segs) < n:
            return None
    return None


FONT_GAMMA = 1.0
_TIGHT = 0.25


def _tight(a, thr=None):
    """Crop a coverage map to the rows/columns reaching `thr` (the same rule for targets and renders)."""
    thr = _TIGHT if thr is None else thr
    rows, cols = np.flatnonzero(a.max(1) > thr), np.flatnonzero(a.max(0) > thr)
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] if len(rows) and len(cols) else a


def _soft_h(a):
    """Sub-pixel ink height: area under the row-max coverage profile."""
    return float(a.max(1).sum())


def _overlay(t, c, metric):
    """Metric of two coverage maps placed on one canvas with their coverage centroids aligned (sub-pixel)."""
    H, W = max(t.shape[0], c.shape[0]) + 6, max(t.shape[1], c.shape[1]) + 6
    ct, cc = np.zeros((H, W)), np.zeros((H, W))
    ct[3:3 + t.shape[0], 3:3 + t.shape[1]] = t
    cen = lambda a: (np.arange(a.shape[0]) @ a.sum(1) / max(a.sum(), 1e-9), np.arange(a.shape[1]) @ a.sum(0) / max(a.sum(), 1e-9))
    (ty, tx), (cy, cx) = cen(t), cen(c)
    dy, dx = 3 + ty - cy, 3 + tx - cx
    im = Image.fromarray(c.astype(np.float32), "F").transform((W, H), Image.AFFINE, (1, 0, -dx, 0, 1, -dy), Image.BILINEAR)
    cc = np.asarray(im, dtype=np.float64)
    return metric(ct, cc)


def _shape_similarity(target, cand, text, metric=None, cand_cols=None):
    """Glyph-shape similarity of two coverage maps of the same text at the same scale; words are aligned separately
    when both split cleanly (tolerates irregular word spacing)."""
    metric = metric or _font_metric
    target = _tight(target)
    if not cand_cols:
        cand = _tight(cand)
    words = len(re.findall(r"\S+", text))
    tcols = _word_cols(target, words) if words > 1 else None
    ccols = (cand_cols or _word_cols(cand, words)) if tcols else None
    if not tcols or not ccols or len(ccols) != len(tcols):
        return _overlay(target, cand, metric), False
    tot = score = 0.0
    for (a0, a1), (c0, c1) in zip(tcols, ccols):
        c0, c1 = int(max(0, math.floor(c0))), int(math.ceil(c1))
        cols = np.flatnonzero((cand[:, c0:c1] > _TIGHT).any(0))
        if not len(cols):
            continue
        cw = cand[:, c0 + cols[0]:c0 + cols[-1] + 1]
        tw_ = target[:, a0:a1]
        score += _overlay(tw_, cw, metric) * tw_.shape[1]
        tot += tw_.shape[1]
    return (score / tot, True) if tot else (_overlay(target, cand, metric), False)


def _fit_font(path, text, target, size0, metric=None):
    """Fit size (by sub-pixel ink height) and tracking (by ink width) of `text` in a font to a target coverage map, then
    score the glyph shapes at that fit."""
    from PIL import ImageFont
    metric = metric or _font_metric
    target = _tight(target)
    th, tw = target.shape
    goal = _soft_h(target)
    size = float(size0)
    for _ in range(3):
        r, _ = _render_line(ImageFont.truetype(str(path), size), text, 0, _kern(path, text, size))
        if r is None:
            return None
        size *= goal / max(_soft_h(r), 1e-6)
    font = ImageFont.truetype(str(path), size)
    kern = _kern(path, text, size)
    r0, _ = _render_line(font, text, 0, kern)
    track = min(0.6, max(-0.15, (tw - r0.shape[1]) / max(1, len(text) - 1) / size))
    words = [(m_.start(), m_.end() - 1) for m_ in re.finditer(r"\S+", text)]
    best = None
    for stroke in sorted({FONT_EMBOLDEN, 0}):  # the mockup's renderer may or may not darken stems: keep the better fit
        r, spans = _render_line(font, text, track * size, kern, stroke=stroke)
        r = r ** FONT_GAMMA
        sim, word_mode = _shape_similarity(target, r, text, metric, [(spans[i][0], spans[j][1]) for i, j in words])
        if best is None or sim > best[0]:
            best = (sim, word_mode, r, stroke)
    sim, word_mode, r, stroke = best
    asc, desc = font.getmetrics()
    return {"similarity": round(sim, 4), "size_px": round(size, 1), "tracking_em": round(track, 3), "embolden": stroke,
            "word_mode": word_mode, "natural_line_height_em": round((asc + desc) / size, 3), "ascent_em": round(asc / size, 3),
            "render": r}


def _match_style(st, t, cands, italic, n_weights, paths, metric=None):
    """All (family, weight, fit) results for one located text style, best first."""
    out = []
    for fam in cands:
        keys = sorted(int(k.rstrip("i")) for k in fam["fonts"] if k.endswith("i") == italic)
        for w in sorted(keys, key=lambda k: abs(k - st.get("weight", 400)))[:n_weights]:
            p = paths.get((fam["family"], w))
            if p and abs(w - st.get("weight", 400)) <= 300:
                fit = _fit_font(p, t["sample"], t["alpha"], st["size_px"], metric)
                if fit:
                    out.append((fam["family"], w, fit))
    return sorted(out, key=lambda r: -r[2]["similarity"])


def _font_group(st):
    g = st.get("family_guess", "")
    named = ",".join(sorted(f["family"] for f in _named_families(g)))
    return (named or re.split(r"[;(]|\bor\b|,", g)[0].strip().lower()) + ("|italic" if re.search(r"italic|slant|oblique", g, re.I) else "")


def _font_sheet(rows, path):
    from PIL import ImageFont
    H, lab = 56, 14
    W = 1500
    im = Image.new("RGB", (W, len(rows) * (H + lab + 6) + 6), "white")
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.load_default(size=12)
    except TypeError:
        f = ImageFont.load_default()
    for i, (title, tiles) in enumerate(rows):
        y = 6 + i * (H + lab + 6)
        x = 4
        for label, tile in tiles:
            t = tile.copy()
            t.thumbnail((360, H))
            if x + t.width > W:
                break
            im.paste(t, (x, y + lab))
            dr.text((x, y), label[:56], fill="black" if label.startswith(("DESIGN", "*")) else "#555555", font=f)
            x += max(t.width, 150) + 14
        dr.text((W - 260, y), title[:40], fill="#0050c8", font=f)
    im.save(path)


_CSS_FALLBACK = {"Serif": "Georgia, 'Times New Roman', serif", "Sans Serif": "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
                 "Display": "system-ui, sans-serif", "Handwriting": "cursive", "Monospace": "ui-monospace, Menlo, Consolas, monospace"}


_WOFF2_WHY = "not attempted"


def _woff2(ttf):
    """Write <name>.woff2 next to a TTF when fontTools and brotli are importable; None (with the reason in _WOFF2_WHY) otherwise."""
    global _WOFF2_WHY
    try:
        import brotli  # noqa: F401
        from fontTools.ttLib import TTFont
    except ImportError as e:
        _WOFF2_WHY = f"{e.name or 'fonttools/brotli'} not installed (pip install fonttools brotli)"
        return None
    out = Path(ttf).with_suffix(".woff2")
    try:
        f = TTFont(str(ttf))
        f.flavor = "woff2"
        f.save(str(out))
    except Exception as e:  # any fontTools/brotli failure: keep the TTF and say why
        _WOFF2_WHY = f"{type(e).__name__}: {str(e)[:80]}"
        return None
    return out


def cmd_fonts(a):
    d, spec, mock = _d2c_load_spec(a.dir)
    try:
        meta = _gf_meta()
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
        sys.exit(f"fonts: {e}")
    forced = _gf_family(a.family, "--family") if a.family else None
    extras = [_gf_family(x, "--extra") for x in (a.extra or "").split(",") if x.strip()]
    out = Path(a.out) if a.out else d / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    rgb = mock.convert("RGB")
    texts = _text_targets(d, spec, mock, a.vision)
    styles = [st for st in spec.get("typography", []) if texts.get(st["role"], {}).get("found")]
    skipped = [st["role"] for st in spec.get("typography", []) if st not in styles]
    if skipped:
        print("  ! no text box for:", ", ".join(skipped),
              "(these keep their spec values; " + ("vision couldn't find them: check the sample text)" if a.vision else "add --vision to locate them)"))
    if any(ord(ch) > 0x024F for st in styles for ch in st.get("sample", "")):
        print("  ! samples contain non-Latin characters: the matcher renders Latin glyphs only (no shaping, ligatures or RTL); "
              "results for those samples are unreliable")
    groups = {}
    for st in styles:
        groups.setdefault(_font_group(st), []).append(st)
    results, report_groups, sheet_rows = {}, [], []
    k = spec.get("px_to_dp", 1.0)
    n_need = n_failed = 0
    last_err = ""
    forced_used = False
    cats = {gkey: _font_shortlist(members[0].get("family_guess", ""), 400, False, 0)[1] for gkey, members in groups.items()}
    force_all = bool(forced) and forced["category"] not in cats.values()
    if force_all and groups:
        print(f"  (--family {forced['family']} is {forced['category']}; no group has that category, so it is tried for every group)")
    for gkey, members in groups.items():
        italic = gkey.endswith("|italic")
        guess = members[0].get("family_guess", "")
        wts = sorted({st.get("weight", 400) for st in members})
        display_ok = any(st["size_px"] * k >= 20 for st in members)
        cands, cat = _font_shortlist(guess, wts[len(wts) // 2], italic, a.candidates, display_ok)
        cands += [f for f in extras if f not in cands]
        if forced and forced not in cands and (forced["category"] == cat or force_all):
            cands.append(forced)
        print(f"group '{guess[:50]}' ({len(members)} styles, {cat}): {len(cands)} candidate families")
        need = sorted({(f["family"], w) for f in cands for st in members
                       for w in sorted((int(x.rstrip("i")) for x in f["fonts"] if x.endswith("i") == italic),
                                       key=lambda x: abs(x - st.get("weight", 400)))[:a.weights]})
        with cf.ThreadPoolExecutor(8) as ex:
            paths = dict(zip(need, ex.map(lambda fw: _gf_ttf(fw[0], fw[1], italic), need)))
        failed = [fw for fw in need if paths[fw] is None]
        n_need += len(need)
        n_failed += len(failed)
        if failed:
            last_err = _GF_TTF_ERRORS.get((failed[-1][0], failed[-1][1], italic), "")
        if need and len(failed) == len(need):
            print(f"  ! none of the {len(need)} candidate TTFs for this group could be downloaded ({last_err}); group skipped")
            skipped += [m["role"] for m in members]
            continue
        if failed:
            print(f"  ! {len(failed)}/{len(need)} candidate fonts could not be downloaded: "
                  + ", ".join(f"{f} {w}" for f, w in failed[:6]) + (" ..." if len(failed) > 6 else "") + f" ({last_err})")
        with cf.ThreadPoolExecutor(8) as ex:
            per = dict(zip([m["role"] for m in members],
                           ex.map(lambda st: _match_style(st, texts[st["role"]], cands, italic, a.weights, paths), members)))
        fam_score = {}
        reliable = [m for m in members if per[m["role"]] and per[m["role"]][0][2]["similarity"] >= 0.45]
        for st in reliable:
            best_by_fam = {}
            for fam, w, fit in per[st["role"]]:
                best_by_fam.setdefault(fam, fit["similarity"])
            top = max(best_by_fam.values())
            area = math.sqrt(texts[st["role"]]["box"][2] * texts[st["role"]]["box"][3])
            for fam, sim in best_by_fam.items():
                fam_score.setdefault(fam, []).append((sim / top, area))
        n_ok = len(reliable)
        ranking = sorted(((fam, sum(sc * ar for sc, ar in v) / sum(ar for _, ar in v)) for fam, v in fam_score.items()
                          if len(v) == n_ok), key=lambda t: -t[1])
        if not ranking:
            best = max((per[m["role"]][0][2]["similarity"] for m in members if per[m["role"]]), default=0)
            print(f"  ! no reliable match (best similarity {best:.2f} < 0.45): likely custom lettering or a family outside "
                  "the shortlist; ship it as an SVG/VectorDrawable, or pass --family/--extra. Styles: "
                  + ", ".join(m["role"] for m in members))
            skipped += [m["role"] for m in members]
            continue
        named = [f["family"] for f in _named_families(guess)]
        best = ranking[0][1]
        prior = next((f for f, sc in ranking if f in named and sc >= best - a.tie), None)
        if forced and forced in cands and not any(f == forced["family"] for f, _ in ranking):
            print(f"  ! --family {forced['family']}: could not be measured for this group (download failed or no reliable fit); "
                  "using the best measured family")
        winner = forced["family"] if forced and any(f == forced["family"] for f, _ in ranking) else (prior or ranking[0][0])
        why = "forced" if forced and winner == forced["family"] else ("spec's named family, within tie margin" if prior and prior != ranking[0][0]
                                                                      else "best measured")
        forced_used |= why == "forced"
        report_groups.append({"guess": guess, "category": cat, "italic": italic, "winner": winner, "decided_by": why,
                              "ranking": [[f, round(sc, 4)] for f, sc in ranking[:8]], "styles": [m["role"] for m in members],
                              "downloads_failed": [f"{f} {w}" for f, w in failed]})
        print(f"  -> {winner} [{why}]   (top: " + ", ".join(f"{f} {sc:.3f}" for f, sc in ranking[:4]) + ")")
        ties = [f for f, sc in ranking[1:6] if sc >= ranking[0][1] - 0.005]
        if ties:
            report_groups[-1]["tied_with"] = ties
            print(f"     measured tie with {', '.join(ties)}: the mockup can't separate them; choose by taste with --family")
        for st in members:
            t = texts[st["role"]]
            rs = per[st["role"]]
            mine = [r for r in rs if r[0] == winner]
            if not mine:
                continue
            fam, w, fit = mine[0]
            tie = None
            spec_w = st.get("weight")
            alt = next((r for r in mine if r[1] == spec_w), None)
            steps = sorted({r[1] for r in mine})
            if alt and alt[1] != w and abs(steps.index(alt[1]) - steps.index(w)) == 1 and alt[2]["similarity"] >= fit["similarity"] - 0.07:
                tie = {"measured": w, "chosen": spec_w, "similarity": [fit["similarity"], alt[2]["similarity"]]}
                fam, w, fit = alt
            ok = rs[0][2]["similarity"] >= 0.45 and -0.15 < fit["tracking_em"] < 0.6
            avail = [int(x.rstrip("i")) for f in cands if f["family"] == winner for x in f["fonts"] if x.endswith("i") == italic]
            spec_w = min(avail, key=lambda x: abs(x - st.get("weight", w))) if avail else w
            results[st["role"]] = {"role": st["role"], "family": winner, "weight": w if ok else spec_w, "italic": italic,
                                   "size_px": fit["size_px"] if ok else None,
                                   "tracking_em": fit["tracking_em"] if ok else st.get("letter_spacing_em", 0),
                                   "reliable": ok, "weight_tie": tie,
                                   "similarity": fit["similarity"], "word_mode": fit["word_mode"],
                                   "natural_line_height_em": fit["natural_line_height_em"], "ascent_em": fit["ascent_em"],
                                   "spec": {"size_px": st["size_px"], "weight": st.get("weight"),
                                            "letter_spacing_em": st.get("letter_spacing_em"), "family_guess": guess},
                                   "alternatives": [[r[0], r[1], r[2]["similarity"]] for r in rs[:6]]}
            if not ok:
                results[st["role"]]["warning"] = (f"unreliable measurement (best fit {rs[0][2]['similarity']:.2f}); kept the spec's "
                                                  "size/weight/tracking; check its box in text_boxes.png")
            elif rs[0][0] != winner and rs[0][2]["similarity"] > fit["similarity"] + 0.1:
                results[st["role"]]["warning"] = (f"fits {winner} poorly ({fit['similarity']:.2f}); {rs[0][0]} {rs[0][1]} fits this "
                                                  f"style much better ({rs[0][2]['similarity']:.2f})")
            if tie:
                results[st["role"]]["note"] = (f"weights {tie['measured']} and {tie['chosen']} tie within 0.07 "
                                               f"({tie['similarity'][0]:.3f} vs {tie['similarity'][1]:.3f}); kept the spec's {tie['chosen']}")
            bg, ink = np.array(_hexi(t["bg"]), float), np.array(_hexi(t["ink"]), float)
            x, y, w_, h_ = t["box"]
            tiles = [("DESIGN (mockup pixels)", rgb.crop((x - 4, y - 4, x + w_ + 4, y + h_ + 4)))]
            seen = set()
            for fam_, wt, ft in [mine[0]] + [r for r in rs if r is not mine[0]]:
                if (fam_, wt) in seen or len(tiles) > a.show:
                    continue
                seen.add((fam_, wt))
                col = (bg + ft["render"][..., None] * (ink - bg)).astype(np.uint8)
                tile = Image.new("RGB", (col.shape[1] + 8, col.shape[0] + 8), tuple(int(v) for v in bg))
                tile.paste(Image.fromarray(col), (4, 4))
                star = "*" if fam_ == winner and wt == w else ""
                tiles.append((f"{star}{fam_} {wt} sim {ft['similarity']:.3f}", tile))
            sheet_rows.append((f"{st['role']} ({t['sample'][:18]})", tiles))
    if n_need and n_failed == n_need:
        sys.exit(f"fonts: could not download any candidate TTF from fonts.googleapis.com/fonts.gstatic.com ({last_err}); "
                 "check network/proxy, then re-run (fonts.json left untouched)")
    if forced and not forced_used and results:
        print(f"  ! --family {forced['family']} was not used by any group")
    _font_sheet(sheet_rows, out / "fonts_review.png")
    families = []
    ttf_dir, res_font, lic_dir = out / "ttf", out / "android" / "res" / "font", out / "licenses"
    if (out / "fonts.json").exists():
        try:
            old = json.loads((out / "fonts.json").read_text()).get("families", [])
        except (OSError, ValueError):
            old = []
        for f in old:
            stale = [(out / x["ttf"]) if not Path(x["ttf"]).is_absolute() else Path(x["ttf"]) for x in f.get("files", []) if x.get("ttf")]
            stale += [res_font / f"{x['android']}.ttf" for x in f.get("files", []) if x.get("android")]
            stale += [res_font / f"{_res_name(f['family'], 'f_')}.xml"] + list(lic_dir.glob(re.sub(r"\W+", "", f["family"]) + "-*.txt"))
            for x in stale:
                x.unlink(missing_ok=True)
    fetch_failed = []
    for fam in sorted({r["family"] for r in results.values()}):
        rs = [r for r in results.values() if r["family"] == fam]
        lic = _gf_license(fam)
        files, ps = [], {}
        for w, it in sorted({(r["weight"], r["italic"]) for r in rs}):
            src = _gf_ttf(fam, w, it)
            if not src:
                fetch_failed.append(f"{fam} {w}{' italic' if it else ''}")
                print(f"  ! could not download {fam} {w}{' italic' if it else ''} ({_GF_TTF_ERRORS.get((fam, w, it), '')}); skipped")
                continue
            psn, full = _ttf_names(src)
            fname = (psn or fam.replace(" ", "") + f"-{w}") + ".ttf"
            ttf_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, ttf_dir / fname)
            wname = {100: "thin", 200: "extralight", 300: "light", 400: "regular", 500: "medium", 600: "semibold", 700: "bold",
                     800: "extrabold", 900: "black"}.get(w, str(w))
            android = _res_name(f"{fam}_{wname}{'_italic' if it else ''}", "f_")
            res_font.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, res_font / f"{android}.ttf")
            entry = {"weight": w, "italic": it, "ttf": str(Path("ttf") / fname), "android": android, "postscript": psn}
            woff2 = _woff2(ttf_dir / fname)
            if woff2:
                entry["woff2"] = str(Path("ttf") / woff2.name)
            files.append(entry)
            ps[f"{w}{'i' if it else ''}"] = psn
        if not files:
            continue
        xml_name = _res_name(fam, "f_")
        if xml_name != re.sub(r"[^a-z0-9]+", "_", fam.lower()).strip("_"):
            print(f"  (Android resource name for {fam}: {xml_name})")
        items = "".join(f'    <font\n        android:font="@font/{f["android"]}"\n        android:fontStyle="'
                        f'{"italic" if f["italic"] else "normal"}"\n        android:fontWeight="{f["weight"]}"\n'
                        f'        app:font="@font/{f["android"]}"\n        app:fontStyle="{"italic" if f["italic"] else "normal"}"\n'
                        f'        app:fontWeight="{f["weight"]}" />\n' for f in files)
        (res_font / f"{xml_name}.xml").write_text('<?xml version="1.0" encoding="utf-8"?>\n<font-family xmlns:android='
                                                  '"http://schemas.android.com/apk/res/android"\n    xmlns:app='
                                                  f'"http://schemas.android.com/apk/res-auto">\n{items}</font-family>\n')
        if lic.get("text"):
            lic_dir.mkdir(parents=True, exist_ok=True)
            (lic_dir / (re.sub(r"\W+", "", fam) + f"-{lic['license']}.txt")).write_text(lic["text"])
        else:
            print(f"  ! license text for {fam} not fetched ({lic.get('error', 'not found on github.com/google/fonts')}); "
                  "add it before shipping (the OFL requires the licence and copyright notice to accompany the fonts)")
        cat = next((f.get("category") for f in meta if f["family"] == fam), None)
        wts = sorted({f["weight"] for f in files})
        has_italic = any(f["italic"] for f in files)
        axis = ("ital,wght@" + ";".join(f"{int(f['italic'])},{f['weight']}" for f in sorted(files, key=lambda f: (f["italic"], f["weight"])))
                if has_italic else "wght@" + ";".join(map(str, wts)))
        families.append({"family": fam, "category": cat, "license": lic["license"], "source": lic.get("dir"), "files": files,
                         "postscript": ps, "android_family": f"@font/{xml_name}",
                         "css": f"https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}:{axis}&display=swap",
                         "css_fallback": _CSS_FALLBACK.get(cat, "sans-serif")})
    web = out / "web"
    web.mkdir(parents=True, exist_ok=True)
    stack = "".join(f"/* {f['family']}: font-family: \"{f['family']}\", {f['css_fallback']}; */\n" for f in families)
    src_of = lambda x: (f"url(\"../{x['woff2']}\") format(\"woff2\"), " if x.get("woff2") else "") + f"url(\"../{x['ttf']}\") format(\"truetype\")"
    (web / "fonts.css").write_text(stack + "".join(
        f"@font-face {{\n  font-family: \"{f['family']}\";\n  font-style: {'italic' if x['italic'] else 'normal'};\n"
        f"  font-weight: {x['weight']};\n  font-display: swap;\n  src: {src_of(x)};\n}}\n"
        for f in families for x in f["files"]))
    if families and not any(x.get("woff2") for f in families for x in f["files"]):
        print(f"  (WOFF2 skipped: {_WOFF2_WHY}; web/fonts.css serves the TTFs)")
    link_html = ""
    if families:
        href = "https://fonts.googleapis.com/css2?" + "&".join(f["css"].split("?", 1)[1].replace("&display=swap", "") for f in families) + "&display=swap"
        link_html = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" '
                     f'crossorigin>\n<link href="{href}" rel="stylesheet">')
        (web / "fonts-link.html").write_text(link_html + "\n")
    ios = out / "ios"
    ios.mkdir(parents=True, exist_ok=True)
    import plistlib
    (ios / "UIAppFonts.plist").write_bytes(plistlib.dumps({"UIAppFonts": [Path(x["ttf"]).name for f in families for x in f["files"]]}))
    # the app: attributes come from androidx.core's declare-styleable FontFamilyFont at build time; a stand-alone link
    # needs that declaration too, or every app:font is "attribute not found"
    aapt2_msg = _aapt2_tree(out / "android" / "res", extra_res={"values/gptkit_fontfamily_attrs.xml": (
        '<resources><declare-styleable name="FontFamilyFont"><attr name="font" format="reference"/>'
        '<attr name="fontStyle"><enum name="normal" value="0"/><enum name="italic" value="1"/></attr>'
        '<attr name="fontWeight" format="integer"/></declare-styleable></resources>')}) if families else None
    if aapt2_msg and aapt2_msg.startswith("aapt2 compile+link ok"):
        aapt2_msg += " (app: attributes against androidx.core's FontFamilyFont declaration)"
    manifest = {"families": families, "groups": report_groups, "styles": list(results.values()), "unmatched": skipped,
                "downloads_failed": fetch_failed, "link_html": link_html, "aapt2": aapt2_msg, "paths_relative_to": str(out)}
    tmp = (out / "fonts.json").with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out / "fonts.json")
    print("\nstyle                           family                 wt  size px (spec)   tracking em (spec)  sim")
    for r in results.values():
        sp = r.get("spec", {})
        print(f"  {r['role'][:30]:30s} {r['family'][:20]:20s} {r['weight']:>4} {r['size_px'] or '-':>6} ({sp.get('size_px', '-')!s:>4})"
              f"   {r['tracking_em']:>7} ({sp.get('letter_spacing_em', '-')!s:>6})   {r['similarity']:.3f}")
    for r in results.values():
        for key in ("warning", "note"):
            if r.get(key):
                print(f"  {'!' if key == 'warning' else '~'} {r['role']}: {r[key]}")
    for f in families:
        print(f"{f['family']}: {f['license']} ({f['source']}); files: " + ", ".join(Path(x["ttf"]).name for x in f["files"]))
    if aapt2_msg:
        print(f"  android/res/font: {aapt2_msg}")
    print(f"-> {out}/fonts.json, fonts_review.png, ttf/, android/res/font/, ios/UIAppFonts.plist, web/fonts.css, web/fonts-link.html, licenses/")
    print(f"iOS: add ttf/*.ttf to the app target (Build Phases > Copy Bundle Resources, or the SwiftPM target's resources), then\n"
          f"     /usr/libexec/PlistBuddy -c \"Merge {ios / 'UIAppFonts.plist'}\" App/Info.plist   (use Font.custom(<PostScript name>))\n"
          "Web: link web/fonts.css (self-hosted TTF), or paste web/fonts-link.html for the Google Fonts CDN. Re-run `tokens` to use these families.")


# ---- redline: pixel-true geometry (edges, corner radii, gaps) for the components of an approved design

def _rise(dev, tau, lab=None, span=4, inside_ok=False):
    """Sub-pixel edge along a profile: the smoothed deviation `dev` first reaches tau at i; the edge is the largest
    colour step (Lab distance between neighbouring pixels of the `lab` profile) within [i-3, i+span], placed where the
    colour is halfway across that step. A drop shadow ramps up slowly before the real edge and anti-aliasing spreads it
    over two pixels; both are handled by the step search. None when the profile never rises, or starts inside (unless
    `inside_ok`, used behind drop shadows, and a clear step is in reach)."""
    idx = np.flatnonzero(dev >= tau)
    if not len(idx) or (idx[0] == 0 and not inside_ok):
        return None
    i = int(idx[0])
    if lab is None:
        return None if i == 0 else i - 1 + (tau - dev[i - 1]) / max(dev[i] - dev[i - 1], 1e-9) + 0.5
    lo, hi = max(0, i - 3), min(len(lab), i + span + 1)
    seg = np.asarray(lab[lo:hi], dtype=np.float64)
    if len(seg) < 3:
        return None
    steps = np.linalg.norm(np.diff(seg, axis=0), axis=-1)
    j = int(np.argmax(steps)) + 1
    if i == 0 and steps[j - 1] < max(2.0, 0.5 * tau):
        return None
    a0, a1 = max(0, j - 2), min(len(seg), j + 2)
    d = seg[a1 - 1] - seg[a0]
    prog = (seg[a0:a1] - seg[a0]) @ d / max(float(d @ d), 1e-9)
    n = next((n for n in range(1, len(prog)) if prog[n - 1] < 0.5 <= prog[n]), None)
    if n is None:
        return lo + j - 0.5 + 0.5
    return lo + a0 + n - 1 + (0.5 - prog[n - 1]) / max(prog[n] - prog[n - 1], 1e-9) + 0.5


def _surface_geometry(rgb, b):
    """Refined edges, per-corner radii and fill of a surface (fill, border or image) around bbox b, or None if its
    outline doesn't stand out from the surroundings near the bbox. Sides within 2 px of the image border are the border
    (full-bleed sheets, bars); corners are measured where the background is visible; a drop shadow is tolerated and
    reported as shadow=True."""
    H, W = rgb.shape[:2]
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    if w < 12 or h < 12:
        return None
    m = max(8, int(0.08 * min(w, h)))
    at_border = {"left": x <= 2, "top": y <= 2, "right": x + w >= W - 2, "bottom": y + h >= H - 2}
    band = lambda y0, y1, x0, x1: rgb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].reshape(-1, 3).astype(np.float64)
    hm = max(3, m // 2)
    outer = [band(y - m, y - hm, x, x + w), band(y + h + hm, y + h + m, x, x + w), band(y, y + h, x - m, x - hm),
             band(y, y + h, x + w + hm, x + w + m)]
    close = [band(y - hm, y - 2, x, x + w), band(y + h + 2, y + h + hm, x, x + w), band(y, y + h, x - hm, x - 2),
             band(y, y + h, x + w + 2, x + w + hm)]
    ring = np.concatenate(outer)
    inner = rgb[y + h // 3:y + h - h // 3, x + w // 3:x + w - w // 3]
    if inner.size == 0 or ring.size == 0:
        return None
    c_out = _dominant(ring)
    fill = _dominant(inner.astype(np.float64))
    c_lab = _lab_arr(c_out)
    contrast = float(np.linalg.norm(_lab_arr(fill) - c_lab))
    ring_dev = np.linalg.norm(_lab_arr(ring) - c_lab, axis=-1)
    tau = max(3.0, min(2 * float(np.percentile(ring_dev, 90)), 0.6 * contrast))
    lum = lambda c: float(np.asarray(c, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722]))
    shadow = False
    for o, c in zip(outer, close):
        if len(o) and len(c):
            od, cd = np.linalg.norm(_lab_arr(o) - c_lab, axis=-1), np.linalg.norm(_lab_arr(c) - c_lab, axis=-1)
            if np.percentile(cd, 50) > max(4.0, 1.5 * np.percentile(od, 50)) and np.percentile(od, 50) > 1.5 \
                    and lum(np.median(c, 0)) < lum(c_out) - 3:
                shadow = True
    oy, ox = max(0, y - m), max(0, x - m)
    lab = _lab_arr(rgb[oy:min(H, y + h + m), ox:min(W, x + w + m)].astype(np.float64))
    raw = np.linalg.norm(lab - c_lab, axis=-1)
    dev = _box_mean(raw, 3)
    edges = {}
    for side in ("top", "bottom", "left", "right"):
        if at_border[side]:
            edges[side] = {"top": 0.0, "left": 0.0, "bottom": float(H), "right": float(W)}[side]
            continue
        vals = []
        for f in np.linspace(0.4, 0.6, 9):
            if side in ("top", "bottom"):
                col = int(x + f * w) - ox
                prof, lprof = (dev[:, col], lab[:, col]) if side == "top" else (dev[::-1, col], lab[::-1, col])
            else:
                row = int(y + f * h) - oy
                prof, lprof = (dev[row, :], lab[row, :]) if side == "left" else (dev[row, ::-1], lab[row, ::-1])
            c = _rise(prof[:2 * m + 2], tau, lprof[:2 * m + 2], m + 4, shadow)
            if c is not None:
                vals.append(c)
        if len(vals) < 5:
            return None
        v = float(np.median(vals))
        edges[side] = {"top": oy + v, "left": ox + v, "bottom": oy + dev.shape[0] - v, "right": ox + dev.shape[1] - v}[side]
    tol_x, tol_y = max(6, 0.05 * w), max(6, 0.05 * h)
    if ((not at_border["left"] and abs(edges["left"] - x) > tol_x) or (not at_border["right"] and abs(edges["right"] - x - w) > tol_x)
            or (not at_border["top"] and abs(edges["top"] - y) > tol_y)
            or (not at_border["bottom"] and abs(edges["bottom"] - y - h) > tol_y)):
        return None
    ew, eh = edges["right"] - edges["left"], edges["bottom"] - edges["top"]
    radii = [None, None, None, None]  # top-left, top-right, bottom-left, bottom-right
    for ci, (flip_x, flip_y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        dd, ll = (dev[:, ::-1], lab[:, ::-1]) if flip_x else (dev, lab)
        dd, ll = (dd[::-1], ll[::-1]) if flip_y else (dd, ll)
        top = (oy + dev.shape[0] - edges["bottom"]) if flip_y else edges["top"] - oy
        left = (ox + dev.shape[1] - edges["right"]) if flip_x else edges["left"] - ox
        pts = []
        lim = int(left + min(ew, eh) / 2)
        for r_ in range(int(math.ceil(top)), min(dd.shape[0], int(top + min(ew, eh) / 2))):
            c = _rise(dd[r_, :lim], tau, ll[r_, :lim], m + 4, shadow)
            if c is not None:
                pts.append((r_ + 0.5 - top, c - left))
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        best = None
        for r in np.arange(0, min(ew, eh) / 2 + 0.25, 0.25):
            yy = np.clip(pts[:, 0], 0, None)
            pred = np.where(yy < r, r - np.sqrt(np.maximum(0, r * r - (r - yy) ** 2)), 0)
            err = float(np.mean((pred - pts[:, 1]) ** 2))
            if best is None or err < best[0]:
                best = (err, r)
        radii[ci] = float(best[1])
    got = [v for v in radii if v is not None]
    if not got:
        return None
    r = float(np.median(got))
    round_ = r > 0.45 * min(ew, eh)
    r = min(ew, eh) / 2 if round_ else r
    return {"box": [round(edges["left"], 1), round(edges["top"], 1), round(ew, 1), round(eh, 1)], "radius": round(r, 1),
            "corner_radii": [None if v is None else round(v, 1) for v in radii], "fill": _to_hex(fill), "outside": _to_hex(c_out),
            "shape": "circle" if round_ and abs(ew - eh) < 0.03 * max(ew, eh) else ("pill" if round_ else "rect"),
            "full_bleed": [s for s, v in at_border.items() if v], "shadow": shadow}


def cmd_redline(a):
    d, spec, mock = _d2c_load_spec(a.dir)
    rgb = np.asarray(mock.convert("RGB"))
    k = spec.get("px_to_dp", 1.0)
    m = _d2c_manifest(d, mock.size)
    texts = _text_targets(d, spec, mock, a.vision) if spec.get("typography") else {}
    out = Path(a.out) if a.out else d / "redline"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for r in _regions(spec, m, mock.width, mock.height):
        b = r["bbox"]
        text_comp = sum(inst["box"][2] * inst["box"][3] for t in texts.values() for inst in t.get("instances", [])
                        if inst.get("component") == r["name"]) > 0.5 * b["w"] * b["h"]
        group = any(cb["w"] * cb["h"] >= 0.5 * b["w"] * b["h"] for cb in r["children"])
        geo = _surface_geometry(rgb, b) if r.get("asset_kind") not in ("icon", "illustration") and not text_comp and not group else None
        item = {"name": r["name"], "asset": r.get("asset"), "spec_px": [b["x"], b["y"], b["w"], b["h"]]}
        if geo:
            item.update(kind="surface", px=geo["box"], radius_px=geo["radius"], corner_radii_px=geo["corner_radii"], fill=geo["fill"],
                        shape=geo["shape"], shadow=geo["shadow"], full_bleed=geo["full_bleed"])
        else:
            pad = 4
            x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
            ob = _object_box(rgb[y0:b["y"] + b["h"] + pad, x0:b["x"] + b["w"] + pad].astype(np.float64))
            if not ob:
                item.update(kind="unmeasured", px=[float(v) for v in item["spec_px"]],
                            reason=("text component" if text_comp else "container of other components" if group else
                                    "no edge or object stands out from its surroundings near the bbox"))
                items.append(item)
                continue
            item.update(kind="object", px=[round(x0 + ob[0], 1), round(y0 + ob[1], 1), round(ob[2], 1), round(ob[3], 1)])
        item["dp"] = [round(v * k, 1) for v in item["px"]]
        if "radius_px" in item:
            item["radius_dp"] = round(item["radius_px"] * k, 1)
        items.append(item)
    fpath = d / "fonts" / "fonts.json"
    fstyles = {}
    if fpath.exists():
        try:
            fstyles = {x["role"]: x for x in json.loads(fpath.read_text()).get("styles", []) if x.get("ascent_em") and x.get("size_px")}
        except (OSError, ValueError, TypeError, KeyError):
            fstyles = {}
    for role, t in texts.items():
        for inst in t.get("instances", []):
            bx = inst["box"]
            items.append({"name": f"text: {inst['sample'][:30]}", "role": role, "kind": "text", "px": bx,
                          "dp": [round(v * k, 1) for v in bx], "baseline_px": inst["baseline"], "baseline_dp": round(inst["baseline"] * k, 1),
                          "ink": inst["ink"], "note": "px/dp are the ink box; pad from baseline_inset_dp or layout_top_inset_dp"})
    measured = [it for it in items if it["kind"] != "unmeasured"]
    for it in measured:
        x, y, w, h = it["px"]
        par = [o for o in items if o is not it and o["kind"] == "surface" and o["px"][0] <= x + 0.5 and o["px"][1] <= y + 0.5
               and o["px"][0] + o["px"][2] >= x + w - 0.5 and o["px"][1] + o["px"][3] >= y + h - 0.5 and o["px"][2] * o["px"][3] > w * h]
        if par:
            p = min(par, key=lambda o: o["px"][2] * o["px"][3])
            it["parent"] = p["name"]
            it["inset_dp"] = {"left": round((x - p["px"][0]) * k, 1), "top": round((y - p["px"][1]) * k, 1),
                              "right": round((p["px"][0] + p["px"][2] - x - w) * k, 1),
                              "bottom": round((p["px"][1] + p["px"][3] - y - h) * k, 1)}
            if it["kind"] == "text":  # ink insets are not padding: give the baseline (paddingFromBaseline) and the layout-box top
                it["baseline_inset_dp"] = round((it["baseline_px"] - p["px"][1]) * k, 1)
                fs = fstyles.get(it.get("role"))
                if fs:
                    it["layout_top_inset_dp"] = round((it["baseline_px"] - fs["ascent_em"] * fs["size_px"] - p["px"][1]) * k, 1)
    col = sorted(measured, key=lambda o: o["px"][1])
    for it in col:
        x, y, w, h = it["px"]
        below = [o for o in col if o["px"][1] >= y + h - 0.5 and min(x + w, o["px"][0] + o["px"][2]) - max(x, o["px"][0]) > 0.3 * min(w, o["px"][2])
                 and o.get("parent") == it.get("parent")]
        if below:
            nb = min(below, key=lambda o: o["px"][1])
            it["gap_below_dp"] = [nb["name"], round((nb["px"][1] - y - h) * k, 1)]
    notes = {c["name"]: c.get("notes", "") for c in spec.get("components", [])}
    native = {c["name"]: c.get("native") for c in spec.get("components", [])}
    surfaces = {i["name"]: i for i in items if i["kind"] == "surface"}
    docs = [{"name": n, "notes": notes.get(n, "")} for n in surfaces]
    pairs = []  # (score, radius index, surface name): each surface is assigned to at most one radius element, best first
    for ri, rr in enumerate(spec.get("radii", [])):
        if not rr.get("element") or not isinstance(rr.get("px"), (int, float)):
            continue
        for sc, reg in _link_regions(rr["element"], docs, scored=True):
            if 0.5 <= (surfaces[reg["name"]]["radius_px"] + 1) / (rr["px"] + 1) <= 2:
                pairs.append((sc, ri, reg["name"]))
    assigned, used = {}, set()
    for sc, ri, name in sorted(pairs, key=lambda t: -t[0]):
        if ri not in assigned and name not in used:
            assigned[ri] = (name, sc)
            used.add(name)
    radii_cmp = []
    for ri, rr in enumerate(spec.get("radii", [])):
        name, sc = assigned.get(ri, (None, None))
        hit = surfaces.get(name)
        radii_cmp.append({"element": rr.get("element"), "spec_px": rr.get("px"), "measured_px": hit["radius_px"] if hit else None,
                          "corner_radii_px": hit["corner_radii_px"] if hit else None, "component": name,
                          "shape": hit["shape"] if hit else None, "native": native.get(name) if hit else None,
                          "match_score": None if sc is None else ("exact" if sc >= 10 else round(sc, 2))})
    ov = mock.convert("RGB")
    dr = ImageDraw.Draw(ov)
    for it in measured:
        x, y, w, h = it["px"]
        c = {"surface": "#00e5ff", "object": "#ffd60a", "text": "#ff2d55"}[it["kind"]]
        dr.rectangle([x, y, x + w, y + h], outline=c, width=1)
        lab = f"{it['dp'][2]:g}x{it['dp'][3]:g}" + (f" r{it['radius_dp']:g}" if "radius_dp" in it else "")
        if it["kind"] != "text":
            dr.text((x + 2, y + 2), lab, fill=c)
        if "gap_below_dp" in it and it["gap_below_dp"][1] > 0:
            dr.line([x + w / 2, y + h, x + w / 2, y + h + it["gap_below_dp"][1] / k], fill="#7CFF6B", width=1)
            dr.text((x + w / 2 + 3, y + h + 1), f"{it['gap_below_dp'][1]:g}", fill="#7CFF6B")
    ov.save(out / "redline.png")
    (out / "redline.json").write_text(json.dumps({"px_to_dp": k, "items": items, "radii": radii_cmp}, indent=2))
    n_s = sum(i["kind"] == "surface" for i in items)
    unm = [i for i in items if i["kind"] == "unmeasured"]
    print(f"{len(items)} items ({n_s} surfaces with measured edges + corner radius, "
          f"{sum(i['kind'] == 'object' for i in items)} objects, {sum(i['kind'] == 'text' for i in items)} text lines"
          + (f", {len(unm)} unmeasured: " + ", ".join(f"{i['name']} ({i['reason']})" for i in unm) if unm else "") + ")")
    for i in items:
        if i["kind"] == "surface" and (i.get("shadow") or i.get("full_bleed")):
            print(f"  {i['name']}: " + ", ".join(filter(None, ["drop shadow detected" if i.get("shadow") else "",
                  ("full-bleed at " + "/".join(i["full_bleed"]) + f"; corner radii {i['corner_radii_px']}") if i.get("full_bleed") else ""])))
    for rc in radii_cmp:
        got = (f"measured {rc['measured_px']:>5}px ({rc['shape']}, {rc['component']}, match {rc['match_score']})" if rc["component"]
               else "no matching surface measured")
        print(f"  radius {str(rc['element'])[:30]:30s} spec {rc['spec_px']!s:>5}px  {got}")
    print(f"-> {out}/redline.json, redline.png (cyan surfaces, yellow objects, red text; green = gap below in dp)")


# ---- compare: implemented screen vs approved mockup

def _gray(rgb):
    return np.asarray(rgb, dtype=np.float64)[..., :3] @ np.array([0.299, 0.587, 0.114])


def _box_mean(a, k):
    """Mean over a k x k window (same size, edge-clamped) via an integral image."""
    p = k // 2
    ap_ = np.pad(a, ((p + 1, p), (p + 1, p)), mode="edge")
    c = ap_.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _ssim_map(a, b, k=7):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _box_mean(a, k), _box_mean(b, k)
    va, vb = _box_mean(a * a, k) - ma ** 2, _box_mean(b * b, k) - mb ** 2
    cov = _box_mean(a * b, k) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def _ncc_map(S, T):
    """Normalised cross-correlation of template T at every valid position of S (FFT + integral images)."""
    h, w = T.shape
    H, W = S.shape
    if h > H or w > W:
        return None
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64) - T.mean()
    tn = np.sqrt((t * t).sum())
    if tn < 1e-6:
        return None
    num = np.fft.irfft2(np.fft.rfft2(S) * np.conj(np.fft.rfft2(t, s=S.shape)), s=S.shape)[:H - h + 1, :W - w + 1]
    c1 = np.pad(S, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    c2 = np.pad(S * S, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    win = lambda c: c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    var = win(c2) - win(c1) ** 2 / (h * w)
    ok = var > h * w
    return np.where(ok, np.clip(num / (tn * np.sqrt(np.where(ok, var, 1.0))), -1, 1), 0.0)


def _find_region(Mg, Sg, b, dy0, R, scales=(0.9, 0.95, 1.0, 1.05, 1.1), ctx=10):
    """Best (score, x, y, scale, score_at_expected) of mockup box b in the screenshot near the global offset.

    Shifts and scales are only accepted when they beat the expected placement by a clear margin."""
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    H, W = Mg.shape
    px0, py0 = max(0, x - ctx), max(0, y - ctx)
    px1, py1 = min(W, x + w + ctx), min(H, y + h + ctx)
    T = Mg[py0:py1, px0:px1]
    if T.size == 0 or T.std() < 2:
        return None
    tw0, th0 = px1 - px0, py1 - py0
    f = 2 if tw0 * th0 > 20000 else 1
    cands, expected = [], None
    for s in scales:
        tw, th = max(4, round(tw0 * s / f)), max(4, round(th0 * s / f))
        Ts = np.asarray(Image.fromarray(T).resize((tw, th), Image.BILINEAR if f == 1 else Image.BOX))
        ex, ey = px0 + (x + w / 2 - px0) * (1 - s), py0 + dy0 + (y + h / 2 - py0) * (1 - s)
        X0, Y0 = int(max(0, ex - R)), int(max(0, ey - R))
        X1, Y1 = int(min(Sg.shape[1], ex + tw0 * s + R)), int(min(Sg.shape[0], ey + th0 * s + R))
        win = Sg[Y0:Y1, X0:X1]
        if f > 1:
            win = np.asarray(Image.fromarray(win).resize((max(1, win.shape[1] // f), max(1, win.shape[0] // f)), Image.BOX))
        m = _ncc_map(win, Ts)
        if m is None or not m.size:
            continue
        iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
        cands.append((float(m[iy, ix]), X0 + ix * f, Y0 + iy * f, s))
        if s == 1.0:
            qy, qx = int(round((ey - Y0) / f)), int(round((ex - X0) / f))
            if 0 <= qy < m.shape[0] and 0 <= qx < m.shape[1]:
                expected = float(m[qy, qx])
    if not cands:
        return None
    best = max(cands, key=lambda c: c[0] - 0.03 * (c[3] != 1.0))
    if expected is not None and expected >= best[0] - 0.05:
        best = (expected, px0, py0 + dy0, 1.0)
    sc, bx, by, s = best
    if f > 1:
        tw, th = max(4, round(tw0 * s)), max(4, round(th0 * s))
        Ts = np.asarray(Image.fromarray(T).resize((tw, th), Image.BILINEAR))
        X0, Y0 = max(0, bx - 3), max(0, by - 3)
        m = _ncc_map(Sg[Y0:min(Sg.shape[0], by + th + 3), X0:min(Sg.shape[1], bx + tw + 3)], Ts)
        if m is not None and m.size:
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            sc, bx, by = float(m[iy, ix]), X0 + ix, Y0 + iy
    return sc, bx + round((x - px0) * s), by + round((y - py0) * s), s, expected


def _global_offset(Mg, Sg):
    """Vertical offset of the screenshot content vs the mockup (row-profile matching) and its cost."""
    k = 64
    ms = np.asarray(Image.fromarray(Mg).resize((k, Mg.shape[0]), Image.BOX))
    ss = np.asarray(Image.fromarray(Sg).resize((k, Sg.shape[0]), Image.BOX))
    H, Hs = len(ms), len(ss)
    best = (1e9, 0)
    for dy in range(-int(0.15 * H), Hs - int(0.6 * H) + 1):
        m0, m1 = max(0, -dy), min(H, Hs - dy)
        if m1 - m0 < 0.6 * H:
            continue
        cost = float(np.abs(ms[m0:m1] - ss[m0 + dy:m1 + dy]).mean())
        if cost < best[0]:
            best = (cost, dy)
    return best[1], best[0]


def _object_box(rgb):
    """Bounding box (x, y, w, h) of what stands out from the background of a region (the dominant colour of the crop's
    2-4 px border ring, so an object filling most of its bbox is not taken for the background), or None."""
    H, W = rgb.shape[:2]
    t = max(2, min(4, min(H, W) // 8))
    ring = np.concatenate([rgb[:t].reshape(-1, 3), rgb[-t:].reshape(-1, 3), rgb[t:-t, :t].reshape(-1, 3), rgb[t:-t, -t:].reshape(-1, 3)])
    d = np.linalg.norm(_lab_arr(rgb) - _lab_arr(_dominant(ring if len(ring) else rgb)), axis=-1)
    m = d > max(12.0, 0.25 * np.percentile(d, 99.5))
    if m.sum() < 12:
        return None
    rows, cols = m.sum(1), m.sum(0)
    ys = np.flatnonzero(rows >= max(2, 0.02 * rows.max()))  # rows/columns with more than isolated specks
    xs = np.flatnonzero(cols >= max(2, 0.02 * cols.max()))
    if not len(ys) or not len(xs):
        return None
    return float(xs[0]), float(ys[0]), float(xs[-1] - xs[0] + 1), float(ys[-1] - ys[0] + 1)


def _tolerant_de(ref, cur):
    """Per-pixel CIEDE2000 after a light blur, minimised over +-1px shifts (ignores anti-aliasing jitter)."""
    blur = lambda x: _lab_arr(np.asarray(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                                         dtype=np.float64))
    lm, lc = blur(ref), blur(cur)
    de = _de00(lm, lc)
    sel = de > 0  # identical pixels stay 0 under every shift: evaluate the shifts only where the images differ
    part = sel.mean() < 0.5
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if (oy, ox) == (0, 0):
                continue
            rolled = np.roll(lc, (oy, ox), (0, 1))
            if part:
                de[sel] = np.minimum(de[sel], _de00(lm[sel], rolled[sel]))
            else:
                de = np.minimum(de, _de00(lm, rolled))
    return de


def _impl_font_check(fonts, role, sample, alpha, size_px, platform=None):
    """Which font explains implemented text best: the one fonts.json chose, a neighbouring weight, another family, or
    (Android) Roboto, the default when fontFamily is not set? SF Pro (the iOS default) is not on Google Fonts.

    Returns (spec (family, weight, sim), best (family, weight, sim)); relative ranking, so rendering noise cancels out."""
    st = next((x for x in fonts.get("styles", []) if x["role"] == role), None)
    if not st:
        return None, None
    it = st["italic"]
    weights = lambda fam: sorted({int(k.rstrip("i")) for f in _gf_meta() if f["family"] == fam for k in f["fonts"] if k.endswith("i") == it})
    cands = [(st["family"], w) for w in weights(st["family"]) if abs(w - st["weight"]) <= 200] or [(st["family"], st["weight"])]
    for f in fonts.get("families", []):
        ws = weights(f["family"])
        if f["family"] != st["family"] and ws:
            cands.append((f["family"], min(ws, key=lambda w: abs(w - st["weight"]))))
    cands += [(fa, w) for fa, w, _ in st.get("alternatives", [])[:3] if (fa, w) not in cands]
    if platform == "android" and st["family"] != "Roboto":
        ws = weights("Roboto")
        if ws:
            cands.append(("Roboto", min(ws, key=lambda w: abs(w - st["weight"]))))

    def one(c):
        try:
            path = _gf_ttf(c[0], c[1], it)
            fit = _fit_font(path, sample, alpha, size_px) if path else None
        except (requests.RequestException, OSError, ValueError):
            return None
        return (c[0], c[1], fit["similarity"]) if fit else None
    with cf.ThreadPoolExecutor(4) as ex:
        res = [r for r in ex.map(one, cands) if r]
    spec = next((r for r in res if r[:2] == (st["family"], st["weight"])), None)
    return spec, (max(res, key=lambda r: r[2]) if res else None)


def _impl_box(box, r, dy0, k):
    """The design crop box moved by the global offset and the region's measured shift (same size, so tiles scale alike)."""
    sx, sy = (round(v / k) for v in r["shift_dp"])
    return box[0] + sx, box[1] + dy0 + sy, box[2] + sx, box[3] + dy0 + sy


def _heat_rgb(v):
    stops = np.array([[0, 0, 0], [80, 0, 110], [200, 30, 40], [255, 150, 0], [255, 255, 160]], dtype=np.float64)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    t = (x - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)


def _regions(spec, m, W, H):
    regs = [{"name": c["name"], "bbox": c["bbox"], "notes": c.get("notes", ""), "kind": "component"}
            for c in spec.get("components", []) if c["bbox"]["w"] * c["bbox"]["h"] < 0.8 * W * H]
    iou = lambda a, b: max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])) * \
        max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])) / max(1, a["w"] * a["h"] + b["w"] * b["h"] -
        max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])) * max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])))
    for x in (m or {}).get("assets", []):
        hit = next((r for r in regs if iou(r["bbox"], x["bbox"]) > 0.5), None)
        if hit:
            hit["asset"], hit["asset_kind"] = x["name"], x.get("kind")
        else:
            regs.append({"name": x["name"], "bbox": x["bbox"], "notes": x.get("description", ""), "kind": "asset", "asset": x["name"],
                         "asset_kind": x.get("kind")})
    for r in regs:
        b = r["bbox"]
        r["children"] = [o["bbox"] for o in regs if o is not r and o["bbox"]["x"] >= b["x"] and o["bbox"]["y"] >= b["y"]
                         and o["bbox"]["x"] + o["bbox"]["w"] <= b["x"] + b["w"] and o["bbox"]["y"] + o["bbox"]["h"] <= b["y"] + b["h"]
                         and o["bbox"]["w"] * o["bbox"]["h"] < 0.9 * b["w"] * b["h"]]
    return regs


COMPARE_PROMPT = ("Each numbered row shows LEFT: a region of the APPROVED design, RIGHT: the same region of the implemented "
                  "app screen. For every row list the concrete visual differences a developer must fix (size, position, "
                  "colour, font family/weight, missing or extra elements, image content). Be specific and brief; say "
                  "'matches' if there is no real difference. Ignore sub-pixel anti-aliasing noise.")
COMPARE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rows"], "properties": {"rows": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["index", "differences"],
                               "properties": {"index": {"type": "integer"},
                                              "differences": {"type": "array", "items": {"type": "string"}}}}}}}


def cmd_compare(a):
    if a.crop_top < 0 or a.crop_bottom < 0:
        sys.exit("compare: --crop-top/--crop-bottom must be >= 0")
    if a.fail_below is not None and not 0 <= a.fail_below <= 100:
        sys.exit("compare: --fail-below is a score between 0 and 100")
    d, spec, mock_rgba = _d2c_load_spec(a.dir, mockup=a.mockup, required=False)
    m = _d2c_manifest(d, mock_rgba.size)
    mock = mock_rgba.convert("RGB")
    W, H = mock.size
    k = spec.get("px_to_dp") or 1.0
    design_dp = spec.get("effective_width_dp") or (m or {}).get("screen_dp") or round(W * k)
    print(f"  (native scale: for a 1:1 check the screenshot should show the screen at ~{design_dp}dp wide; "
          "a narrower device reflows the layout)")
    shot = _open_image(a.screenshot, "screenshot").convert("RGB")
    if a.crop_top + a.crop_bottom >= shot.height:
        sys.exit(f"compare: --crop-top {a.crop_top} + --crop-bottom {a.crop_bottom} leaves nothing of the {shot.height}px screenshot")
    if a.crop_top or a.crop_bottom:
        shot = shot.crop((0, a.crop_top, shot.width, shot.height - a.crop_bottom))
    shot_dp = a.shot_width_dp or (shot.width / a.density if a.density else None)
    if shot_dp and abs(shot_dp - design_dp) / design_dp > 0.05:
        print(f"  ! the screenshot shows a ~{shot_dp:.0f}dp-wide screen, the design is {design_dp}dp: the layout reflows and every "
              f"size reads {shot_dp / design_dp:.2f}x; re-shoot at {design_dp}dp (Preview widthDp={design_dp})")
    shot = shot.resize((W, round(shot.height * W / shot.width)), Image.LANCZOS)
    out = Path(a.out) if a.out else d / "compare"
    out.mkdir(parents=True, exist_ok=True)
    M, S = np.asarray(mock, dtype=np.float64), np.asarray(shot, dtype=np.float64)
    Mg, Sg = _gray(M), _gray(S)
    dy0, cost = _global_offset(Mg.astype(np.float32), Sg.astype(np.float32))
    top_extra = max(0, dy0)
    bottom_extra = max(0, S.shape[0] - (H + dy0))
    A = np.zeros_like(M)
    y0, y1 = max(0, -dy0), min(H, S.shape[0] - dy0)
    A[y0:y1] = S[y0 + dy0:y1 + dy0]
    valid = np.zeros((H, W), bool)
    valid[y0:y1] = True
    de = _tolerant_de(M, A)
    de[~valid] = 0
    ssim = _ssim_map(Mg, _gray(A))
    overall = {"ssim": round(float(ssim[valid].mean()), 4), "mean_de00": round(float(de[valid].mean()), 2),
               "pct_pixels_de_gt_10": round(float((de[valid] > 10).mean() * 100), 1),
               "vertical_offset_px": int(dy0), "status_bar_px": int(top_extra), "extra_bottom_px": int(bottom_extra),
               "coverage": round(float(valid.mean()), 3), "compared_rows": [int(y0), int(y1)]}
    overall["score"] = round(100 * overall["ssim"] * max(0.0, 1 - overall["mean_de00"] / 50), 1)
    overall["ssim_method"] = "SSIM (7px box window, luma)"
    texts = _text_targets(d, spec, mock_rgba) if spec.get("typography") else {}
    role_of = {}
    for r_, t_ in texts.items():
        for inst in t_.get("instances", []):
            if inst.get("component"):
                role_of.setdefault(inst["component"], []).append((r_, inst))
    fpath = Path(a.fonts) if a.fonts else d / "fonts" / "fonts.json"
    if a.fonts:
        _need(fpath, "fonts.json (--fonts)", "run `fonts DIR` first")
    fonts_json = _json_lists(_read_json(fpath, "fonts.json", ("families", "styles")), ("families", "styles"), fpath,
                             f"fonts {d}") if fpath.exists() else None
    tk_path = d / "tokens" / "tokens.json"
    token_names = {}
    if tk_path.exists():
        try:
            tk = json.loads(tk_path.read_text())
            token_names = {t["role"]: f"{tk.get('prefix', 'App')}Typography.{t['kt']}" for t in tk.get("typography", []) if t.get("kt")}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    platform = spec.get("platform") or (m or {}).get("platform")
    styles = {st["role"]: st for st in spec.get("typography", [])}
    Sg32, Mg32 = Sg.astype(np.float32), Mg.astype(np.float32)
    R = max(24, int(0.04 * H))
    found = []
    font_check_off = False
    size_hints = []  # (region, hint) pairs, dropped when every text is off by the same factor (wrong screenshot scale)
    for r in _regions(spec, m, W, H):
        b = r["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        if w < 4 or h < 4:
            continue
        rep = {"name": r["name"], "asset": r.get("asset"), "bbox": b}
        inside = float(valid[y:y + h, x:x + w].mean()) if valid[y:y + h, x:x + w].size else 0.0
        if inside < 0.5:
            rep.update(status="outside screenshot", inside=round(inside, 2), hints=[], badness=0.0, match=None, shift_dp=[0.0, 0.0],
                       scale=1.0, de00_mean=None, ssim=None)
            found.append(rep)
            continue
        hit = _find_region(Mg32, Sg32, b, dy0, R)
        if hit and hit[0] < 0.5 and bottom_extra + top_extra > R:
            far = _find_region(Mg32, Sg32, b, dy0 + (bottom_extra + top_extra) // 2, (bottom_extra + top_extra) // 2 + R)
            hit = far if far and far[0] > hit[0] + 0.1 else hit
        if hit:
            sc, sx, sy, s, _ = hit
            sw, sh = max(1, round(w * s)), max(1, round(h * s))
            cur = S[sy:sy + sh, sx:sx + sw]
            cur = np.asarray(Image.fromarray(cur.astype(np.uint8)).resize((w, h), Image.LANCZOS), dtype=np.float64)
            rep.update(match=round(sc, 3), shift_dp=[round((sx + (sw - w) / 2 - x) * k, 1), round((sy + (sh - h) / 2 - y - dy0) * k, 1)],
                       scale=round(s, 3), impl_bbox=[int(sx), int(sy - dy0), int(sw), int(sh)])
        else:
            cur = A[y:y + h, x:x + w]
            rep.update(match=None, shift_dp=[0.0, 0.0], scale=1.0)
        own = np.ones((h, w), bool)
        for cb in r["children"]:
            own[max(0, cb["y"] - y):cb["y"] - y + cb["h"], max(0, cb["x"] - x):cb["x"] - x + cb["w"]] = False
        if own.mean() < 0.15:
            own[:] = True
        ref = M[y:y + h, x:x + w]
        dmap = _tolerant_de(ref, cur)
        rep["de00_mean"] = round(float(dmap[own].mean()), 2)
        rep["ssim"] = round(float(_ssim_map(_gray(ref), _gray(cur))[own].mean()), 3)
        fill_m, fill_i = _dominant(ref), _dominant(cur)
        share = lambda img, c: float((np.linalg.norm(_lab_arr(img) - _lab_arr(c), axis=-1) < 6).mean())
        rep["fill"] = [_to_hex(fill_m), _to_hex(fill_i), round(float(_de00(_lab_arr(fill_m), _lab_arr(fill_i))), 1),
                       round(min(share(ref, fill_m), share(cur, fill_i)), 2)]
        image = r.get("asset_kind") in ("background", "photo")
        hints = []
        text_boxes = []
        for role, t in role_of.get(r["name"], []):
            st = styles[role]
            ib = rep.get("impl_bbox") or [x, y, w, h]
            tx, ty, tw_, th_ = t["box"]
            px_, py_ = int(0.6 * st["size_px"]) + 6, int(0.35 * st["size_px"]) + 4
            found_ = []
            for ox, oy in {(0, dy0), (ib[0] - x, ib[1] + dy0 - y)}:
                g_ = _locate_text(S, [tx + ox - px_, ty + oy - py_, tw_ + 2 * px_, th_ + 2 * py_], t["sample"], st["size_px"],
                                  st.get("letter_spacing_em", 0), t["ink"])
                if g_ and _de_hex(g_["ink"], t["ink"]) < 20 and 0.6 < g_["box"][3] / th_ < 1.6:
                    found_.append((abs(math.log(g_["box"][3] / th_)) + abs(math.log(g_["box"][2] / tw_)), g_))
            got = min(found_, key=lambda f_: f_[0])[1] if found_ else None
            text_boxes.append((tx - x, ty - y, tw_, th_))
            if not got:
                hints.append(f"text '{t['sample'][:24]}' not found in the implementation")
                continue
            gx, gy, gw, gh = got["box"]
            text_boxes.append((gx - (ib[0] - x) - x, gy - dy0 - (ib[1] - y) - y, gw, gh))
            hr, wr = gh / th_, gw / tw_
            dei = round(float(_de00(_lab_arr(_hexi(t["ink"])), _lab_arr(_hexi(got["ink"])))), 1)
            shape = round(_shape_similarity(t["alpha"], got["alpha"], t["sample"])[0], 3)
            fspec, fbest = None, None
            if fonts_json and th_ >= 16 and not font_check_off:
                try:
                    fspec, fbest = _impl_font_check(fonts_json, role, t["sample"], got["alpha"], st["size_px"], platform)
                except (requests.RequestException, RuntimeError, ValueError, KeyError, OSError) as e:
                    font_check_off = True
                    print(f"  (font verdict skipped: Google Fonts metadata unavailable: {str(e)[:160]})")
            sh = [round((gx - tx) * k, 1), round((got["baseline"] - dy0 - t["baseline"]) * k, 1)]
            rep.setdefault("text", []).append({"role": role, "sample": t["sample"], "height_ratio": round(hr, 3),
                                               "width_ratio": round(wr, 3), "ink": [t["ink"], got["ink"]], "ink_de00": dei,
                                               "glyph_similarity_vs_design": shape, "spec_font_fit": fspec,
                                               "best_font_fit": fbest, "shift_dp": sh})
            lab = f"'{t['sample'][:20]}'"
            token = f", token {token_names[role]}" if role in token_names else ""
            if abs(hr - 1) >= 0.04 and abs(gh - th_) >= 2:
                design_sp = st["size_px"] * k
                hint = (f"{lab} {abs(hr - 1) * 100:.0f}% {'larger' if hr > 1 else 'smaller'}: renders like ~{design_sp * hr:.1f}sp "
                        f"(design {design_sp:g}sp{token}; ink height {th_} -> {gh}px)")
                if fspec and fbest and fbest[:2] == fspec[:2]:
                    hint += f"; if it already uses {design_sp:g}sp the font runs {abs(hr - 1) * 100:.0f}% {'large' if hr > 1 else 'small'}: use {design_sp / hr:.1f}sp"
                hints.append(hint)
                size_hints.append((rep, hint))
            elif abs(wr - 1) >= 0.05 and abs(gw - tw_) >= 3:
                hints.append(f"{lab} {abs(wr - 1) * 100:.0f}% {'wider' if wr > 1 else 'narrower'} at the same height "
                             "(letter-spacing or font family/weight differs)")
            if fspec and fbest and fbest[2] > fspec[2] + (0.08 if fbest[0] != fspec[0] else 0.05):
                what = f"{fbest[0]} {fbest[1]}" if fbest[0] != fspec[0] else f"weight {fbest[1]}"
                hints.append(f"{lab} looks like {what} (glyph fit {fbest[2]:.2f}) rather than the specified {fspec[0]} {fspec[1]} "
                             f"({fspec[2]:.2f})" + (" (fontFamily not set?)" if fbest[0] == "Roboto" and platform == "android" else ""))
            if dei >= 3:
                hints.append(f"{lab} colour off by dE {dei} ({t['ink']} -> {got['ink']})")
            if abs(sh[0]) >= 2 or abs(sh[1]) >= 2:
                hints.append(f"{lab} shifted {sh[0]:+.0f}dp x, {sh[1]:+.0f}dp y")
        if rep.get("text"):
            rep["shift_dp"] = rep["text"][0]["shift_dp"]
            rep["scale"] = 1.0
        blob_mask = (dmap > 12) & own
        for bx_, by_, bw_, bh_ in text_boxes:
            blob_mask[max(0, int(by_) - 3):int(by_ + bh_) + 3, max(0, int(bx_) - 3):int(bx_ + bw_) + 3] = False
        if blob_mask.any():
            bl = max(_blobs(blob_mask), key=lambda t: len(t[0])) if blob_mask.sum() < 60000 else None
            if bl is not None and len(bl[0]) >= max(30, 0.01 * own.sum()):
                ys_, xs_ = bl
                bb = (int(xs_.min()), int(ys_.min()), int(xs_.max() - xs_.min() + 1), int(ys_.max() - ys_.min() + 1))
                sub = lambda img: img[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
                inkish = lambda img: float((np.linalg.norm(_lab_arr(sub(img)) - _lab_arr(_dominant(img)), axis=-1) > 15).mean())
                dm, di = inkish(ref), inkish(cur)
                what = ("differs inside the artwork (redrawn image)" if image else
                        "missing in the implementation" if dm > 0.15 and di < 0.5 * dm else
                        "extra in the implementation" if di > 0.15 and dm < 0.5 * di else "drawn differently")
                rep["diff_blob"] = {"bbox": [x + bb[0], y + bb[1], bb[2], bb[3]], "px": int(len(ys_)), "what": what}
                hints.append(f"element at x={x + bb[0]},y={y + bb[1]} ({bb[2]}x{bb[3]}px) {what}")
        strong = (rep["match"] is not None and rep["match"] < 0.6) or \
            rep.get("diff_blob", {}).get("what", "").startswith(("missing", "extra"))
        if strong:  # a missing/replaced element: its "shift" and "scale" are matches against noise, keep the expected position
            rep["shift_dp"], rep["scale"] = [0.0, 0.0], 1.0
            rep.pop("impl_bbox", None)
        sx_, sy_ = rep["shift_dp"]
        if not strong and (abs(sx_) >= 2 or abs(sy_) >= 2):
            hints.append(f"shifted {'%+.0fdp horizontally' % sx_ if abs(sx_) >= 2 else ''}"
                         f"{' and ' if abs(sx_) >= 2 and abs(sy_) >= 2 else ''}{'%+.0fdp vertically' % sy_ if abs(sy_) >= 2 else ''}")
        if not strong and "text" not in rep and not r["children"] and r.get("asset_kind") not in ("background", "photo"):
            ob_m, ob_i = _object_box(ref), _object_box(cur)
            if ob_m and ob_i:
                ratio = rep["scale"] * math.sqrt((ob_i[2] * ob_i[3]) / (ob_m[2] * ob_m[3]))
                rep["object"] = {"design_px": [round(v) for v in ob_m[2:]], "impl_px": [round(v * rep["scale"]) for v in ob_i[2:]],
                                 "size_ratio": round(ratio, 3)}
                rep["scale"] = round(ratio, 3)
            if abs(rep["scale"] - 1) >= 0.06:
                o = rep.get("object")
                hints.append(f"element ~{abs(rep['scale'] - 1) * 100:.0f}% {'larger' if rep['scale'] > 1 else 'smaller'}"
                             + (f" (visible {o['design_px'][0]}x{o['design_px'][1]} -> {o['impl_px'][0]}x{o['impl_px'][1]}px)" if o else ""))
        if 3 <= rep["fill"][2] < 25 and rep["fill"][3] >= 0.3:
            hints.append(f"fill colour off by dE {rep['fill'][2]} ({rep['fill'][0]} -> {rep['fill'][1]})")
        if rep["match"] is not None and rep["match"] < 0.6 and not rep.get("diff_blob", {}).get("what", "").startswith(("missing", "extra")):
            hints.append(f"content differs strongly (match {rep['match']}): missing, replaced or restyled element")
        elif rep["ssim"] < 0.75 and not hints:
            hints.append(f"structure differs (SSIM {rep['ssim']}): check shapes, icon/image content, font")
        tx_ = rep.get("text", [])
        badness = ((0.4 if image else 1) * (min(1, (1 - rep["ssim"]) / 0.5) + min(1, rep["de00_mean"] / 12))
                   + min(1, math.hypot(sx_, sy_) / 8) + min(1, abs(math.log(rep["scale"])) / 0.1)
                   + (min(1.5, rep["fill"][2] / 5) if 3 <= rep["fill"][2] < 25 and rep["fill"][3] >= 0.3 else 0)
                   + sum(min(1.5, abs(math.log(t_["height_ratio"])) / 0.06) + min(1, t_["ink_de00"] / 10)
                         + (min(1.5, max(0, t_["best_font_fit"][2] - t_["spec_font_fit"][2] - 0.05) / 0.05)
                            * (1 if t_["best_font_fit"][0] != t_["spec_font_fit"][0] else 0.4)
                            if t_["spec_font_fit"] and t_["best_font_fit"] else 0) for t_ in tx_)
                   + ((0.3 if image else 0.5 if r.get("asset") else 1) * min(1, rep["diff_blob"]["px"] / 300)
                      + (1.0 if rep["diff_blob"]["what"].startswith(("missing", "extra")) else 0) if "diff_blob" in rep else 0)
                   + (1.0 if strong else 0))
        rep["badness"] = round(badness, 3)
        rep["hints"] = hints
        found.append(rep)
    ratios = [t_["height_ratio"] for r in found for t_ in r.get("text", [])]
    global_scale = None
    if len(ratios) >= 3:
        med = float(np.median(ratios))
        if abs(med - 1) > 0.05 and sum(abs(v - med) <= 0.05 for v in ratios) >= max(3, 0.75 * len(ratios)):
            global_scale = med
            for rep, hint in size_hints:
                rep["hints"] = [h_ for h_ in rep["hints"] if h_ != hint]
            overall["text_scale"] = round(med, 3)
    found.sort(key=lambda r: -r["badness"])
    covered = np.zeros((H, W), bool)
    for r in found:
        b = r["bbox"]
        covered[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = True
    g = 32
    unl = []
    for yy in range(0, H - g + 1, g):
        for xx in range(0, W - g + 1, g):
            cell = de[yy:yy + g, xx:xx + g]
            if valid[yy:yy + g, xx:xx + g].all() and not covered[yy:yy + g, xx:xx + g].any() and cell.mean() > 8:
                unl.append({"bbox": [xx, yy, g, g], "de00_mean": round(float(cell.mean()), 1)})
    ranked = [r for r in found if r.get("status") != "outside screenshot"]
    if a.vision and ranked:
        top = ranked[:min(6, len(ranked))]
        rows = []
        for i, r in enumerate(top):
            b = r["bbox"]
            pad = 12
            box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(W, b["x"] + b["w"] + pad), min(H, b["y"] + b["h"] + pad))
            rows.append([(f"{i + 1}. design", mock.crop(box)), (f"{i + 1}. implemented", shot.crop(_impl_box(box, r, dy0, k)))])
        vis = out / "vision_rows.png"
        sheet(rows, vis, T=300)
        try:
            res = ask([COMPARE_PROMPT, Image.open(vis)], COMPARE_SCHEMA, "compare_rows")
            for row in res["rows"]:
                if isinstance(row, dict) and isinstance(row.get("index"), int) and 1 <= row["index"] <= len(top) \
                        and isinstance(row.get("differences"), list):
                    top[row["index"] - 1]["vision"] = [str(v) for v in row["differences"]]
        except Exception as e:
            print(f"  ! vision pass failed: {e}")
    heat = (_heat_rgb(de / 20) * 0.75 + M * 0.25).astype(np.uint8)
    hm = Image.fromarray(heat)
    dr = ImageDraw.Draw(hm)
    for i, r in enumerate(ranked[:a.top]):
        b = r["bbox"]
        dr.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline="#00e5ff", width=3)
        dr.text((b["x"] + 3, b["y"] + 2), str(i + 1), fill="#00e5ff")
    for u in unl:
        x_, y_, w_, h_ = u["bbox"]
        dr.rectangle([x_, y_, x_ + w_, y_ + h_], outline="#ffffff", width=1)
    panel = Image.new("RGB", (W * 3 + 40, H), "white")
    panel.paste(mock, (0, 0))
    panel.paste(Image.fromarray(A.astype(np.uint8)), (W + 20, 0))
    panel.paste(hm, (2 * W + 40, 0))
    panel.save(out / "heatmap.png")
    hm.save(out / "heatmap_only.png")
    rows = []
    for i, r in enumerate(ranked[:a.top]):
        b = r["bbox"]
        pad = 10
        box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(W, b["x"] + b["w"] + pad), min(H, b["y"] + b["h"] + pad))
        rows.append([(f"#{i + 1} {r['name'][:30]} design", mock.crop(box)), ("implemented", shot.crop(_impl_box(box, r, dy0, k))),
                     ("dE heat", hm.crop(box))])
    if rows:
        sheet(rows, out / "regions.png", T=260)
    report = {"overall": overall, "regions": found, "unlabelled": unl, "mockup": str(a.mockup or d / (m or {}).get("mockup", "mockup.png")),
              "screenshot": str(a.screenshot), "px_to_dp": k, "design_width_dp": design_dp, "screenshot_width_dp": shot_dp,
              "fonts": str(fpath) if fonts_json else None, "fail_below": a.fail_below}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"overall score {overall['score']}/100  (SSIM (7px box) {overall['ssim']}, mean dE00 {overall['mean_de00']}, "
          f"{overall['pct_pixels_de_gt_10']}% px dE>10)")
    if top_extra or bottom_extra:
        print(f"  aligned with a {top_extra}px ({top_extra * k:.0f}dp) top offset (status bar / extra top inset); "
              f"{bottom_extra}px extra at the bottom (taller screen or nav bar) ignored")
    if overall["coverage"] < 0.98:
        print(f"  ! compared {overall['coverage'] * 100:.0f}% of the design (screenshot covers rows {y0}-{y1} of {H}); "
              f"scroll and capture the rest ({sum(r.get('status') == 'outside screenshot' for r in found)} regions outside the screenshot)")
    if global_scale:
        est = design_dp * global_scale
        print(f"  ! all text is ~{abs(global_scale - 1) * 100:.0f}% {'larger' if global_scale > 1 else 'smaller'}: the screenshot shows a "
              f"~{est:.0f}dp-wide screen (or a {global_scale:.2f}x font scale), the design is {design_dp}dp; re-shoot at {design_dp}dp "
              f"(Preview widthDp={design_dp}, fontScale=1) -- per-text size hints suppressed")
    if platform == "ios" and fonts_json:
        print("  (SF Pro, the iOS default, is not on Google Fonts: an unset font can't be named; check the 'looks like' verdicts)")
    print(f"worst regions (of {len(ranked)}" + (f"; {len(found) - len(ranked)} outside the screenshot" if len(found) > len(ranked) else "") + "):")
    for i, r in enumerate(ranked[:a.top]):
        hint = "; ".join(r["hints"]) or f"minor (dE {r['de00_mean']}, SSIM {r['ssim']})"
        print(f"  {i + 1:2d}. {r['name'][:34]:34s} bad {r['badness']:.2f}  {hint}")
        for v in r.get("vision", []):
            print(f"        vision: {v}")
    if unl:
        print(f"  + {len(unl)} unlabelled 32px cells with dE>8 (white boxes on the heatmap)")
    print(f"-> {out}/report.json, heatmap.png (design | implementation aligned | dE heatmap), regions.png")
    if a.fail_below is not None:
        ok = overall["score"] >= a.fail_below
        print(f"gate: score {overall['score']} {'>=' if ok else '<'} --fail-below {a.fail_below:g}: {'pass' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


# ---------------------------------------------------------------- hybrid screens: code builds the UI, GPT paints the materials
# GPT image models draw a plausible screen, not a designed one: chunky type, no hairlines, loud fills. The hybrid route
# keeps layout, type, states and motion in code (HTML/CSS/JS, measured and crisp) and asks GPT only for what code cannot
# fake: materials and real objects (foil, paper, glass, food, fabric). `material` paints them, `state` derives a state
# variant that stays pixel-identical outside the change, `stage` renders the page to stills and a scrubbed motion clip.

HYBRID_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=412, initial-scale=1">
<title>Screen</title>
<style>
/* Animate a mask radius smoothly: gradients do not interpolate, a registered length does. */
@property --hole { syntax: '<length>'; inherits: true; initial-value: 0px; }
:root { --paper: #F1EFEA; --ink: #151716; --muted: #55595A; --hair: #D9D5CC; --accent: #FF7A1A; }
* { box-sizing: border-box; margin: 0; padding: 0 }
body { width: 412px; height: 915px; overflow: hidden; position: relative; background: var(--paper); color: var(--ink);
  font-family: system-ui, sans-serif; -webkit-font-smoothing: antialiased; font-feature-settings: "tnum" 1 }
.hero { position: absolute; left: 24px; top: 72px; font-size: 34px; line-height: 40px; font-weight: 500; letter-spacing: -.035em }
.object { position: absolute; left: 24px; top: 160px; width: 160px; filter: drop-shadow(0 14px 22px rgba(60,52,40,.13)) }
.object img { display: block; width: 100% }
</style>
</head>
<body>
<div class="hero" data-in>One idea, executed quietly.</div>
<div class="object" data-in><img id="material" src="assets/material.png" alt=""></div>
<script>
// Page contract for `gptkit stage`:
//   body[data-ready="1"] once fonts, images and layout are done.
//   ?capture: lay out only; window.demo() builds the whole timeline as Web Animations and pauses it;
//   window.seek(ms) sets every animation's currentTime, so the recorder scrubs exact frames.
// Every motion is a Web Animation. Fill "forwards" by default; only the entrance fills "both", or a later
// animation's first keyframe leaks backwards into the idle frames.
const A = (el, kf, o) => el.animate(kf, Object.assign({fill: "forwards", easing: "cubic-bezier(.2,.8,.2,1)"}, o));
const params = new URLSearchParams(location.search);
function entrance(t0 = 0) {
  document.querySelectorAll("[data-in]").forEach((el, i) =>
    A(el, [{opacity: 0, transform: "translateY(10px)"}, {opacity: 1, transform: "none"}], {duration: 700, delay: t0 + 80 + i * 45, fill: "both"}));
}
window.demo = () => { entrance(0); document.getAnimations().forEach(a => a.pause()); };
window.seek = ms => document.getAnimations().forEach(a => { a.currentTime = ms; });
(async () => {
  await Promise.all([...document.images].map(i => i.decode().catch(() => {})));
  await document.fonts.ready;
  // Optional measured asset geometry. geo.js may assign window.GEO = {...}; missing file is harmless.
  try { const r = await fetch('geo.js'); if (r.ok) (0, eval)(await r.text()); } catch (_) {}
  window.GEO = window.GEO || {};
  if (!params.has("capture") && !params.has("still")) entrance(0);
  document.body.dataset.ready = "1";
})();
</script>
</body>
</html>
"""


def _merge_json(path, key, value):
    """Merge one entry into a JSON object file (re-read + temp + os.replace, so parallel runs keep each other's entries)."""
    p = Path(path)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except ValueError:
            data = {}
        data = data if isinstance(data, dict) else {}
    data[key] = value
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def _hybrid_flags(img, transparent=True):
    """Flags for a free-standing asset (no mockup to compare with): EMPTY, OPAQUE, CUT-OFF; plus its content box."""
    cb = content_box(img, 40)
    empty = cb is None or (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.002 * img.width * img.height
    flags = ["EMPTY"] if empty else []
    if transparent:
        share, corners = _opaque_share(img)
        if share > 0.95 or corners:
            flags.append("OPAQUE")
        elif not empty and (cb[0] <= 2 or cb[1] <= 2 or cb[2] >= img.width - 2 or cb[3] >= img.height - 2):
            flags.append("CUT-OFF")
    return flags, cb


def _image_bytes_checked(path, what):
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"{what}: no such file {p}")
    raw = p.read_bytes()
    return _decode_result(raw, f"{what} {p}"), raw


def cmd_material(a):
    """New material/object art for a code-built screen: N candidates in parallel, flagged and ranked; the best
    unflagged one is written to OUT/NAME.png (trimmed to its content with a small margin)."""
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    bg = "transparent" if a.transparent else "auto"
    prompt = a.prompt
    if a.transparent:
        prompt += (" One isolated object on a fully transparent background, whole and centred with clear margin on every side;"
                   " no text, no labels, no backdrop, no cast shadow outside the object.")
    ref = None
    if a.ref:
        _, ref = _image_bytes_checked(a.ref, "--ref")
        prompt = ("Use the reference only for its material, lighting and rendering quality, not its layout or content. " + prompt)
    _plan(a.candidates, 0, image_model=a.model, size=a.size, edits=bool(a.ref))

    def one(i):
        try:
            raw, meta = (edit(ref, prompt, size=a.size, background=bg, model=a.model, fidelity="low", task="concept") if ref
                         else generate(prompt, size=a.size, background=bg, model=a.model))
        except ApiError as e:
            return i, None, f"c{i} failed: {e}"
        img = _decode_result(raw, f"{a.name} c{i}")
        f = out / "raw" / f"{a.name}_c{i}.png"
        img.save(f)
        flags, cb = _hybrid_flags(img, a.transparent)
        return i, {"file": str(f.relative_to(out)), "size": list(img.size), "bbox": list(cb) if cb else None,
                   "flags": flags, "source_format": (meta or {}).get("source_format")}, None

    results = {}
    with cf.ThreadPoolExecutor(max_workers=min(a.jobs, a.candidates)) as ex:
        for fut in cf.as_completed([ex.submit(one, i) for i in range(1, a.candidates + 1)]):
            i, r, err = fut.result()
            if err:
                print(f"! {a.name} {err}")
                continue
            results[i] = r
            print(f"  {a.name} c{i}: {r['size'][0]}x{r['size'][1]} {' '.join(r['flags']) or 'ok'}")
    if not results:
        sys.exit(f"gptkit material: every candidate of {a.name} failed; re-run with --candidates 1 to see one error")
    usable = [i for i in sorted(results) if "EMPTY" not in results[i]["flags"]]
    best = min(usable, key=lambda i: (_flag_weight(results[i]["flags"]), i)) if usable else None
    tiles = [(f"c{i} {' '.join(r['flags']) or 'ok'}{' BEST' if i == best else ''}", Image.open(out / r["file"]))
             for i, r in sorted(results.items())]
    rev = sheet([tiles], out / "review" / f"material_{a.name}.png", T=360)
    entry = {"prompt": a.prompt, "ref": a.ref, "size": a.size, "transparent": a.transparent, "model": a.model or "auto",
             "candidates": {f"c{i}": r for i, r in sorted(results.items())}, "best": f"c{best}" if best else None}
    if best is None:
        _merge_json(out / "material.json", a.name, entry)
        sys.exit(f"gptkit material: no usable candidate for {a.name} (all EMPTY); see {rev}")
    img = Image.open(out / results[best]["file"]).convert("RGBA")
    final = trim(img, pad=a.margin) if a.transparent and a.trim else img
    final.save(out / f"{a.name}.png")
    entry["file"] = f"{a.name}.png"
    _merge_json(out / "material.json", a.name, entry)
    flags = results[best]["flags"]
    print(f"material {a.name}: c{best} -> {out / (a.name + '.png')} ({final.width}x{final.height})"
          f"{'  ! ' + ' '.join(flags) if flags else ''}\n  review: {rev}  (Read it: the flags rank, your eyes decide; "
          f"`gptkit material ... --name {a.name}` again re-rolls)")


def _region_mask(size, region, grow, feather):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if region[0] == "circle":
        cx, cy, r = region[1]
        r *= grow
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    else:
        x0, y0, x1, y1 = region[1]
        gx, gy = (x1 - x0) * (grow - 1) / 2, (y1 - y0) * (grow - 1) / 2
        d.rectangle((x0 - gx, y0 - gy, x1 + gx, y1 + gy), fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather else m


def _parse_region(a):
    try:
        if a.circle:
            cx, cy, r = (float(v) for v in a.circle.split(","))
            return ("circle", (cx, cy, r))
        x0, y0, x1, y1 = (float(v) for v in a.box.split(","))
        return ("box", (x0, y0, x1, y1))
    except ValueError:
        raise ValueError("state: --circle takes CX,CY,R and --box takes X0,Y0,X1,Y1, in pixels of SRC")


def _best_shift(src_l, cand_l, outside, search):
    """Integer (dx, dy) that best registers the candidate onto the source outside the changed region (coarse, then fine)."""
    def err(dx, dy, s=1):
        c = np.roll(np.roll(cand_l, dy, 0), dx, 1)
        return float(np.abs(c - src_l)[::s, ::s][outside[::s, ::s]].mean())
    best = min(((err(dx, dy, 3), dx, dy) for dx in range(-search, search + 1, 2) for dy in range(-search, search + 1, 2)))
    _, bx, by = best
    best = min(((err(dx, dy), dx, dy) for dx in range(bx - 2, bx + 3) for dy in range(by - 2, by + 3)))
    return best[1], best[2], best[0]


def cmd_state(a):
    """A state variant of an approved asset (popped, filled, lit, open...): one-reference edits, registered onto the
    source, and only the changed region composited back, so everything outside it stays pixel-identical."""
    src, raw_src = _image_bytes_checked(a.src, "state")
    region = _parse_region(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rawdir = out.parent / "raw"
    rawdir.mkdir(exist_ok=True)
    transparent = _opaque_share(src)[0] < 0.95
    prompt = ("Edit this image. Keep it identical in every pixel: the same object, position, size, framing, lighting and "
              f"texture. Change ONLY this: {a.change}. Everything else stays exactly as it is."
              + (" Transparent background." if transparent else ""))
    inner = np.asarray(_region_mask(src.size, region, 1.0, 0)) > 127
    outside = np.asarray(_region_mask(src.size, region, a.grow * 1.25, 0)) < 1
    src_l = np.asarray(src.convert("L"), dtype=np.float32)
    mask = _region_mask(src.size, region, a.grow, a.feather)

    def one(i):
        try:
            raw, _ = edit(raw_src, prompt, size=_state_size(src), background="transparent" if transparent else "auto",
                          model=a.model, fidelity="high", task="precise")
        except ApiError as e:
            return i, None, f"c{i} failed: {e}"
        c = _decode_result(raw, f"state c{i}")
        c.save(rawdir / f"{out.stem}_c{i}.png")
        if abs(c.width / c.height - src.width / src.height) > 0.03:
            return i, None, f"c{i}: returned {c.width}x{c.height}, not the source's aspect ({src.width}x{src.height})"
        c = c.resize(src.size, Image.LANCZOS) if c.size != src.size else c
        cl = np.asarray(c.convert("L"), dtype=np.float32)
        dx, dy, drift = _best_shift(src_l, cl, outside, a.search)
        c = Image.fromarray(np.roll(np.roll(np.asarray(c), dy, 0), dx, 1))
        change = float(np.abs(np.asarray(c.convert("L"), dtype=np.float32) - src_l)[inner].mean())
        comp = Image.composite(c, src, mask)
        flags = (["NO-CHANGE"] if change < a.min_change else []) + (["DRIFT"] if drift > 25 else [])
        return i, {"comp": comp, "shift": [dx, dy], "drift_outside": round(drift, 1), "change_inside": round(change, 1),
                   "flags": flags}, None

    res = {}
    with cf.ThreadPoolExecutor(max_workers=min(a.jobs, a.candidates)) as ex:
        for fut in cf.as_completed([ex.submit(one, i) for i in range(1, a.candidates + 1)]):
            i, r, err = fut.result()
            if err:
                print(f"! {err}")
                continue
            res[i] = r
            print(f"  c{i}: shift {r['shift']}, re-render drift outside {r['drift_outside']}, change inside "
                  f"{r['change_inside']} {' '.join(r['flags']) or 'ok'}")
    if not res:
        sys.exit("gptkit state: no candidate came back; re-run with --candidates 1 to see one error")
    best = min(res, key=lambda i: (1000 * ("NO-CHANGE" in res[i]["flags"]), res[i]["drift_outside"]))
    r = res[best]
    if "NO-CHANGE" in r["flags"]:
        sys.exit(f"gptkit state: no candidate changed the region (best change {r['change_inside']} < --min-change "
                 f"{a.min_change}); describe the change more concretely, or check --circle/--box")
    r["comp"].save(out)
    after = np.abs(np.asarray(r["comp"].convert("RGBA"), dtype=np.int16) - np.asarray(src.convert("RGBA"), dtype=np.int16))
    untouched = np.asarray(mask) == 0
    max_out = int(after[untouched].max()) if untouched.any() else 0
    if region[0] == "circle":
        cx, cy, rr = region[1]
        rr *= a.grow * 1.6
        box = (int(cx - rr), int(cy - rr), int(cx + rr), int(cy + rr))
    else:
        x0, y0, x1, y1 = region[1]
        gx, gy = (x1 - x0) * 0.5, (y1 - y0) * 0.5
        box = (int(x0 - gx), int(y0 - gy), int(x1 + gx), int(y1 + gy))
    tiles = [("source", src.crop(box))] + [(f"c{i} {' '.join(v['flags']) or 'ok'}{' BEST' if i == best else ''}",
                                             v["comp"].crop(box)) for i, v in sorted(res.items())]
    rev = sheet([tiles], out.parent / f"{out.stem}_state_review.png", T=320)
    _merge_json(out.parent / "state.json", out.name, {
        "src": a.src, "change": a.change, "region": {region[0]: list(region[1])}, "grow": a.grow, "feather": a.feather,
        "best": f"c{best}", "max_diff_outside_mask": max_out,
        "candidates": {f"c{i}": {k: v for k, v in c.items() if k != "comp"} for i, c in sorted(res.items())}})
    print(f"state {out.name}: c{best} -> {out}  (max difference outside the mask: {max_out}; pixel-identical there)\n"
          f"  review: {rev}")


def _state_size(img):
    w, h = img.size
    return "1536x1024" if w / h > 1.2 else "1024x1536" if h / w > 1.2 else "1024x1024"


def _wh(s, flag):
    try:
        w, h = (int(v) for v in s.lower().split("x"))
        if w > 0 and h > 0:
            return w, h
    except ValueError:
        pass
    raise ValueError(f"stage: {flag} takes WIDTHxHEIGHT in CSS px, e.g. 412x915")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        # The starter page optionally loads geo.js; a missing optional file is not a page error.
        if self.path.split("?", 1)[0].endswith("/geo.js") and not os.path.isfile(self.translate_path(self.path)):
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


def _motion_easing(value, warnings=None):
    """CSS easing to Compose expression. Returns (Kotlin expression, warning)."""
    value = (value or "linear").strip().lower()
    known = {"linear": "LinearEasing", "ease": "CubicBezierEasing(0.25f, 0.1f, 0.25f, 1f)",
             "ease-in": "CubicBezierEasing(0.42f, 0f, 1f, 1f)",
             "ease-out": "CubicBezierEasing(0f, 0f, 0.58f, 1f)",
             "ease-in-out": "CubicBezierEasing(0.42f, 0f, 0.58f, 1f)"}
    if value in known:
        return known[value], None
    m = re.fullmatch(r"cubic-bezier\(\s*([-+\d.e]+)\s*,\s*([-+\d.e]+)\s*,\s*([-+\d.e]+)\s*,\s*([-+\d.e]+)\s*\)", value)
    if m:
        vals = [float(x) for x in m.groups()]
        if all(math.isfinite(x) for x in vals):
            return "CubicBezierEasing(" + ", ".join(_kt_float(x) for x in vals) + ")", None
    if value.startswith("steps("):
        return "LinearEasing", f"unsupported easing {value}: using linear"
    return "LinearEasing", f"unknown easing {value}: using linear"


def _kt_float(value):
    """Finite Kotlin Float literal with an explicit suffix."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite motion value")
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return (s if "." in s else s + ".0") + "f"


def _transform_components(value):
    """Parse supported CSS transforms into Compose property values; matrix skew is rejected."""
    result = {"translationX": 0.0, "translationY": 0.0, "scaleX": 1.0, "scaleY": 1.0, "rotationZ": 0.0}
    if not value or value.strip().lower() in ("none", "initial"):
        return result
    def length(s):
        m = re.fullmatch(r"\s*([-+\d.e]+)(?:px)?\s*", s)
        if not m:
            raise ValueError(f"unsupported transform length {s!r}")
        return float(m.group(1))
    matrix = re.fullmatch(r"matrix\(([^)]+)\)", value.strip())
    if matrix:
        v = [float(x.strip()) for x in matrix.group(1).split(",")]
        if len(v) != 6 or not all(math.isfinite(x) for x in v):
            raise ValueError(f"invalid matrix transform {value!r}")
        a, b, c, d, tx, ty = v
        sx = math.hypot(a, b)
        if sx == 0:
            raise ValueError("matrix has zero x scale")
        det = a * d - b * c
        sy = det / sx
        skew = (a * c + b * d) / (sx * sx)
        if abs(skew) > 1e-5:
            raise ValueError("matrix contains skew (unsupported)")
        result.update(translationX=tx, translationY=ty, scaleX=sx, scaleY=sy,
                      rotationZ=math.degrees(math.atan2(b, a)))
        return result
    for name, args in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", value):
        vals = [x.strip() for x in re.split(r"\s*,\s*|\s+", args.strip()) if x.strip()]
        if name in ("translate", "translateX", "translateY"):
            if name == "translateX": result["translationX"] = length(vals[0])
            elif name == "translateY": result["translationY"] = length(vals[0])
            else:
                result["translationX"] = length(vals[0]); result["translationY"] = length(vals[1]) if len(vals) > 1 else 0.0
        elif name in ("scale", "scaleX", "scaleY"):
            if name == "scaleX": result["scaleX"] = float(vals[0])
            elif name == "scaleY": result["scaleY"] = float(vals[0])
            else:
                result["scaleX"] = float(vals[0]); result["scaleY"] = float(vals[1]) if len(vals) > 1 else float(vals[0])
        elif name == "rotate":
            m = re.fullmatch(r"\s*([-+\d.e]+)(deg|rad|turn)?\s*", vals[0])
            if not m: raise ValueError(f"unsupported rotation {vals[0]!r}")
            n, unit = float(m.group(1)), m.group(2) or "deg"
            result["rotationZ"] = math.degrees(n) if unit == "rad" else n * 360 if unit == "turn" else n
        else:
            raise ValueError(f"unsupported transform function {name}")
    if not re.fullmatch(r"\s*(?:[a-zA-Z]+\([^)]*\)\s*)+", value):
        raise ValueError(f"unsupported transform {value!r}")
    if not all(math.isfinite(x) for x in result.values()): raise ValueError("non-finite transform")
    return result


def _motion_property_values(prop, value):
    if prop == "transform":
        return _transform_components(str(value))
    if prop == "opacity":
        return {"opacity": float(value)}
    if prop.startswith("--"):
        m = re.fullmatch(r"\s*([-+\d.e]+)(?:px)?\s*", str(value))
        if not m: raise ValueError(f"unsupported custom property value {value!r}")
        return {re.sub(r"[^A-Za-z0-9]+", " ", prop).strip().title().replace(" ", ""): float(m.group(1))}
    return {}


def _emit_motion_kt(doc, prefix, package, author=None):
    """Emit a compact Compose keyframe object from the neutral extracted timeline."""
    class A: pass
    a = A(); a.author = author
    lines = [_kt_header(a, package), "import androidx.compose.animation.core.CubicBezierEasing",
             "import androidx.compose.animation.core.LinearEasing", "import androidx.compose.animation.core.keyframes",
             "import androidx.compose.animation.core.KeyframesSpec", "\n", f"object {prefix}Motion {{"]
    emitted, warnings = [], []
    property_counts = {}
    elems = {}
    for i, anim in enumerate(doc.get("animations", []), 1):
        target = anim.get("target", {})
        label = target.get("id") or target.get("dataName") or target.get("selector") or f"element{i}"
        elem = _ident(label, "camel")
        elems.setdefault(label, elem)
        timing = anim.get("timing", {})
        dur = max(0, int(round(float(timing.get("duration", 0)))))
        delay = int(round(float(timing.get("delay", 0))))
        if timing.get("iterations", 1) not in (1, 1.0):
            warnings.append(f"{label}: iterations={timing.get('iterations')} is preserved in motion.json but not repeated by Compose keyframes")
        if timing.get("direction", "normal") != "normal":
            warnings.append(f"{label}: direction={timing.get('direction')} is preserved in motion.json but not applied by Compose keyframes")
        name = _ident(f"animation {i} {label}", "camel")
        lines.append(f"    const val {name}DelayMillis = {delay}")
        lines.append(f"    const val {name}DurationMillis = {dur}")
        for frame in anim.get("keyframes", []):
            for prop, raw in frame.get("properties", {}).items():
                if prop not in ("transform", "opacity") and not prop.startswith("--"):
                    warnings.append(f"unsupported property {prop} on {label}")
        effect_easing = (timing.get("easing") or "linear").strip().lower()
        if effect_easing != "linear" and len(anim.get("keyframes", [])) > 2:
            warnings.append(f"{label}: the effect easing {effect_easing} spans the whole animation in CSS; it is applied "
                            "per keyframe segment in Compose, so the timing differs: move it onto the keyframes")
        mapped = {}
        for frame in anim.get("keyframes", []):
            offset = frame.get("offset")
            if offset is None:
                offset = frame.get("computedOffset")
            if offset is None: continue
            for prop, raw in frame.get("properties", {}).items():
                try: props = _motion_property_values(prop, raw)
                except ValueError as e:
                    warnings.append(f"{label} {prop}: {e}"); continue
                frame_easing = frame.get("easing") or "linear"
                # WAAPI reports the computed keyframe default as linear; inherit the effect easing in that case.
                easing, warn = _motion_easing(anim.get("timing", {}).get("easing")
                                              if frame_easing == "linear" else frame_easing)
                if warn: warnings.append(f"{label}: {warn}")
                for key, val in props.items(): mapped.setdefault(key, []).append((int(round(float(offset) * dur)), val, easing))
        for prop, frames in mapped.items():
            if len({round(v, 6) for _, v, _ in frames}) < 2:
                continue  # constant: transform() decomposes into all five channels, most of which do not move
            base = elems[label] + prop[:1].upper() + prop[1:]
            property_counts[base] = property_counts.get(base, 0) + 1
            fn = base if property_counts[base] == 1 else f"{base}{property_counts[base]}"
            lines.append(f"    val {fn}: KeyframesSpec<Float> get() = keyframes {{")
            lines.append(f"        durationMillis = {dur}")
            lines.append(f"        delayMillis = {delay}")
            for ms, val, easing in frames:
                lines.append(f"        {_kt_float(val)} at {max(0, ms)} using {easing}")
            lines.append("    }")
            emitted.append((label, prop))
    lines.append("}")
    return "\n".join(lines) + "\n", emitted, warnings, elems


def _emit_geo_kt(geo, prefix, package, author=None):
    class A: pass
    a = A(); a.author = author
    lines = [_kt_header(a, package)]
    def value(name, val, indent, used=None):
        used = used if used is not None else []
        ident = _uniq(used + [_ident(name, "pascal")])[-1]
        used.append(ident)
        if isinstance(val, bool): return [" " * indent + f"const val {ident} = {str(val).lower()}"]
        if isinstance(val, (int, float)):
            return [" " * indent + f"const val {ident} = {_kt_float(val)}"]
        if isinstance(val, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val):
            return [" " * indent + f"val {ident} = listOf(" + ", ".join(_kt_float(x) for x in val) + ")"]
        if isinstance(val, dict):
            rows = [" " * indent + f"object {ident} {{"]
            child_used = []
            for k, v in val.items(): rows += value(k, v, indent + 4, child_used)
            return rows + [" " * indent + "}"]
        return []
    lines.append(f"object {prefix}Geo {{")
    used = [f"{prefix}Geo"]
    for k, v in (geo or {}).items(): lines += value(k, v, 4, used)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _lint_stage_compose(files):
    findings = []
    decl = re.compile(r"^\s*(?:const\s+)?(?:val|var|fun|object|class)\s+(\w+)")
    for f in map(Path, files):
        text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), f.read_text(), flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
        if text.count("{") != text.count("}"):
            findings.append(f"{f.name}: unbalanced braces")
        names = {}
        scopes = ["<top>"]
        for n, line in enumerate(text.splitlines(), 1):
            m = decl.match(line)
            if m:
                key = (tuple(scopes), m.group(1))
                if key in names: findings.append(f"{f.name}:{n}: duplicate identifier {m.group(1)}")
                names[key] = n
                if line.lstrip().startswith("object "):
                    scopes.append(m.group(1))
            if line.strip() == "}" and len(scopes) > 1:
                scopes.pop()
            if re.search(r"(?<![A-Za-z])(?:NaN|Infinity)(?![A-Za-z])", line): findings.append(f"{f.name}:{n}: non-finite literal")
            for lit in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", line):
                if re.search(re.escape(lit) + r"\s*(?:at\s+\d+|\n|$)", line) and "Millis =" not in line:
                    findings.append(f"{f.name}:{n}: float literal {lit} requires f suffix")
    return findings


def _stage_compose_files(doc, out, a, page):
    out.mkdir(parents=True, exist_ok=True)
    kt, properties, warnings, _ = _emit_motion_kt(doc, a.prefix, a.package, a.author)
    motion = out / f"{a.prefix}Motion.kt"; motion.write_text(kt)
    files = [motion]
    if isinstance(doc.get("geo"), dict):
        geo = out / f"{a.prefix}Geo.kt"; geo.write_text(_emit_geo_kt(doc["geo"], a.prefix, a.package, a.author)); files.append(geo)
    # capture where the motion changes: every keyframe of every animation, delay included (absolute ms on the
    # page's timeline), capped so the template stays a handful of screenshots rather than one per video frame
    times = sorted({int(round(float(x.get("timing", {}).get("delay", 0) or 0)
                              + float(k.get("offset") if k.get("offset") is not None else k.get("computedOffset") or 0)
                              * float(x.get("timing", {}).get("duration", 0) or 0)))
                    for x in doc.get("animations", []) for k in x.get("keyframes", [])} | {0})
    if len(times) > 24:
        times = [times[round(i * (len(times) - 1) / 23)] for i in range(24)]
    class A: pass
    h = A(); h.author = a.author
    test = [_kt_header(h, a.package), "import androidx.compose.ui.test.junit4.createComposeRule",
            "import com.github.takahirom.roborazzi.captureRoboImage",
            "import org.junit.Rule", "import org.junit.Test", "import androidx.compose.runtime.Composable", "",
            f"class {a.prefix}MotionScreenshotTest {{", "    @get:Rule val composeTestRule = createComposeRule()", "",
            f"    @Composable private fun {a.prefix}Screen() {{ /* TODO: port the approved screen to Compose */ }}", "",
            "    @Test fun captureTimeline() {", "        composeTestRule.mainClock.autoAdvance = false", f"        composeTestRule.setContent {{ {a.prefix}Screen() }}"]
    prev = 0
    for ms in times:  # advanceTimeBy takes a delta, the names carry the absolute time
        if ms > prev:
            test.append(f"        composeTestRule.mainClock.advanceTimeBy({ms - prev})")
        test.append(f'        composeTestRule.onRoot().captureRoboImage("{a.prefix.lower()}_{ms}.png")')
        prev = ms
    test += ["    }", "}"]
    tf = out / f"{a.prefix}MotionScreenshotTest.kt"; tf.write_text("\n".join(test) + "\n"); files.append(tf)
    (out / "motion.json").write_text(json.dumps(doc, indent=2))
    lint = _lint_stage_compose(files)
    print(f"  Compose lint: {'FAILED: ' + '; '.join(lint[:5]) if lint else 'passed'}")
    verdict = _kotlinc_check(files, a.package, [])
    print(f"  Compose compiler: {verdict}")
    return properties, warnings, files


def cmd_stage(a):
    """Render a code-built page: stills at --dpr, and with --motion a scrubbed clip (demo()/seek(ms) page contract),
    encoded to MP4 with ffmpeg (animated WebP without it) plus a labelled keyframe sheet."""
    page = Path(a.page).resolve()
    if a.compose:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", a.prefix or ""):
            sys.exit(f"gptkit stage: --prefix {a.prefix!r} is not a Kotlin identifier prefix")
        pkg_re = r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
        kt_keywords = _JAVA_KEYWORDS | {"as", "fun", "in", "is", "object", "typealias", "typeof", "val", "var", "when"}
        if not re.fullmatch(pkg_re, a.package or "") or any(x in kt_keywords for x in (a.package or "").split(".")):
            sys.exit(f"gptkit stage: --package {a.package!r} is not a valid Kotlin package")
    if a.init:
        if page.exists():
            sys.exit(f"gptkit stage: {page} already exists; --init only writes a new page")
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(HYBRID_PAGE)
        print(f"wrote {page}: the page contract (data-ready, ?capture, demo(), seek(ms)) and the animation helper are in it")
        return
    if not page.is_file():
        raise RuntimeError(f"stage: no such page {page} (create one with --init)")
    try:
        from playwright.sync_api import sync_playwright, Error as PwError
    except ImportError:
        sys.exit("gptkit stage: needs Playwright: pip install playwright (it drives your installed Chrome; "
                 "without one, also run `playwright install chromium`)")
    w, h = _wh(a.size, "--size")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stills = []
    for s in a.still or ["screen="]:
        name, _, query = s.partition("=")
        if not name or "/" in name:
            raise ValueError(f"stage: --still takes NAME=QUERY (e.g. after=state=after), got {s!r}")
        stills.append((name, query))
    root = page.parent
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root)))
    server_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    server_thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/{page.name}"
    errors, report = [], {"page": str(page), "size": [w, h], "dpr": a.dpr, "stills": {}}
    try:
        with sync_playwright() as p:
            exe = _chrome()
            b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()

            def open_page(dpr, query):
                pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=dpr)
                pg.on("pageerror", lambda e: errors.append(f"page error: {e}"))
                pg.on("console", lambda m: m.type == "error" and not m.text.startswith("Failed to load resource")
                      and errors.append(f"console: {m.text}"))
                pg.on("response", lambda r: r.status >= 400 and not r.url.endswith("/favicon.ico")
                      and errors.append(f"HTTP {r.status} {r.url.split(str(srv.server_address[1]), 1)[-1]}"))
                pg.goto(base + ("?" + query if query else ""))
                try:
                    pg.wait_for_selector("body[data-ready='1']", state="attached", timeout=a.timeout * 1000)
                except PwError:
                    raise RuntimeError(f"stage: {page.name}{'?' + query if query else ''} never set body[data-ready='1'] "
                                       f"within {a.timeout}s (see the page contract: `gptkit stage PAGE --init`)")
                return pg

            if a.compose:
                pg = open_page(1, "capture")
                if not pg.evaluate("typeof window.demo === 'function'"):
                    raise RuntimeError("stage: --compose needs window.demo() on the page (?capture); see `gptkit stage PAGE --init`")
                pg.evaluate("window.demo()")
                doc = pg.evaluate("""() => {
                  const animations = [...document.getAnimations()].map((a, i) => {
                    const e = a.effect, t = e && e.target, r = t && t.getBoundingClientRect();
                    let selector = null;
                    if (t && !t.id && !t.dataset.name) {
                      const parts = []; let n = t;
                      while (n && n.nodeType === 1 && parts.length < 4) {
                        let p = n.tagName.toLowerCase();
                        if (n.classList.length) p += '.' + [...n.classList].slice(0, 2).join('.');
                        const siblings = n.parentElement ? [...n.parentElement.children].filter(x => x.tagName === n.tagName) : [];
                        if (siblings.length > 1) p += `:nth-of-type(${siblings.indexOf(n) + 1})`;
                        parts.unshift(p); n = n.parentElement;
                      }
                      selector = parts.join(' > ');
                    }
                    const frames = e ? e.getKeyframes().map(k => {
                      const properties = {};
                      for (const [key, value] of Object.entries(k))
                        if (!['offset','computedOffset','easing','composite'].includes(key)) properties[key] = value;
                      // offset is null when the page did not set one; computedOffset is where Chrome places it
                      return {offset: k.offset ?? k.computedOffset, easing: k.easing || 'linear', properties};
                    }) : [];
                    const timing = e ? e.getTiming() : {};
                    return {index: i + 1, target: {id: t && t.id || null, dataName: t && t.dataset.name || null,
                      selector, warning: t && !t.id ? 'animation target has no id' : null,
                      rect: r ? {x:r.x,y:r.y,width:r.width,height:r.height} : null},
                      keyframes: frames, timing: {delay: timing.delay || 0, duration: typeof timing.duration === 'number' ? timing.duration : 0,
                        iterations: Number.isFinite(timing.iterations) ? timing.iterations : 'Infinity',
                        direction: timing.direction, fill: timing.fill, easing: timing.easing || 'linear'}};
                  });
                  return {schema: 'gptkit-motion-v1', viewport: {width: innerWidth, height: innerHeight},
                    geo: window.GEO && Object.keys(window.GEO).length ? window.GEO : null, animations};
                }""")
                warnings = []
                for anim in doc["animations"]:
                    if anim["target"].get("warning"):
                        warnings.append(f"animation {anim['index']} target has no id: {anim['target'].get('dataName') or anim['target'].get('selector')}")
                props, compose_warnings, files = _stage_compose_files(doc, Path(a.compose), a, page)
                warnings.extend(compose_warnings)
                pg.close()
                b.close()
                print(f"stage compose: {len(doc['animations'])} animation(s) on {len({x['target'].get('id') or x['target'].get('dataName') or x['target'].get('selector') for x in doc['animations']})} element(s); "
                      f"properties {', '.join(sorted({p for _, p in props})) or 'none'}; wrote {', '.join(f.name for f in files)} and motion.json")
                for warning in sorted(set(warnings)): print(f"! {warning}")
                return

            for name, query in stills:
                pg = open_page(a.dpr, query)
                pg.wait_for_timeout(a.settle)
                f = out / f"{name}.png"
                pg.screenshot(path=str(f))
                pg.close()
                report["stills"][name] = {"query": query, "file": f.name}
                print(f"  still {name}: {f} ({w * a.dpr}x{h * a.dpr})")
            if a.motion:
                pg = open_page(a.motion_dpr, "capture" + ("&" + a.query if a.query else ""))
                if not pg.evaluate("typeof window.demo === 'function' && typeof window.seek === 'function'"):
                    raise RuntimeError("stage: --motion needs window.demo() and window.seek(ms) on the page (?capture); "
                                       "see `gptkit stage PAGE --init`")
                pg.evaluate("window.demo()")
                frames = out / "frames"
                shutil.rmtree(frames, ignore_errors=True)
                frames.mkdir()
                n = max(1, int(round(a.motion * a.fps)))
                for i in range(n):
                    pg.evaluate(f"window.seek({i * 1000 / a.fps})")
                    pg.screenshot(path=str(frames / f"{i:04d}.png"))
                pg.close()
                report["motion"] = {"fps": a.fps, "frames": n, "seconds": a.motion, "dpr": a.motion_dpr}
                ff = shutil.which("ffmpeg")
                if ff:
                    mp4 = out / "motion.mp4"
                    r = subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "%04d.png"),
                                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                        "-crf", "16", "-movflags", "+faststart", str(mp4)], capture_output=True, text=True)
                    if r.returncode:
                        raise RuntimeError(f"stage: ffmpeg failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
                    report["motion"]["file"] = mp4.name
                else:
                    ims = [Image.open(frames / f"{i:04d}.png").convert("RGB") for i in range(n)]
                    durs = [round((i + 1) * 1000 / a.fps) - round(i * 1000 / a.fps) for i in range(n)]
                    webp = out / "motion.webp"
                    ims[0].save(webp, save_all=True, append_images=ims[1:], duration=durs, loop=0, quality=90, method=6)
                    report["motion"]["file"] = webp.name
                    print("  (no ffmpeg: wrote an animated WebP instead of MP4)")
                picks = sorted({min(n - 1, round(k * (n - 1) / max(1, a.sheet - 1))) for k in range(a.sheet)})
                rows, row = [], []
                for k in picks:
                    row.append((f"{round(k * 1000 / a.fps)} ms", Image.open(frames / f"{k:04d}.png")))
                    if len(row) == 6:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
                sheet(rows, out / "motion_sheet.png", T=300)
                print(f"  motion: {n} frames at {a.fps} fps -> {out / report['motion']['file']}; keyframes: {out / 'motion_sheet.png'}")
            b.close()
    finally:
        srv.shutdown()
        srv.server_close()
        server_thread.join(timeout=2)
    report["errors"] = errors
    (out / "stage.json").write_text(json.dumps(report, indent=2))
    for e in errors:
        print(f"! {e}")
    print(f"stage: {len(stills)} still(s){' + motion' if a.motion else ''} -> {out}")


def _hybrid_parsers(sub):
    p = sub.add_parser("material", help="new material/object art for a code-built screen (transparent candidates, flagged)")
    p.add_argument("prompt", help="the object and its material, light and camera; no UI")
    p.add_argument("--out", required=True, help="assets dir: NAME.png, raw/, review/, material.json")
    p.add_argument("--name", required=True, help="asset name (file stem)")
    p.add_argument("--size", default="1536x1024", help="WIDTHxHEIGHT canvas: 1536x1024, 1024x1536 or 1024x1024")
    p.add_argument("--candidates", type=_pos_int, default=2, help="images generated in parallel")
    p.add_argument("--jobs", type=_pos_int, default=4)
    p.add_argument("--ref", help="one approved image to match in material and rendering quality (not layout)")
    p.add_argument("--opaque", dest="transparent", action="store_false", help="a full-bleed texture/background instead")
    p.add_argument("--no-trim", dest="trim", action="store_false", help="keep the full canvas instead of trimming to content")
    p.add_argument("--margin", type=_pos_float, default=0.02, help="margin kept around the trimmed content (share of size)")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_material)
    p = sub.add_parser("state", help="state variant of an approved asset, pixel-identical outside the changed region")
    p.add_argument("src", help="the approved asset PNG")
    p.add_argument("change", help='what changes, e.g. "the third cavity has been pushed through: foil torn, empty"')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--circle", help="changed region CX,CY,R in SRC pixels")
    g.add_argument("--box", help="changed region X0,Y0,X1,Y1 in SRC pixels")
    p.add_argument("--out", required=True, help="output PNG (review sheet, raw/ and state.json go next to it)")
    p.add_argument("--candidates", type=_pos_int, default=2); p.add_argument("--jobs", type=_pos_int, default=4)
    p.add_argument("--grow", type=_pos_float, default=1.16, help="composite radius/size as a multiple of the region")
    p.add_argument("--feather", type=float, default=7, help="mask feather in px")
    p.add_argument("--search", type=_pos_int, default=8, help="max registration shift in px")
    p.add_argument("--min-change", type=float, default=4, help="mean grey change inside the region below which nothing changed")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_state)
    p = sub.add_parser("stage", help="render a code-built page: stills, scrubbed motion, or Compose motion files")
    p.add_argument("page", help="the HTML page (its folder is served over a local http server)")
    p.add_argument("--out", default="shots", help="output dir (default shots)")
    p.add_argument("--size", default="412x915", help="viewport WIDTHxHEIGHT in CSS px (412x915 phone; 1600x1200 a presentation shot)")
    p.add_argument("--dpr", type=_pos_int, default=3, help="device pixel ratio of the stills")
    p.add_argument("--still", action="append", help="NAME=QUERY, repeatable (e.g. before=still --still after=state=after)")
    p.add_argument("--motion", type=_pos_float, help="seconds of the demo() timeline to record")
    p.add_argument("--fps", type=_pos_int, default=30); p.add_argument("--motion-dpr", type=_pos_int, default=2)
    p.add_argument("--query", help="extra query string for the motion page (added after capture)")
    p.add_argument("--sheet", type=_pos_int, default=12, help="keyframes on the motion sheet")
    p.add_argument("--settle", type=int, default=300, help="ms to wait after ready before a still")
    p.add_argument("--timeout", type=_pos_int, default=30, help="seconds to wait for body[data-ready]")
    p.add_argument("--init", action="store_true", help="write a starter page with the page contract, then stop")
    p.add_argument("--compose", metavar="OUTDIR", help="extract the Web Animation timeline and emit Compose Kotlin")
    p.add_argument("--package", default="com.acme.app.ui.hybrid", help="Kotlin package for generated Compose files")
    p.add_argument("--prefix", default="Hybrid", help="Kotlin identifier prefix")
    p.add_argument("--author", help="adds an @author/@since header to generated Kotlin files")
    p.set_defaults(fn=cmd_stage)


def _design_to_code_parsers(sub):
    p = sub.add_parser("tokens", help="design_spec.json -> Compose/SwiftUI/CSS/Tailwind theme code (palette snapped to pixels)")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/tokens")
    p.add_argument("--package", default="com.example.app.ui.theme", help="Kotlin package for the Compose files")
    p.add_argument("--r-package", help="package of the app's R class: the module namespace in AGP 8 (default: --package minus .ui.theme)")
    p.add_argument("--author", help="adds an @author/@since header to Kotlin files")
    p.add_argument("--prefix", default="App", help="identifier prefix for the generated objects (AppColors, AppTypography, ...)")
    p.add_argument("--platforms", default="compose,swiftui,web",
                   help="comma-separated subset of compose, swiftui, web (aliases: android, ios)")
    p.add_argument("--fonts", help="fonts.json from `fonts` (default DIR/fonts/fonts.json when it exists)")
    p.add_argument("--no-snap", action="store_true", help="keep the spec's hex guesses")
    p.add_argument("--dtcg", action="store_true", help="also write tokens.tokens.json (W3C Design Tokens, Style Dictionary v4 subset)")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.add_argument("--verify", action="store_true", help="lint the emitted code; kotlinc when on PATH, swiftc on macOS, "
                                                         "Tailwind config in node, CSS in headless Chrome")
    p.set_defaults(fn=cmd_tokens)
    p = sub.add_parser("fonts", help="match typography to Google Fonts by rendering + measuring; download winners")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/fonts")
    p.add_argument("--candidates", type=int, default=24, help="similar families per group besides the guessed ones (0 = named only)")
    p.add_argument("--weights", type=_pos_int, default=3, help="weights tried per family and style")
    p.add_argument("--extra", help="comma-separated Google Fonts families to add to every shortlist")
    p.add_argument("--family", help="force this Google Fonts family for the groups of its category (every group when none matches)")
    p.add_argument("--tie", type=float, default=0.02, help="keep the spec's named family if it scores within this of the best")
    p.add_argument("--show", type=_pos_int, default=4, help="candidates per row on the review sheet")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.set_defaults(fn=cmd_fonts)
    p = sub.add_parser("compare", help="implemented-screen screenshot vs approved mockup: heatmap, score, ranked fixes")
    p.add_argument("dir", help="build dir with assets.json / design_spec.json"); p.add_argument("screenshot")
    p.add_argument("--mockup", help="default: the dir's mockup"); p.add_argument("--out", help="default DIR/compare")
    p.add_argument("--fonts", help="fonts.json from `fonts` (default DIR/fonts/fonts.json when it exists)")
    p.add_argument("--top", type=_pos_int, default=10, help="regions listed and boxed on the heatmap")
    p.add_argument("--vision", action="store_true", help="one gpt vision call describing the top regions")
    p.add_argument("--crop-top", type=int, default=0, help="screenshot px to drop at the top (status bar)")
    p.add_argument("--crop-bottom", type=int, default=0, help="screenshot px to drop at the bottom (nav bar)")
    p.add_argument("--shot-width-dp", type=_pos_int, help="dp width of the screen in the screenshot (warns when it differs "
                                                          "from the design's effective width by more than 5%%)")
    p.add_argument("--density", type=_pos_float, help="screenshot px per dp (alternative to --shot-width-dp)")
    p.add_argument("--fail-below", type=float, help="exit 1 when the overall score is below this (CI gate; default: no gate)")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("redline", help="pixel-true geometry: refined edges, corner radii, insets, gaps (dp) -> DIR/redline")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/redline")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.set_defaults(fn=cmd_redline)


def cmd_doctor(a):
    c = _config()
    print(f"config file : {c['config_path']} ({'found' if Path(c['config_path']).exists() else 'not found'})")
    print(f"base url    : {c['base']}")
    print(f"api key     : {'set' if c['key'] else 'MISSING'}")
    print(f"image model : {c['image_model']}" + (" (pinned)" if c["image_model_pinned"] else
          f" (default; auto-routed per task: concept {image_chain('concept')[0]}, precise {image_chain('precise')[0]}, first listed wins)"))
    print(f"text model  : {c['text_model']}")
    today = _today()
    for label, mid in (("image", c["image_model"]), ("text", c["text_model"])):
        retirement = RETIREMENTS.get(mid)
        if retirement:
            days = (date.fromisoformat(retirement[0]) - date.fromisoformat(today)).days
            if days <= 60:
                status = "retired" if days < 0 else f"retires in {days} days"
                print(f"  ! {label} model {mid} {status} (shutdown {retirement[0]}; replacement: {retirement[1]})")
    print(f"retries     : {c['retries']}")
    print(f"debug       : {'on' if c['debug'] else 'off'}")
    renderer, exe = _renderer_name(), _chrome()
    if renderer == "chrome" or (renderer is None and exe):
        ok, why = _chrome_probe()
        hint = " (running as root needs --no-sandbox: set GPTKIT_CHROME_NO_SANDBOX=1)" if "sandbox" in why.lower() else ""
        line = "chrome (ok)" if ok else f"none: chrome found at {exe} but failed: {why}{hint}"
    elif renderer:
        line = renderer + (" (no alpha: matted from two renders)" if renderer == "qlmanage" else "")
    else:
        line = "none (install Chrome/Chromium and set GPTKIT_CHROME, or pip install cairosvg)"
    print(f"svg render  : {line}")
    sdk = _android_sdk()
    if sdk and sdk[1]:
        print(f"aapt2       : {sdk[0]} (link checks: {Path(sdk[1]).parent.name})")
    elif sdk:
        print(f"aapt2       : {sdk[0]} (compile only: no platforms/android-*/android.jar; attribute names/enums not checked)")
    else:
        print("aapt2       : not found (set ANDROID_HOME to validate VectorDrawables)")
    env_eb = os.environ.get("GPTKIT_EBSYNTH")
    if env_eb and not Path(env_eb).exists():
        eb_line = f"GPTKIT_EBSYNTH={env_eb} does not exist (sequence propagate refuses to run until it does)"
    else:
        eb = _find_ebsynth(None)
        eb_line = eb or ("not found: git clone https://github.com/jamriska/ebsynth ~/.cache/gptkit/ebsynth && cd ~/.cache/gptkit/ebsynth"
                         f" && sh build-{'macos' if sys.platform == 'darwin' else 'linux'}-cpu_only.sh, or set GPTKIT_EBSYNTH "
                         "(sequence then renders every frame with GPT)")
    print(f"ebsynth     : {eb_line}")
    lj = _lottie_js_cache()
    try:
        txt = lj.read_text() if lj.exists() else ""
    except OSError as e:
        txt, lj_line = "", f"cannot read {lj}: {e}"
    else:
        lj_line = (f"cached ({lj}, {len(txt) // 1024} kB)" if txt and _lottie_js_ok(txt) else
                   f"{lj} is not lottie-web ({len(txt)} bytes): refetched on the next preview" if txt else
                   f"not cached: fetched from {LOTTIE_JS} on the first vectorize/rig preview (SVG fallback without it)")
    print(f"lottie-web  : {lj_line}")
    gf = _cache_dir() / "gf_metadata.json"
    if gf.exists():
        try:
            fams = json.loads(gf.read_text())["familyMetadataList"]
            age = (time.time() - gf.stat().st_mtime) / 86400
            gf_line = (f"cached ({gf}: {len(fams)} families, {age:.0f} days old"
                       + ("; refreshed by the next online fonts run)" if age > 7 else ")"))
        except (OSError, ValueError, KeyError, TypeError) as e:
            gf_line = f"{gf} is corrupt ({type(e).__name__}): the next fonts run deletes and refetches it"
    else:
        gf_line = f"not cached: fonts/compare need {GF_METADATA} once (7-day cache)"
    print(f"fonts cache : {gf_line}")
    if not c["key"]:
        sys.exit(1)
    base, headers = _auth()
    try:
        r = requests.get(base + "/models", headers=headers, timeout=30)
        r.raise_for_status()
        ids = {m["id"] for m in r.json().get("data", [])}
    except Exception as e:
        sys.exit(f"models      : request failed ({_redact(e)})")
    print(f"models      : {len(ids)} available")
    for label, mid, like in (("image", c["image_model"], lambda i: "image" in i), ("text", c["text_model"], _is_text_model)):
        if mid in ids:
            print(f"  {label:5s} {mid}: ok")
        else:
            alt = sorted(i for i in ids if like(i))
            print(f"  {label:5s} {mid}: NOT LISTED (set GPTKIT_{label.upper()}_MODEL"
                  + (f" to one of: {', '.join(alt[:6])})" if alt else " to an id this endpoint lists)"))
    images = sorted(i for i in ids if "image" in i)
    if images:
        print("  image models on this endpoint:", ", ".join(images))


_NON_TEXT_MODEL = ("image", "audio", "realtime", "codex", "transcribe", "tts")


def _is_text_model(i):
    """Chat/vision-capable ids in the gpt-6 / gpt-5 families (skips image, audio, realtime and codex variants)."""
    return i.startswith(("gpt-6", "gpt-5")) and not any(t in i for t in _NON_TEXT_MODEL)


def cmd_score(a):
    d, m, mock = _load(a.dir)
    asset = _asset(m, a.name)
    ref = ref_crop(mock, asset["bbox"])
    js = lambda o: json.dumps(o, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    for f in a.files:
        p = _need(f, "file")
        if p.suffix.lower() == ".svg":
            r = render_svgs([p.read_text()], 256)[0]
            if r is None:
                print(f"{f}: not renderable (no working SVG renderer: see the 'svg render' line of doctor)")
                continue
            sc = svg_fidelity(mock, asset["bbox"], r)
            print(f, js(sc), *(f"\n  ! {i}" for i in _svg_issues(sc)))
        else:
            print(f, js(score(mock, asset["bbox"], _open_image(p, "file"), ref, asset["transparent"])))


def _color_args(p):
    p.add_argument("--color-match", type=_unit_float, default=0.75, metavar="STRENGTH",
                   help="pull the final file's colours toward the design's (0-1, default 0.75); kept only if it scores better")
    p.add_argument("--no-color-match", dest="color_match", action="store_const", const=0.0, help="keep the raw colours")


# ---- market research + feature blueprint

RESEARCH_MODEL = "gpt-6-sol"
DEPTHS = {"quick": 5, "standard": 10, "deep": 16}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def think(prompt, schema=None, name="result", model=None, effort="high", search=False):
    """Reasoning call (optionally with web search). Returns (text or parsed JSON, [url citations]); a refusal,
    truncation, empty or non-JSON answer raises ApiError (see _responses_text)."""
    if effort not in EFFORTS:
        raise ValueError(f"think: effort must be one of {', '.join(EFFORTS)}, got {effort!r}")
    chosen_model = model or RESEARCH_MODEL
    _warn_retired_model(chosen_model)
    body = {"model": chosen_model, "store": False, "reasoning": {"effort": effort},
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    if search:
        body["tools"] = [{"type": "web_search"}]
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}
    try:
        return _responses_text(_responses(body, timeout=1200), schema)
    except ApiError as e:
        e.prompt = e.prompt or prompt[:200]
        raise


def _check_keys(data, schema, what):
    """Exit when a structured response lacks a top-level key the schema requires (a gateway ignoring strict mode)."""
    missing = [k for k in schema["required"] if not isinstance(data, dict) or k not in data]
    if missing:
        sys.exit(f"gptkit: the {what} response is missing {', '.join(missing)}: the endpoint may not honour strict "
                 "json_schema; try --model gpt-6-sol or another gateway")
    return data


def _mdcell(s):
    """A Markdown table cell: pipes escaped, newlines collapsed."""
    return " ".join(str(s).replace("|", "\\|").split())


def _obj(props, **extra):
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props, **extra}


_S, _N, _B = {"type": "string"}, {"type": "number"}, {"type": "boolean"}
_SA = {"type": "array", "items": _S}

PLAN_SCHEMA = _obj({"product_frame": _S, "questions": {"type": "array", "items": _obj(
    {"id": _S, "area": _S, "question": _S, "why": _S})}})

REPORT_SCHEMA = _obj({
    "executive_summary": _S,
    "market": _obj({"size_and_growth": _S, "trends": _SA, "target_segments": _SA}),
    "competitors": {"type": "array", "items": _obj({
        "name": _S, "platforms": _S, "positioning": _S, "pricing": _S, "rating_and_scale": _S,
        "key_features": _SA, "strengths": _SA, "weaknesses": _SA})},
    "user_pain_points": {"type": "array", "items": _obj({"pain": _S, "evidence": _S, "severity": {"type": "string", "enum": ["high", "medium", "low"]}})},
    "unmet_needs_and_gaps": _SA,
    "monetization": _obj({"benchmarks": _S, "recommended_model": _S, "price_points": _S}),
    "regulatory_and_platform": _SA,
    "risks": _SA,
    "differentiation_opportunities": _SA,
    "open_questions": _SA,
    "confidence_notes": _S})

BLUEPRINT_SCHEMA = _obj({
    "product": _obj({"name_suggestion": _S, "one_liner": _S, "positioning": _S, "north_star_metric": _S,
                     "differentiators": _SA}),
    "personas": {"type": "array", "items": _obj({"name": _S, "description": _S, "jobs_to_be_done": _SA, "pains": _SA})},
    "features": {"type": "array", "items": _obj({
        "id": _S, "name": _S, "description": _S, "personas": _SA, "solves": _S, "evidence": _S,
        "priority": {"type": "string", "enum": ["must", "should", "could", "wont"]},
        "rice": _obj({"reach": _N, "impact": _N, "confidence": _N, "effort": _N, "score": _N}),
        "release": {"type": "string", "enum": ["mvp", "v1", "v2", "later"]},
        "user_stories": {"type": "array", "items": _obj({"story": _S, "acceptance_criteria": _SA})},
        "platform_notes": _S, "dependencies": _SA, "risks": _SA})},
    "screens": {"type": "array", "items": _obj({
        "id": _S, "name": _S, "purpose": _S, "features": _SA, "native_components": _SA, "content": _SA,
        "mockup_brief": _S, "states": _SA})},
    "flows": {"type": "array", "items": _obj({"name": _S, "steps": _SA})},
    "monetization": _obj({"model": _S, "tiers": _SA, "paywall_placement": _S}),
    "analytics_events": _SA,
    "roadmap": _obj({"mvp": _SA, "v1": _SA, "v2": _SA}),
    "non_goals": _SA,
    "open_risks": _SA})


def _md_cites(cites):
    seen, out = set(), []
    for c in cites:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)
    return out


def cmd_research(a):
    n = DEPTHS[a.depth]
    _plan(0, n + 2, model=a.model, tokens_in=15000, tokens_out=15000, web_calls=n * 4)  # reasoning-heavy, ~4 searches each
    if getattr(a, "dry_run", False):
        print(f"would make 1 planning call, {n} web-searched questions, and 1 synthesis call")
        return
    out = Path(a.out)
    (out / "findings").mkdir(parents=True, exist_ok=True)
    ctx = (f"Product idea: {a.idea}\nTarget platform(s): {a.platform or 'unspecified'}\nRegion/market: {a.region}\n"
           + (f"Extra context: {a.context}\n" if a.context else ""))
    print(f"planning {n} research questions ({a.model}, effort {a.effort})...")
    plan, _ = think(ctx + f"\nYou are a senior product strategist. Plan an in-depth market study for this idea as exactly "
                    f"{n} precise, answerable web-research questions covering: market size/growth and trends; the top "
                    "direct and indirect competitors (names, pricing, ratings, downloads, features); real user pain points "
                    "and complaints from app-store reviews, Reddit and forums; unmet needs; monetization and pricing "
                    "benchmarks; platform/regulatory constraints; acquisition channels and ASO keywords; differentiation "
                    "angles. Also restate the product frame in 2 sentences.", PLAN_SCHEMA, "research_plan",
                    a.model, a.effort)
    _check_keys(plan, PLAN_SCHEMA, "research plan")
    (out / "plan.json").write_text(json.dumps(plan, indent=2))

    def run(q):
        text, cites = think(ctx + f"\nResearch question ({q['area']}): {q['question']}\nWhy it matters: {q['why']}\n"
                            "Search the web thoroughly (several queries; prefer primary sources, app-store listings, "
                            "official pricing pages, recent reviews and reputable reports). Answer in concise markdown "
                            "with concrete facts, numbers, names and dates, and cite every claim. Say explicitly what "
                            "you could not verify.", model=a.model, effort=a.effort, search=True)
        cites = _md_cites(cites)
        (out / "findings" / f"{q['id']}.md").write_text(f"# {q['question']}\n\n{text}\n\n## Sources\n" +
                                                        "".join(f"- [{c['title'] or c['url']}]({c['url']})\n" for c in cites))
        print(f"  done {q['id']}: {q['question'][:70]} ({len(cites)} sources)")
        return q, text, cites
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        found = list(ex.map(run, plan["questions"]))
    corpus = "\n\n".join(f"## {q['id']} [{q['area']}] {q['question']}\n{t}" for q, t, _ in found)
    sources = _md_cites([c for _, _, cs in found for c in cs])
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    print(f"synthesising report from {len(found)} findings, {len(sources)} sources...")
    report, _ = think(ctx + "\nYou are a senior product strategist. Synthesise these research findings into a decision-grade "
                      "market report. Use ONLY facts in the findings (keep their numbers and names), mark uncertainty, "
                      "and make the gaps/opportunities specific and actionable.\n\nFINDINGS:\n" + corpus,
                      REPORT_SCHEMA, "market_report", a.model, a.synth_effort)
    _check_keys(report, REPORT_SCHEMA, "market report")
    report["idea"], report["platform"], report["region"] = a.idea, a.platform, a.region
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(_report_md(report, sources))
    print(f"-> {out / 'report.md'}, report.json, findings/, sources.json")


def _report_md(r, sources):
    L = [f"# Market research: {r['idea']}", "", f"_Platform: {r['platform'] or 'unspecified'} · Region: {r['region']}_", "",
         "## Executive summary", r["executive_summary"], "", "## Market", r["market"]["size_and_growth"], "",
         "**Trends**", *[f"- {x}" for x in r["market"]["trends"]], "", "**Target segments**",
         *[f"- {x}" for x in r["market"]["target_segments"]], "", "## Competitors", "",
         "| App | Platforms | Pricing | Rating / scale | Positioning |", "|---|---|---|---|---|",
         *["| " + " | ".join(_mdcell(c[k]) for k in ("name", "platforms", "pricing", "rating_and_scale", "positioning")) + " |"
           for c in r["competitors"]], ""]
    for c in r["competitors"]:
        L += [f"### {c['name']}", "**Key features:** " + "; ".join(c["key_features"]),
              "**Strengths:** " + "; ".join(c["strengths"]), "**Weaknesses:** " + "; ".join(c["weaknesses"]), ""]
    L += ["## User pain points", *[f"- **{p['pain']}** ({p['severity']}): {p['evidence']}" for p in r["user_pain_points"]],
          "", "## Unmet needs and gaps", *[f"- {x}" for x in r["unmet_needs_and_gaps"]], "",
          "## Monetization", r["monetization"]["benchmarks"], "", f"**Recommended:** {r['monetization']['recommended_model']}",
          f"**Price points:** {r['monetization']['price_points']}", "", "## Platform and regulatory",
          *[f"- {x}" for x in r["regulatory_and_platform"]], "", "## Risks", *[f"- {x}" for x in r["risks"]], "",
          "## Differentiation opportunities", *[f"- {x}" for x in r["differentiation_opportunities"]], "",
          "## Open questions", *[f"- {x}" for x in r["open_questions"]], "", "## Confidence", r["confidence_notes"], "",
          "## Sources", *[f"{i + 1}. [{s['title'] or s['url']}]({s['url']})" for i, s in enumerate(sources)]]
    return "\n".join(L) + "\n"


def cmd_blueprint(a):
    d = Path(a.dir)
    report = (_read_json(d / "report.json", "research report", required=REPORT_SCHEMA["required"])
              if (d / "report.json").exists() else None)
    idea = a.idea or (report or {}).get("idea")
    if not idea:
        sys.exit("gptkit: give --idea or a research dir with report.json")
    platform = a.platform or (report or {}).get("platform")
    native = (f"Target platform: {PLATFORMS[platform]['label']}; name each screen's native components from: "
              f"{PLATFORMS[platform]['components']}.") if platform in PLATFORMS else "Platform: unspecified."
    print(f"generating the feature blueprint ({a.model}, effort {a.effort})...")
    bp, _ = think(f"Product idea: {idea}\n{native}\n" + (f"Constraints: {a.constraints}\n" if a.constraints else "") +
                  ("MARKET RESEARCH REPORT (ground every feature in it; cite the pain point or gap it addresses in "
                   "'evidence'):\n" + json.dumps(report) if report else "No research report: reason from first principles "
                   "and mark evidence as 'assumption'.") +
                  "\n\nYou are a principal product manager. Produce a complete, buildable feature blueprint: product "
                  "positioning, personas with jobs-to-be-done, a prioritised feature list (MoSCoW + RICE with reach per "
                  "quarter, impact 0.25-3, confidence 0-1, effort in person-weeks, score = R*I*C/E; release mvp/v1/v2), "
                  "user stories with testable acceptance criteria, a screen inventory where every screen has a precise "
                  "one-paragraph mockup_brief a designer can render directly (layout, content, states, native "
                  "components), key flows, monetization, analytics events, a roadmap and non-goals. Keep the MVP "
                  "small and differentiated.", BLUEPRINT_SCHEMA, "feature_blueprint", a.model, a.effort)
    _check_keys(bp, BLUEPRINT_SCHEMA, "blueprint")
    for f in bp["features"]:
        r = f["rice"]
        r["score"] = round(r["reach"] * r["impact"] * r["confidence"] / max(r["effort"], 0.1), 1)
    bp["features"].sort(key=lambda f: (["mvp", "v1", "v2", "later"].index(f["release"]), -f["rice"]["score"]))
    bp["idea"], bp["platform"] = idea, platform
    out = Path(a.out) if a.out else d
    out.mkdir(parents=True, exist_ok=True)
    (out / "blueprint.json").write_text(json.dumps(bp, indent=2))
    (out / "screens.json").write_text(json.dumps([{"id": s["id"], "name": s["name"], "platform": platform,
                                                   "mockup_brief": s["mockup_brief"]} for s in bp["screens"]], indent=2))
    (out / "blueprint.md").write_text(_blueprint_md(bp))
    print(f"-> {out / 'blueprint.md'}, blueprint.json, screens.json ({len(bp['features'])} features, "
          f"{len(bp['screens'])} screens; MVP: {sum(f['release'] == 'mvp' for f in bp['features'])} features)")


def _blueprint_md(b):
    p = b["product"]
    L = [f"# Feature blueprint: {p['name_suggestion']}", "", f"**{p['one_liner']}**", "", p["positioning"], "",
         f"**North-star metric:** {p['north_star_metric']}", "", "**Differentiators**", *[f"- {x}" for x in p["differentiators"]],
         "", "## Personas"]
    for x in b["personas"]:
        L += [f"### {x['name']}", x["description"], "**Jobs to be done:** " + "; ".join(x["jobs_to_be_done"]),
              "**Pains:** " + "; ".join(x["pains"]), ""]
    L += ["## Features", "", "| ID | Feature | Priority | Release | RICE | Solves |", "|---|---|---|---|---|---|",
          *["| " + " | ".join(_mdcell(v) for v in (f["id"], f["name"], f["priority"], f["release"], f["rice"]["score"], f["solves"])) + " |"
            for f in b["features"]], ""]
    for f in b["features"]:
        L += [f"### {f['id']} {f['name']} ({f['release']}, {f['priority']})", f["description"], "",
              f"_Evidence:_ {f['evidence']}", ""]
        for us in f["user_stories"]:
            L += [f"- {us['story']}", *[f"  - [ ] {c}" for c in us["acceptance_criteria"]]]
        if f["platform_notes"]:
            L += ["", f"_Platform:_ {f['platform_notes']}"]
        L.append("")
    L += ["## Screens"]
    for s in b["screens"]:
        L += [f"### {s['id']} {s['name']}", s["purpose"], "", "**Native components:** " + ", ".join(s["native_components"]),
              "**States:** " + ", ".join(s["states"]), "", f"> Mockup brief: {s['mockup_brief']}", ""]
    L += ["## Flows", *[f"- **{x['name']}:** " + " → ".join(x["steps"]) for x in b["flows"]], "", "## Monetization",
          b["monetization"]["model"], *[f"- {t}" for t in b["monetization"]["tiers"]],
          f"Paywall: {b['monetization']['paywall_placement']}", "", "## Roadmap",
          "**MVP:** " + "; ".join(b["roadmap"]["mvp"]), "**v1:** " + "; ".join(b["roadmap"]["v1"]),
          "**v2:** " + "; ".join(b["roadmap"]["v2"]), "", "## Analytics events", *[f"- `{e}`" for e in b["analytics_events"]],
          "", "## Non-goals", *[f"- {x}" for x in b["non_goals"]], "", "## Open risks", *[f"- {x}" for x in b["open_risks"]]]
    return "\n".join(L) + "\n"


def _color_arg(s):
    """argparse type: a colour PIL and CSS both read (#RGB, #RRGGBB, #RRGGBBAA or a CSS colour name)."""
    from PIL import ImageColor
    try:
        ImageColor.getrgb(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a colour such as #1b1d33 or white, got {s!r}")
    if not re.fullmatch(r"#[0-9a-fA-F]{3,8}|[a-zA-Z]+", s):
        raise argparse.ArgumentTypeError(f"expected a colour such as #1b1d33 or white, got {s!r}")
    return s


def cmd_cost(a):
    """Summarise the local JSONL ledger by call, model or time bucket."""
    ledger = a.ledger or os.environ.get("GPTKIT_LEDGER", str(Path.home() / ".cache/gptkit/ledger.jsonl"))
    if ledger == "off":
        print("no ledger entries: ledger disabled")
        return
    path = Path(ledger).expanduser()
    if not path.exists():
        print(f"no ledger entries: {path}")
        return
    since = None
    if a.since:
        try:
            if re.fullmatch(r"\d+[dh]", a.since):
                n, unit = int(a.since[:-1]), a.since[-1]
                since = datetime.now(timezone.utc) - (timedelta(days=n) if unit == "d" else timedelta(hours=n))
            else:
                since = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            sys.exit("gptkit cost: --since must be 7d, 24h or YYYY-MM-DD")
    rows, malformed = [], 0
    for line in path.read_text(encoding="utf8").splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("ts"), str):
                raise ValueError("not a ledger object")
            if row.get("usage") is not None and not isinstance(row.get("usage"), dict):
                raise ValueError("invalid usage")
            if not isinstance(row.get("est_usd", 0), (int, float)):
                raise ValueError("invalid estimate")
            ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
            if since is None or ts >= since:
                rows.append(row)
        except (ValueError, KeyError, TypeError, AttributeError):
            malformed += 1
    if not rows and not malformed:
        print(f"no ledger entries: {path}")
        return
    groups = {}
    for row in rows:
        key = {"cmd": row.get("cmd") or "unknown", "model": row.get("model") or "unknown",
               "day": row.get("ts", "")[:10], "dir": row.get("dir") or "(none)"}[a.by]
        g = groups.setdefault(key, {"calls": 0, "image_calls": 0, "text_calls": 0, "tokens_in": 0,
                                    "tokens_out": 0, "est_usd": 0.0})
        image = str(row.get("endpoint", "")).startswith("/images/")
        usage = row.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        tokens_in = usage.get("input_tokens") or details.get("text_tokens") or 0
        tokens_out = usage.get("output_tokens") or 0
        if not isinstance(tokens_in, (int, float)):
            tokens_in = 0
        if not isinstance(tokens_out, (int, float)):
            tokens_out = 0
        g["calls"] += 1
        g["image_calls" if image else "text_calls"] += 1
        g["tokens_in"] += tokens_in
        g["tokens_out"] += tokens_out
        g["est_usd"] += row.get("est_usd", 0) or 0
    total = {k: sum(g[k] for g in groups.values()) for k in next(iter(groups.values()))} if groups else {
        "calls": 0, "image_calls": 0, "text_calls": 0, "tokens_in": 0, "tokens_out": 0, "est_usd": 0.0}
    note = "list-price equivalent; through a subscription gateway (e.g. sub2api) this is quota, not money billed" \
        if "api.openai.com" not in _config()["base"] else None
    if a.json:
        print(json.dumps({"groups": groups, "total": total, "malformed": malformed, "note": note}, indent=2))
    else:
        print(f"{a.by:24} calls image text tokens in/out est USD")
        for key, g in sorted(groups.items()):
            print(f"{key:24} {g['calls']:5} {g['image_calls']:5} {g['text_calls']:4} "
                  f"{g['tokens_in']:9}/{g['tokens_out']:<9} ${g['est_usd']:.4f}")
        print(f"{'TOTAL':24} {total['calls']:5} {total['image_calls']:5} {total['text_calls']:4} "
              f"{total['tokens_in']:9}/{total['tokens_out']:<9} ${total['est_usd']:.4f}")
        if malformed:
            print(f"skipped {malformed} malformed ledger lines")
        if note:
            print(note)


def _status_json(path):
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError, UnicodeError) as e:
        return None, f"{path.name}: invalid JSON ({type(e).__name__})"


def _status_sha(path):
    """Hash of the decoded pixels: manifest re-encodes the mockup it copies, so the file bytes differ."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return hashlib.sha1(repr(im.size).encode() + im.tobytes()).hexdigest()
    except (OSError, ValueError):
        return None


def _status_unit(root, kind, label, checks, problems=None, next_=None):
    return {"type": kind, "path": str(root), "name": label,
            "checks": checks, "problems": problems or [], "next": next_ or []}


def _status_arg(path):
    """A path for a printed command: relative to the cwd when inside it (short and still runnable), else absolute."""
    path = Path(path)
    try:
        rel = os.path.relpath(path, os.getcwd())
        path = rel if not rel.startswith("..") else path
    except ValueError:  # another drive on Windows
        pass
    return shlex.quote(str(path))


def _status_scan(base):
    """Discover bounded project dirs; status never opens images or makes network calls."""
    base = Path(base).resolve()
    dirs = [base]
    for here, names, files in os.walk(base):
        p = Path(here)
        depth = len(p.relative_to(base).parts)
        names[:] = [n for n in names if n not in ("node_modules", ".git", "raw", "frames")]
        if depth >= 4:
            names[:] = []
        dirs.append(p)
    dirs = sorted(set(dirs), key=lambda p: (len(p.parts), str(p)))
    units = []
    seen = set()
    # manifest copies the approved mockup into the build dir: matching bytes link a design to its build
    build_mockups = {}
    for d in dirs:
        if (d / "assets.json").exists() and (d / "mockup.png").exists():
            build_mockups.setdefault(_status_sha(d / "mockup.png"), []).append(d)
    def add(d, kind, label, checks, problems=None, nxt=None):
        key = (str(d), kind)
        if key not in seen:
            seen.add(key)
            units.append(_status_unit(d, kind, label, checks, problems, nxt))
    for d in dirs:
        # Ignore individual oversized artifacts and JSONs which cannot be read.
        children = []
        try:
            children = [p for p in d.iterdir() if p.is_file() and p.stat().st_size <= 50 * 1024 * 1024]
        except OSError:
            continue
        names = {p.name for p in children}
        if {"report.json", "sources.json"} <= names or "findings" in {p.name for p in d.iterdir() if p.is_dir()}:
            probs, checks = [], []
            for n in ("report.json", "report.md", "findings", "sources.json"):
                exists = (d / n).exists()
                checks.append({"name": n, "status": "done" if exists else "missing"})
                if n.endswith(".json") and exists:
                    _, err = _status_json(d / n)
                    if err: probs.append(err)
            nxt = [] if (d / "report.json").exists() else [f"python3 $K research \"<idea>\" --out {_status_arg(d)}"]
            if (d / "report.json").exists() and not (d / "screens.json").exists() and not (d / "blueprint.md").exists():
                nxt = [f"python3 $K blueprint {_status_arg(d)}"]
            add(d, "research", d.name, checks, probs, nxt)
        if "blueprint.md" in names or "screens.json" in names:
            checks = [{"name": n, "status": "done" if (d / n).exists() else "missing"}
                      for n in ("blueprint.md", "screens.json")]
            probs = []
            for n in ("screens.json",):
                if (d / n).exists():
                    _, err = _status_json(d / n)
                    if err: probs.append(err)
            add(d, "blueprint", d.name, checks, probs, [] if all(c["status"] == "done" for c in checks)
                else [f"python3 $K blueprint {_status_arg(d)}"])
        mockups = [p for p in children if p.suffix.lower() == ".png" and p.name.startswith("mockup")]
        if (mockups or "design.json" in names) and "assets.json" not in names:  # a build dir keeps its own mockup.png
            design = d / "design.json"
            checks = [{"name": "mockup PNGs", "status": "done" if mockups else "missing"},
                      {"name": "design.json", "status": "done" if design.exists() else "missing"}]
            probs = []
            if design.exists():
                _, err = _status_json(design)
                if err: probs.append(err)
            latest = max(mockups, key=lambda p: p.stat().st_mtime) if mockups else None
            built = [b for m in mockups for b in build_mockups.get(_status_sha(m), [])]
            if built:
                checks.append({"name": "manifest (" + ", ".join(_status_arg(b) for b in built) + ")", "status": "done"})
            nxt = [f"python3 $K mockup \"<brief>\" --out {_status_arg(d)}"] if not latest else [] if built else \
                [f"once the user approves a variant: python3 $K manifest {_status_arg(latest)} --out <build dir>"]
            add(d, "design", d.name, checks, probs, nxt)
        critiques = [p for p in children if p.name.endswith("_critique.json")]
        for p in critiques:
            data, err = _status_json(p)
            probs = [err] if err else []
            issues = []
            if isinstance(data, dict):
                for rnd in data.get("rounds", []):
                    if isinstance(rnd, dict):
                        for batch in (rnd.get("issues"), rnd.get("new_issues")):
                            if isinstance(batch, list): issues.extend(x for x in batch if isinstance(x, dict))
            high = sum(1 for x in issues if isinstance(x, dict) and x.get("severity") == "high")
            if high: probs.append(f"{high} high-severity critique issue(s)")
            add(d, "critique", p.stem, [{"name": p.name, "status": "done"}], probs, [])
        if "stage.json" in names or "material.json" in names or "stage" in names:
            probs, checks = [], []
            for n in ("material.json", "stage.json", "frames", "motion_sheet.png"):
                exists = (d / n).exists()
                checks.append({"name": n, "status": "done" if exists else "missing"})
                if n == "material.json" and exists:
                    data, err = _status_json(d / n)
                    if err: probs.append(err)
                    elif isinstance(data, dict):
                        for name, entry in data.items():
                            if not isinstance(entry, dict):
                                probs.append(f"material.json: damaged entry {name}"); continue
                            candidates = entry.get("candidates") or {}
                            flags = sorted({flag for c in candidates.values() if isinstance(c, dict)
                                            for flag in c.get("flags", [])}) if isinstance(candidates, dict) else []
                            if flags: probs.append(f"{name}: {' '.join(flags)}")
            if (d / "stage.json").exists():
                data, err = _status_json(d / "stage.json")
                if err: probs.append(err)
                elif isinstance(data, dict): probs.extend(str(x) for x in data.get("errors", []))
            add(d, "hybrid", d.name, checks, probs, [])
        if "assets.json" in names:
            probs, checks = [], []
            core = ("assets.json", "manifest_overlay.png", "design_spec.json", "forge_report.json", "assets/", "svg_report.json",
                    "svg/", "tokens/tokens.json", "compare/report.json")
            for n in core + ("icon_spec.json", "review/", "icons/", "traced/", "rig/", "anim/", "vector/", "frames/"):
                npath = d / n.rstrip("/")
                is_dir = n.endswith("/")
                exists = npath.is_dir() if is_dir else npath.exists()
                if not exists and n not in core:
                    continue  # optional outputs are listed only once they exist
                state = "done" if exists else "missing"
                checks.append({"name": n, "status": state})
                if n.endswith(".json") and exists:
                    data, err = _status_json(npath)
                    if err: probs.append(err)
                    if n == "forge_report.json" and isinstance(data, dict):
                        flagged, incomplete = [], []
                        for name, entry in data.items():
                            if not isinstance(entry, dict):
                                probs.append(f"forge_report.json: damaged entry {name}"); continue
                            # the same test forge uses to re-queue: flagged winner, unfinished run or no final file
                            if entry.get("needs_attention"): flagged.append(name)
                            elif _incomplete(entry) or not (d / "assets" / f"{name}.png").exists(): incomplete.append(name)
                        if flagged: probs.append("NEEDS ATTENTION: " + ", ".join(flagged))
                        if incomplete: probs.append("no final file: " + ", ".join(incomplete))
                    if n == "svg_report.json" and isinstance(data, dict):
                        for name, entry in data.items():
                            if isinstance(entry, dict) and entry.get("error"): probs.append(f"svg {name}: {entry['error']}")
            def stale(output, inputs, label):
                op = d / output
                ins = [d / x for x in inputs if (d / x).exists()]
                if op.exists() and ins and op.stat().st_mtime < max(x.stat().st_mtime for x in ins):
                    probs.append(f"STALE: {label} older than " + ", ".join(x.name for x in ins if x.stat().st_mtime > op.stat().st_mtime))
                    for c in checks:
                        if c["name"] == output: c["status"] = "STALE"
            stale("design_spec.json", ["assets.json"], "design_spec.json")
            stale("svg", ["design_spec.json"], "svg/")
            stale("tokens/tokens.json", ["design_spec.json", "fonts/fonts.json"], "tokens")
            stale("compare/report.json", ["tokens/tokens.json"], "compare")
            manifest, _ = _status_json(d / "assets.json")
            report, _ = _status_json(d / "forge_report.json") if (d / "forge_report.json").exists() else ({}, None)
            report = report if isinstance(report, dict) else {}
            pngs = [x.get("name") for x in (manifest or {}).get("assets", []) if isinstance(x, dict) and x.get("format") == "png"] \
                if isinstance(manifest, dict) else []
            unforged = [n for n in pngs if n and n not in report and not (d / "assets" / f"{n}.png").exists()]
            if unforged and (d / "forge_report.json").exists():
                probs.append("not forged yet: " + ", ".join(unforged))
            svgs = [x.get("name") for x in (manifest or {}).get("assets", []) if isinstance(x, dict) and x.get("format") == "svg"] \
                if isinstance(manifest, dict) else []
            nxt = []
            if not (d / "design_spec.json").exists() or any(c["name"] == "design_spec.json" and c["status"] == "STALE" for c in checks):
                nxt = [f"python3 $K spec {_status_arg(d)}"]
            elif (pngs and not (d / "forge_report.json").exists()) or any("NEEDS ATTENTION" in x or "no final file" in x for x in probs):
                flagged = next((x.partition(": ")[2] for x in probs if x.startswith("NEEDS ATTENTION:")), "")
                nxt = ([f"read {_status_arg(d / 'review')}/forge_*.png, then python3 $K pick {_status_arg(d)} <name> <N>, "
                        f"python3 $K fix {_status_arg(d)} <name>, or python3 $K forge {_status_arg(d)} --only "
                        f"{flagged.replace(', ', ',')} --force --candidates 3"] if flagged else []) + \
                      ([f"python3 $K forge {_status_arg(d)}"] if not flagged or any(x.startswith(("no final file", "not forged")) for x in probs) else [])
                if svgs and not (d / "svg_report.json").exists():
                    nxt.append(f"python3 $K svg {_status_arg(d)}   (in parallel with forge)")
            elif unforged:
                nxt = [f"python3 $K forge {_status_arg(d)}"] + ([f"python3 $K svg {_status_arg(d)}   (in parallel with forge)"]
                                                               if svgs and not (d / "svg_report.json").exists() else [])
            elif svgs and not (d / "svg_report.json").exists(): nxt = [f"python3 $K svg {_status_arg(d)}"]
            elif not (d / "tokens" / "tokens.json").exists(): nxt = [f"python3 $K tokens {_status_arg(d)}"]
            elif not (d / "compare" / "report.json").exists(): nxt = [f"python3 $K compare {_status_arg(d)} <screenshot>"]
            else: nxt = ["review reports; no pending kit step"]
            add(d, "build", d.name, checks, probs, nxt)
        if "appicon.json" in names:
            data, err = _status_json(d / "appicon.json")
            probs = [err] if err else []
            if isinstance(data, dict):
                for key in ("safe_zone", "glyph"):
                    if isinstance(data.get(key), dict) and data[key].get("ok") is False: probs.append(f"{key}.ok is false")
                probs.extend(str(x) for x in data.get("warnings", []))
            add(d, "appicon", d.name, [{"name": "appicon.json", "status": "done"}], probs, [])
        concept_pngs = [p for p in children if p.suffix.lower() == ".png" and re.match(r"concept(?:_v\d+)?\.png$", p.name)]
        if "concepts" in names or concept_pngs:
            checks = [{"name": "concept PNGs", "status": "done" if concept_pngs else "missing"},
                      {"name": "review/", "status": "done" if (d / "review").is_dir() else "missing"}]
            chosen = concept_pngs[0] if len(concept_pngs) == 1 else None
            nxt = [f"python3 $K appicon build {_status_arg(chosen)} --out {_status_arg(d / 'build')}" ] if chosen else []
            add(d, "appicon concepts", d.name, checks, [], nxt)
        if "sequence.json" in names:
            meta, err = _status_json(d / "sequence.json")
            probs = [err] if err else []
            checks = [{"name": "sequence.json", "status": "done"}]
            if isinstance(meta, dict) and isinstance(meta.get("frames"), int) and 0 <= meta["frames"] <= 10000:
                if meta.get("route"):
                    checks.append({"name": "route", "status": str(meta["route"].get("engine", meta["route"]))})
                missing = []
                for i in range(meta["frames"]):
                    raw = d / "raw" / f"{i:03d}.png"
                    side = raw.with_suffix(".json")
                    if not raw.exists(): missing.append(i)
                    elif side.exists():
                        side_data, _ = _status_json(side)
                        guide = meta.get("guide_sha1") or []
                        if not isinstance(side_data, dict) or i >= len(guide) or side_data.get("guide_sha1") != guide[i]:
                            probs.append(f"stale frame {i}")
                if missing: probs.append(f"frames missing: {missing[:10]}")
                checks.append({"name": "raw frames", "status": "missing" if missing else "done"})
                qa, qe = _status_json(d / "report.json") if (d / "report.json").exists() else (None, None)
                if qe: probs.append(qe)
                elif isinstance(qa, dict):
                    if qa.get("route"): checks.append({"name": "route", "status": str(qa["route"])})
                    q = qa.get("outliers") or (qa.get("sequence", {}).get("outliers", []) if isinstance(qa.get("sequence"), dict) else [])
                    if q: probs.append(f"outlier frames: {q}")
            nxt = [f"python3 $K sequence {_status_arg(d)} <name> render"] if any("missing" in p for p in probs) else []
            add(d, "sequence", d.name, checks, probs, nxt)
        if "tokens.json" in names and d.name == "tokens":
            data, err = _status_json(d / "tokens.json")
            probs = [err] if err else []
            if isinstance(data, dict):
                probs.extend(f"verify {k}: {v}" for k, v in (data.get("verify") or {}).items()
                             if v is False or (isinstance(v, dict) and not v.get("ok", True))
                             or (isinstance(v, str) and v.startswith("FAILED")))
            check = {"name": "tokens.json", "status": "done"}
            inputs = [p for p in (d.parent / "design_spec.json", d.parent / "fonts" / "fonts.json") if p.exists()]
            if inputs and (d / "tokens.json").stat().st_mtime < max(p.stat().st_mtime for p in inputs):
                check["status"] = "STALE"
                probs.append("STALE: tokens.json older than " + ", ".join(p.name for p in inputs if p.stat().st_mtime > (d / "tokens.json").stat().st_mtime))
            add(d, "tokens", d.name, [check], probs, [f"python3 $K tokens {_status_arg(d.parent)}"] if check["status"] == "STALE" else [])
        if "report.json" in names and d.name == "compare":
            data, err = _status_json(d / "report.json")
            probs = [err] if err else []
            score = data.get("overall", {}).get("score") if isinstance(data, dict) and isinstance(data.get("overall"), dict) else None
            if score is not None: probs.append(f"overall.score {score}/100")
            check = {"name": "report.json", "status": "done"}
            tokens = d.parent / "tokens" / "tokens.json"
            if tokens.exists() and (d / "report.json").stat().st_mtime < tokens.stat().st_mtime:
                check["status"] = "STALE"
                probs.append("STALE: compare report.json older than tokens.json")
            add(d, "compare", d.name, [check], probs,
                [f"python3 $K compare {_status_arg(d.parent)} <screenshot>"] if check["status"] == "STALE" else [])
        if "fonts.json" in names and d.name == "fonts":
            _, err = _status_json(d / "fonts.json")
            add(d, "fonts", d.name, [{"name": "fonts.json", "status": "done"}], [err] if err else [], [])
        if "redline.json" in names and d.name == "redline":
            _, err = _status_json(d / "redline.json")
            add(d, "redline", d.name, [{"name": "redline.json", "status": "done"}], [err] if err else [], [])
        if "animation_report.json" in names:
            data, err = _status_json(d / "animation_report.json")
            probs = [err] if err else []
            if isinstance(data, dict):
                for name, motion in (data.get("motion") or {}).items():
                    if isinstance(motion, dict) and motion.get("max", 1) < 0.2:
                        probs.append(f"{name}: max travel {motion.get('max')} (< 0.2)")
            add(d, "animation", d.name, [{"name": "animation_report.json", "status": "done"}], probs, [])
        for p in (p for p in children if p.name.endswith(".trace.json")):
            data, err = _status_json(p)
            probs = [err] if err else []
            if isinstance(data, dict):
                advice = data.get("advice")
                if isinstance(advice, dict) and advice.get("prefer") in ("png", "webp"):
                    probs.append(f"advice.prefer: {advice['prefer']}")
            add(d, "trace", p.stem, [{"name": p.name, "status": "done"}], probs, [])
        if "layers.json" in names:
            _, err = _status_json(d / "layers.json")
            add(d, "layers", d.name, [{"name": "layers.json", "status": "done"}], [err] if err else [], [])
        if "frames_review.png" in names:
            add(d, "frames", d.name, [{"name": "frames_review.png", "status": "done"}], [], [])
    return units


def cmd_status(a):
    base = Path(a.dir or os.getcwd()).resolve()
    units = _status_scan(base)
    result = {"dir": str(base), "units": units}
    if not units:
        result["next"] = [f"python3 $K mockup \"<brief>\" --out {_status_arg(base)}",
                           f"python3 $K research \"<idea>\" --out {_status_arg(base / 'research')}"]
        if a.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"no gptkit artifacts under {base}")
            print("next: " + " | ".join(result["next"]))
        return
    result["next"] = [command for unit in units for command in unit["next"]]
    if a.json:
        print(json.dumps(result, indent=2))
        return
    for unit in units:
        print(f"{unit['type']} {_status_arg(unit['path'])}")
        for c in unit["checks"]:
            print(f"  {c['status']:7s} {c['name']}")
        for problem in unit["problems"]:
            print(f"  ! {problem}")
        for i, command in enumerate(unit["next"] or ["nothing pending in the kit: review the reports or continue to the next phase"]):
            print(("  next: " if i == 0 else "        ") + command)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    efforts = ["high", "xhigh", "max"]
    p = sub.add_parser("cost", help="summarise API usage and estimated list-price cost")
    p.add_argument("--since", help="7d, 24h or YYYY-MM-DD")
    p.add_argument("--by", choices=["cmd", "model", "day", "dir"], default="cmd")
    p.add_argument("--json", action="store_true")
    p.add_argument("--ledger", help="ledger JSONL path (default GPTKIT_LEDGER or ~/.cache/gptkit/ledger.jsonl)")
    p.set_defaults(fn=cmd_cost)
    p = sub.add_parser("research", help="in-depth market research with web search (gpt-6-sol, high effort) -> report.md")
    p.add_argument("idea", help='e.g. "Android budgeting app for young couples"')
    p.add_argument("--out", required=True, help="output dir: report.md, report.json, plan.json, findings/, sources.json")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform (also carried into blueprint)")
    p.add_argument("--region", default="global", help='market/region to study, e.g. "US + UK" (default global)')
    p.add_argument("--context", help="extra constraints or known facts")
    p.add_argument("--depth", choices=list(DEPTHS), default="standard", help="5 / 10 / 16 research questions")
    p.add_argument("--model", default=RESEARCH_MODEL, help=f"reasoning model with web search (default {RESEARCH_MODEL})")
    p.add_argument("--effort", choices=efforts, default="high", help="reasoning effort for the plan and each question")
    p.add_argument("--synth-effort", choices=efforts, default="xhigh", help="effort for the final synthesis")
    p.add_argument("--jobs", type=_pos_int, default=4, help="parallel research questions"); p.set_defaults(fn=cmd_research)
    p.add_argument("--dry-run", action="store_true", help="show the planned research calls and estimated cost only")
    p = sub.add_parser("blueprint", help="feature blueprint (features, RICE, stories, screens with mockup briefs) from research")
    p.add_argument("dir", help="research dir with report.json (or an empty dir with --idea)")
    p.add_argument("--idea", help="product idea when DIR has no report.json (first-principles blueprint)")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform for native components (default: the report's)")
    p.add_argument("--constraints", help="constraints the blueprint must respect (team, timeline, tech, budget)")
    p.add_argument("--out", help="output dir (default DIR): blueprint.md, blueprint.json, screens.json")
    p.add_argument("--model", default=RESEARCH_MODEL, help=f"reasoning model (default {RESEARCH_MODEL})")
    p.add_argument("--effort", choices=efforts, default="xhigh", help="reasoning effort"); p.set_defaults(fn=cmd_blueprint)
    p = sub.add_parser("doctor", help="check config, API access, models and optional tools (Chrome probe, aapt2 level)")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("status", help="offline project progress, recorded problems and next commands")
    p.add_argument("dir", nargs="?", help="project root (default: current directory)")
    p.add_argument("--json", action="store_true", help="print a machine-readable status object")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("mockup", help="generate UI mockup variants (or --edit an existing one)")
    p.add_argument("prompt"); p.add_argument("--out", required=True); p.add_argument("--name", default="mockup")
    p.add_argument("--size", default="1024x1536", help="WIDTHxHEIGHT: 1024x1536 (portrait), 1536x1024, 1024x1024")
    p.add_argument("--variants", type=_pos_int, default=1, help="images generated in parallel")
    p.add_argument("--edit", help="existing mockup PNG to revise with the prompt"); p.add_argument("--raw", action="store_true")
    p.add_argument("--style-ref", help="approved screen PNG: design a NEW screen in its exact design language")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform: native components, system UI, metrics")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_mockup)
    p = sub.add_parser("critique", help="vision design QA of a mockup; --apply edits the fixes in, kept only if better")
    p.add_argument("mockup"); p.add_argument("--out", help="output dir (default: the mockup's dir)")
    p.add_argument("--style-ref", help="approved screen of the same app: also check consistency with it")
    p.add_argument("--platform", choices=list(PLATFORMS), help="default: design.json next to the mockup, else android")
    p.add_argument("--screen-dp", type=_pos_float, help="logical screen width (default: the platform's)")
    p.add_argument("--note", help="extra focus for the review")
    p.add_argument("--apply", action="store_true", help="edit the issues' fixes into the mockup, verify, keep if better")
    p.add_argument("--min-severity", choices=["high", "medium", "low"], default="medium")
    p.add_argument("--rounds", type=_pos_int, default=1, help="max --apply rounds (each kept only if the weight drops)")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_critique)
    p = sub.add_parser("manifest", help="inventory assets of an approved mockup -> DIR/assets.json")
    p.add_argument("mockup"); p.add_argument("--out", required=True)
    p.add_argument("--platform", choices=list(PLATFORMS), help="default: design.json written by mockup")
    p.add_argument("--screen-dp", type=_pos_float, help="logical screen width (default: the platform's, else 390)")
    p.add_argument("--model", default=None, help="text/vision model (default GPTKIT_TEXT_MODEL)")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("spec", help="extract palette/typography/spacing/native widgets -> DIR/design_spec.json")
    p.add_argument("dir")
    p.add_argument("--scale", choices=["auto", "native", "design"], default="auto",
                   help="px->dp from native widget sizes (auto: when GPT drew them >12%% off) or the screen width")
    p.add_argument("--recalibrate", action="store_true", help="only redo the px->dp scale of an existing spec (no API)")
    p.add_argument("--model", default=None, help="text/vision model (default GPTKIT_TEXT_MODEL)")
    p.set_defaults(fn=cmd_spec)
    p = sub.add_parser("forge", help="redraw png assets from scratch, 1:1, with candidates + scoring")
    p.add_argument("dir", help="project dir with assets.json (from manifest)")
    p.add_argument("--only", help="comma-separated asset names (default: every png-format asset without a final file)")
    p.add_argument("--candidates", type=_pos_int, default=2, help="redraws per asset, best one wins (default 2)")
    p.add_argument("--jobs", type=_pos_int, default=6, help="parallel image calls (default 6)")
    p.add_argument("--dry-run", action="store_true", help="show pending candidate calls and estimated cost only")
    p.add_argument("--force", action="store_true", help="redo assets that already have a final file")
    p.add_argument("--include-svg", action="store_true", help="also raster-forge svg-format assets")
    p.add_argument("--same-prompt", action="store_true", help="no per-candidate colour/shape emphasis")
    _color_args(p); p.add_argument("--model", default=None, help="image model (default GPTKIT_IMAGE_MODEL)"); p.set_defaults(fn=cmd_forge)
    p = sub.add_parser("fix", help="vision-diff the chosen asset against the design and apply one correction")
    p.add_argument("dir"); p.add_argument("name", help="asset name from assets.json")
    p.add_argument("--rounds", type=_pos_int, default=1, help="diff+edit rounds (default 1); stops early when nothing differs")
    p.add_argument("--note", help="extra user feedback to include"); p.add_argument("--accept", action="store_true",
                                                                                   help="keep the result even if the score is worse")
    _color_args(p); p.add_argument("--model", default=None, help="image model (default GPTKIT_IMAGE_MODEL)"); p.set_defaults(fn=cmd_fix)
    p = sub.add_parser("svg", help="write svg-format assets as SVG: scored vs the design, retried, 24-grid, previews")
    p.add_argument("dir"); p.add_argument("--only")
    p.add_argument("--no-retry", dest="retry", action="store_false", help="skip the vision-guided retry of weak icons")
    p.add_argument("--grid", type=float, default=24, help="normalise icon viewBoxes to this grid (0 = keep as drawn)")
    p.add_argument("--pad", type=float, default=2, help="grid padding around the icon's content")
    p.add_argument("--check", action="store_true", help="only score existing DIR/svg files (no API calls)")
    p.set_defaults(fn=cmd_svg)
    p = sub.add_parser("iconspec", help="extract the icon-set style spec + Material/SF symbol names -> DIR/icon_spec.json")
    p.add_argument("dir"); p.add_argument("--model", default=None, help="text/vision model (default GPTKIT_TEXT_MODEL)")
    p.set_defaults(fn=cmd_iconspec)
    p = sub.add_parser("icons", help="draw NEW icons (not in the mockup) in the design's icon style as SVG")
    p.add_argument("dir"); p.add_argument("concepts", help='comma-separated, e.g. "notifications, sleep timer, download"')
    p.add_argument("--out", help="default DIR/icons")
    p.add_argument("--treatment", choices=["auto", "outlined", "filled"], default="auto")
    p.add_argument("--stroke", type=_pos_float, help="stroke width on the 24 grid (default: measured from the set)")
    p.add_argument("--color", help="hex colour (default: the set's most common)")
    p.add_argument("--snap", action="store_true", help="force every stroke-width (attribute, style or inherited default) to the set's stroke")
    p.add_argument("--vd", help="also write VectorDrawables to this res/drawable dir")
    p.add_argument("--dp", type=_pos_float, help="VectorDrawable size in dp for --vd (default: the SVG's width)")
    p.set_defaults(fn=cmd_icons)
    p = sub.add_parser("vd", help="convert SVG files (or directories of them) to Android VectorDrawable XML")
    p.add_argument("svgs", nargs="+", help="SVG files or directories (*.svg)")
    p.add_argument("--out", required=True, help="output dir, e.g. app/src/main/res/drawable")
    p.add_argument("--dp", type=_pos_float, help="android:width in dp (default: the SVG's width, or svg_report.json's per-icon dp)")
    p.add_argument("--prefix", default="ic_", help="resource name prefix (names are lower-cased to [a-z0-9_])")
    p.add_argument("--no-check", action="store_true", help="skip the render round-trip and aapt2 validation")
    p.set_defaults(fn=cmd_vd)
    p = sub.add_parser("frames", help="frame-by-frame animation from a forged asset (moving layer sprite sheet, "
                                      "or --procedural local motion)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving", help="part that moves (generated route)")
    p.add_argument("--static", help="part that stays still (generated once, composited)")
    p.add_argument("--static-order", choices=["front", "back"], default="front")
    p.add_argument("--motion"); p.add_argument("--count", type=_pos_int, help="frames, at least 2 (default 8; procedural: one loop)")
    p.add_argument("--cols", type=_pos_int, default=4, help="sheet columns (clamped to --count)")
    p.add_argument("--anchor", choices=["bottom", "center"], default="bottom")
    p.add_argument("--fps", type=_pos_float, help="default 12 (procedural 24, or 12 for slow loops)")
    p.add_argument("--procedural", choices=sorted(MOTIONS), help="no API calls: eased affine motion of the asset")
    p.add_argument("--amount", type=_pos_float, default=1.0, help="procedural motion strength multiplier")
    p.add_argument("--reuse", choices=["all", "static"], help="reuse sheet_raw/static_raw.png from a previous run")
    p.add_argument("--reroll", action="store_true", help="redraw QA-flagged frames from a good neighbour (1 call each)")
    p.add_argument("--match-colors", action="store_true", help="shift frame colours onto the approved asset (Lab)")
    p.add_argument("--formats", default="gif,webp,apng,lottie", help="comma list of gif,webp,apng,lottie")
    p.add_argument("--px", type=_pos_int, default=480, help="long side of the animated WebP/APNG/Lottie/GIF and the "
                                                            "android/drawable-nodpi frames")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="backdrop of preview.gif and preview.html")
    p.add_argument("--model", default=None, help="image model (default $GPTKIT_IMAGE_MODEL)"); p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("sequence", help="layout-driven animation: code lays the motion out, GPT renders frames "
                                        "(stages: layout, plan, render, build | propagate, reroll)")
    p.add_argument("dir"); p.add_argument("name")
    p.add_argument("stage", choices=["layout", "plan", "render", "build", "propagate", "reroll"])
    p.add_argument("--guides", help="layout: use these guide frames (one PNG per frame, square, one size, transparent, "
                                    "naturally ordered 0.png..N.png) instead of a plume")
    p.add_argument("--base", help="layout: the locked artwork on its own canvas; build cuts it away")
    p.add_argument("--sprites", help="layout: comma list of the plume's element sprites, drawn in the art's own style")
    p.add_argument("--vent", help="layout: emission point x,y on the base canvas")
    p.add_argument("--loop", type=_pos_float, default=3.0, help="layout: loop length in seconds")
    p.add_argument("--fps", type=_pos_float, default=24)
    p.add_argument("--chuff", type=_pos_float, default=0.75, help="layout: seconds between emissions (the loop must be a multiple)")
    p.add_argument("--life", type=_pos_float, default=5.4, help="layout: seconds each element lives")
    p.add_argument("--velocity", default="0,-85", help="layout: launch velocity x,y in canvas units per second")
    p.add_argument("--spread", default="12,18", help="layout: launch jitter x,y")
    p.add_argument("--drag", type=_pos_float, default=1.5, help="layout: air drag per second toward --wind")
    p.add_argument("--wind", default="45,-3", help="layout: air velocity x,y (a moving engine's slipstream, a breeze); "
                                                  "a negative x needs the = form: --wind=-40,-3")
    p.add_argument("--buoyancy", type=float, default=-10, help="layout: vertical acceleration; negative rises")
    p.add_argument("--size", type=_pos_float, default=96, help="layout: sprite width at scale 1, canvas units")
    p.add_argument("--grow", default="0.45,1.5", help="layout: scale at birth,death")
    p.add_argument("--front", action="store_true", help="layout: draw the plume in front of the base")
    p.add_argument("--moving", default="The moving element", help="render: what the layout shows, e.g. 'The smoke'")
    p.add_argument("--locked", default="The rest of the artwork", help="render: what must not change, e.g. 'The toy steam train'")
    p.add_argument("--look", help="render: the finish to aim for (default: the same finish as the locked artwork)")
    p.add_argument("--jobs", type=_pos_int, default=12, help="render/reroll: parallel image calls; propagate: parallel EbSynth runs")
    p.add_argument("--force", action="store_true", help="render: redo frames that are already rendered; propagate: redo "
                                                        "finished syntheses")
    p.add_argument("--every", type=_pos_int, help="render/propagate: GPT renders every Nth frame only (the keys) and "
                                                  "propagate synthesises the rest with EbSynth (default: the plan's; "
                                                  "N must divide the frame count)")
    p.add_argument("--engine", choices=["auto", "ebsynth", "gpt"], default=None,
                   help="plan: auto picks EbSynth keys when they can carry the layout, else GPT renders every frame; "
                        "remembered in sequence.json for the later stages")
    p.add_argument("--ebsynth", help="propagate: path to the ebsynth binary (else $GPTKIT_EBSYNTH, PATH, ~/.cache/gptkit)")
    p.add_argument("--ebsynth-timeout", type=_pos_float, help="propagate: seconds per EbSynth run before it counts as "
                                                              "hung (default 600)")
    p.add_argument("--keep-work", action="store_true", help="propagate: keep propagate/syn_* and pos_* after a successful "
                                                            "run (a later run then makes no EbSynth runs)")
    p.add_argument("--guide-weight", type=_pos_float, default=2.0, help="propagate: weight of the appearance guide")
    p.add_argument("--position-weight", type=_pos_float, default=1.5, help="propagate: weight of the positional guide (plumes)")
    p.add_argument("--ref", help="build/propagate: approved picture of the moving part; each frame (or key) is "
                                 "colour-matched to it (remembered for later builds)")
    p.add_argument("--below", type=_pos_float, help="build/propagate: drop anything below this canvas row (under a chimney "
                                                    "mouth; remembered for later builds)")
    p.add_argument("--quality", type=_pos_int, help="build: WebP quality 1-100 (default 80, remembered)")
    p.add_argument("--frames", help="reroll: comma list of frames (default: the outliers build reported)")
    p.add_argument("--candidates", type=_pos_int, default=2, help="reroll: renders per frame to choose from")
    p.add_argument("--preview-bg", type=_color_arg, help="build: backdrop of preview.gif and contact.png (default #a0aabe, remembered)")
    p.add_argument("--model", default=None, help="render/reroll: image model (default $GPTKIT_IMAGE_MODEL)")
    p.add_argument("--dry-run", action="store_true", help="render: list pending frames and estimated cost only")
    p.set_defaults(fn=cmd_sequence)
    p = sub.add_parser("vectorize", help="vector layers (morphs, transforms, glows) -> Lottie + AnimatedVectorDrawable")
    p.add_argument("dir", help="design dir (assets.json) or any dir for the outputs with --svg / --rebuild")
    p.add_argument("name", help="asset name: DIR/assets/NAME.png or DIR/svg/NAME.svg; outputs in DIR/vector/NAME/")
    p.add_argument("--moving", help="what moves, e.g. 'the flame and its glow' (required unless --rebuild)")
    p.add_argument("--static", help="what stays still, e.g. 'the logs'")
    p.add_argument("--motion", help="how it moves, e.g. 'gentle flicker, a slow pulse'")
    p.add_argument("--keyframes", type=_pos_int, default=4, help="morph keyframes per animated layer (>= 2)")
    p.add_argument("--size", type=_pos_int, default=512, help="Lottie canvas long side in px")
    p.add_argument("--seconds", type=_pos_float, default=1.2, help="loop length (AVD loops longer than 5 s are subsampled)")
    p.add_argument("--fps", type=_pos_int, default=30, help="Lottie frame rate")
    p.add_argument("--strength", choices=list(MOTION_STRENGTH), default="normal", help="how much the moving parts move")
    p.add_argument("--min-motion", type=_unit_float, help="retry once below this share of a layer's size (default per --strength)")
    p.add_argument("--svg", help="animate this SVG (default DIR/svg/NAME.svg when DIR/assets/NAME.png is missing); no assets.json needed")
    p.add_argument("--dp", type=_pos_float, help="AnimatedVectorDrawable width in dp (default: the asset's dp, else 48)")
    p.add_argument("--rebuild", action="store_true", help="rebuild outputs from an edited layers.json, no API call")
    p.add_argument("--no-fit", action="store_true", help="keep the viewBox even if the motion leaves it (clips)")
    p.add_argument("--review-frames", type=_pos_int, default=8, help="frames in keys_review.png")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="background colour of keys_review.png and preview.html")
    p.set_defaults(fn=cmd_vectorize)
    p = sub.add_parser("export", help="export a PNG to android densities / ios imageset / web")
    p.add_argument("file", help="source image (PNG with alpha; any size)"); p.add_argument("--dp", type=_pos_float, required=True, help="width in dp/pt/css px")
    p.add_argument("--hdp", type=_pos_float, help="height in dp/pt/css px (default: keeps the aspect ratio)")
    p.add_argument("--name", help="asset name (default: the file stem; sanitised to [a-z0-9_] for Android)")
    p.add_argument("--res", required=True, help="android: res/ dir; ios: *.xcassets dir; web: output dir")
    p.add_argument("--platform", choices=["android", "ios", "web"], default="android")
    p.add_argument("--format", choices=["webp", "png"], help="android/web file format (default webp; ios is always png)")
    p.add_argument("--quality", type=int, help="WebP quality 0-100; 100 = lossless (default: lossless for <= 256 colours, else 90)")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("appicon", help="app launcher icons: concepts, then Android adaptive / iOS / web icon sets")
    isub = p.add_subparsers(dest="icmd", required=True)
    q = isub.add_parser("concepts", help="square full-bleed icon concepts + mask review sheet")
    q.add_argument("brief", help='brand brief, e.g. "Hush, a calm sleep app, moon motif"'); q.add_argument("--out", required=True)
    q.add_argument("--mockup", help="approved design to take the visual language from (sent as the single reference)")
    q.add_argument("--crop", help="x,y,w,h region of --mockup to use as the reference instead of the whole screen")
    q.add_argument("--variants", type=_pos_int, default=3, help="number of concepts (1 image call each)")
    q.add_argument("--directions", help="'|'-separated concept directions")
    q.add_argument("--text", help="wordmark/letter to include (default: no text)"); q.add_argument("--colors")
    q.add_argument("--name", default="concept", help="file stem of the concepts: <name>_v1.png, ...")
    q.add_argument("--jobs", type=_pos_int, default=2, help="concurrent image calls")
    q.add_argument("--fidelity", choices=["high", "low"], default="low"); q.add_argument("--model", default=None)
    q.set_defaults(fn=cmd_appicon_concepts)
    q = isub.add_parser("build", help="Android adaptive (fg/bg/monochrome) + legacy + Play, iOS AppIcon, web icons")
    q.add_argument("concept", help="chosen square concept PNG"); q.add_argument("--out", required=True)
    q.add_argument("--subject", default="the main subject", help='what the foreground is, e.g. "the crescent moon and stars"')
    q.add_argument("--separate", choices=["auto", "local", "api"], default="auto",
                   help="foreground: local matte against the shipped background (exact, 0 calls), an image-edit redraw "
                        "(1 call), or auto: local unless it leaves the subject mostly semi-transparent")
    q.add_argument("--bg", choices=["auto", "color", "gradient", "model", "image"], default="auto",
                   help="background layer: colour resource, VectorDrawable gradient, fitted bitmap, or an image-edit "
                        "plate (1 call; auto picks it only for busy backgrounds)")
    q.add_argument("--bg-color", help="hex for --bg color (default: sampled)")
    q.add_argument("--mono", choices=["glyph", "silhouette", "both", "none"], default="glyph",
                   help="themed-icon layer: gpt-6 glyph -> VectorDrawable (1 text call, falls back to the silhouette), "
                        "the foreground silhouette bitmap, or both (glyph referenced)")
    q.add_argument("--fg-scale", type=_pos_float, default=1.0, help="fraction of the 66dp safe circle the subject may fill")
    q.add_argument("--ios-from", choices=["concept", "composite"], default="concept")
    q.add_argument("--name", default="ic_launcher", help="Android resource name ([a-z0-9_]): <name>, <name>_round, ...")
    q.add_argument("--no-validate", dest="validate", action="store_false", help="skip the aapt2 / actool checks")
    q.add_argument("--model", default=None); q.set_defaults(fn=cmd_appicon_build)
    p = sub.add_parser("pick", help="override the chosen candidate (the cN number from the review-sheet label, or an image file)")
    p.add_argument("dir"); p.add_argument("name", help="asset name from assets.json")
    p.add_argument("candidate", help="N of the tile labelled cN in review/forge_*.png, or a path to any PNG")
    _color_args(p); p.set_defaults(fn=cmd_pick)
    _design_to_code_parsers(sub)
    _hybrid_parsers(sub)
    p = sub.add_parser("score", help="score arbitrary PNG or SVG files against a manifest asset (JSON per file, no API)")
    p.add_argument("dir"); p.add_argument("name", help="asset name from assets.json"); p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_score)
    _add_trace_rig_parsers(sub)
    a = ap.parse_args()
    global _LEDGER_CMD, _LEDGER_DIR
    _LEDGER_CMD = ("sequence " + a.stage) if a.cmd == "sequence" else (
        "appicon " + a.icmd if a.cmd == "appicon" else a.cmd)
    dir_arg = getattr(a, "dir", None) or getattr(a, "out", None)
    _LEDGER_DIR = str(Path(dir_arg).expanduser().resolve()) if dir_arg else None
    try:
        a.fn(a)
    except KeyboardInterrupt:
        if __name__ == "__main__":  # the CLI: pool threads still inside a 900 s API call would hold the exit until they return
            try:  # the interrupt may have landed inside a print: never let a flush error skip the exit
                sys.stdout.flush()
                sys.stderr.flush()
            except BaseException:
                pass
            finally:
                os.write(2, f"gptkit {a.cmd}: interrupted\n".encode())
                os._exit(130)
        print(f"gptkit {a.cmd}: interrupted", file=sys.stderr)
        sys.exit(130)
    except (ApiError, RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError,
            ArithmeticError, MemoryError) as e:
        debug = os.environ.get("GPTKIT_DEBUG", "").lower() in _TRUE
        try:
            debug = debug or _config()["debug"]
        except SystemExit:
            pass
        if debug:
            raise
        msg = (f"missing key {e.args[0]!r} (older JSON file or non-conforming model response?)" if isinstance(e, KeyError)
               else " ".join(str(e).split()) or type(e).__name__)
        if isinstance(e, (FileExistsError, NotADirectoryError)):
            msg += " (a file is where the kit needs a directory: check --out/--res and the project dir)"
        if isinstance(e, MemoryError):
            msg = "out of memory (an input image, or a size/--px/--dp flag, is far larger than the kit needs)"
        elif not isinstance(e, (ApiError, RuntimeError, OSError, ValueError, KeyError)):  # a bug or a JSON file of the wrong shape
            msg = (f"unexpected {type(e).__name__}: {msg} (a hand-edited or older JSON file in the project? re-create it; "
                   "else re-run with GPTKIT_DEBUG=1 and report the traceback)")
        sys.exit(f"gptkit {a.cmd}: {msg}")


if __name__ == "__main__":
    main()
