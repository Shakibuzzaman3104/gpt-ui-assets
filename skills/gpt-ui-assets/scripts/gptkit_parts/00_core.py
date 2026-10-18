# gptkit part 00_core: the API layer: config, retries, image/text calls, model routing, cost ledger, shared helpers.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

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
    def area(s):
        try:
            w, h = [int(x) for x in str(s).split("x")]
            return w * h
        except ValueError:
            return 1024 * 1024
    sizes = list(size) if isinstance(size, (list, tuple)) else [size]  # one size per call when they differ (forge)
    mean_area = sum(area(s) for s in sizes) / max(1, len(sizes))
    per_image = (1756 * mean_area / 1024 ** 2 * ip.get("image_out", 30) + 300 * ip.get("text_in", 5)
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
