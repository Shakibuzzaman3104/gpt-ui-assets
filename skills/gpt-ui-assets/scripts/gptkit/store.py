"""Store screenshots: capture real screens, plan the set, paint plates and lettering, composite pixel-exact, check."""
import concurrent.futures as cf
import functools
import http.server
import io
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .core import ApiError, _plan, _pos_float, _pos_int, ask, edit, generate
from .imaging import _lab
from .frames import _decode_result
from .previews import _chrome
from .hybrid import _QuietHandler
from .research import _B, _N, _S, _SA, _mdcell, _obj

# ---------------------------------------------------------------- store screenshots
# GPT redraws any UI it is shown, so the app's screens are never sent through an image model: real captures are
# composited pixel-exact in HTML (one Playwright browser), and GPT paints only what sits behind and above them, the
# art plates and the optional display lettering. Copy, sizes, contrast and fidelity are checked locally.

TARGETS = {
    "ios-phone": {"store": "ios", "device": "phone", "css": (402, 874), "dpr": 3},  # the required medium Dynamic Island slot
    "ios-phone-large": {"store": "ios", "device": "phone", "css": (440, 956), "dpr": 3, "optional": True},
    "ios-tablet": {"store": "ios", "device": "tablet", "css": (1032, 1376), "dpr": 2},
    "ios-tablet-landscape": {"store": "ios", "device": "tablet", "css": (1376, 1032), "dpr": 2},
    "play-phone": {"store": "play", "device": "phone", "css": (360, 640), "dpr": 4},
    "play-tablet": {"store": "play", "device": "tablet", "css": (720, 1280), "dpr": 2},
    "play-tablet-landscape": {"store": "play", "device": "tablet", "css": (1280, 720), "dpr": 2},
}
FEATURE_SIZE = (1024, 500)
MAX_BYTES = 8 * 1024 * 1024
CONTRAST_MIN = 4.5
FID_MEAN_MAX, FID_P99_MAX = 2.5, 20.0
SEARCH_SCALE, SEARCH_MIN_PT = 0.261, 9.0  # an iPhone search result shows a portrait shot at ~0.26 of its width
TAGLINE_MAX = 0.20  # Play: taglines under 20% of the image
TAGLINE_FIT, HEAD_MAX = 0.18, 13.0  # headlines grow to HEAD_MAX (% of width) while Play's tagline stays under 18%
BARE_W = 420  # captionless copies for the hide-the-copy check, about search-result size
FLOATS_MAX, FLOAT_COVER_MAX = 3, 0.03  # floating art per slot, and the share of the screen front floats may hide
GPT_SIZES = {(1024, 1024), (1024, 1536), (1536, 1024), (1024, 1792), (1792, 1024), (1152, 2496), (2496, 1152)}
IOS_PHONE_SIZES = {(1320, 2868), (1290, 2796), (1260, 2736), (1206, 2622), (1179, 2556), (1284, 2778), (1170, 2532),
                   (1242, 2688), (1125, 2436), (1080, 2340), (828, 1792), (750, 1334), (1242, 2208), (640, 1136)}
IOS_TABLET_SIZES = {(2064, 2752), (2048, 2732), (1668, 2420), (1668, 2388), (1640, 2360), (1620, 2160), (1488, 2266),
                    (1536, 2048), (1668, 2224)}

PLAY_WORDS = re.compile(r"(#\s?1\b|\bno\.?\s?1\b|\bnumber one\b|\bbest\b|\btop\b|\bnew\b|\bfree\b|\bsale\b|\bdiscount\b|"
                        r"\bdeal\b|% off|\baward|\beditor'?s choice\b|\bmillions?\b|\branked\b)", re.I)
CTA_WORDS = re.compile(r"\b(download|install|play now|get it|try (?:it )?now|sign up|subscribe|tap here|click)\b", re.I)
CLAIM_WORDS = re.compile(r"(#\s?1\b|\bno\.?\s?1\b|\bnumber one\b|\bbest\b|\baward|\beditor'?s choice\b|\bmillions?\b|"
                         r"\bleading\b|\btrusted by\b|\bworld'?s\b)", re.I)
PRICE = re.compile(r"[$£€¥₹]\s?\d|\d\s?(usd|eur|gbp)\b|\bper month\b|/mo\b", re.I)
ANDROID_WORDS = re.compile(r"\b(android|google play|play store|pixel|galaxy)\b", re.I)
TUTORIAL = re.compile(r"\b(?:tap|swipe|click)\s+(?:the|a|an|on|to|here|through|left|right|up|down)\b|"
                      r"\bstep\s?(?:\d+|one|two|three)\b|\bhow to\b", re.I)
APPLE_WORDS = re.compile(r"\b(iphone|ipad|ios|app store|apple)\b", re.I)


def _sh(args, binary=False, timeout=60):
    """Run a device tool; its stdout (bytes with binary=True). RuntimeError naming the command when it fails."""
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(f"store capture: {args[0]} not found (iOS needs Xcode's xcrun, Android needs adb on PATH)")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"store capture: {' '.join(args[:4])} timed out after {timeout}s")
    if r.returncode:
        err = (r.stderr or b"").decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"store capture: {' '.join(args[:4])} failed: {err[-1] if err else r.returncode}")
    return r.stdout if binary else r.stdout.decode(errors="replace")


def _fetch(url, timeout=30):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "gptkit-store/1"})
    if r.status_code >= 400:
        raise RuntimeError(f"store research: HTTP {r.status_code} for {url.split('?')[0]}")
    return r.content


def _store_json(path, what):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise RuntimeError(f"store: no {what} at {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"store: {path} is not valid JSON ({e.msg}, line {e.lineno}); fix or regenerate it")


def _captures(d):
    f = Path(d) / "captures.json"
    if not f.exists():
        return []
    caps = _store_json(f, "captures.json").get("captures", [])
    if not isinstance(caps, list):
        raise RuntimeError(f"store: {f} has no captures list; re-run `store capture`")
    return caps


def _save_captures(d, caps):
    caps = sorted(caps, key=lambda c: (c["platform"], c["device"], c["orientation"], c["name"]))
    (Path(d) / "captures.json").write_text(json.dumps({"captures": caps}, indent=1))


def _storyboard_path(d, locale=None):
    return Path(d) / (f"storyboard.{locale}.json" if locale else "storyboard.json")


def _storyboard(d, locale=None):
    f = _storyboard_path(d, locale)
    sb = _store_json(f, f"storyboard (run `store plan {d}` first)" if not locale else f"storyboard for locale {locale}")
    if not isinstance(sb.get("slots"), list) or not sb["slots"]:
        raise RuntimeError(f"store: {f} has no slots")
    for s in sb["slots"]:
        for k in ("id", "capture", "caption"):
            if not s.get(k):
                raise RuntimeError(f"store: a slot in {f} has no {k!r}")
        if not re.fullmatch(r"[a-z0-9_]+", s["id"]):
            raise RuntimeError(f"store: slot id {s['id']!r} in {f} must be [a-z0-9_]")
    return sb


# ---------------------------------------------------------------- capture

def _status_bar_px(platform, device, w, h, serial=None):
    """Height of the system status bar in capture px (cropped off frameless cards)."""
    if platform == "ios":
        if device == "tablet":
            return round(24 * (2 if min(w, h) >= 1400 else 1))
        return round(h * 0.066) if h / w > 2 else round(20 * (3 if w >= 1080 else 2))
    if serial:
        try:
            out = _sh(["adb", "-s", serial, "shell", "dumpsys", "window"], timeout=20)
            m = re.search(r"(?:ITYPE_STATUS_BAR|statusBars)[^\n]*?frame=\[0,0\]\[\d+,(\d+)\]", out)
            if m:
                return int(m.group(1))
        except RuntimeError:
            pass
    return round(max(w, h) * 0.032)


def _booted_ios():
    data = json.loads(_sh(["xcrun", "simctl", "list", "devices", "booted", "-j"]) or "{}")
    return [d for devs in data.get("devices", {}).values() for d in devs if d.get("state") == "Booted"]


def _adb_devices():
    try:
        out = _sh(["adb", "devices"], timeout=20)
    except RuntimeError:
        return []
    return [ln.split("\t")[0] for ln in out.splitlines()[1:] if ln.strip().endswith("\tdevice")]


def _capture_ios(udid, a):
    clean = a.status_bar
    if clean:
        _sh(["xcrun", "simctl", "status_bar", udid, "override", "--time", a.time, "--dataNetwork", "wifi",
             "--wifiMode", "active", "--wifiBars", "3", "--cellularMode", "active", "--cellularBars", "4",
             "--operatorName", "", "--batteryState", "discharging", "--batteryLevel", "100"])
    try:
        if a.url:
            _sh(["xcrun", "simctl", "openurl", udid, a.url])
        if a.wait or a.url:
            time.sleep(a.wait if a.wait is not None else 2.0)
        tmp = Path(a.dir) / "captures" / f".{a.name}.tmp.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        _sh(["xcrun", "simctl", "io", udid, "screenshot", "--type=png", str(tmp)])
        img = Image.open(tmp)
        img.load()
        tmp.unlink(missing_ok=True)
        return img
    finally:
        if clean:
            _sh(["xcrun", "simctl", "status_bar", udid, "clear"])


def _demo(serial, *extras):
    _sh(["adb", "-s", serial, "shell", "am", "broadcast", "-a", "com.android.systemui.demo", *extras], timeout=20)


def _capture_android(serial, a):
    clean = a.status_bar
    if clean:
        _sh(["adb", "-s", serial, "shell", "settings", "put", "global", "sysui_demo_allowed", "1"])
        _demo(serial, "-e", "command", "enter")
        _demo(serial, "-e", "command", "clock", "-e", "hhmm", a.time.replace(":", "").zfill(4))
        _demo(serial, "-e", "command", "battery", "-e", "level", "100", "-e", "plugged", "false")
        _demo(serial, "-e", "command", "network", "-e", "wifi", "show", "-e", "level", "4")
        _demo(serial, "-e", "command", "network", "-e", "mobile", "show", "-e", "datatype", "none", "-e", "level", "4")
        _demo(serial, "-e", "command", "notifications", "-e", "visible", "false")
    try:
        if a.url:
            _sh(["adb", "-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", a.url])
        if a.wait or a.url:
            time.sleep(a.wait if a.wait is not None else 2.0)
        raw = _sh(["adb", "-s", serial, "exec-out", "screencap", "-p"], binary=True)
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except OSError:
            raise RuntimeError("store capture: adb screencap returned no PNG (is the screen on and unlocked?)")
        return img
    finally:
        if clean:
            _demo(serial, "-e", "command", "exit")


def _android_kind(serial, w, h):
    try:
        dens = re.findall(r"(\d+)", _sh(["adb", "-s", serial, "shell", "wm", "density"], timeout=20))
        dpi = int(dens[-1]) if dens else 420
    except RuntimeError:
        dpi = 420
    return "tablet" if min(w, h) / (dpi / 160) >= 600 else "phone"


def _record_capture(d, caps, img, name, platform, device, source, status=None):
    img = img.convert("RGB")
    w, h = img.size
    orient = "landscape" if w > h else "portrait"
    rel = Path("captures") / f"{platform}-{device}{'-landscape' if orient == 'landscape' else ''}" / f"{name}.png"
    (Path(d) / rel).parent.mkdir(parents=True, exist_ok=True)
    img.save(Path(d) / rel)
    rec = {"name": name, "platform": platform, "device": device, "orientation": orient, "size": [w, h],
           "status_bar": status if status is not None else _status_bar_px(platform, device, w, h),
           "file": str(rel), "source": source, "at": datetime.now().isoformat(timespec="seconds")}
    caps[:] = [c for c in caps if not (c["name"] == name and c["platform"] == platform and c["device"] == device
                                       and c["orientation"] == orient)] + [rec]
    return rec


def _import_folder(a, caps):
    src = Path(a.from_dir)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if src.is_dir() else []
    if not files:
        raise RuntimeError(f"store capture: no PNG/JPEG screenshots in {src}")
    done = 0
    for f in files:
        img = Image.open(f)
        img.load()
        w, h = img.size
        key = (min(w, h), max(w, h))
        if (w, h) in GPT_SIZES:
            print(f"! {f.name}: {w}x{h} is an image-model size, not a device capture; skipped (real captures only)")
            continue
        platform = a.platform if a.platform != "auto" else (
            "ios" if key in IOS_PHONE_SIZES | IOS_TABLET_SIZES else None)
        if not platform:
            raise RuntimeError(f"store capture: can't tell the platform of {f.name} ({w}x{h}); pass --platform ios|android")
        device = a.kind if a.kind != "auto" else (
            "tablet" if key in IOS_TABLET_SIZES or max(w, h) / min(w, h) < 1.7 else "phone")
        if platform == "ios" and key not in IOS_PHONE_SIZES | IOS_TABLET_SIZES:
            print(f"! {f.name}: {w}x{h} is not a native iPhone/iPad capture size (scaled or cropped?); kept, check it")
        name = re.sub(r"[^a-z0-9_]+", "_", f.stem.lower()).strip("_") or "screen"
        rec = _record_capture(a.dir, caps, img, name, platform, device, f"import:{f}")
        print(f"  {name}: {platform} {device} {rec['orientation']} {w}x{h}")
        done += 1
    return done


def cmd_store_capture(a):
    """Capture the current screen of a booted iOS simulator or Android device with a clean status bar, or import a
    folder of real captures; records each in DIR/captures.json."""
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    caps = _captures(d)
    if a.from_dir:
        n = _import_folder(a, caps)
        _save_captures(d, caps)
        print(f"store capture: imported {n} capture(s) -> {d / 'captures.json'}")
        return
    if not a.name:
        raise ValueError("store capture: pass --name NAME for this screen (or --from FOLDER to import captures)")
    if not re.fullmatch(r"[a-z0-9_]+", a.name):
        raise ValueError(f"store capture: --name {a.name!r} must be [a-z0-9_]")
    platform = a.platform
    if platform == "auto":
        ios = a.udid or (_booted_ios() if shutil.which("xcrun") else [])
        android = a.serial or (_adb_devices() if shutil.which("adb") else [])
        if ios and android and not (a.udid or a.serial):
            raise RuntimeError("store capture: both an iOS simulator and an Android device are up; pass --platform")
        platform = "ios" if ios else "android" if android else None
        if not platform:
            raise RuntimeError("store capture: no booted iOS simulator or Android device found")
    if platform == "ios":
        sims = [] if a.udid else _booted_ios()
        if not a.udid and len(sims) != 1:
            raise RuntimeError(f"store capture: {len(sims)} booted simulators; pass --udid "
                               f"({', '.join(s['name'] + ' ' + s['udid'] for s in sims) or 'boot one first'})")
        udid = a.udid or sims[0]["udid"]
        sim_name = next((s["name"] for s in sims if s["udid"] == udid), "")
        img = _capture_ios(udid, a)
        device = a.kind if a.kind != "auto" else ("tablet" if "ipad" in sim_name.lower() or (
            (min(img.size), max(img.size)) in IOS_TABLET_SIZES) else "phone")
        rec = _record_capture(d, caps, img, a.name, "ios", device, f"simctl:{sim_name or udid}")
    else:
        devs = [] if a.serial else _adb_devices()
        if not a.serial and len(devs) != 1:
            raise RuntimeError(f"store capture: {len(devs)} adb devices; pass --serial ({', '.join(devs) or 'none'})")
        serial = a.serial or devs[0]
        img = _capture_android(serial, a)
        device = a.kind if a.kind != "auto" else _android_kind(serial, *img.size)
        rec = _record_capture(d, caps, img, a.name, "android", device, f"adb:{serial}",
                      status=_status_bar_px("android", device, *img.size, serial=serial))
    _save_captures(d, caps)
    print(f"store capture: {rec['name']} {rec['platform']} {rec['device']} {rec['orientation']} "
          f"{rec['size'][0]}x{rec['size'][1]} -> {d / rec['file']}\n"
          "  Stores ask for fictional account data (Apple 2.3.9): sign in as a demo persona, not a real account.")


# ---------------------------------------------------------------- research

PLAYBOOK_SCHEMA = _obj({
    "category": _S, "slot1": _S, "caption_voice": _S, "caption_case": _S, "caption_position": _S, "frame_style": _S,
    "background": _S, "palette": _S, "density": _S, "callouts": _S, "common": _SA, "avoid": _SA,
    "recommendations": _SA,
    "apps": {"type": "array", "items": _obj({"name": _S, "slot1_type": _S, "frame": _S, "caption_example": _S})}})


def _contact(rows, out, h=520, label_h=34):
    """rows: [(label, [PIL images])] -> one JPEG, a labelled row per app."""
    widths = [sum(round(im.width * h / im.height) + 8 for im in ims) for _, ims in rows]
    sheet = Image.new("RGB", (max(widths + [400]) + 8, len(rows) * (h + label_h + 8) + 8), "white")
    dr = ImageDraw.Draw(sheet)
    y = 8
    for label, ims in rows:
        dr.text((8, y + 8), label[:120], fill=(20, 20, 20))
        x = 8
        for im in ims:
            t = im.convert("RGB").resize((round(im.width * h / im.height), h), Image.LANCZOS)
            sheet.paste(t, (x, y + label_h))
            x += t.width + 8
        y += h + label_h + 8
    sheet.save(out, quality=88)
    return out


def cmd_store_research(a):
    """Top App Store results for a search term: slots 1-3 of each, contact sheets, and a vision-coded playbook of what
    the category's listings do (slot-1 type, caption voice, frame, palette)."""
    from urllib.parse import quote
    d = Path(a.dir) / "research"
    (d / "shots").mkdir(parents=True, exist_ok=True)
    url = (f"https://itunes.apple.com/search?term={quote(a.term)}&country={a.country}&entity=software"
           f"&limit={min(200, a.limit * 2)}")
    try:
        found = json.loads(_fetch(url)).get("results", [])
    except json.JSONDecodeError:
        raise RuntimeError("store research: the iTunes Search API did not return JSON; try again later")
    apps = [r for r in found if r.get("screenshotUrls")][:a.limit]
    if not apps:
        raise RuntimeError(f"store research: no App Store results with screenshots for {a.term!r} in {a.country}")
    rows, meta = [], []
    for r in apps:
        ims = []
        for i, u in enumerate(r["screenshotUrls"][:a.per_app]):
            f = d / "shots" / f"{r['trackId']}_{i + 1}.jpg"
            if not f.exists():
                try:
                    f.write_bytes(_fetch(u))
                except (RuntimeError, OSError) as e:
                    print(f"! {r.get('trackName')}: shot {i + 1} not downloaded ({e})")
                    continue
            try:
                ims.append(Image.open(f))
            except OSError:
                continue
        if ims:
            label = f"{r.get('trackName', '?')} · {r.get('primaryGenreName', '')} · {r.get('userRatingCount', 0):,} ratings"
            rows.append((label, ims))
            meta.append({"name": r.get("trackName"), "id": r.get("trackId"), "genre": r.get("primaryGenreName"),
                         "ratings": r.get("userRatingCount"), "rating": r.get("averageUserRating"),
                         "url": r.get("trackViewUrl"), "shots": len(ims)})
    (d / "apps.json").write_text(json.dumps(meta, indent=1))
    sheets = [_contact(rows[i:i + 6], d / f"competitors_{i // 6 + 1}.jpg") for i in range(0, len(rows), 6)]
    print(f"store research: {len(rows)} apps, sheets {', '.join(str(s) for s in sheets)}")
    if a.no_ai:
        return
    _plan(0, 1, a.model, tokens_in=8000 + 6000 * len(sheets), tokens_out=3000)
    prompt = (f"These are the first {a.per_app} App Store screenshots of the top results for '{a.term}' "
              f"({a.country.upper()}), one labelled row per app. Code what the category's listings do, as a designer "
              "would brief a new app in this category: the slot-1 type (benefit headline + UI, brand statement, social "
              "proof, key art, lifestyle photo, before/after), caption voice and case, caption position, frame style "
              "(device frame, frameless cards, full-bleed), background treatment, palette, density, use of zoomed "
              "callouts, what most do (common), what to avoid, and 5-8 concrete recommendations. Describe only what "
              "is visible; one entry per app in apps.")
    pb = ask([prompt, *[Image.open(s) for s in sheets]], schema=PLAYBOOK_SCHEMA, name="playbook", model=a.model)
    pb["term"], pb["country"] = a.term, a.country
    (Path(a.dir) / "playbook.json").write_text(json.dumps(pb, indent=1))
    md = [f"# Store playbook: {a.term} ({a.country.upper()})", ""]
    for k in ("category", "slot1", "caption_voice", "caption_case", "caption_position", "frame_style", "background",
              "palette", "density", "callouts"):
        md.append(f"- **{k.replace('_', ' ')}**: {pb.get(k, '')}")
    for k in ("common", "avoid", "recommendations"):
        md += ["", f"## {k.title()}", *[f"- {x}" for x in pb.get(k, [])]]
    md += ["", "| App | Slot 1 | Frame | Caption |", "|---|---|---|---|"]
    md += [f"| {_mdcell(x['name'])} | {_mdcell(x['slot1_type'])} | {_mdcell(x['frame'])} | {_mdcell(x['caption_example'])} |"
           for x in pb.get("apps", [])]
    (Path(a.dir) / "playbook.md").write_text("\n".join(md) + "\n")
    print(f"  playbook -> {Path(a.dir) / 'playbook.md'}")


# ---------------------------------------------------------------- plan

_BOX = {"type": "array", "items": _N}
STORYBOARD_SCHEMA = _obj({
    "app": _S, "category": _S, "voice": _S, "promise": _S,
    "brand": _obj({"colors": _SA, "ink_dark": _S, "ink_light": _S, "head_font": _S, "body_font": _S,
                   "art_style": _S, "lettering_style": _S}),
    "slots": {"type": "array", "items": _obj({
        "id": _S, "capture": _S, "caption": _S, "sub": _S, "caption_play": _S, "sub_play": _S,
        "arch": {"type": "string", "enum": ["framed", "callout", "fan"]},
        "ink": {"type": "string", "enum": ["dark", "light"]},
        "calls": {"type": "array", "items": _obj({"capture": _S, "box": _BOX})},
        "cards": {"type": "array", "items": _obj({"capture": _S, "box": _BOX})},
        "plate_brief": _S, "alt": _S, "why": _S})},
    "feature": _obj({"subtitle": _S, "plate_brief": _S}),
    "privacy": {"type": "array", "items": _obj({"capture": _S, "what": _S})},
    "notes": _SA})

PLAN_PROMPT = """You are planning the App Store and Google Play screenshot set for this app from its real screen captures.
App: {app}
{playbook}{claims}
Captures (name, platform, device, size) follow, each image after its label. Plan {slots} slots, most important first.
Rules:
- Slots 1-3 are the ad: they show in search results and are all most visitors see. One idea per slot.
- promise: one line, what the app does and the outcome its users want, for whom. Slot 1 says it, so a shopper who
  sees only slot 1 knows what the app is. Slots 2-3 back it up as a story (the key moment, then the payoff), not a
  tutorial: no "tap", "swipe" or "step 1".
- caption: 2-6 words selling the outcome people want (calmer nights, money that arrives), not the feature's name,
  in words users would say and in the app's own voice. Shorter is better: a short caption is set bigger. No outcome
  the app can't back (cures, guaranteed results). sub: one short sentence, at most 12 words, naming the feature
  that delivers it.
- caption_play/sub_play: the same idea for Google Play with no ranking, award, price or promo words (best, #1, top,
  new, free, sale, million) and no calls to action (download, play now). Repeat caption/sub when they already comply.
- No prices anywhere. No claims (#1, best, award, million users) unless they are in the allowed claims above.
- arch: framed (the whole screen in a device), callout (framed plus 1-2 enlarged crops of the one element that proves
  the caption; calls[].box is [x0,y0,x1,y1] as fractions 0-1 of that capture, a full-width band reads best; keep it
  between 18% and 85% of the capture's height, at most 45% tall and no thinner than about 3.5:1), or fan
  (2-3 screens as tilted cards, cards[0] is the hero; boxes as fractions, the region worth showing). Use fan at most
  once, and only with several captures. Leave calls/cards empty when unused.
- capture: the capture name that slot shows. Prefer screens doing their job with real content (a list with items, a
  result, a game mid-play) over near-empty ones (a blank form, a bare question, a loading state).
- plate_brief: the background scene in the app's own art style; the top third must stay calm for the caption. For an
  app with no illustration style, describe a soft abstract treatment in the brand colours. Never repeat the scenery
  or imagery shown inside the slot's capture: the plate is a different, simpler place. No characters or mascots
  unless they are the app's own art. ink: dark text on a light
  plate top, light text on a dark one.
- alt: what the screenshot shows, under 140 characters, not starting with "Image of".
- brand: colours (hex) taken from the captures, ink_dark (a deep brand tone for text) and ink_light (a near-white or
  cream from the app), the closest Google Fonts to the app's display and body type, art_style (one line describing
  the app's illustration look) and lettering_style (one line: how display lettering in this style would look).
- feature: a Google Play feature graphic subtitle (2-5 words) and its plate_brief (calm centre for the logo).
- privacy: any real-looking personal data visible in a capture (names, handles, emails, photos of people); stores
  require fictional account data.
- notes: anything missing from the captures that the set needs (a dark-mode screen, a result screen, a tablet).
"""


def _lint(sb, claims=""):
    """[(level, slot id, message)] for store copy rules; level is FAIL or WARN."""
    out = []
    allowed = (claims or "").lower()
    for s in sb.get("slots", []):
        sid = s.get("id", "?")
        ios_text = " ".join(x for x in (s.get("caption"), s.get("sub")) if x)
        play_text = " ".join(x for x in (s.get("caption_play") or s.get("caption"), s.get("sub_play") or s.get("sub")) if x)
        for store, text in (("ios", ios_text), ("play", play_text)):
            if PRICE.search(text):
                out.append(("FAIL", sid, f"{store}: a price in the copy ({PRICE.search(text).group(0)!r}); stores reject prices (Apple 2.3.7)"))
            for m in CLAIM_WORDS.finditer(text):
                if m.group(0).lower() not in allowed:
                    out.append(("WARN", sid, f"{store}: claim {m.group(0)!r} needs proof; pass it in --claims if you can back it (Apple 2.3.14)"))
        if ANDROID_WORDS.search(ios_text):
            out.append(("FAIL", sid, f"ios: names another platform ({ANDROID_WORDS.search(ios_text).group(0)!r}; Apple 2.3.10)"))
        if APPLE_WORDS.search(play_text):
            out.append(("WARN", sid, f"play: names Apple ({APPLE_WORDS.search(play_text).group(0)!r})"))
        for m in PLAY_WORDS.finditer(play_text):
            if m.group(0).lower() not in allowed:
                out.append(("WARN", sid, f"play: {m.group(0)!r} is a ranking/promo word Play asks listings to avoid"))
        if CTA_WORDS.search(play_text):
            out.append(("WARN", sid, f"play: call to action {CTA_WORDS.search(play_text).group(0)!r}; Play asks for none"))
        if TUTORIAL.search(ios_text):
            out.append(("WARN", sid, f"{TUTORIAL.search(ios_text).group(0)!r} reads as a tutorial; show what the user "
                                     "gets, not how to use the app"))
        if len((s.get("caption") or "").split()) > 7:
            out.append(("WARN", sid, f"caption has {len(s['caption'].split())} words; 2-6 read at search size"))
        if len((s.get("sub") or "").split()) > 14:
            out.append(("WARN", sid, f"sub has {len(s['sub'].split())} words; keep it under 12"))
        for c in s.get("calls") or []:
            x0, y0, x1, y1 = c.get("box") or (0, 0, 1, 1)
            if y0 < 0.18 or y1 > 0.85 or y1 - y0 > 0.45:
                out.append(("WARN", sid, f"callout box {c.get('box')} sits at the screen's top or bottom or is too tall; "
                                         "an enlarged card can't sit over its source there (keep it within 18-85%, <=45% tall)"))
        alt = s.get("alt") or ""
        if not alt:
            out.append(("WARN", sid, "no alt text (Play reads it to screen-reader users)"))
        elif len(alt) > 140:
            out.append(("FAIL", sid, f"alt text is {len(alt)} chars; Play allows 140"))
        elif re.match(r"\s*(an? )?(image|screenshot|picture) of", alt, re.I):
            out.append(("WARN", sid, "alt text starts with 'image of'; describe the content"))
    for p in sb.get("privacy", []) or []:
        out.append(("WARN", p.get("capture", "?"), f"personal data on screen: {p.get('what')}; recapture with a demo persona (Apple 2.3.9)"))
    return out


def _print_lint(issues):
    for level, sid, msg in issues:
        print(f"  {level} {sid}: {msg}")
    if not issues:
        print("  copy lint: clean")


def _fraction_box(box, w, h):
    """A model box (fractions, or px when any value > 1) -> clamped capture px [x0,y0,x1,y1], or None."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        b = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in b):
        return None
    if max(b) <= 1.0:
        b = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    x0, y0, x1, y1 = max(0, min(b[0], b[2])), max(0, min(b[1], b[3])), min(w, max(b[0], b[2])), min(h, max(b[1], b[3]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]


def _storyboard_md(sb, issues):
    md = [f"# {sb.get('app', 'App')} store storyboard", "", f"Promise: {sb.get('promise', '')}  ",
          f"Voice: {sb.get('voice', '')}  ", f"Category: {sb.get('category', '')}", "",
          "| # | Slot | Capture | Arch | Caption | Sub | Play caption | Alt |", "|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(sb["slots"], 1):
        md.append(f"| {i} | {s['id']} | {s['capture']} | {s.get('arch', 'framed')} | {_mdcell(s['caption'])} | "
                  f"{_mdcell(s.get('sub', ''))} | {_mdcell(s.get('caption_play') or s['caption'])} | {_mdcell(s.get('alt', ''))} |")
    if sb.get("notes"):
        md += ["", "## Notes", *[f"- {n}" for n in sb["notes"]]]
    md += ["", "## Copy lint", *([f"- {lv} {sid}: {m}" for lv, sid, m in issues] or ["- clean"])]
    return "\n".join(md) + "\n"


def _primary(caps, name):
    """The capture a slot's name means for art and planning: iOS phone first, then Android phone, then any."""
    order = {("ios", "phone"): 0, ("android", "phone"): 1}
    found = [c for c in caps if c["name"] == name]
    return min(found, key=lambda c: (order.get((c["platform"], c["device"]), 2), c["orientation"] != "portrait")) if found else None


def cmd_store_plan(a):
    """A storyboard from the captures (one vision call): slots, captions per store, archetypes, callout boxes, plate
    briefs and alt text, then a copy lint. --lint re-lints an edited storyboard.json without a call."""
    d = Path(a.dir)
    if a.lint:
        sb = _storyboard(d, a.locale)
        issues = _lint(sb, sb.get("claims", ""))
        _storyboard_path(d, a.locale).with_suffix(".md").write_text(_storyboard_md(sb, issues))
        _print_lint(issues)
        return
    if not a.app:
        raise ValueError("store plan: pass --app 'Name: what it does, for whom' (or --lint to check an edited storyboard)")
    caps = _captures(d)
    if not caps:
        raise RuntimeError(f"store plan: no captures in {d}; run `store capture {d} --name NAME` first")
    names = sorted({c["name"] for c in caps})
    pb = d / "playbook.json"
    playbook = f"Category playbook from the top listings: {pb.read_text()[:6000]}\n" if pb.exists() else ""
    claims = f"Allowed claims (verified by the developer): {a.claims}\n" if a.claims else "Allowed claims: none.\n"
    slots = a.slots or min(6, len(names))
    _plan(0, 1, a.model, tokens_in=6000 + 1500 * len(names), tokens_out=4000)
    if a.dry_run:
        return
    parts = [PLAN_PROMPT.format(app=a.app, playbook=playbook, claims=claims, slots=slots)]
    for n in names:
        c = _primary(caps, n)
        img = Image.open(d / c["file"]).convert("RGB")
        k = 1024 / max(img.size)
        parts += [f"Capture '{n}' ({c['platform']} {c['device']}, {c['size'][0]}x{c['size'][1]}):",
                  img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)]
    sb = ask(parts, schema=STORYBOARD_SCHEMA, name="storyboard", model=a.model)
    seen = set()
    for i, s in enumerate(sb.get("slots", [])):
        sid = re.sub(r"[^a-z0-9_]+", "_", (s.get("id") or "").lower()).strip("_")
        if not sid or re.fullmatch(r"(slot_?)?\d+", sid):
            sid = re.sub(r"[^a-z0-9_]+", "_", str(s.get("capture") or f"slot{i + 1}").lower()).strip("_") or f"slot{i + 1}"
        while sid in seen:
            sid += "_b"
        seen.add(sid)
        s["id"] = sid
        if s.get("capture") not in names:
            print(f"! slot {sid}: unknown capture {s.get('capture')!r}; using {names[0]!r}")
            s["capture"] = names[0]
        for key in ("calls", "cards"):
            fixed = []
            for item in s.get(key) or []:
                cname = item.get("capture") if item.get("capture") in names else s["capture"]
                c = _primary(caps, cname)
                box = _fraction_box(item.get("box"), *c["size"])
                if box:
                    fixed.append({"capture": cname, "box": box})
            s[key] = fixed
        if s.get("arch") == "callout" and not s["calls"]:
            s["arch"] = "framed"
        if s.get("arch") == "fan" and len(s["cards"]) < 2:
            s["arch"] = "framed"
    sb["claims"] = a.claims or ""
    sb["slots"] = sb.get("slots", [])[:10]
    issues = _lint(sb, a.claims)
    out = _storyboard_path(d, a.locale)
    out.write_text(json.dumps(sb, indent=1))
    out.with_suffix(".md").write_text(_storyboard_md(sb, issues))
    print(f"store plan: {len(sb['slots'])} slots -> {out} (edit captions, boxes and order there, then `store plan "
          f"{d} --lint`)")
    for i, s in enumerate(sb["slots"], 1):
        print(f"  {i}. {s['id']:<12} {s.get('arch', 'framed'):<8} {s['caption']}")
    _print_lint(issues)


# ---------------------------------------------------------------- art

ART_PROMPT = ("Paint a full-bleed background plate for an app store screenshot of this app, in exactly the app's own "
              "art style ({style}): the same rendering, light and palette as the reference. {scene} Composition: the "
              "top 30% is calm, low-detail sky, haze or soft gradient where a headline will sit; the middle stays soft "
              "and out of focus because a phone will cover it; detail only near the edges and bottom. Absolutely no "
              "text, letters, numbers, logos, UI, buttons, phones, hands or people.")
FEATURE_COMPOSITION = ("Composition: a wide banner; the centre is calm and soft for a logo and title; detail only near "
                       "the left and right edges. Absolutely no text, letters, numbers, logos, UI, phones or people.")
LETTER_PROMPT = ("A single app store screenshot caption, isolated on a fully transparent background: the words \"{text}\" "
                 "spelled exactly as written, all uppercase, on {lines}, centred, {style}. Nothing else in the image.")
OCR_SCHEMA = _obj({"text": _S})
TEXT_SCHEMA = _obj({"has_text": _B, "what": _S})


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def _ref_png(path, long_side=1536):
    im = Image.open(path).convert("RGB")
    k = long_side / max(im.size)
    return _png(im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS))


def _wide_ref(path):
    """A 3:2 reference (the capture centred on its own blurred fill) so an edit keeps a landscape canvas."""
    im = Image.open(path).convert("RGB")
    W, H = 1536, 1024
    bg = im.resize((W, round(im.height * W / im.width)))
    top = max(0, (bg.height - H) // 2)
    bg = bg.crop((0, top, W, top + H)).filter(ImageFilter.GaussianBlur(30))
    fg = im.resize((round(im.width * H / im.height), H))
    bg.paste(fg, ((W - fg.width) // 2, 0))
    return _png(bg)


def _lines(text):
    words = text.upper().split()
    if len(words) < 3:
        return "one line"
    best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return f"two lines: '{' '.join(words[:best])}' over '{' '.join(words[best:])}'"


def _norm(s):
    s = unicodedata.normalize("NFKC", s).upper()
    return " ".join(re.sub(r"[^A-Z0-9 ]+", " ", s).split())


def _busy_top(img):
    """True when the top 30% (the caption's place) carries more detail than the plate overall."""
    g = np.asarray(img.convert("L").resize((256, round(256 * img.height / img.width))), np.float32)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    top = e[: int(e.shape[0] * 0.3)].mean()
    return bool(top > 1.25 * e.mean() and top > 6)


def _plate_job(name, ref, prompt, size, a, tries):
    log = []
    for t in range(1, tries + 1):
        try:
            raw, _ = edit(ref, prompt, size=size, model=a.model, fidelity="low", task="concept")
        except ApiError as e:
            log.append({"try": t, "error": str(e)})
            continue
        img = _decode_result(raw, f"plate {name}").convert("RGB")
        flags = ["BUSY-TOP"] if _busy_top(img) else []
        if a.text_check:
            got = ask(["Does this image contain any text, letters, digits, logos, user interface, phones or people? "
                       "has_text is true if it has any of them; what names them.", img], schema=TEXT_SCHEMA, name="plate_check")
            if got.get("has_text"):
                flags.append("TEXT")
        img.save(Path(a.dir) / "art" / f"{name}_t{t}.png")
        log.append({"try": t, "flags": flags})
        if "TEXT" not in flags or t == tries:
            img.save(Path(a.dir) / "art" / f"{name}.png")
            return name, {"file": f"art/{name}.png", "size": list(img.size), "flags": flags, "tries": log}
    return name, {"file": None, "flags": ["FAILED"], "tries": log}


def _letter_job(name, text, style, a):
    log = []
    for t in range(1, a.tries + 1):
        try:
            raw, _ = generate(LETTER_PROMPT.format(text=text, lines=_lines(text), style=style), size="1536x1024",
                              background="transparent", model=a.model)
        except ApiError as e:
            log.append({"try": t, "error": str(e)})
            continue
        im = _decode_result(raw, f"lettering {name}")
        bb = im.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
        if not bb:
            log.append({"try": t, "ocr": "", "ok": False})
            continue
        im = im.crop(bb)
        chk = Image.new("RGB", im.size, (40, 90, 160))
        chk.paste(im, (0, 0), im)
        got = ask(["Transcribe exactly every character of the text in this image, in reading order, nothing else. Report "
                   "what is drawn, not what was probably meant.", chk], schema=OCR_SCHEMA, name="ocr")["text"]
        ok = _norm(got) == _norm(text)
        im.save(Path(a.dir) / "art" / f"{name}_t{t}.png")
        log.append({"try": t, "ocr": got, "ok": ok})
        if ok:
            im.save(Path(a.dir) / "art" / f"{name}.png")
            return name, {"file": f"art/{name}.png", "text": text, "ok": True, "tries": log}
    return name, {"file": None, "text": text, "ok": False, "tries": log}


def cmd_store_art(a):
    """Art plates (an edit of each slot's own capture, so they keep the app's style; flagged TEXT/BUSY-TOP) and, with
    --lettering, display captions drawn transparent, trimmed and OCR-checked (re-rolled up to --tries)."""
    d = Path(a.dir)
    sb, caps = _storyboard(d), _captures(d)
    brand = sb.get("brand", {})
    want = set(a.slots.split(",")) if a.slots else None
    slots = [s for s in sb["slots"] if not want or s["id"] in want]
    if want and len(slots) != len(want):
        raise ValueError(f"store art: unknown slot(s) {', '.join(sorted(want - {s['id'] for s in slots}))}")
    (d / "art").mkdir(exist_ok=True)
    plates, letters = [], []
    for s in slots:
        c = _primary(caps, s["capture"])
        if not c:
            raise RuntimeError(f"store art: slot {s['id']} shows capture {s['capture']!r}, which is not in captures.json")
        if a.plates and s.get("plate_brief") and (a.force or not (d / "art" / f"plate_{s['id']}.png").exists()):
            plates.append((f"plate_{s['id']}", _ref_png(d / c["file"]),
                           ART_PROMPT.format(style=brand.get("art_style") or "its own illustration style",
                                             scene=s["plate_brief"]), "1152x2496"))
        if a.lettering:
            texts = {f"letter_{s['id']}": s["caption"]}
            if (s.get("caption_play") or s["caption"]) != s["caption"]:
                texts[f"letter_{s['id']}_play"] = s["caption_play"]
            letters += [(n, t) for n, t in texts.items() if a.force or not (d / "art" / f"{n}.png").exists()]
    if a.feature and (a.force or not (d / "art" / "plate_feature.png").exists()):
        c = _primary(caps, sb["slots"][0]["capture"])
        brief = (sb.get("feature") or {}).get("plate_brief") or sb["slots"][0].get("plate_brief") or "A wide calm scene."
        prompt = ART_PROMPT.format(style=brand.get("art_style") or "its own illustration style", scene=brief)
        plates.append(("plate_feature", _wide_ref(d / c["file"]), prompt.split(" Composition:")[0] + " " + FEATURE_COMPOSITION,
                       "1536x1024"))
    if not plates and not letters:
        print("store art: nothing to paint (every plate and lettering exists; --force repaints)")
        return
    tries = max(1, min(a.tries, 2))
    _plan(len(plates) + len(letters), len(letters) + (len(plates) if a.text_check else 0), a.model,
          image_model=a.model, size=[p[3] for p in plates] + ["1536x1024"] * len(letters), edits=bool(plates),
          tokens_in=3000, tokens_out=200)
    if a.dry_run:
        for p in plates:
            print(f"  {p[0]}: edit of its capture at {p[3]}")
        for n, t in letters:
            print(f"  {n}: \"{t}\" (up to {a.tries} tries)")
        return
    style = brand.get("lettering_style") or ("every letter the same chunky display style with a dark outline and a "
                                             "soft contact shadow, matching the app's own logo")
    results = {}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_plate_job, n, r, p, sz, a, tries) for n, r, p, sz in plates]
        futs += [ex.submit(_letter_job, n, t, style, a) for n, t in letters]
        for f in cf.as_completed(futs):
            n, r = f.result()
            results[n] = r
            state = ("ok" if r.get("ok") else "FAILED OCR") if "text" in r else (" ".join(r["flags"]) or "ok")
            print(f"  {n}: {state}{'' if 'text' not in r else ' after ' + str(len(r['tries'])) + ' tries'}")
    art = (d / "art" / "art.json")
    prev = json.loads(art.read_text()) if art.exists() else {}
    prev.update(results)
    art.write_text(json.dumps(prev, indent=1))
    tiles = [(n, Image.open(d / r["file"])) for n, r in sorted(results.items()) if r.get("file")]
    if tiles:
        _contact([(n, [im]) for n, im in tiles], d / "art" / "review.jpg", h=360)
    bad = [n for n, r in results.items() if not r.get("file")]
    print(f"store art: {len(results) - len(bad)} of {len(results)} done -> {d / 'art'}"
          + (f"; failed: {', '.join(sorted(bad))} (build falls back to HTML captions / brand gradient)" if bad else ""))


# ---------------------------------------------------------------- build

STORE_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
__FONTS__
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{overflow:hidden;background:__BG__}
body{position:relative;-webkit-font-smoothing:antialiased}
.plate{position:absolute;inset:0;background-size:cover;background-position:center top}
.scrim{position:absolute;left:0;right:0;top:0;height:42%}
.cap{position:absolute;left:50%;transform:translateX(-50%);text-align:center}
.cap h1{text-wrap:balance;font-family:__HEAD__;font-weight:900;letter-spacing:-.01em;line-height:1.02}
.cap p{text-wrap:balance;font-family:__BODY__;font-weight:600;line-height:1.3;margin-top:.55em;opacity:.92}
.cap img{display:block;margin:0 auto;max-width:100%}
.dev{position:absolute;left:50%;transform:translateX(-50%)}
.scr{overflow:hidden;position:relative;background:#000}
.scr img{display:block;width:100%}
.isle{position:absolute;left:50%;transform:translateX(-50%);width:29%;top:2.4%;aspect-ratio:3.4;border-radius:99px;background:#050505}
.call{position:absolute;left:50%;transform:translateX(-50%);background-repeat:no-repeat}
body.bare .cap h1,body.bare .cap p{color:transparent!important;text-shadow:none!important}
body.bare .cap img{visibility:hidden}
body.no-caption .cap{visibility:hidden}
body.fid .over{visibility:hidden}
</style></head>
<body>
<script>
const q=new URLSearchParams(location.search), W=innerWidth, H=innerHeight, wu=W/100, u=Math.min(W,H*.5625)/100;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const rect=e=>{if(!e)return null;const b=e.getBoundingClientRect();return {x:b.x,y:b.y,w:b.width,h:b.height};};
const tight=e=>{if(!e)return null;const r=document.createRange();r.selectNodeContents(e);return rect(r);};
fetch(q.get('data')||'data.json').then(r=>r.json()).then(D=>{
  const T=D.targets[q.get('t')], S=T.slots[q.get('s')], I=D.ink[S.ink], B=D.brand, mode=q.get('mode')||'';
  const card=S.frame==='card', phone=T.device==='phone', html=[];
  html.push(`<div class="plate" style="${S.plate?`background-image:url(${S.plate})`:`background:${B.gradient}`}"></div>`);
  if(!S.letter) html.push(`<div class="scrim" style="background:${I.scrim}"></div>`);
  const capTop=H*(W>H?.045:.052), capW=Math.min(.88*W,106*u);
  const pill=S.pill?`;background:${I.pill};padding:${2.2*u}px ${3.4*u}px;border-radius:${4*u}px`:'';
  if(S.letter) html.push(`<div class="cap over" style="top:${capTop}px;width:${capW}px"><img id="letter" src="${S.letter}" style="max-height:${H*.17}px;filter:drop-shadow(0 ${.6*u}px ${.8*u}px ${B.shadow})"></div>`);
  else html.push(`<div class="cap over" style="top:${capTop}px;width:${capW}px${pill}"><h1 style="color:${I.h};font-size:${(T.fs||D.fit_max)*u}px">${esc(S.cap)}</h1>${S.sub?`<p style="color:${I.p};font-size:${4.1*u}px">${esc(S.sub)}</p>`:''}</div>`);
  const devTop=H*(S.letter?.245:.262);
  const crop=(c,w)=>{const [x0,y0,x1,y1]=c.box,k=w/(x1-x0);
    return `width:${w}px;height:${(y1-y0)*k}px;background-image:url(${c.src});background-size:${c.w*k}px ${c.h*k}px;background-position:${-x0*k}px ${-y0*k}px`;};
  const shadow=`0 ${4*u}px ${7*u}px ${B.shadow}`;
  const float=f=>`<img class="float over" data-front="${f.front?1:0}" src="${f.src}" style="position:absolute;left:${f.x*W}px;top:${f.y*H}px;width:${f.w*100*u}px;`
    +`transform:translate(-50%,-50%) rotate(${f.rot}deg);filter:drop-shadow(0 ${1.6*u}px ${2.4*u}px ${B.shadow})${f.blur?` blur(${f.blur*u}px)`:''}">`;
  (S.floats||[]).filter(f=>!f.front).forEach(f=>html.push(float(f)));
  if(S.arch==='fan'){
    const hero=S.cards[0], r=(hero.box[3]-hero.box[1])/(hero.box[2]-hero.box[0]);
    const wc=Math.min((card?58:62)*u,.62*H/r), base=H*(S.letter?.28:.30);
    const lay=S.cards.length>2?[[1,-23,-8,.8],[2,23,8,.8],[0,0,0,1]]:[[1,-16,-7,.85],[0,10,3,1]];
    lay.forEach(([i,dx,rot,k])=>{const w=wc*k, dy=k<1?wc*Math.min(r,1.2)*.46:0;
      html.push(`<div class="call fan" style="${crop(S.cards[i],w)};top:${base+dy}px;border-radius:${6*u}px;box-shadow:0 ${3*u}px ${5*u}px ${B.shadow},0 0 0 ${.7*u}px ${B.ring};transform:translateX(calc(-50% + ${dx*u}px)) rotate(${rot}deg)"></div>`);});
  } else {
    const sc=S.screen, cut=card?sc.status:0, asp=(sc.h-cut)/sc.w;
    const fr=card?null:(phone?{p:.0325,r:'16.5%/7.6%',s:'14%/6.5%'}:{p:.03,r:'6.2%/4.6%',s:'3.8%/2.8%'}), p=fr?fr.p:0;
    const pct=phone?(card?(S.letter?70:68):(S.letter?82:80)):(card?66:74);
    const D=Math.min(pct*wu,(H*1.10-devTop)/(asp*(1-2*p)+2*p));
    const img=`<img id="shot" src="${sc.src}" style="${cut?`margin-top:-${cut/sc.w*100}%`:''}">`;
    if(card) html.push(`<div class="dev scr" id="scr" style="width:${D}px;top:${devTop}px;border-radius:${phone?'7.5%/3.5%':'4%/3%'};box-shadow:0 ${3.5*u}px ${6*u}px ${B.shadow},0 0 0 ${.6*u}px ${B.ring}">${img}</div>`);
    else html.push(`<div class="dev" style="width:${D}px;top:${devTop}px;padding:${p*D}px;border-radius:${fr.r};background:${B.bezel};box-shadow:${shadow},inset 0 0 0 ${.35*u}px rgba(255,255,255,.25)"><div class="scr" id="scr" style="border-radius:${fr.s}">${img}${sc.island?'<div class="isle over"></div>':''}</div></div>`);
  }
  document.body.insertAdjacentHTML('beforeend',html.join(''));
  document.body.style.width=W+'px';document.body.style.height=H+'px';
  if(mode) document.body.classList.add(mode);
  return Promise.all([document.fonts.ready,...[...document.images].map(i=>i.decode().catch(()=>{}))]).then(async()=>{
    const h1=document.querySelector('.cap h1');
    if(h1){let fs=parseFloat(h1.style.fontSize);const lh=()=>h1.getBoundingClientRect().height/(fs*1.02);
      const capEl=h1.parentElement, share=()=>{const b=[tight(h1),tight(document.querySelector('.cap p'))].filter(Boolean);
        return (Math.max(...b.map(t=>t.x+t.w))-Math.min(...b.map(t=>t.x)))*(Math.max(...b.map(t=>t.y+t.h))-Math.min(...b.map(t=>t.y)))/(W*H);};
      const over=()=>lh()>2.05||capEl.getBoundingClientRect().bottom>devTop-1.8*u||(T.tag_max&&share()>T.tag_max);
      while(over()&&fs>6*u){fs-=.2*u;h1.style.fontSize=fs+'px';}}
    if(S.arch==='callout'){const r=document.getElementById('shot').getBoundingClientRect(), k=r.width/S.screen.w;
      S.calls.forEach(c=>{const bw=c.box[2]-c.box[0], bh=c.box[3]-c.box[1], w=Math.min(94*wu,bw*k*1.2), hh=bh*w/bw;
        const cy=r.top+(c.box[1]+c.box[3])/2*k, el=document.createElement('div');
        el.className='call over';el.setAttribute('style',crop(c,w)+`;border-radius:${2.6*u}px;box-shadow:0 ${3*u}px ${5*u}px ${B.shadow},0 0 0 ${.7*u}px ${B.ring}`);
        el.style.top=Math.min(cy-hh/2,H-hh-3*u)+'px';document.body.appendChild(el);});}
    const fronts=(S.floats||[]).filter(f=>f.front);
    if(fronts.length){document.body.insertAdjacentHTML('beforeend',fronts.map(float).join(''));
      await Promise.all([...document.querySelectorAll('.float')].map(i=>i.decode().catch(()=>{})));}
    const p=document.querySelector('.cap p'), ht=tight(h1), pt=tight(p);
    const text=[ht,pt].filter(Boolean), tag=text.length?(()=>{const x0=Math.min(...text.map(t=>t.x)),y0=Math.min(...text.map(t=>t.y)),
      x1=Math.max(...text.map(t=>t.x+t.w)),y1=Math.max(...text.map(t=>t.y+t.h));return {x:x0,y:y0,w:x1-x0,h:y1-y0};})():rect(document.getElementById('letter'));
    window.REPORT={screen:rect(document.getElementById('scr')),shot:rect(document.getElementById('shot')),
      h1:h1?{...ht,font:parseFloat(h1.style.fontSize)}:null,sub:p?{...pt,font:parseFloat(p.style.fontSize)}:null,
      tagline:tag,calls:[...document.querySelectorAll('.call')].map(rect),
      floats:[...document.querySelectorAll('.float')].map(e=>({...rect(e),front:e.dataset.front==='1'}))};
    document.body.dataset.ready='1';
  });
}).catch(e=>{document.title='error: '+e;console.error('store page: '+e);});
</script>
</body></html>
"""

FEATURE_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
__FONTS__
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1024px;height:500px;overflow:hidden;background:__BG__}
.plate{position:absolute;inset:0;background-size:cover;background-position:center}
.glow{position:absolute;inset:0}
.mid{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);text-align:center;width:640px}
.mid img{display:block;margin:0 auto;max-width:400px;max-height:220px}
h1{font-family:__HEAD__;font-weight:900;font-size:76px;line-height:1}
p{font-family:__BODY__;font-weight:700;font-size:30px;margin-top:14px}
</style></head><body>
<div class="plate" style="__PLATE__"></div><div class="glow" style="background:__GLOW__"></div>
<div class="mid" id="mid">__LOGO__<p style="color:__INK__">__SUB__</p></div>
<script>
const rect=e=>{const b=e.getBoundingClientRect();return {x:b.x,y:b.y,w:b.width,h:b.height};};
Promise.all([document.fonts.ready,...[...document.images].map(i=>i.decode().catch(()=>{}))]).then(()=>{
  window.REPORT={content:rect(document.getElementById('mid'))};document.body.dataset.ready='1';});
</script></body></html>
"""


def _store_hex(h, fallback):
    h = (h or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", h):
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if re.fullmatch(r"[0-9a-fA-F]{3}", h):
        return tuple(int(c * 2, 16) for c in h)
    return fallback


def _mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _scale(c, k):
    return tuple(min(255, round(x * k)) for x in c)


def _css(c, alpha=None):
    return f"rgba({c[0]},{c[1]},{c[2]},{alpha})" if alpha is not None else "#%02x%02x%02x" % c


def _store_palette(brand):
    """Ink, scrim, pill, ring, bezel and fallback-plate colours from the storyboard's brand block."""
    dark = _store_hex(brand.get("ink_dark"), (31, 27, 36))
    light = _store_hex(brand.get("ink_light"), (255, 248, 238))
    cols = [_store_hex(c, None) for c in brand.get("colors") or []]
    cols = [c for c in cols if c] or [_mix(light, dark, 0.15), _mix(light, dark, 0.35)]
    deep = _mix(dark, (0, 0, 0), 0.25)
    bezel = _store_hex(brand.get("bezel"), _scale(dark, 1.55))
    ink = {"dark": {"h": _css(dark), "p": _css(_mix(dark, (255, 255, 255), 0.12)),
                    "scrim": f"linear-gradient(180deg,{_css(light, .78)} 0%,{_css(light, .45)} 55%,{_css(light, 0)} 100%)",
                    "pill": _css(light, .9), "rgb": [list(dark), list(_mix(dark, (255, 255, 255), 0.12))]},
           "light": {"h": _css(light), "p": _css(_mix(light, dark, 0.04)),
                     "scrim": f"linear-gradient(180deg,{_css(deep, .62)} 0%,{_css(deep, .30)} 55%,{_css(deep, 0)} 100%)",
                     "pill": _css(deep, .72), "rgb": [list(light), list(_mix(light, dark, 0.04))]}}
    two = cols + cols[:1] if len(cols) == 1 else cols
    return {"ink": ink, "brand": {
        "gradient": f"linear-gradient(180deg,{_css(two[0])},{_css(two[1])})", "bg": _css(two[0]),
        "shadow": _css(_mix(dark, (0, 0, 0), 0.3), .4), "ring": _css(light, .92),
        "bezel": (f"linear-gradient(150deg,{_css(_scale(bezel, 1.7))},{_css(bezel)} 38%,{_css(_scale(bezel, 1.55))} 68%,"
                  f"{_css(_scale(bezel, .81))})")}}


def _fonts_link(brand, webfonts):
    head, body = brand.get("head_font") or "Nunito", brand.get("body_font") or "DM Sans"
    stack = lambda f: f"\"{f}\",system-ui,-apple-system,sans-serif"  # noqa: E731
    link = ""
    if webfonts:
        fam = lambda f, w: "family=" + f.replace(" ", "+") + ":wght@" + w  # noqa: E731
        link = (f"<link href=\"https://fonts.googleapis.com/css2?{fam(head, '800;900')}&{fam(body, '600;700')}"
                "&display=block\" rel=\"stylesheet\">")
    return link, stack(head), stack(body)


def _pick(caps, name, platform, device, orient):
    """(capture, stand_in) for a slot on a target: the target platform's own capture, or for Play an iOS one."""
    for plat in ([platform] if platform == "ios" else ["android", "ios"]):
        for c in caps:
            if c["name"] == name and c["platform"] == plat and c["device"] == device and c["orientation"] == orient:
                return c, plat != platform
    return None, False


def _targets(a, caps):
    stores = [s.strip() for s in a.stores.split(",") if s.strip()]
    bad = [s for s in stores if s not in ("ios", "play")]
    if bad:
        raise ValueError(f"store build: --stores takes ios,play (got {', '.join(bad)})")
    devices = ["phone", "tablet"] if a.devices == "auto" else [x.strip() for x in a.devices.split(",")]
    out = []
    for key, t in TARGETS.items():
        if t["store"] not in stores or t["device"] not in devices or (t.get("optional") and not a.iphone_large):
            continue
        orient = "landscape" if t["css"][0] > t["css"][1] else "portrait"
        plats = ["ios"] if t["store"] == "ios" else ["android", "ios"]
        if t["device"] == "tablet" and not any(c["device"] == "tablet" and c["orientation"] == orient
                                               and c["platform"] in plats for c in caps):
            if a.devices != "auto":
                print(f"! {key}: no {orient} tablet captures; skipped")
            continue
        out.append(key)
    return out


def _slot_data(d, sb, s, key, caps, variant, frame, notes):
    """The page's data for one slot on one target, or None when the target lacks a capture it needs."""
    t = TARGETS[key]
    orient = "landscape" if t["css"][0] > t["css"][1] else "portrait"
    platform = "ios" if t["store"] == "ios" else "android"
    store_frame = frame if frame != "auto" else ("device" if t["store"] == "ios" else "card")

    def cap(name):
        c, standin = _pick(caps, name, platform, t["device"], orient)
        if c and standin:
            notes.add(f"standin|{key}|{name}")
        return c

    def rel(f):
        return "../" + f

    def boxed(items):
        out = []
        for it in items or []:
            c = cap(it.get("capture") or s["capture"])
            if not c:
                return None
            w, h = c["size"]
            b = it["box"]
            px = [b[0] * w, b[1] * h, b[2] * w, b[3] * h] if max(b) <= 1.0 else list(b)
            out.append({"src": rel(c["file"]), "w": w, "h": h, "box": px})
        return out

    main = cap(s["capture"])
    if not main:
        return None
    arch = s.get("arch") or "framed"
    play = t["store"] == "play"
    picks = (["_play"] if play else []) + (["_tablet"] if t["device"] == "tablet" else []) + [""]

    def override(name):
        return next((s[name + p] for p in picks if s.get(name + p)), None)
    calls = boxed(override("calls")) if arch == "callout" else []
    cards = boxed(override("cards")) if arch == "fan" else []
    if calls is None or cards is None or (arch == "fan" and len(cards) < 2):
        notes.add(f"{key}: {s['id']} lacks a capture for its {arch}; drawn framed")
        arch, calls, cards = "framed", [], []
    if arch == "callout" and not calls:
        arch = "framed"
    caption = (s.get("caption_play") or s["caption"]) if play else s["caption"]
    sub = (s.get("sub_play") or s.get("sub") or "") if play else (s.get("sub") or "")
    letter = None
    if variant == "lettering":
        for n in ([f"letter_{s['id']}_play"] if play and caption != s["caption"] else []) + [f"letter_{s['id']}"]:
            if (Path(d) / "art" / f"{n}.png").exists() and (n.endswith("_play") or caption == s["caption"]):
                letter = rel(f"art/{n}.png")
                break
        if not letter:
            notes.add(f"{key}: {s['id']} has no OCR-checked lettering; HTML caption used")
    floats = []
    for fl in (override("floats") or [])[:FLOATS_MAX]:
        f = _float_file(d, fl.get("src") or "")
        if not f:
            notes.add(f"{key}: {s['id']} float {fl.get('src')!r} not found; left out")
            continue
        floats.append({"src": rel(f), "x": float(fl.get("x", .5)), "y": float(fl.get("y", .5)), "w": float(fl.get("w", .2)),
                       "rot": float(fl.get("rot", 0)), "blur": float(fl.get("blur", 0)), "front": fl.get("layer") == "front"})
    plate = s.get("plate") or f"plate_{s['id']}"
    plate_file = Path(d) / "art" / f"{plate}.png"
    w, h = main["size"]
    return {"id": s["id"], "arch": arch, "ink": s.get("ink") if s.get("ink") in ("dark", "light") else "dark",
            "cap": caption, "sub": sub, "letter": letter, "pill": bool(s.get("pill")),
            "plate": rel(f"art/{plate}.png") if plate_file.exists() else None,
            "frame": "card" if store_frame == "card" else "device",
            "screen": {"src": rel(main["file"]), "w": w, "h": h, "status": main.get("status_bar", 0),
                       "island": main["platform"] == "ios" and t["device"] == "phone" and h / w > 2.1},
            "calls": calls, "cards": cards, "floats": floats}


def _rect_overlap(a, b):
    """Overlap area of two {x, y, w, h} rects."""
    w = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return max(w, 0) * max(h, 0)


def _float_file(d, src):
    """A float image as a path relative to the project; files from elsewhere are copied into floats/."""
    p = Path(src).expanduser()
    p = p if p.is_absolute() else Path(d) / p
    if not src or not p.is_file():
        return None
    try:
        return str(p.resolve().relative_to(Path(d).resolve()))
    except ValueError:
        dst = Path(d) / "floats" / p.name
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists() or dst.stat().st_mtime < p.stat().st_mtime:
            shutil.copy2(p, dst)
        return f"floats/{p.name}"


def _group_notes(notes):
    stand, out = {}, []
    for n in sorted(notes):
        if n.startswith("standin|"):
            _, key, name = n.split("|", 2)
            stand.setdefault(key, []).append(name)
        else:
            out.append(n)
    out += [f"{key}: iOS captures stand in for Android ({', '.join(v)}); capture them on Android for release"
            for key, v in stand.items()]
    return out


class _Renderer:
    """One Chrome for a whole build: serves DIR and shoots build/PAGE?QUERY at a viewport and DPR."""

    def __init__(self, root, timeout):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("store: needs Playwright: pip install playwright (it drives your installed Chrome; "
                               "without one, also run `playwright install chromium`)")
        self.timeout, self.errors, self.pages = timeout, [], {}
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root)))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.pw = sync_playwright().start()
        exe = _chrome()
        self.browser = self.pw.chromium.launch(executable_path=exe) if exe else self.pw.chromium.launch()

    def shot(self, page, query, size, dpr):
        pg = self._load(page, query, size, dpr)
        img = Image.open(io.BytesIO(pg.screenshot(type="png"))).convert("RGB")
        return img, pg.evaluate("window.REPORT")

    def layout(self, page, query, size):
        """The page's REPORT without a screenshot (layout only, at DPR 1)."""
        return self._load(page, query, size, 1).evaluate("window.REPORT")

    def _load(self, page, query, size, dpr):
        from playwright.sync_api import Error as PwError
        key = (size, dpr)
        pg = self.pages.get(key)
        if pg is None:
            pg = self.pages[key] = self.browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=dpr)
            pg.on("pageerror", lambda e: self.errors.append(f"page error: {e}"))
            pg.on("console", lambda m: m.type == "error" and not m.text.startswith("Failed to load resource")
                  and self.errors.append(f"console: {m.text}"))
            pg.on("response", lambda r: r.status >= 400 and "fonts.g" not in r.url and not r.url.endswith("/favicon.ico")
                  and self.errors.append(f"HTTP {r.status} {r.url.split(str(self.srv.server_address[1]), 1)[-1]}"))
        pg.goto(f"http://127.0.0.1:{self.srv.server_address[1]}/build/{page}?{query}")
        try:
            pg.wait_for_selector("body[data-ready='1']", state="attached", timeout=self.timeout * 1000)
        except PwError:
            raise RuntimeError(f"store: build/{page}?{query} never got ready within {self.timeout}s"
                               + (f" ({self.errors[-1]})" if self.errors else ""))
        return pg

    def close(self):
        try:
            self.browser.close()
            self.pw.stop()
        finally:
            self.srv.shutdown()


def _wcag(rgb):
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ np.array([0.2126, 0.7152, 0.0722])


def _contrast_ratio(a, b):
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _contrast(bare, r, ink_rgb, dark, dpr):
    """WCAG ratio of the ink against the worst 5% of the caption's background (from a render with the text hidden)."""
    if not r or r["w"] < 1 or r["h"] < 1:
        return None
    x0, y0 = max(0, int(r["x"] * dpr)), max(0, int(r["y"] * dpr))
    x1, y1 = min(bare.width, int((r["x"] + r["w"]) * dpr)), min(bare.height, int((r["y"] + r["h"]) * dpr))
    if x1 <= x0 or y1 <= y0:
        return None
    L = _wcag(np.asarray(bare.crop((x0, y0, x1, y1))).reshape(-1, 3))
    worst = np.percentile(L, 5 if dark else 95)
    return round(float(_contrast_ratio(float(_wcag(ink_rgb)), worst)), 2)


def _fidelity(fid, rep, cap_img, dpr):
    """Mean / p99 CIELAB dE between the visible screen in a render with overlays hidden and the capture itself."""
    s, sh = rep.get("screen"), rep.get("shot")
    if not s or not sh:
        return None
    inset = 0.09 * s["w"]
    x0, y0 = max(s["x"], 0) + inset, max(s["y"], 0) + inset
    x1, y1 = s["x"] + s["w"] - inset, min(s["y"] + s["h"] - inset, fid.height / dpr - 2)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    X0, Y0, X1, Y1 = (round(v * dpr) for v in (x0, y0, x1, y1))
    k = cap_img.width / sh["w"]
    box = ((X0 / dpr - sh["x"]) * k, (Y0 / dpr - sh["y"]) * k, (X1 / dpr - sh["x"]) * k, (Y1 / dpr - sh["y"]) * k)
    ref = _lab(np.asarray(cap_img.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=box)))
    arr = np.asarray(fid)
    pad = 2
    if Y0 < pad or X0 < pad or Y1 + pad > arr.shape[0] or X1 + pad > arr.shape[1]:
        pad = 0
    big = _lab(arr[Y0 - pad:Y1 + pad, X0 - pad:X1 + pad])
    # pick the sub-pixel offset on a lightness subsample, then measure dE there in full
    offsets = [(dy, dx) for dy in range(-pad, pad + 1) for dx in range(-pad, pad + 1)]
    h, w = ref.shape[:2]
    rl = ref[::3, ::3, 0]
    dy, dx = min(offsets, key=lambda o: float(np.abs(big[pad + o[0]:pad + o[0] + h:3, pad + o[1]:pad + o[1] + w:3, 0] - rl).mean()))
    de = np.linalg.norm(big[pad + dy:pad + dy + h, pad + dx:pad + dx + w] - ref, axis=-1)
    return {"mean": round(float(de.mean()), 2), "p99": round(float(np.percentile(de, 99)), 2), "over": round(float((de > 2.5).mean()), 3)}


def _save_store_png(img, path):
    """24-bit RGB, no alpha; JPEG when the PNG passes Play's 8 MB."""
    img = img.convert("RGB")
    img.save(path)
    if path.stat().st_size <= MAX_BYTES:
        return path
    jpg = path.with_suffix(".jpg")
    img.save(jpg, quality=92, subsampling=0)
    path.unlink()
    return jpg


def _set_sheet(files, out, h=900):
    ims = [Image.open(f).convert("RGB") for f in files]
    if not ims:
        return None
    return _contact([(out.parent.name, ims)], out, h=h)


def _search_row(files, out):
    """The first three iPhone shots at search-result size on a phone at 3x, to judge legibility."""
    ims = [Image.open(f).convert("RGB") for f in files[:3]]
    tw = round(ims[0].width * SEARCH_SCALE)
    th = round(tw * ims[0].height / ims[0].width)
    row = Image.new("RGB", (ims[0].width, th + 120), "white")
    x = (ims[0].width - (3 * tw + 2 * 24)) // 2
    for im in ims:
        t = im.resize((tw, th), Image.LANCZOS)
        mask = Image.new("L", t.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, tw - 1, th - 1), radius=30, fill=255)
        row.paste(t, (x, 60), mask)
        x += tw + 24
    row.save(out, quality=90)
    return out


def cmd_store_build(a):
    """Composite every slot for each store and device from the real captures (pixel-exact) with the plates and
    captions, measure contrast and UI fidelity, and write store-ready RGB images, alt text and review sheets."""
    d = Path(a.dir).resolve()
    sb, caps = _storyboard(d, a.locale), _captures(d)
    if not caps:
        raise RuntimeError(f"store build: no captures in {d}")
    if a.frame == "device" and "play" in a.stores:
        raise ValueError("store build: --frame device would put an iPhone on Google Play; Play asks for no device "
                         "imagery and Apple forbids the reverse, so Play is always frameless (use --stores ios)")
    want = set(a.slots.split(",")) if a.slots else None
    slots = [s for s in sb["slots"] if not want or s["id"] in want]
    if not slots:
        raise ValueError("store build: --slots matched no slot in the storyboard")
    targets = _targets(a, caps)
    if not targets:
        raise RuntimeError("store build: no target has the captures it needs (iOS needs iOS captures; Apple 2.3.10)")
    build = d / ("build" if not a.locale else f"build-{a.locale}")
    build.mkdir(exist_ok=True)
    pal = _store_palette(sb.get("brand", {}))
    link, head, body = _fonts_link(sb.get("brand", {}), a.webfonts)
    notes, data = set(), {"ink": pal["ink"], "brand": pal["brand"], "fit_max": HEAD_MAX, "targets": {}}
    for key in targets:
        rows = {}
        for s in slots:
            sd = _slot_data(d, sb, s, key, caps, a.variant, a.frame, notes)
            if sd:
                rows[s["id"]] = sd
            else:
                notes.add(f"{key}: {s['id']} has no {TARGETS[key]['device']} capture of {s['capture']!r}; left out")
        data["targets"][key] = {"store": TARGETS[key]["store"], "device": TARGETS[key]["device"], "slots": rows,
                                "tag_max": TAGLINE_FIT if TARGETS[key]["store"] == "play" else None}
    page = (STORE_PAGE.replace("__FONTS__", link).replace("__HEAD__", head).replace("__BODY__", body)
            .replace("__BG__", pal["brand"]["bg"]))
    pname, dname = "store.html", "data.json" if not a.locale else f"data-{a.locale}.json"
    (d / "build").mkdir(exist_ok=True)
    (d / "build" / pname).write_text(page)
    data_file = d / "build" / dname

    def write_data():
        data_file.write_text(json.dumps(data, indent=1))

    write_data()
    report = {"storyboard": str(_storyboard_path(d, a.locale)), "variant": a.variant, "no_caption": a.no_caption, "targets": {}}
    r = _Renderer(d, a.timeout)
    try:
        for key in targets:
            t, td = TARGETS[key], data["targets"][key]
            size, dpr = t["css"], t["dpr"]
            out = build / key
            if out.exists() and any(out.iterdir()):  # an earlier set must not linger beside this one
                prev = build / ".previous"
                prev.mkdir(exist_ok=True)
                out.rename(prev / f"{key}-{datetime.now():%Y%m%d-%H%M%S}")
            out.mkdir(exist_ok=True)
            shutil.rmtree(build / "bare" / key, ignore_errors=True)
            fits = [(r.layout(pname, f"t={key}&s={sid}" + (f"&data={dname}" if a.locale else ""), size).get("h1") or {})
                    .get("font") for sid, sd in td["slots"].items() if not sd["letter"]]
            fits = [f for f in fits if f]
            if fits:  # one headline size per set: the largest every caption fits at
                td["fs"] = math.floor(min(fits) / (min(size[0], size[1] * 0.5625) / 100) * 10) / 10
                write_data()
            entries = []
            for i, (sid, sd) in enumerate(td["slots"].items(), 1):
                q = f"t={key}&s={sid}" + (f"&data={dname}" if a.locale else "")
                ink_rgb = pal["ink"][sd["ink"]]["rgb"]
                contrast = None
                if not sd["letter"]:
                    bare, rep = r.shot(pname, q + "&mode=bare", size, dpr)
                    contrast = {"h1": _contrast(bare, rep.get("h1"), ink_rgb[0], sd["ink"] == "dark", dpr),
                                "sub": _contrast(bare, rep.get("sub"), ink_rgb[1], sd["ink"] == "dark", dpr)}
                    low = [v for v in contrast.values() if v is not None and v < CONTRAST_MIN]
                    if low and not sd["pill"] and a.pill:
                        sd["pill"] = True
                        write_data()
                        bare, rep = r.shot(pname, q + "&mode=bare", size, dpr)
                        contrast = {"h1": _contrast(bare, rep.get("h1"), ink_rgb[0], sd["ink"] == "dark", dpr),
                                    "sub": _contrast(bare, rep.get("sub"), ink_rgb[1], sd["ink"] == "dark", dpr),
                                    "pill": True}
                if sd["letter"]:
                    bare = r.shot(pname, q + "&mode=bare", size, 1)[0]
                (build / "bare" / key).mkdir(parents=True, exist_ok=True)
                bare.resize((BARE_W, round(bare.height * BARE_W / bare.width)), Image.LANCZOS).save(
                    build / "bare" / key / f"{i:02d}_{sid}.jpg", quality=88)
                img, rep = r.shot(pname, q + ("&mode=no-caption" if a.no_caption else ""), size, dpr)
                fid = None
                if sd["arch"] != "fan":
                    fimg, frep = r.shot(pname, q + "&mode=fid", size, dpr)
                    src = d / sd["screen"]["src"][3:]
                    fid = _fidelity(fimg, frep, Image.open(src).convert("RGB"), dpr)
                f = _save_store_png(img, out / f"{i:02d}_{sid}.png")
                slot = next(s for s in slots if s["id"] == sid)
                tag, sr, fl = rep.get("tagline"), rep.get("screen"), rep.get("floats") or []
                cover = sum(_rect_overlap(x, sr) for x in fl if x["front"]) / (sr["w"] * sr["h"]) if fl and sr else 0
                entries.append({"file": str(f.relative_to(d)), "slot": sid, "size": list(img.size), "bytes": f.stat().st_size,
                                "mode": "RGB", "screen_box": {k: v * dpr for k, v in rep["screen"].items()} if rep.get("screen") else None,
                                "arch": sd["arch"], "lettering": bool(sd["letter"]), "contrast": contrast,
                                "fidelity": fid, "h1_px": (rep.get("h1") or {}).get("font"),
                                "tagline_share": round(tag["w"] * tag["h"] / (size[0] * size[1]), 3) if tag else None,
                                "alt": slot.get("alt", ""), "pill": sd["pill"], "floats": len(fl),
                                "float_cover": round(cover, 3), "float_on_caption": bool(tag and any(_rect_overlap(x, tag) for x in fl))})
                c = contrast or {}
                print(f"  {key} {i:02d} {sid:<12} " + (f"contrast {c.get('h1')}/{c.get('sub')}" if contrast else "lettering")
                      + (f"  dE {fid['mean']}/{fid['p99']}" if fid else "") + ("  pill" if sd["pill"] else ""))
            files = [d / e["file"] for e in entries]
            (out / "alt.txt").write_text("".join(f"{Path(e['file']).name}\t{e['alt']}\n" for e in entries))
            if files:
                _set_sheet(files, out / "sheet.jpg", h=900 if size[1] > size[0] else 600)
                if key == "ios-phone":
                    _search_row(files, out / "search_row.jpg")
            report["targets"][key] = {"css": list(size), "dpr": dpr, "shots": entries}
    finally:
        r.close()
    report["notes"] = _group_notes(notes)
    report["errors"] = r.errors[:20]
    (build / "report.json").write_text(json.dumps(report, indent=1))
    for n in report["notes"]:
        print(f"  note: {n}")
    for e in r.errors[:5]:
        print(f"  ! {e}")
    print(f"store build: {sum(len(t['shots']) for t in report['targets'].values())} shots in {len(targets)} set(s) -> "
          f"{build} (review the sheet.jpg in each; then `store check {a.dir}`)")


# ---------------------------------------------------------------- feature graphic

def cmd_store_feature(a):
    """The Google Play feature graphic (1024x500): the feature plate, a centred logo or title and subtitle."""
    d = Path(a.dir).resolve()
    sb = _storyboard(d)
    brand = sb.get("brand", {})
    pal = _store_palette(brand)
    link, head, body = _fonts_link(brand, a.webfonts)
    (d / "build").mkdir(exist_ok=True)
    plate = d / "art" / "plate_feature.png"
    plate_css = f"background-image:url(../art/plate_feature.png)" if plate.exists() else f"background:{pal['brand']['gradient']}"
    ink = pal["ink"]["dark"]
    light = _store_hex(brand.get("ink_light"), (255, 248, 238))
    if a.logo:
        lp = Path(a.logo)
        if not lp.is_file():
            raise RuntimeError(f"store feature: no logo at {lp}")
        shutil.copyfile(lp, d / "build" / "feature_logo.png")
        logo = ('<img src="feature_logo.png" style="filter:drop-shadow(0 6px 14px rgba(0,0,0,.35))">')
    else:
        title = a.title or sb.get("app") or "App"
        logo = f"<h1 style=\"color:{ink['h']}\">{_esc(title)}</h1>"
    if a.icon:
        ip = Path(a.icon)
        if not ip.is_file():
            raise RuntimeError(f"store feature: no icon at {ip}")
        shutil.copyfile(ip, d / "build" / "feature_icon.png")
        logo = ('<img src="feature_icon.png" style="width:112px;height:112px;border-radius:25px;margin-bottom:18px;'
                'box-shadow:0 10px 24px rgba(0,0,0,.28)">') + logo
    sub = a.subtitle if a.subtitle is not None else (sb.get("feature") or {}).get("subtitle", "")
    glow = f"radial-gradient(ellipse 46% 60% at 50% 50%,{_css(light, .55)} 0%,{_css(light, 0)} 100%)"
    page = (FEATURE_PAGE.replace("__FONTS__", link).replace("__HEAD__", head).replace("__BODY__", body)
            .replace("__BG__", pal["brand"]["bg"]).replace("__PLATE__", plate_css).replace("__GLOW__", glow)
            .replace("__LOGO__", logo).replace("__INK__", ink["h"]).replace("__SUB__", _esc(sub)))
    (d / "build" / "feature.html").write_text(page)
    r = _Renderer(d, a.timeout)
    try:
        img, rep = r.shot("feature.html", "", FEATURE_SIZE, 1)
    finally:
        r.close()
    out = d / "build" / "play-feature"
    out.mkdir(exist_ok=True)
    f = _save_store_png(img, out / "feature.png")
    checks = _feature_checks(img, rep)
    (out / "feature.json").write_text(json.dumps({"file": str(f.relative_to(d)), "checks": checks}, indent=1))
    for lv, msg in checks:
        print(f"  {lv} {msg}")
    print(f"store feature: {f}")


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _feature_checks(img, rep):
    out = []
    if img.size != FEATURE_SIZE:
        out.append(("FAIL", f"feature graphic is {img.size[0]}x{img.size[1]}, Play needs 1024x500"))
    L = _wcag(np.asarray(img.resize((256, 125))).reshape(-1, 3))
    if L.mean() > 0.9 or L.mean() < 0.03 or L.std() < 0.02:
        out.append(("WARN", "the graphic reads as plain white, black or flat; Play asks for vibrant colour"))
    c = (rep or {}).get("content")
    if c and (c["x"] < 0.12 * 1024 or c["x"] + c["w"] > 0.88 * 1024 or c["y"] < 0.1 * 500 or c["y"] + c["h"] > 0.9 * 500):
        out.append(("WARN", "logo/title reach the edges Play may crop; keep them in the centre"))
    return out or [("PASS", "1024x500, coloured, content centred")]


# ---------------------------------------------------------------- check

IDEA_SCHEMA = _obj({"app": _S, "audience": _S, "unclear": _B, "slots": {"type": "array", "items": _obj({"idea": _S})}})
JUDGE_SCHEMA = _obj({"app_match": _B, "app_note": _S,
                     "slots": {"type": "array", "items": _obj({"match": _B, "note": _S})}})
IDEA_PROMPT = ("These are the first three App Store screenshots of one app, in order, with their headlines removed, at "
               "about the size a shopper sees them in search results. From the pictures alone, say in one line what the "
               "app does and the outcome it promises, who it is for, and for each screenshot the one idea it shows. Read "
               "only what a shopper would catch at a glance. Set unclear when you cannot tell what the app does.")
JUDGE_PROMPT = ("The developer meant these screenshots to say:\nApp: {intent}\n{slots}\nA viewer who saw them with the "
                "headlines hidden read them as:\nApp: {app} (for {audience})\n{seen}\nFor the app and for each screenshot, "
                "does the viewer's reading carry the intended message in substance (the same job or outcome; the wording "
                "may differ)? Give a short note for every mismatch.")


def _idea_check(d, build, rep, sb, model):
    """Hide-the-copy test: a blind read of the captionless first three shots, then a text-only comparison with the plan.
    [(level, where, msg)]."""
    key = next((k for k in ("ios-phone", "play-phone") if k in rep.get("targets", {})), None)
    if not key:
        return [("WARN", "idea", "no phone set in this build; the hide-the-copy test skipped")]
    shots = rep["targets"][key]["shots"][:3]
    files = [build / "bare" / key / Path(e["file"]).with_suffix(".jpg").name for e in shots]
    if not all(f.exists() for f in files):
        return [("WARN", "idea", f"captionless copies missing in {build / 'bare' / key}; re-run `store build`")]
    by_id = {s["id"]: s for s in sb.get("slots", [])}
    caps = [by_id.get(e["slot"], {}) for e in shots]
    intent = sb.get("promise") or f"{sb.get('app', '')} ({sb.get('category', '')})"
    _plan(0, 2, model, tokens_in=4000, tokens_out=800)
    seen = ask([IDEA_PROMPT, *[Image.open(f) for f in files]], schema=IDEA_SCHEMA, name="idea", model=model)
    ideas = (seen.get("slots") or []) + [{"idea": ""}] * 3
    judge = ask([JUDGE_PROMPT.format(
        intent=intent, app=seen.get("app", ""), audience=seen.get("audience", ""),
        slots="\n".join(f"Screenshot {i}: {c.get('caption', '')}. {c.get('sub', '')}" for i, c in enumerate(caps, 1)),
        seen="\n".join(f"Screenshot {i}: {ideas[i - 1].get('idea', '')}" for i in range(1, len(caps) + 1)))],
        schema=JUDGE_SCHEMA, name="idea-judge", model=model)
    (build / "idea.json").write_text(json.dumps({"target": key, "intent": intent, "seen": seen, "judge": judge}, indent=1))
    out = []
    if seen.get("unclear") or not judge.get("app_match"):
        out.append(("WARN", "idea", f"with the captions hidden, the first shots read as {seen.get('app', '?')!r}, not "
                    f"{intent!r} ({judge.get('app_note') or 'unclear'}); make slot 1's picture show the outcome"))
    else:
        out.append(("PASS", "idea", f"with the captions hidden: {seen.get('app', '')!r}"))
    for i, (e, j) in enumerate(zip(shots, (judge.get("slots") or []) + [{}] * 3), 1):
        if j.get("match") is False:
            out.append(("WARN", f"idea {e['slot']}", f"reads as {ideas[i - 1].get('idea', '')!r} without its caption: "
                        f"{j.get('note', '')}"))
    return out


def cmd_store_check(a):
    """Store rules over the last build: sizes, counts, no alpha, file size, UI fidelity, caption contrast, search
    legibility, Play tagline share and copy. Exits 1 on any FAIL."""
    d = Path(a.dir)
    build = d / ("build" if not a.locale else f"build-{a.locale}")
    rep = _store_json(build / "report.json", f"build report (run `store build {a.dir}` first)")
    sb = _storyboard(d, a.locale)
    game = "game" in (sb.get("category") or "").lower()
    res = []

    def add(level, where, msg):
        res.append({"level": level, "where": where, "msg": msg})

    for key, t in rep.get("targets", {}).items():
        spec = TARGETS.get(key)
        if not spec:
            continue
        want = (spec["css"][0] * spec["dpr"], spec["css"][1] * spec["dpr"])
        shots = t.get("shots", [])
        n = len(shots)
        if spec["store"] == "ios":
            if not 1 <= n <= 10:
                add("FAIL", key, f"{n} screenshots; App Store Connect takes 1-10")
            elif n < 3:
                add("WARN", key, f"only {n} screenshots; search shows the first 3")
        else:
            if not 2 <= n <= 8:
                add("FAIL", key, f"{n} screenshots; Play takes 2-8 per device type")
            need = 3 if game and spec["device"] == "phone" else 4
            if n < need:
                add("WARN", key, f"{n} screenshots; Play's large-format surfaces need at least {need}")
        for i, e in enumerate(shots):
            where = f"{key} {Path(e['file']).name}"
            f = d / e["file"]
            if not f.exists():
                add("FAIL", where, "file missing; re-run build")
                continue
            im = Image.open(f)
            if im.size != want:
                add("FAIL", where, f"{im.size[0]}x{im.size[1]}, the {key} slot needs {want[0]}x{want[1]}")
            if im.mode != "RGB":
                add("FAIL", where, f"mode {im.mode}; stores need 24-bit RGB with no alpha")
            if f.stat().st_size > MAX_BYTES:
                add("FAIL", where, f"{f.stat().st_size / 1e6:.1f} MB; Play allows 8 MB")
            if spec["store"] == "play" and (max(im.size) > 3840 or min(im.size) < 320 or max(im.size) > 2 * min(im.size)):
                add("FAIL", where, "outside Play's 320-3840 px or 2:1 limits")
            fid = e.get("fidelity")
            if fid and (fid["mean"] > FID_MEAN_MAX or fid["p99"] > FID_P99_MAX):
                add("FAIL", where, f"the app UI drifted from the capture (dE mean {fid['mean']}, p99 {fid['p99']})")
            c = e.get("contrast") or {}
            for part in ("h1", "sub"):
                if c.get(part) is not None and c[part] < CONTRAST_MIN:
                    add("FAIL" if part == "h1" else "WARN", where, f"{'headline' if part == 'h1' else 'subhead'} contrast "
                        f"{c[part]}:1 (needs {CONTRAST_MIN}:1); change the slot's ink or set \"pill\": true")
            if spec["store"] == "ios" and spec["device"] == "phone" and i < 3 and e.get("h1_px") and e["h1_px"] * SEARCH_SCALE < SEARCH_MIN_PT:
                add("WARN", where, f"headline is {e['h1_px'] * SEARCH_SCALE:.1f} pt in search results (aim for "
                    f"{SEARCH_MIN_PT:g}+); shorten the caption")
            if spec["store"] == "play" and e.get("tagline_share") and e["tagline_share"] > TAGLINE_MAX:
                add("WARN", where, f"caption covers {e['tagline_share']:.0%} of the image; Play asks for under 20%")
            if e.get("float_on_caption"):
                add("WARN", where, "a floating element touches the caption; move it below the headline")
            if (e.get("float_cover") or 0) > FLOAT_COVER_MAX:
                add("WARN", where, f"front floats hide {e['float_cover']:.0%} of the app screen; keep them on its edges")
            if spec["store"] == "play" and not e.get("alt"):
                add("WARN", where, "no alt text")
    if "play-phone" in rep.get("targets", {}):
        shots = rep["targets"]["play-phone"]["shots"]
        need = 3 if game else 4
        big = sum(1 for e in shots if min(e["size"]) >= 1080)
        add("PASS" if big >= need else "WARN", "play-phone",
            f"{big} shots at 1080 px+; promotion on Play needs {need}+")
    for note in rep.get("notes", []):
        add("WARN", "build", note)
    listed = {Path(e["file"]).name for t in rep.get("targets", {}).values() for e in t.get("shots", [])}
    for key in TARGETS:
        folder = build / key
        if not folder.is_dir():
            continue
        if key not in rep.get("targets", {}):
            add("WARN", key, f"{folder} is from an earlier build, not this report; don't upload it")
            continue
        for f in folder.glob("[0-9][0-9]_*.*"):
            if f.suffix in (".png", ".jpg") and f.name not in listed:
                add("WARN", key, f"{f.name} is not in this build; don't upload it")
    for level, sid, msg in _lint(sb, sb.get("claims", "")):
        add(level, f"copy {sid}", msg)
    feat = build / "play-feature" / "feature.json"
    if feat.exists():
        for lv, msg in json.loads(feat.read_text()).get("checks", []):
            add(lv, "feature", msg)
    if a.idea:
        for lv, where, msg in _idea_check(d, build, rep, sb, a.model):
            add(lv, where, msg)
            if lv == "PASS":
                print(f"  PASS {where}: {msg}")
    (build / "check.json").write_text(json.dumps(res, indent=1))
    fails = [x for x in res if x["level"] == "FAIL"]
    warns = [x for x in res if x["level"] == "WARN"]
    for x in fails + warns:
        print(f"  {x['level']} {x['where']}: {x['msg']}")
    n = sum(len(t.get("shots", [])) for t in rep.get("targets", {}).values())
    print(f"store check: {n} shots, {len(fails)} FAIL, {len(warns)} WARN -> {build / 'check.json'}")
    if fails:
        sys.exit(1)


# ---------------------------------------------------------------- parsers

def _store_parsers(sub):
    p = sub.add_parser("store", help="App Store / Google Play screenshots from real captures "
                                     "(stages: capture, research, plan, art, build, check, feature, scene, compose)")
    ssub = p.add_subparsers(dest="scmd", required=True)
    q = ssub.add_parser("capture", help="capture the booted simulator/device with a clean status bar, or import a folder")
    q.add_argument("dir", help="store project dir (captures/, captures.json)")
    q.add_argument("--name", help="screen name, e.g. hunt ([a-z0-9_]); a slot refers to it")
    q.add_argument("--from", dest="from_dir", help="import every PNG/JPEG in this folder (names from the file stems)")
    q.add_argument("--platform", choices=["auto", "ios", "android"], default="auto")
    q.add_argument("--kind", choices=["auto", "phone", "tablet"], default="auto", help="device class (default: detected)")
    q.add_argument("--udid", help="iOS simulator UDID (default: the one booted simulator)")
    q.add_argument("--serial", help="adb serial (default: the one connected device)")
    q.add_argument("--url", help="open this deep link first (simctl openurl / am start VIEW)")
    q.add_argument("--wait", type=_pos_float, help="seconds to wait before the screenshot (default 2 after --url, else 0)")
    q.add_argument("--time", default="9:41", help="status bar clock")
    q.add_argument("--no-status-bar", dest="status_bar", action="store_false",
                   help="leave the status bar as it is (no simctl override / Android demo mode)")
    q.set_defaults(fn=cmd_store_capture)
    q = ssub.add_parser("research", help="top App Store listings for a term: slots 1-3, sheets and a category playbook")
    q.add_argument("dir")
    q.add_argument("--term", required=True, help='search term, e.g. "word game"')
    q.add_argument("--country", default="us", help="App Store country code")
    q.add_argument("--limit", type=_pos_int, default=12, help="apps to study")
    q.add_argument("--per-app", type=_pos_int, default=3, help="screenshots per app")
    q.add_argument("--no-ai", action="store_true", help="download and sheet only; no playbook call")
    q.add_argument("--model", default=None)
    q.set_defaults(fn=cmd_store_research)
    q = ssub.add_parser("plan", help="storyboard.json from the captures: slots, captions per store, layouts, alt text")
    q.add_argument("dir")
    q.add_argument("--app", help="'Name: what it does, for whom'")
    q.add_argument("--slots", type=_pos_int, help="number of slots (default min(6, captures))")
    q.add_argument("--claims", help="claims you can prove (awards, rankings), allowed in the copy")
    q.add_argument("--lint", action="store_true", help="re-lint an edited storyboard.json; no call")
    q.add_argument("--locale", help="storyboard.LOCALE.json instead of storyboard.json")
    q.add_argument("--model", default=None)
    q.add_argument("--dry-run", action="store_true", help="print the planned call and cost only")
    q.set_defaults(fn=cmd_store_plan)
    q = ssub.add_parser("art", help="art plates from each slot's capture, optional OCR-checked lettering, feature plate")
    q.add_argument("dir")
    q.add_argument("--slots", help="comma list of slot ids (default: all)")
    q.add_argument("--lettering", action="store_true", help="also draw each caption as display lettering")
    q.add_argument("--feature", action="store_true", help="also paint the Play feature graphic plate")
    q.add_argument("--no-plates", dest="plates", action="store_false", help="skip the slot plates")
    q.add_argument("--tries", type=_pos_int, default=3, help="lettering attempts until OCR reads it back exactly "
                                                            "(plates re-roll once on TEXT)")
    q.add_argument("--no-text-check", dest="text_check", action="store_false", help="skip the TEXT check on plates")
    q.add_argument("--jobs", type=_pos_int, default=4)
    q.add_argument("--force", action="store_true", help="repaint art that exists")
    q.add_argument("--model", default=None)
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_store_art)
    q = ssub.add_parser("build", help="composite every slot per store and device, measure, write store-ready images")
    q.add_argument("dir")
    q.add_argument("--stores", default="ios,play", help="ios,play")
    q.add_argument("--devices", default="auto", help="phone, tablet or both (default: tablets when captured)")
    q.add_argument("--variant", choices=["brand", "lettering"], default="brand",
                   help="HTML captions in the brand fonts, or the OCR-checked lettering from `store art --lettering`")
    q.add_argument("--frame", choices=["auto", "device", "card"], default="auto",
                   help="iOS: device frame or frameless card (Play is always frameless)")
    q.add_argument("--iphone-large", action="store_true", help="also the optional 1320x2868 large-display iPhone set "
                                                                  "(the required slot is 1206x2622)")
    q.add_argument("--no-caption", action="store_true", help="hide captions and lettering while keeping their layout")
    q.add_argument("--slots", help="comma list of slot ids (default: all, in storyboard order)")
    q.add_argument("--locale", help="build from storyboard.LOCALE.json into build-LOCALE/")
    q.add_argument("--no-pill", dest="pill", action="store_false", help="don't put a low-contrast caption on a pill")
    q.add_argument("--no-webfonts", dest="webfonts", action="store_false", help="system fonts instead of Google Fonts")
    q.add_argument("--timeout", type=_pos_float, default=60, help="seconds per render")
    q.set_defaults(fn=cmd_store_build)
    q = ssub.add_parser("check", help="store rules over the last build; exits 1 on a FAIL")
    q.add_argument("dir")
    q.add_argument("--locale")
    q.add_argument("--idea", action="store_true", help="hide-the-copy test: can a viewer tell what the app does from "
                                                       "the first three shots without their captions (2 text calls)")
    q.add_argument("--model", default=None)
    q.set_defaults(fn=cmd_store_check)
    q = ssub.add_parser("feature", help="Google Play feature graphic 1024x500")
    q.add_argument("dir")
    q.add_argument("--logo", help="transparent logo PNG (default: the app name in the headline font)")
    q.add_argument("--icon", help="app icon PNG, drawn as a rounded tile above the title or logo")
    q.add_argument("--title", help="title text when there is no --logo")
    q.add_argument("--subtitle", help="default: the storyboard's feature subtitle")
    q.add_argument("--no-webfonts", dest="webfonts", action="store_false")
    q.add_argument("--timeout", type=_pos_float, default=60)
    q.set_defaults(fn=cmd_store_feature)
    from .scenes import _scene_parsers
    _scene_parsers(ssub)
