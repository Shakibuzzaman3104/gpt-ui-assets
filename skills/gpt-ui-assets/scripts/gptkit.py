#!/usr/bin/env python3
"""gptkit: design-to-assets pipeline on any OpenAI-compatible API — UI mockups, asset inventory, 1:1 from-scratch
redraws, SVG icons, Android VectorDrawables, animation frames, Lottie and app launcher icon sets. Run `gptkit.py -h`.

Config (environment variables win over the config file):
  GPTKIT_BASE_URL / OPENAI_BASE_URL   default https://api.openai.com/v1
  GPTKIT_API_KEY  / OPENAI_API_KEY    required
  GPTKIT_IMAGE_MODEL                  default gpt-image-2
  GPTKIT_TEXT_MODEL                   default gpt-6 (vision + structured outputs)
  GPTKIT_CONFIG                       config file path, default ~/.config/gptkit/env (KEY=value lines)
"""
import argparse
import base64
import concurrent.futures as cf
import functools
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat

SVG_NS = "{http://www.w3.org/2000/svg}"
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


# ---------------------------------------------------------------- config + API

@functools.cache
def _config():
    cfg = {}
    path = Path(os.environ.get("GPTKIT_CONFIG", "~/.config/gptkit/env")).expanduser()
    if path.exists():
        for line in path.read_text().splitlines():
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
    return {"base": pick("GPTKIT_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1").rstrip("/"),
            "key": pick("GPTKIT_API_KEY", "OPENAI_API_KEY"),
            "image_model": pick("GPTKIT_IMAGE_MODEL", default="gpt-image-2"),
            "text_model": pick("GPTKIT_TEXT_MODEL", default="gpt-6"),
            "config_path": str(path)}


def _auth():
    c = _config()
    if not c["key"]:
        sys.exit("gptkit: no API key. Set GPTKIT_API_KEY (or OPENAI_API_KEY), or add GPTKIT_API_KEY=... to "
                 f"{c['config_path']}")
    return c["base"], {"Authorization": f"Bearer {c['key']}"}


def _call(path, *, json_body=None, files=None, data=None, timeout=900, retries=2):
    base, headers = _auth()
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(base + path, headers=headers, json=json_body, files=files, data=data, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(8 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise ApiError(r.status_code, f"HTTP {r.status_code} {path}: {r.text[:500]}")
            return r.json()
        except requests.RequestException as e:
            last = str(e)
            if attempt < retries:
                time.sleep(8 * (attempt + 1))
                continue
    raise RuntimeError(f"{path} failed after retries: {last}")


class ApiError(RuntimeError):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


OPTIONAL_PARAMS = ("input_fidelity", "quality", "output_format", "background")


def _with_fallback(send, params):
    """Retry without optional params a provider rejects (e.g. input_fidelity on models that lack it)."""
    params = dict(params)
    while True:
        try:
            return send(params)
        except ApiError as e:
            dropped = next((p for p in OPTIONAL_PARAMS if p in params and p in str(e)), None)
            if e.status != 400 or not dropped:
                raise
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
    item = resp["data"][0]
    meta = {k: resp.get(k) for k in ("quality", "size", "background")}
    meta["usage"] = resp.get("usage")
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"]), meta
    return requests.get(item["url"], timeout=120).content, meta


def generate(prompt, size="1024x1024", background="auto", model=None):
    body = {"model": model or _config()["image_model"], "prompt": prompt, "size": size,
            "output_format": "png", "quality": "high"}
    if background != "auto":
        body["background"] = background
    return _image_result(_with_fallback(lambda b: _call("/images/generations", json_body=b), body))


def edit(ref_png: bytes, prompt, size="1024x1024", background="auto", model=None, fidelity="high"):
    # One reference image only: multi-image edits were observed to return the reference itself.
    data = {"model": model or _config()["image_model"], "prompt": prompt, "size": size, "output_format": "png",
            "quality": "high", "input_fidelity": fidelity}
    if background != "auto":
        data["background"] = background
    files = [("image", ("ref.png", ref_png, "image/png"))]
    return _image_result(_with_fallback(lambda d: _call("/images/edits", data=d, files=files), data))


def _data_url(img_or_bytes):
    raw = img_or_bytes if isinstance(img_or_bytes, bytes) else _png_bytes(img_or_bytes)
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def ask(parts, schema=None, name="result", model=None):
    """parts: list of str | PIL.Image | bytes. Returns parsed JSON when schema/json requested, else text."""
    content = []
    for p in parts:
        if isinstance(p, str):
            content.append({"type": "input_text", "text": p})
        else:
            content.append({"type": "input_image", "image_url": _data_url(p)})
    body = {"model": model or _config()["text_model"], "store": False, "input": [{"role": "user", "content": content}]}
    if schema == "json":
        body["text"] = {"format": {"type": "json_object"}}
    elif schema:
        body["text"] = {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}
    resp = _call("/responses", json_body=body)
    text = "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []))
    return json.loads(text) if schema else text


# ---------------------------------------------------------------- image utils

def _rgba(p):
    return Image.open(p).convert("RGBA")


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


def ref_crop(mock, bbox, margin=0.6, long_side=1024):
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    box = (max(0, int(x - w * margin)), max(0, int(y - h * margin)),
           min(mock.width, int(x + w * (1 + margin))), min(mock.height, int(y + h * (1 + margin))))
    c = mock.crop(box).convert("RGB")
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
        d.rectangle([0, 0, T, 14], fill="white")
        d.text((3, 1), label[:48], fill="black")
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


def _design_view(mock, bbox):
    """The design element at WORK scale in Lab, its backdrop and a foreground mask segmented from it."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    s = WORK / max(w, h)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    arr = np.asarray(mock.crop((x, y, x + w, y + h)).convert("RGB").resize(size, Image.LANCZOS))
    ring = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    backdrop = tuple(int(v) for v in np.median(ring, 0))
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
        span = range(-M // 2, M // 2 + 1) if search else [0]
        for dy in span:
            for dx in span:
                am = BM[M + dy:M + dy + h, M + dx:M + dx + w]
                iou = (am & dm).sum() / max((am | dm).sum(), 1)
                if best is None or iou > best[0]:
                    best = (iou, BL[M + dy:M + dy + h, M + dx:M + dx + w], am)
    return best


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
    k = np.clip(dst.std(0) / np.maximum(src.std(0), 1e-3), 0.75, 1.33)
    a = np.asarray(asset.convert("RGBA"))
    L = _lab(a)
    L = L + strength * (dst.mean(0) + (L - src.mean(0)) * k - L)
    out = np.dstack([_lab_to_rgb(L), a[..., 3]])
    out[a[..., 3] == 0, :3] = 0
    return Image.fromarray(out, "RGBA")


def score(mock, bbox, asset, ref=None, transparent=True):
    """Fidelity vs the mockup element plus copy/crop red flags. Lower 'fid' (and 'diff') = closer to the design."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    crop = mock.crop((x, y, x + w, y + h)).convert("RGB")
    corners = [crop.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    backdrop = tuple(sum(c[i] for c in corners) // 4 for i in range(3))
    flags = []
    A = asset.getchannel("A")
    transp = sum(A.histogram()[:8]) / (asset.width * asset.height)
    bb = content_box(asset, 40) or (0, 0, asset.width, asset.height)
    g = asset.crop(bb)
    if transparent:
        if transp < 0.15:
            flags.append("OPAQUE")
        if bb[0] <= 2 or bb[1] <= 2 or bb[2] >= asset.width - 2 or bb[3] >= asset.height - 2:
            flags.append("CUT-OFF")
        small_g = g.resize((64, 64))
        pixels = small_g.get_flattened_data() if hasattr(small_g, "get_flattened_data") else small_g.getdata()
        solid = [p for p in pixels if p[3] > 200]
        near = sum(1 for p in solid if sum(abs(p[i] - backdrop[i]) for i in range(3)) < 36)
        if solid and near / len(solid) > 0.3:
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


def _rank(sc, meta):
    return (len(sc["flags"]), sc.get("fid", sc["diff"]) + (6 if (meta or {}).get("quality") == "low" else 0))


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
            return json.loads(f.read_text())
    return {}


def _screen_dp(platform, given=None):
    return given or (PLATFORMS[platform]["screen_dp"] if platform else 390)


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
- contrast/accessibility: text or icons below WCAG AA (4.5:1 body, 3:1 large text and icons), text under 11dp,
  controls that cannot get a {touch}dp touch target (a smaller visible pill or icon with room around it is fine).
- platform: {platform} guideline problems (navigation pattern, tab count, safe areas, standard controls).
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
                                                          "platform", "implementability", "consistency"]},
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
                "Do NOT upscale, trace or copy the screenshot pixels. Do NOT include the backdrop, container shape, "
                "text, or any neighbouring element. {bg} Centered with about 10% empty padding on every side.")

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
corner rounding, colours). Hard rules so it converts to Android VectorDrawable:
- Elements allowed: path, circle, ellipse, rect, line, polyline, polygon, g (no transform attr), defs,
  linearGradient, radialGradient, stop. No text, image, filter, mask, clipPath, use, style, CSS classes.
- Colours as hex. Gradients must use gradientUnits="userSpaceOnUse" with explicit coordinates.
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
- fill is a hex colour, url(#id) of a gradient in "defs" (gradientUnits="userSpaceOnUse", no gradientTransform) or
  "none"; stroke is a hex colour or "none", stroke_width in viewBox units (0 without stroke).
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
_MOTION = {"type": "object", "additionalProperties": False, "required": ["anchor_x", "anchor_y", "keys"],
           "properties": {"anchor_x": _NUMBER_T, "anchor_y": _NUMBER_T, "keys": {"type": "array", "items": {
               "type": "object", "additionalProperties": False,
               "required": ["t", "dx", "dy", "sx", "sy", "rotate", "opacity", "ease"],
               "properties": {**{k: _NUMBER_T for k in ("t", "dx", "dy", "sx", "sy", "rotate", "opacity")},
                              "ease": {"type": "string", "enum": list(EASES)}}}}}}
_PAINT = {"fill": {"type": "string"}, "opacity": _NUMBER_T, "stroke": {"type": "string"}, "stroke_width": _NUMBER_T}

VECTOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["viewBox", "defs", "static_layers", "animated_layers", "glows", "layer_order"],
    "properties": {
        "viewBox": {"type": "string"}, "defs": {"type": "string"},
        "layer_order": {"type": "array", "items": {"type": "string"}},
        "static_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "paths", "motion"],
            "properties": {"name": {"type": "string"}, "motion": _MOTION, "paths": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["d", *_PAINT],
                "properties": {"d": {"type": "string"}, **_PAINT}}}}}},
        "animated_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", *_PAINT, "keyframes", "motion"],
            "properties": {"name": {"type": "string"}, **_PAINT, "motion": _MOTION,
                           "keyframes": {"type": "array", "items": {"type": "string"}}}}},
        "glows": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "cx", "cy", "rx", "ry", "color", "opacity", "motion"],
            "properties": {"name": {"type": "string"}, **{k: _NUMBER_T for k in ("cx", "cy", "rx", "ry", "opacity")},
                           "color": {"type": "string"}, "motion": _MOTION}}},
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

def _load(d):
    d = Path(d)
    m = json.loads((d / "assets.json").read_text())
    return d, m, _rgba(d / m["mockup"])


def _select(m, only):
    names = set(only.split(",")) if only else None
    return [a for a in m["assets"] if not names or a["name"] in names]


ASPECTS = {"1024x1536": "portrait 2:3 (phone screen, taller than wide)", "1536x1024": "landscape 3:2 (desktop/tablet, wider than tall)",
           "1024x1024": "square 1:1"}


def _aspect_ok(img_bytes, size):
    w, h = Image.open(io.BytesIO(img_bytes)).size
    tw, th = map(int, size.split("x"))
    return (w > h) == (tw > th) and (w < h) == (tw < th)


def _layout_diff(a, b):
    return round(_mean_diff(a.convert("L").resize((64, 96)), b.convert("L").resize((64, 96))), 1)


def cmd_mockup(a):
    if a.edit and a.style_ref:
        sys.exit("gptkit: use --edit (revise a screen) or --style-ref (new screen in its style), not both")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    aspect = ASPECTS.get(a.size, a.size)
    platform = a.platform or (_design_meta(a.edit or a.style_ref).get("platform") if (a.edit or a.style_ref) else None)
    if not platform:
        print("  (no --platform: designing platform-neutral; pass --platform android|ios|web for native components)")
    rule = _platform_rule(platform)
    prompt = a.prompt if a.raw else MOCKUP_WRAP.format(prompt=a.prompt, aspect=aspect) + rule
    edit_prompt = f"{a.prompt} Keep the {aspect} canvas and everything not mentioned exactly as it is."
    style_prompt = a.prompt if a.raw else STYLE_REF_PROMPT.format(prompt=a.prompt, aspect=aspect) + rule
    if platform:
        meta_path = out / "design.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update({"platform": platform, "screen_dp": _screen_dp(platform)})
        meta_path.write_text(json.dumps(meta, indent=2))
    ref = _rgba(a.style_ref) if a.style_ref else None

    def one(i):
        # the backend picks aspect from the wording, not from `size`, and a style-ref edit can return the reference
        for _ in range(2):
            if a.edit:
                img, meta = edit(Path(a.edit).read_bytes(), edit_prompt, a.size, model=a.model)
            elif ref is not None:
                img, meta = edit(Path(a.style_ref).read_bytes(), style_prompt, a.size, model=a.model)
            else:
                img, meta = generate(prompt, a.size, model=a.model)
            meta["layout_diff"] = _layout_diff(ref, Image.open(io.BytesIO(img))) if ref is not None else None
            copied = ref is not None and meta["layout_diff"] < 10
            if copied:
                print(f"  ! variant {i + 1} copied the style reference (layout_diff {meta['layout_diff']}), retrying")
            if _aspect_ok(img, a.size) and not copied:
                break
        p = out / f"{a.name}_v{i + 1}.png"
        p.write_bytes(img)
        return p, meta
    with cf.ThreadPoolExecutor(a.variants) as ex:
        for p, meta in ex.map(one, range(a.variants)):
            extra = f" layout_diff={meta['layout_diff']}" if ref is not None else ""
            print(f"{p}  quality={meta['quality']} size={meta['size']}{extra}")


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
    issues = ask(parts, CRITIQUE_SCHEMA, "design_qa")["issues"]
    for i, x in enumerate(issues):
        x["id"] = i + 1
        if x["category"] == "contrast" and x["bbox"]["w"] > 2 and x["bbox"]["h"] > 2:
            x["measured_contrast"] = measured_contrast(img, x["bbox"])
    return issues


def _weight(issues):
    return sum(SEVERITY[x["severity"]] for x in issues)


def _issue_overlay(img, issues, out):
    ov = img.convert("RGB")
    d = ImageDraw.Draw(ov)
    col = {"high": "#ff2d55", "medium": "#ff9f0a", "low": "#0a84ff"}
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for x in issues:
        b = x["bbox"]
        d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=col[x["severity"]], width=4)
        ty = max(0, b["y"] - 36)
        d.rectangle([b["x"], ty, b["x"] + 40, ty + 36], fill=col[x["severity"]])
        d.text((b["x"] + 6, ty + 3), str(x["id"]), fill="white", font=font)
    ov.save(out)


def _print_issues(issues):
    for x in issues:
        mc = f"  [measured {x['measured_contrast']}:1]" if "measured_contrast" in x else ""
        print(f"  {x['id']:>3}. {x['severity']:6s} {x['category']:16s} {x['problem']}{mc}\n      edit: {x['edit']}")


def cmd_critique(a):
    src = Path(a.mockup)
    out = Path(a.out) if a.out else src.parent
    out.mkdir(parents=True, exist_ok=True)
    ref = _rgba(a.style_ref) if a.style_ref else None
    cur_path, cur = src, _rgba(src)
    a.platform = a.platform or _design_meta(src).get("platform") or "android"
    a.screen_dp = _screen_dp(a.platform, a.screen_dp)
    issues = critique(cur, a.screen_dp, a.platform, ref, a.note)
    log = {"mockup": str(src), "rounds": [{"image": str(src), "issues": issues, "weight": _weight(issues)}]}
    print(f"{src.name}: {len(issues)} issues, weight {_weight(issues)}")
    _print_issues(issues)
    _issue_overlay(cur, issues, out / f"{src.stem}_critique.png")
    min_w = SEVERITY[a.min_severity]
    for rnd in range(a.rounds if a.apply else 0):
        todo = [x for x in issues if SEVERITY[x["severity"]] >= min_w]
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
        listing = "\n".join(f"{x['id']}. [{x['severity']}] {x['problem']}" for x in todo)
        res = ask([f"FIRST image: a UI design. SECOND image: its revision, which was asked to fix these issues:\n{listing}\n"
                   "For each id say whether the SECOND image fixed it. Then list NEW defects the revision introduced "
                   "(changed layout or copy, garbled text, lost elements, style drift) in the same issue format, "
                   "bbox in the SECOND image's pixels. Do not re-list issues that exist in both.", cur.convert("RGB"),
                   new.convert("RGB")], VERIFY_SCHEMA, "design_qa_verify")
        fixed = {c["id"] for c in res["checks"] if c["fixed"]}
        left = [x for x in todo if x["id"] not in fixed]
        before, after = _weight(todo), _weight(left) + _weight(res["new_issues"])
        drift = round(_mean_diff(cur.resize((128, 192)), new.resize((128, 192))), 1)
        kept = after < before
        print(f"round {rnd + 1}: fixed {len(fixed)}/{len(todo)}, new {len(res['new_issues'])}, weight {before} -> {after}, "
              f"drift {drift} -> {'KEPT' if kept else 'discarded'}  ({p})")
        _print_issues([dict(x, id=f"n{i + 1}") for i, x in enumerate(res["new_issues"])])
        log["rounds"].append({"image": str(p), "fixed": sorted(fixed), "new_issues": res["new_issues"],
                              "weight_before": before, "weight_after": after, "drift": drift, "kept": kept})
        if not kept:
            break
        rest = [x for x in issues if SEVERITY[x["severity"]] < min_w]
        issues = [dict(x, id=i + 1) for i, x in enumerate(left + res["new_issues"] + rest)]
        cur_path, cur = p, new
    log["best"] = str(cur_path)
    (out / f"{src.stem}_critique.json").write_text(json.dumps(log, indent=2))
    print(f"-> {out / (src.stem + '_critique.json')}, overlay {out / (src.stem + '_critique.png')}, best: {cur_path}")


def cmd_manifest(a):
    mock_path = Path(a.mockup).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mock = _rgba(mock_path)
    local = out / "mockup.png"
    if mock_path != local.resolve():
        mock.save(local)
    platform = a.platform or _design_meta(mock_path).get("platform")
    screen_dp = _screen_dp(platform, a.screen_dp)
    native = (f"\n- Target platform: {PLATFORMS[platform]['label']}. Native components ({PLATFORMS[platform]['components']}) "
              "are built in code with the platform toolkit: never list a component itself (tab bar, app bar, switch, slider, "
              "chip, button, sheet) as an asset, only custom glyphs/artwork inside it (e.g. the tab icons, not the tab bar).") \
        if platform else ""
    res = ask([MANIFEST_PROMPT.format(W=mock.width, H=mock.height) + native, mock], MANIFEST_SCHEMA, "asset_manifest")
    system = re.compile(r"wifi|wi_fi|signal|cellular|battery|status_bar|clock|dynamic_island|home_indicator|gesture")
    dropped = [x["name"] for x in res["assets"] if system.search(x["name"].lower()) and
               (x["bbox"]["y"] + x["bbox"]["h"] < 0.05 * mock.height or x["bbox"]["y"] > 0.95 * mock.height)]
    if dropped:
        print(f"  (dropped system UI the OS draws: {', '.join(dropped)})")
        res["assets"] = [x for x in res["assets"] if x["name"] not in dropped]
    seen = set()
    for x in res["assets"]:
        n = re.sub(r"_(svg|png|icon|asset)$", "", re.sub(r"[^a-z0-9_]", "_", x["name"].lower()).strip("_"))
        while n in seen:
            n += "_2"
        seen.add(n)
        x["name"] = n
        x["material_symbol"], x["sf_symbol"] = _symbol_names(x.get("material_symbol", ""), x.get("sf_symbol", ""))
        x["dp"] = {"w": round(x["bbox"]["w"] * screen_dp / mock.width, 1),
                   "h": round(x["bbox"]["h"] * screen_dp / mock.width, 1)}
    manifest = {"mockup": "mockup.png", "width": mock.width, "height": mock.height, "platform": platform,
                "screen_dp": screen_dp, "assets": res["assets"]}
    (out / "assets.json").write_text(json.dumps(manifest, indent=2))
    ov = mock.convert("RGB")
    d = ImageDraw.Draw(ov)
    for x in res["assets"]:
        b = x["bbox"]
        col = "#ff2d55" if x["format"] == "png" else "#34c759"
        d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=col, width=3)
        d.text((b["x"] + 2, max(0, b["y"] - 12)), x["name"], fill=col)
    ov.save(out / "manifest_overlay.png")
    for x in res["assets"]:
        b = x["bbox"]
        print(f"{x['name']:34s} {x['kind']:12s} {x['format']}  bbox={b['x']},{b['y']},{b['w']},{b['h']}  "
              f"~{x['dp']['w']}x{x['dp']['h']}dp  transparent={x['transparent']} animated={x['animated']}")
    print(f"\n{len(res['assets'])} assets -> {out / 'assets.json'}   overlay -> {out / 'manifest_overlay.png'} "
          f"(red=png, green=svg)")


def cmd_spec(a):
    if a.recalibrate:
        return cmd_recalibrate(a.dir, a.scale)
    d, m, mock = _load(a.dir)
    platform = m.get("platform")
    target = (f" Target platform: {PLATFORMS[platform]['label']}. For every component set 'native' to the native "
              f"component that should implement it (from: {PLATFORMS[platform]['components']}; add its variant, e.g. "
              "'Card (elevated)'), 'system' for UI the OS draws (status bar, home indicator), or 'custom' when none fits. For every text style set 'platform_style' to the closest "
              f"style of {PLATFORMS[platform]['type']}.") if platform else \
        " Set 'native' to 'custom' and 'platform_style' to '' (no target platform given)."
    res = ask([f"This is an APPROVED UI design ({mock.width}x{mock.height}px; the screen is {m['screen_dp']}dp wide). "
               "Extract the design spec a developer needs to rebuild it 1:1: palette (hex), every text style "
               "(role, a sample string, likely font family, size in px of THIS image, weight, letter spacing, colour), "
               "corner radii, key spacings, and the component list with pixel bboxes." + target, mock], SPEC_SCHEMA, "design_spec")
    res["platform"] = platform
    res["px_to_dp"] = round(m["screen_dp"] / mock.width, 5)
    _calibrate(d, res, m, mock.width, a.scale)
    (d / "design_spec.json").write_text(json.dumps(res, indent=2))
    print(f"-> {d / 'design_spec.json'}  (multiply px values by px_to_dp={res['px_to_dp']} for dp/pt)")


# native widgets whose platform size is fixed: (pattern on spec 'native', dp, measured bbox side)
NATIVE_ANCHORS = {
    "android": [(r"navigation ?bar", 80, "h"), (r"extended ?(fab|floating)", 56, "h"),
                (r"^(?!.*extended).*(\bfab\b|floating ?action)", 56, "h"),
                (r"top ?app ?bar(?!.*(medium|large))", 64, "h"), (r"search ?bar", 56, "h"), (r"navigation ?rail", 80, "w")],
    "ios": [(r"toggle|switch", 31, "h"), (r"segmented|picker \(segmented", 32, "h")],
}


def _calibrate(d, spec, m, width, scale="auto"):
    """Scale px -> dp from native widgets of known size; GPT draws them smaller than the platform does."""
    design_k = round(m["screen_dp"] / width, 5)
    spec["px_to_dp_design"] = design_k
    spec["px_to_dp"] = design_k
    anchors = []
    for c in spec.get("components", []):
        for pat, dp, side in NATIVE_ANCHORS.get(m.get("platform") or "", []):
            px = c["bbox"][side]
            if px > 4 and re.search(pat, c.get("native", ""), re.I):
                anchors.append({"component": c["name"], "native": c["native"], "px": px, "dp": dp, "k": dp / px})
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
        for x in m["assets"]:
            x["dp"] = {"w": round(x["bbox"]["w"] * k, 1), "h": round(x["bbox"]["h"] * k, 1)}
        m["px_to_dp"] = spec["px_to_dp"]
        m["effective_width_dp"] = spec["effective_width_dp"]
        (d / "assets.json").write_text(json.dumps(m, indent=2))
        print(f"  native scale: GPT drew {', '.join(a['component'] for a in anchors[:3])} at {1 / factor:.2f}x platform size; "
              f"px_to_dp {design_k} -> {spec['px_to_dp']} (design depicts a ~{spec['effective_width_dp']}dp-wide screen; "
              "asset dp estimates updated). `spec --scale design` keeps the proportional scale.")
    else:
        print(f"  native scale: design matches platform sizes (factor {factor:.2f}); kept px_to_dp {design_k}")


def cmd_recalibrate(d, scale):
    d = Path(d)
    spec = json.loads((d / "design_spec.json").read_text())
    m = json.loads((d / "assets.json").read_text())
    _calibrate(d, spec, m, m["width"], scale)
    (d / "design_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"-> {d / 'design_spec.json'}  px_to_dp={spec['px_to_dp']}")


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
        ref = ref_crop(mock, b)
        bg = "Transparent background (real alpha)." if asset["transparent"] else "Opaque background matching the design."
        prompt = ASSET_PROMPT.format(kind=asset["kind"], name=asset["name"].replace("_", " "),
                                     description=asset["description"], colors=colors, bg=bg)
        prompt += CANDIDATE_FOCUS[cand % len(CANDIDATE_FOCUS)] if vary else ""
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", model)
    return asset["name"], cand, img, meta, ref


def _color_final(mock, asset, path, sc, strength, out_dir):
    """Colour-match a chosen candidate into out_dir/<stem>_cm.png, kept only if it scores better (the original stays)."""
    # opaque backgrounds are skipped: UI drawn over them skews the stats and desaturated the accents in tests
    if strength <= 0 or sc["flags"] or not asset["transparent"]:
        return None
    img = color_match(mock, asset["bbox"], _rgba(path), strength)
    new = score(mock, asset["bbox"], img, None, True)
    info = {"strength": strength, "fid_before": sc["fid"], "fid_after": new["fid"], "applied": new["fid"] < sc["fid"]}
    if info["applied"]:
        p = Path(out_dir) / f"{Path(path).stem}_cm.png"
        img.save(p)
        info.update(path=str(p), score=new)
    return info


def _current(entry):
    """Path of the image behind an asset's final file: the colour-matched copy when one was kept."""
    cm = entry.get("color_match") or {}
    return Path(cm["path"]) if cm.get("applied") else Path(entry["chosen"]["path"])


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


def cmd_forge(a):
    d, m, mock = _load(a.dir)
    todo = [x for x in _select(m, a.only) if x["format"] == "png" or a.include_svg]
    if not a.force:
        todo = [x for x in todo if not (d / "assets" / f"{x['name']}.png").exists()]
    if not todo:
        print("nothing to forge (use --force to redo)")
        return
    (d / "raw").mkdir(exist_ok=True)
    (d / "assets").mkdir(exist_ok=True)
    report_path = d / "forge_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    by_name = {x["name"]: x for x in todo}
    results = {x["name"]: [] for x in todo}
    jobs = [(x, c) for x in todo for c in range(a.candidates)]
    print(f"forging {len(todo)} assets x {a.candidates} candidates = {len(jobs)} image calls, {a.jobs} parallel")
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(_forge_job, mock, x, c, a.model, not a.same_prompt) for x, c in jobs]
        for f in cf.as_completed(futs):
            try:
                name, cand, raw, meta, ref = f.result()
            except Exception as e:
                print(f"  ! job failed: {e}")
                continue
            asset = by_name[name]
            p = d / "raw" / f"{name}_c{cand + 1}.png"
            p.write_bytes(raw)
            img = _rgba(p)
            sc = score(mock, asset["bbox"], img, ref, asset["transparent"])
            results[name].append({"path": str(p), "meta": {"quality": meta["quality"], "size": meta["size"]}, "score": sc})
            print(f"  {name} c{cand + 1}: fid={sc['fid']} (diff={sc['diff']} iou={sc['iou']} de={sc['de']} col={sc['col']}) "
                  f"flags={sc['flags'] or '-'} warn={sc['warn'] or '-'} quality={meta['quality']}")
    rows = []
    for name, cands in results.items():
        asset = by_name[name]
        if not cands:
            continue
        cands.sort(key=lambda c: _rank(c["score"], c["meta"]))
        best = cands[0]
        cm = _color_final(mock, asset, best["path"], best["score"], a.color_match, d / "raw")
        report[name] = {"chosen": best, "candidates": cands, "color_match": cm}
        if cm:
            print(f"  {name}: colour match {cm['strength']} fid {cm['fid_before']} -> {cm['fid_after']} "
                  f"({'kept' if cm['applied'] else 'discarded'})")
        final = _finalize(asset, _rgba(_current(report[name])))
        out = d / "assets" / f"{name}.png"
        final.save(out)
        report[name].update(final=str(out), final_size=final.size, dp=asset.get("dp"),
                            needs_attention=bool(best["score"]["flags"]))
        b = asset["bbox"]
        pad = int(max(b["w"], b["h"]) * 0.25)
        target = mock.crop((max(0, b["x"] - pad), max(0, b["y"] - pad), b["x"] + b["w"] + pad, b["y"] + b["h"] + pad))
        rows.append([(f"{name} (design)", target)] +
                    [(f"{'BEST ' if c is best else ''}fid={c['score']['fid']} {' '.join(c['score']['flags'])}", _rgba(c["path"]))
                     for c in cands] +
                    ([(f"BEST+colour fid={cm['fid_after']}", _rgba(cm["path"]))] if cm and cm["applied"] else []))
    report_path.write_text(json.dumps(report, indent=2))
    (d / "review").mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for i in range(0, len(rows), 6):
        print("review sheet ->", sheet(rows[i:i + 6], d / "review" / f"forge_{stamp}_{i // 6 + 1}.png"))
    flagged = [n for n, r in report.items() if r.get("needs_attention") and n in results]
    if flagged:
        print("NEEDS ATTENTION (every candidate flagged):", ", ".join(flagged))


def cmd_fix(a):
    d, m, mock = _load(a.dir)
    asset = next(x for x in m["assets"] if x["name"] == a.name)
    report = json.loads((d / "forge_report.json").read_text())
    cur = _rgba(_current(report[a.name]))
    cur_score = score(mock, asset["bbox"], cur, None, asset["transparent"])
    b = asset["bbox"]
    for rnd in range(a.rounds):
        T = 512
        pad = 6
        tgt = mock.crop((max(0, b["x"] - pad), max(0, b["y"] - pad), b["x"] + b["w"] + pad, b["y"] + b["h"] + pad))
        tgt = tgt.resize((round(tgt.width * T / tgt.height), T), Image.LANCZOS).convert("RGB")
        corner = tgt.getpixel((2, 2))
        g = trim(cur)
        cv = Image.new("RGBA", g.size, corner + (255,))
        cv.alpha_composite(g)
        cv = cv.resize((round(cv.width * T / cv.height), T), Image.LANCZOS).convert("RGB")
        sbs = Image.new("RGB", (tgt.width + cv.width + 30, T), "white")
        sbs.paste(tgt, (0, 0))
        sbs.paste(cv, (tgt.width + 30, 0))
        extra = f" The user also says: {a.note}" if a.note else ""
        res = ask(["LEFT is the target graphic from an approved design (low-res screenshot on its backdrop). RIGHT is "
                   "a recreated asset on a similar backdrop. List every visible difference RIGHT must fix to match LEFT "
                   "1:1 (silhouette, proportions, parts, colours with hex, gradients, glow, relative sizes). Ignore "
                   "resolution/sharpness and the backdrop. Then write ONE concise corrective edit instruction (<=100 "
                   f"words) limited to the real differences.{extra} JSON: {{\"differences\":[...],\"edit_instruction\":\"...\"}}",
                   sbs], "json")
        print(f"round {rnd + 1} differences:", *[f"  - {x}" for x in res["differences"]], sep="\n")
        prompt = ("Revise the attached standalone asset. Keep everything that already matches; change only what is listed. "
                  "Clean high-resolution redraw, " + ("transparent background, " if asset["transparent"] else "") +
                  "no backdrop, no text. Corrections: " + res["edit_instruction"])
        raw, meta = edit(_png_bytes(cur), prompt, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", a.model)
        p = d / "raw" / f"{a.name}_fix{int(time.time())}.png"
        p.write_bytes(raw)
        new = _rgba(p)
        sc = score(mock, b, new, None, asset["transparent"])
        cm = _color_final(mock, asset, p, sc, a.color_match, d / "raw")
        final_img, final_sc = (_rgba(cm["path"]), cm["score"]) if cm and cm["applied"] else (new, sc)
        better = _rank(final_sc, meta) < _rank(cur_score, None)
        print(f"round {rnd + 1}: fid {cur_score['fid']} -> {final_sc['fid']} (diff {cur_score['diff']} -> {final_sc['diff']}"
              f"{', colour-matched' if final_img is not new else ''}) flags={sc['flags'] or '-'} -> "
              f"{'KEPT' if better or a.accept else 'discarded (worse)'}  ({p})")
        if better or a.accept:
            cur, cur_score = final_img, final_sc
            report[a.name]["chosen"] = {"path": str(p), "meta": {"quality": meta["quality"], "size": meta["size"]}, "score": sc}
            report[a.name]["color_match"] = cm
            _finalize(asset, final_img).save(d / "assets" / f"{a.name}.png")
    (d / "forge_report.json").write_text(json.dumps(report, indent=2))


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
    if _chrome():
        return "chrome"
    if _cairosvg():
        return "cairosvg"
    return next((t for t in ("rsvg-convert", "qlmanage") if shutil.which(t)), None)


def _sized_svg(txt, size):
    m_ = re.search(r"<svg\b[^>]*>", txt)
    root_tag = re.sub(r'\s(width|height)="[^"]*"', "", m_.group(0)).replace("<svg", f'<svg width="{size}" height="{size}"', 1)
    return txt[:m_.start()] + root_tag + txt[m_.end():]


def _chrome_render(svgs, size):
    cols = max(1, min(len(svgs), 2048 // size))
    rows = math.ceil(len(svgs) / cols)
    cells = "".join(f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}" style="position:absolute;'
                    f'left:{i % cols * size}px;top:{i // cols * size}px;width:{size}px;height:{size}px">'
                    for i, s in enumerate(svgs))
    with tempfile.TemporaryDirectory() as td:
        html, png = Path(td) / "r.html", Path(td) / "r.png"
        html.write_text(f"<!doctype html><style>html,body{{margin:0;background:transparent}}</style>{cells}")
        subprocess.run([_chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--default-background-color=00000000", "--use-mock-keychain",
                        f"--window-size={cols * size},{rows * size}", "--virtual-time-budget=3000",
                        f"--screenshot={png}", html.as_uri()], capture_output=True, timeout=120)
        grid = Image.open(png).convert("RGBA")
    return [grid.crop((i % cols * size, i // cols * size, i % cols * size + size, i // cols * size + size))
            for i in range(len(svgs))]


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
    """Rasterise SVG texts with real alpha (headless Chrome, cairosvg, rsvg-convert, or Quick Look + matting)."""
    global _warned_render
    sized = []
    for s in svgs:
        try:
            ET.fromstring(s)
            sized.append(_sized_svg(s, size))
        except (ET.ParseError, AttributeError):
            sized.append(None)
    ok = [i for i, s in enumerate(sized) if s]
    out = [None] * len(svgs)
    if ok and _chrome():
        try:
            for i, im in zip(ok, _chrome_render([sized[i] for i in ok], size)):
                out[i] = im
            return out
        except Exception as e:
            print(f"  ! chrome render failed ({e}); falling back")
    if ok and not _renderer_name() and not _warned_render:
        print("  (no SVG renderer found: install Chrome/Chromium or pip install cairosvg for previews)")
        _warned_render = True
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
              "fill-opacity", "stroke-opacity", "stroke-dasharray", "color", "display", "visibility")
NAMED_COLORS = {"white": "#ffffff", "black": "#000000", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
                "yellow": "#ffff00", "orange": "#ffa500", "purple": "#800080", "gray": "#808080", "grey": "#808080",
                "silver": "#c0c0c0", "navy": "#000080", "teal": "#008080", "maroon": "#800000", "olive": "#808000",
                "lime": "#00ff00", "aqua": "#00ffff", "cyan": "#00ffff", "fuchsia": "#ff00ff", "magenta": "#ff00ff",
                "pink": "#ffc0cb", "transparent": "#00000000"}


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


def _inherit(el, parent):
    """Effective presentation attributes of el (style beats attributes; opacity multiplies down the tree)."""
    attrs = dict(parent)
    style = _style_dict(el)
    for k in PRES_ATTRS:
        v = style.get(k, el.get(k))
        if v is not None:
            attrs[k] = v.strip()
    op = style.get("opacity", el.get("opacity"))
    attrs["opacity"] = _num(parent.get("opacity"), 1.0) * (_num(op, 1.0) if op is not None else 1.0)
    return attrs


def _color(v, opacity=1.0, current=None):
    v = (v or "").strip().lower()
    if v == "currentcolor":
        v = (current or "#000000").strip().lower()
    v = NAMED_COLORS.get(v, v)
    m_ = re.match(r"rgba?\(([^)]*)\)", v)
    if m_:
        parts = [p.strip() for p in re.split(r"[\s,/]+", m_.group(1).strip()) if p.strip()]
        rgb = [round(_num(p) * 2.55) if p.endswith("%") else round(_num(p)) for p in parts[:3]]
        a = (_num(parts[3]) / 100 if parts[3].endswith("%") else _num(parts[3])) if len(parts) > 3 else 1.0
        v = "#" + "".join(f"{max(0, min(255, c)):02x}" for c in rgb) + f"{round(a * 255):02x}"
    if not v.startswith("#") or len(v) not in (4, 5, 7, 9):
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
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM.findall(args)]
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
            groups.append({"translateX": e, "translateY": f, "rotation": math.degrees(math.atan2(b, a)),
                           "scaleX": sx, "scaleY": (a * d - b * c) / sx})
        else:
            raise ValueError(f"transform {name}() is not expressible in VectorDrawable")
    return groups


def svg_to_vd(svg_text, width_dp=None, warnings=None):
    """Convert an SVG to VectorDrawable XML. Unsupported features raise ValueError; lossy ones append to warnings."""
    warns = warnings if warnings is not None else []
    root = ET.fromstring(svg_text)
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if v] or \
         [0, 0, _num(root.get("width"), 24), _num(root.get("height"), 24)]
    vx, vy, vw, vh = vb
    wdp = width_dp or vw
    hdp = round(wdp * vh / vw, 2)
    ids = {el.get("id"): el for el in root.iter() if el.get("id")}
    grads = {}
    for gel in root.iter():
        tag = gel.tag.replace(SVG_NS, "")
        if tag not in ("linearGradient", "radialGradient"):
            continue
        src = gel
        href = gel.get("href") or gel.get("{http://www.w3.org/1999/xlink}href")
        if not any(s.tag.replace(SVG_NS, "") == "stop" for s in gel) and href and href[1:] in ids:
            src = ids[href[1:]]
        if (gel.get("gradientUnits") or src.get("gradientUnits")) != "userSpaceOnUse":
            raise ValueError(f"gradient #{gel.get('id')} must use gradientUnits=userSpaceOnUse")
        if gel.get("gradientTransform"):
            raise ValueError(f"gradient #{gel.get('id')} uses gradientTransform (unsupported)")
        stops = []
        for s in src:
            if s.tag.replace(SVG_NS, "") == "stop":
                st = _style_dict(s)
                col = st.get("stop-color", s.get("stop-color", "#000000"))
                op = _num(st.get("stop-opacity", s.get("stop-opacity")), 1.0)
                off = s.get("offset", "0")
                off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
                stops.append((min(1.0, max(0.0, off)), _color(col, op)))
        grads[gel.get("id")] = (tag, gel, stops)
    used_grad = False

    def grad_xml(ref, attr, indent):
        nonlocal used_grad
        if ref not in grads:
            raise ValueError(f"gradient #{ref} not found")
        used_grad = True
        tag, gel, stops = grads[ref]
        if tag == "linearGradient":
            head = (f'android:type="linear" android:startX="{_num(gel.get("x1")):g}" android:startY="{_num(gel.get("y1")):g}" '
                    f'android:endX="{_num(gel.get("x2")):g}" android:endY="{_num(gel.get("y2")):g}"')
        else:
            head = (f'android:type="radial" android:centerX="{_num(gel.get("cx")):g}" android:centerY="{_num(gel.get("cy")):g}" '
                    f'android:gradientRadius="{_num(gel.get("r")):g}"')
        tile = {"reflect": "mirror", "repeat": "repeat"}.get(gel.get("spreadMethod"))
        if tile:
            head += f' android:tileMode="{tile}"'
        items = "".join(f'\n{indent}      <item android:offset="{o:g}" android:color="{c}"/>' for o, c in stops)
        return (f'\n{indent}  <aapt:attr name="android:{attr}">\n{indent}    <gradient {head}>{items}\n'
                f'{indent}    </gradient>\n{indent}  </aapt:attr>')

    def path_xml(d, attrs, ind):
        fill = attrs.get("fill", "#000000")
        stroke = attrs.get("stroke", "none")
        op = attrs.get("opacity", 1.0)
        cur = attrs.get("color")
        out = [f'{ind}<path', f'{ind}    android:pathData="{path_str(parse_path(d))}"']
        children = ""
        if fill not in ("none", "transparent"):
            m_ = re.match(r"url\(#([^)]+)\)", fill)
            if m_:
                children += grad_xml(m_.group(1), "fillColor", ind)
            else:
                out.append(f'{ind}    android:fillColor="{_color(fill, 1.0, cur)}"')
            fo = _num(attrs.get("fill-opacity"), 1.0) * op
            if fo < 1:
                out.append(f'{ind}    android:fillAlpha="{fo:g}"')
            if attrs.get("fill-rule") == "evenodd":
                out.append(f'{ind}    android:fillType="evenOdd"')
        if stroke not in ("none", "transparent"):
            m_ = re.match(r"url\(#([^)]+)\)", stroke)
            if m_:
                children += grad_xml(m_.group(1), "strokeColor", ind)
            else:
                out.append(f'{ind}    android:strokeColor="{_color(stroke, 1.0, cur)}"')
            out.append(f'{ind}    android:strokeWidth="{_num(attrs.get("stroke-width"), 1.0):g}"')
            so = _num(attrs.get("stroke-opacity"), 1.0) * op
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
        return "\n".join(out) + (f">{children}\n{ind}</path>" if children else " />")

    def wrap(body, tf, ind):
        for g in reversed(_transforms(tf)):
            attrs = " ".join(f'android:{k}="{v:g}"' for k, v in g.items() if v or k.startswith("scale"))
            body = f"{ind}<group {attrs}>\n" + "\n".join("    " + ln for ln in body.split("\n")) + f"\n{ind}</group>"
        return body

    def walk(el, inherited, ind):
        lines = []
        for ch in el:
            tag = ch.tag.replace(SVG_NS, "")
            if tag in ("defs", "linearGradient", "radialGradient", "title", "desc", "metadata"):
                continue
            if tag in ("text", "image", "use", "mask", "clipPath", "filter", "style", "pattern", "symbol"):
                raise ValueError(f"<{tag}> is not convertible")
            attrs = _inherit(ch, inherited)
            if attrs.get("display") == "none" or attrs.get("visibility") == "hidden":
                continue
            if ch.get("clip-path") or ch.get("mask") or ch.get("filter"):
                raise ValueError(f"<{tag}> uses clip-path/mask/filter (not convertible)")
            if tag in ("g", "a", "switch"):
                body = walk(ch, attrs, ind)
            else:
                d = _shape_path(ch)
                body = path_xml(d, attrs, ind) if d else ""
            if body:
                lines.append(wrap(body, ch.get("transform"), ind) if ch.get("transform") else body)
        return "\n".join(lines)
    body = walk(root, _inherit(root, {}), "    ")
    if root.get("transform"):
        body = wrap(body, root.get("transform"), "    ")
    if vx or vy:
        body = wrap(body, f"translate({-vx:g} {-vy:g})", "    ")
    ns = ' xmlns:aapt="http://schemas.android.com/aapt"' if used_grad else ""
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
        out = []
        for ch in el:
            if ch.tag == "group":
                tx, ty = float(g(ch, "translateX", 0)), float(g(ch, "translateY", 0))
                px, py = float(g(ch, "pivotX", 0)), float(g(ch, "pivotY", 0))
                tf = (f"translate({tx + px:g} {ty + py:g}) rotate({g(ch, 'rotation', 0)}) "
                      f"scale({g(ch, 'scaleX', 1)} {g(ch, 'scaleY', 1)}) translate({-px:g} {-py:g})")
                out.append(f'<g transform="{tf}">{walk(ch)}</g>')
            elif ch.tag == "path":
                fill, fa = paint(ch, "fillColor", float(g(ch, "fillAlpha", 1)))
                stroke, sa = paint(ch, "strokeColor", float(g(ch, "strokeAlpha", 1)))
                out.append(f'<path d="{g(ch, "pathData")}" fill="{fill}" fill-opacity="{fa:g}" stroke="{stroke}" '
                           f'stroke-opacity="{sa:g}" stroke-width="{g(ch, "strokeWidth", 0)}" '
                           f'stroke-linecap="{g(ch, "strokeLineCap", "butt")}" stroke-linejoin="{g(ch, "strokeLineJoin", "miter")}" '
                           f'stroke-miterlimit="{g(ch, "strokeMiterLimit", 4)}" '
                           f'fill-rule="{"evenodd" if g(ch, "fillType") == "evenOdd" else "nonzero"}"/>')
        return "".join(out)
    body = walk(root)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {g(root, "viewportWidth")} {g(root, "viewportHeight")}">'
            f'<defs>{"".join(defs)}</defs>{body}</svg>')


def vd_roundtrip(pairs, size=128):
    """[(svg, vd xml)] -> mean RGBA difference (0-255) between each SVG and its VectorDrawable rendered, or None."""
    ims = render_svgs([s for s, _ in pairs] + [vd_to_svg(x) for _, x in pairs], size)
    out = []
    for a, b in zip(ims[:len(pairs)], ims[len(pairs):]):
        if a is None or b is None:
            out.append(None)
            continue
        pa, pb = (np.asarray(im, dtype=float) for im in (a, b))
        pa[..., :3] *= pa[..., 3:] / 255
        pb[..., :3] *= pb[..., 3:] / 255
        out.append(round(float(np.abs(pa - pb).mean()), 2))
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


def aapt2_check(xml_files):
    """Compile + link VectorDrawable files with the Android SDK's aapt2; {file name: error} (empty = all valid)."""
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
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True)
            for line in (r.stderr + r.stdout).splitlines():
                m_ = re.search(r"drawable/([^/:]+\.xml)(?::\d+)?: error: (.*)", line)
                if m_:
                    errs.setdefault(m_.group(1), m_.group(2))
            if r.returncode and not errs:
                errs["*"] = (r.stderr or r.stdout).strip()[:300]
            if errs:
                break
    return errs


def cmd_vd(a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for f in a.svgs:
        p = Path(f)
        warns = []
        try:
            svg = p.read_text()
            xml = svg_to_vd(svg, a.dp, warns)
        except (ValueError, ET.ParseError) as e:
            print(f"  ! {p.name}: {e}")
            continue
        out = out_dir / f"{a.prefix + p.stem}.xml"
        out.write_text(xml)
        done.append((out, svg, xml, warns))
    rts = vd_roundtrip([(s, x) for _, s, x, _ in done]) if done and not a.no_check else [None] * len(done)
    for (out, _, _, warns), rt in zip(done, rts):
        note = "".join(f"  ! {w}" for w in dict.fromkeys(warns))
        if rt is not None and rt > 2:
            note += f"  ! render differs from the SVG (mean diff {rt})"
        print(f"  {out}{'  (render match ' + str(rt) + ')' if rt is not None and rt <= 2 else ''}{note}")
    written = [x[0] for x in done]
    if written and not a.no_check:
        errs = aapt2_check(written)
        if errs is None:
            print("  (aapt2 not found: set ANDROID_HOME to validate with the Android SDK)")
        elif errs:
            for k, v in errs.items():
                print(f"  ! aapt2: {k}: {v}")
        else:
            print(f"  aapt2 compile+link: {len(written)} file(s) valid")


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

SVG_FIX_PROMPT = """You wrote these SVGs for assets of an APPROVED design, but they do not match it closely enough. For each
asset the image shows LEFT the design element (zoomed screenshot on its backdrop) and RIGHT your SVG rendered at the same
scale on the same backdrop; then come the measured problems and your current SVG. Rewrite each SVG so RIGHT matches LEFT
1:1: fix the listed problems, keep what already matches, do not simplify, return the same names."""

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


def _units_box(render, vb):
    """Alpha bounding box of a render (made by render_svgs) in viewBox units."""
    bb = content_box(render, 25)
    if not bb:
        return None
    R = render.width
    s = R / max(vb[2], vb[3])
    ox, oy = (R - vb[2] * s) / 2, (R - vb[3] * s) / 2
    return (vb[0] + (bb[0] - ox) / s, vb[1] + (bb[1] - oy) / s, vb[0] + (bb[2] - ox) / s, vb[1] + (bb[3] - oy) / s)


def normalize_svg(txt, box, grid=24, pad=2.0):
    """Scale coordinates (never adding transforms) so content box `box` fits a grid x grid viewBox inside `pad`."""
    root = ET.fromstring(txt)
    if any(el.get("transform") or el.get("gradientTransform") for el in root.iter()):
        return None
    s = (grid - 2 * pad) / max(box[2] - box[0], box[3] - box[1], 1e-6)
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
        if st:
            el.set("style", ";".join(f"{k}:{v}" for k, v in st.items()))
        stroked |= (st.get("stroke") or el.get("stroke") or "none") != "none"
    if stroked and root.get("stroke-width") is None and "stroke-width" not in _style_dict(root):
        root.set("stroke-width", _fmt(s))
    for k in ("width", "height"):
        root.set(k, str(grid))
    root.set("viewBox", f"0 0 {grid} {grid}")
    return ET.tostring(root, encoding="unicode")


def svg_style(txt):
    """Effective style facts of an SVG: viewBox size, stroke widths, caps/joins, fill/stroke use, colours."""
    root = ET.fromstring(txt)
    vb = _vb(root)
    info = {"viewBox": [vb[2], vb[3]], "strokes": [], "caps": set(), "joins": set(), "filled": False,
            "stroked": False, "colors": set(), "transform": False}

    def walk(el, attrs):
        for ch in el:
            tag = ch.tag.replace(SVG_NS, "")
            at = _inherit(ch, attrs)
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
    walk(root, _inherit(root, {}))
    return info


def _stroke24(st):
    return round(float(np.median(st["strokes"])) * 24 / max(st["viewBox"]), 2) if st["strokes"] else None


def icon_set_report(entries, tol=0.15):
    """entries: [{name, svg, size (design px or None)}] -> consistency notes (stroke outliers per size group, caps, viewBox)."""
    rows, notes = [], []
    for e in entries:
        try:
            st = svg_style(e["svg"])
        except ET.ParseError:
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
    return json.loads(p.read_text()) if p.exists() else None


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
            svg_to_vd(svg, None, warns)
            issues += [f"not VectorDrawable-safe: {w}" for w in dict.fromkeys(warns)]
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


def _grid_normalize(svgs, grid=24, pad=2.0):
    """[svg] -> [svg rescaled onto the grid, or None], using one batch render to measure each content box."""
    out = []
    for s, r in zip(svgs, render_svgs(svgs, 512)):
        try:
            box = r and _units_box(r, _vb(ET.fromstring(s)))
            out.append(normalize_svg(s, box, grid, pad) if box else None)
        except (ValueError, ET.ParseError):
            out.append(None)
    return out


def cmd_svg(a):
    d, m, mock = _load(a.dir)
    todo = [x for x in _select(m, a.only) if x["format"] == "svg" or a.only]
    for sub in ("preview", "raw"):
        (d / "svg" / sub).mkdir(parents=True, exist_ok=True)
    spec = _load_spec(d)
    per_icon = {i["name"]: i for i in (spec or {}).get("icons", [])}
    by_name = {x["name"]: x for x in todo}
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
        return ask(parts, SVG_SCHEMA, "svg_files")["files"]

    def fix(batch):
        parts = [SVG_FIX_PROMPT + "\n" + SVG_RULES + _spec_text(spec)]
        for name, e in batch:
            parts += [f"Asset \"{name}\": {by_name[name]['description']}"]
            if e["image"] is not None:
                parts.append(e["image"])
            parts.append("Problems: " + "; ".join(e["issues"]) + f"\nCurrent SVG:\n{e['svg']}")
        return ask(parts, SVG_SCHEMA, "svg_files")["files"]
    first = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for files in ex.map(run, batches):
            for f in files:
                first[_norm_name(f["name"])] = f["svg"]
    evals = _evaluate_svgs(mock, by_name, first)
    history = {n: [e] for n, e in evals.items()}
    todo_fix = [(n, e) for n, e in evals.items() if e["issues"] and n in by_name] if a.retry else []
    if todo_fix:
        print(f"retrying {len(todo_fix)} below threshold: {', '.join(n for n, _ in todo_fix)}")
        with cf.ThreadPoolExecutor(4) as ex:
            fixed = {}
            for files in ex.map(fix, [todo_fix[i:i + 4] for i in range(0, len(todo_fix), 4)]):
                fixed.update({_norm_name(f["name"]): f["svg"] for f in files})
        for n, e in _evaluate_svgs(mock, by_name, {k: v for k, v in fixed.items() if k in history}).items():
            history[n].append(e)
    report_path = d / "svg_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    rows = []
    best = {n: min(t, key=_svg_rank) for n, t in history.items()}
    final = {n: b["svg"] for n, b in best.items()}
    grid_ok = [n for n, b in best.items() if a.grid and by_name.get(n, {}).get("kind") == "icon" and
               not any(i.startswith("invalid") for i in b["issues"])]
    for n, norm in zip(grid_ok, _grid_normalize([final[n] for n in grid_ok], a.grid, a.pad)):
        if norm:
            final[n] = norm
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
        if grid and asset.get("dp"):
            line += f"  (24-grid; use at ~{round(max(asset['dp']['w'], asset['dp']['h']) * grid / (grid - 2 * a.pad))}dp)"
        if asset.get("material_symbol") or asset.get("sf_symbol"):
            line += f"  [material: {asset.get('material_symbol') or '-'}, sf: {asset.get('sf_symbol') or '-'}]"
        print(line)
        for i in b["issues"]:
            print(f"      ! {i}")
        report[name] = {"svg": str(p), "grid": grid, "score": sc, "issues": b["issues"], "tries": len(tries),
                        "kept_try": tries.index(b) + 1}
        if sc and prev:
            rows.append([(f"{name}: design | svg", b["image"])] +
                        [(f"try {i + 1}: rel={t['score']['rel']}" + (" KEPT" if t is b else ""), t["render"])
                         for i, t in enumerate(tries) if t["score"]] +
                        [(f"final ink={sc['ink']} dE={sc['dE']}", prev)])
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


def _svg_check(d, m, mock, by_name):
    have = {n: (d / "svg" / f"{n}.svg").read_text() for n in by_name if (d / "svg" / f"{n}.svg").exists()}
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
    _, notes = icon_set_report([{"name": n, "svg": s, "size": max(by_name[n]["bbox"]["w"], by_name[n]["bbox"]["h"])}
                                for n, s in have.items() if by_name[n].get("kind") == "icon"])
    print("icon set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")


def cmd_iconspec(a):
    d, m, mock = _load(a.dir)
    icons = [x for x in m["assets"] if x["format"] == "svg"]
    parts = [ICON_SPEC_PROMPT.format(W=mock.width, H=mock.height), mock.convert("RGB")]
    for x in icons:
        parts += [f"Icon \"{x['name']}\" ({x['description'][:160]})", ref_crop(mock, x["bbox"], margin=0.25, long_side=256)]
    spec = ask(parts, ICON_SPEC_SCHEMA, "icon_spec")
    for i in spec["icons"]:
        i["name"] = _norm_name(i["name"])
        i["material_symbol"], i["sf_symbol"] = _symbol_names(i["material_symbol"], i["sf_symbol"])
    (d / "icon_spec.json").write_text(json.dumps(spec, indent=2))
    names = {i["name"]: i for i in spec["icons"]}
    for x in m["assets"]:
        if x["name"] in names:
            x["material_symbol"], x["sf_symbol"] = names[x["name"]]["material_symbol"], names[x["name"]]["sf_symbol"]
    (d / "assets.json").write_text(json.dumps(m, indent=2))
    print(f"{spec['style']} icons, stroke {spec['stroke_px']:g}px ({spec['stroke_24']:g} on 24 grid), {spec['linecap']} caps, "
          f"{spec['linejoin']} joins, radius {spec['corner_radius_24']:g}")
    print(f"traits: {spec['traits']}")
    for s in spec["states"]:
        print(f"  state {s['state']}: {s['treatment']} {s['color']}")
    for i in spec["icons"]:
        print(f"  {i['name']:22s} {i['state']:18s} {i['treatment']:9s} material={i['material_symbol'] or '-':16s} "
              f"sf={i['sf_symbol'] or '-'}")
    print(f"-> {d / 'icon_spec.json'} (symbol names merged into assets.json)")


def _symbol_names(material, sf):
    material = material.strip() if re.fullmatch(r"[a-z0-9_]+", material.strip()) else ""
    sf = sf.strip() if re.fullmatch(r"[a-z0-9]+(\.[a-z0-9]+)*", sf.strip()) else ""
    return material, sf


def cmd_icons(a):
    d, m, mock = _load(a.dir)
    concepts = [c.strip() for c in a.concepts.split(",") if c.strip()]
    spec = _load_spec(d)
    out = Path(a.out) if a.out else d / "icons"
    out.mkdir(parents=True, exist_ok=True)
    have = [x for x in m["assets"] if x["format"] == "svg" and x.get("kind") == "icon" and (d / "svg" / f"{x['name']}.svg").exists()]
    if not have:
        sys.exit("icons: no existing svg icons in DIR/svg; run `svg` first so the new icons can match them")
    srcs = [(d / "svg" / f"{x['name']}.svg").read_text() for x in have]
    srcs = [n or s for s, n in zip(srcs, _grid_normalize(srcs))]
    existing = [{"asset": x, "svg": s, "style": svg_style(s), "sw": _stroke24(svg_style(s))} for x, s in zip(have, srcs)]
    size = float(np.median([max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) for e in existing]))
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
    prompt = ICONS_PROMPT.format(concepts=", ".join(concepts), treatment=tr, sw=f"{sw:g}", cap=cap, join=join, color=color)
    exemplars = "\n".join(f"--- {e['asset']['name']}.svg\n{e['svg']}" for e in pool[:5])
    parts = [prompt + "\n" + SVG_RULES + _spec_text(spec), mock.convert("RGB"), _rgba(ref).convert("RGB"),
             f"Existing icon SVG sources, on the same 24 grid:\n{exemplars}"]
    files = {}
    for attempt in range(2):
        for f in ask(parts, SVG_SCHEMA, "svg_files")["files"]:
            files[_norm_name(f["name"])] = f["svg"]
        bad = {}
        for n, s in files.items():
            try:
                svg_to_vd(s)
            except (ValueError, ET.ParseError) as e:
                bad[n] = str(e)
        if not bad:
            break
        print("  retrying invalid:", "; ".join(f"{k}: {v}" for k, v in bad.items()))
        parts = parts + [f"These files were invalid, fix them and return ALL icons again: {bad}"]
    for n, s in files.items():
        if _vb(ET.fromstring(s))[2:] != [24.0, 24.0]:
            s = _grid_normalize([s])[0] or s
        if a.snap:
            s = re.sub(r'stroke-width="[^"]*"', f'stroke-width="{sw:g}"', s)
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
        st = svg_style(files[n])
        print(f"  {out / (n + '.svg')}  stroke={_stroke24(st)} caps={'/'.join(sorted(st['caps'])) or '-'} "
              f"{'filled' if st['filled'] else ''}{' outlined' if st['stroked'] else ''}")
    print(f"target: {treatment}, stroke {sw:g}, {cap} caps, {join} joins, {color}; preview -> {out / 'preview.png'}")
    print("set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")
    if a.vd:
        a2 = argparse.Namespace(svgs=[str(out / f"{n}.svg") for n in names], out=a.vd, dp=a.dp, prefix="ic_", no_check=False)
        cmd_vd(a2)


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
    """Best scale/offset placing `layer` (a part of `target`) onto target's canvas; returns the placed layer."""
    tm = target.getchannel("A").resize((grid, grid)).point(lambda v: 255 if v > 100 else 0)
    lm_full = layer.getchannel("A").point(lambda v: 255 if v > 100 else 0)
    best = (-1, 1.0, 0, 0)
    for s10 in range(85, 116, 3):
        s = s10 / 100
        size = (max(1, round(grid * s)), max(1, round(grid * s)))
        lm = lm_full.resize(size)
        for dx in range(-grid // 10, grid // 10 + 1, 2):
            for dy in range(-grid // 10, grid // 10 + 1, 2):
                canvas = Image.new("L", (grid, grid), 0)
                canvas.paste(lm, ((grid - size[0]) // 2 + dx, (grid - size[1]) // 2 + dy))
                inter = ImageStat.Stat(ImageChops.multiply(canvas, tm)).sum[0]
                outside = ImageStat.Stat(ImageChops.subtract(canvas, tm)).sum[0]
                sc = inter - 2 * outside
                if sc > best[0]:
                    best = (sc, s, dx, dy)
    _, s, dx, dy = best
    W, H = target.size
    placed = layer.resize((round(W * s), round(H * s)), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(placed, ((W - placed.width) // 2 + round(dx * W / grid), (H - placed.height) // 2 + round(dy * H / grid)))
    return out


def _fit_moving(src, static, frame, grid=128):
    """Scale + bottom-centre anchor that best overlays `frame` (moving part) on `src`, given its static layer."""
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
    best = None
    for f in np.linspace(0.55, 1.25, 15):
        sc = s0 * f
        fw, fh = max(1, round(fcrop.width * sc * k)), max(1, round(fcrop.height * sc * k))
        if fw > tw or fh > th + th // 2:
            continue
        fm = np.asarray(fcrop.getchannel("A").resize((fw, fh))) > 100
        for y in range(-fh // 2, th - fh // 2):
            for x in range(0, tw - fw + 1):
                y0, y1 = max(0, y), min(th, y + fh)
                if y1 <= y0:
                    continue
                sub = fm[y0 - y:y1 - y]
                hit = (sub & visible[y0:y1, x:x + fw]).sum()
                spill = (sub & ~allowed[y0:y1, x:x + fw]).sum() + fm[:max(0, -y)].sum()
                miss = visible.sum() - hit
                score = hit - 2 * spill - 0.7 * miss
                if best is None or score > best[0]:
                    best = (score, sc, x, y, fw, fh)
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


def _write_anim(out, anim, seq, fps, a, labels=None, lottie=None):
    """preview.gif, animated WebP/APNG, Lottie, animation-list XML and the review sheet for a frame sequence."""
    formats = {f.strip() for f in a.formats.split(",")}
    dur = round(1000 / fps)
    small = [_fit_px(f, a.px) for f in seq]
    wrote = []
    if "gif" in formats:
        bg = Image.new("RGBA", small[0].size, a.preview_bg)
        gif = [Image.alpha_composite(bg, f).convert("P", palette=Image.ADAPTIVE) for f in small]
        gif[0].save(out / "preview.gif", save_all=True, append_images=gif[1:], duration=dur, loop=0, disposal=2)
        wrote.append("preview.gif")
    if "webp" in formats:
        small[0].save(out / f"{anim}.webp", save_all=True, append_images=small[1:], duration=dur, loop=0, quality=90,
                      method=4)
        wrote.append(f"{anim}.webp")
    if "apng" in formats:
        small[0].save(out / f"{anim}.apng", format="PNG", save_all=True, append_images=small[1:], duration=dur, loop=0)
        wrote.append(f"{anim}.apng")
    if "lottie" in formats:
        lot = lottie or lottie_image_sequence(small, fps, anim)
        (out / f"{anim}.json").write_text(json.dumps(lot, separators=(",", ":")))
        (out / "preview.html").write_text(_lottie_html(lot, a.preview_bg))
        wrote += [f"{anim}.json", "preview.html"]
    items = "\n".join(f'    <item android:drawable="@drawable/{anim}_frame_{i + 1:02d}" android:duration="{dur}" />'
                      for i in range(len(seq)))
    (out / f"{anim}_anim.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<animation-list xmlns:android="http://schemas.android.com/apk/res/android"\n'
        f'    android:oneshot="false">\n{items}\n</animation-list>\n')
    labels = labels or [f"f{i + 1}" for i in range(len(seq))]
    sheet([list(zip(labels, seq))[j:j + 8] for j in range(0, len(seq), 8)], out / "frames_review.png", T=200)
    print(f"{len(seq)} frames @ {fps:g} fps -> {out}  ({', '.join(wrote + ['frames_review.png', anim + '_anim.xml'])})")


def _frames_procedural(a, src, out):
    period = MOTIONS[a.procedural]
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
        return i, j, edit(_png_bytes(_square(crops[j])[0]), prompt, "1024x1024", "transparent", a.model)[0]
    with cf.ThreadPoolExecutor(2) as ex:
        redraws = list(ex.map(job, bad))
    for i, j, raw in redraws:
        p = out / f"reroll_f{i + 1:02d}.png"
        p.write_bytes(raw)
        img = _rgba(p)
        bb = _robust_box(np.asarray(img.getchannel("A")))
        if not bb:
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


def cmd_frames(a):
    d, m, mock = _load(a.dir)
    src = _rgba(d / "assets" / f"{a.name}.png")
    out = d / "frames" / (f"{a.name}_{a.procedural}" if a.procedural else a.name)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("frame_*.png")) + list(out.glob("moving_f*.png")):
        old.unlink()
    if a.procedural:
        return _frames_procedural(a, src, out)
    if not a.moving:
        sys.exit("frames: --moving is required (or use --procedural)")
    count, fps = a.count or 8, a.fps or 12
    rows_n = math.ceil(count / a.cols)
    sheet_size = _size_for(a.cols * src.width, rows_n * src.height)
    canvas = {"1536x1024": "wide landscape", "1024x1536": "tall portrait"}.get(sheet_size, "square")
    motion = a.motion or "a gentle, natural looping motion"
    moving_prompt = (f"The attached image is an APPROVED asset. Produce ONLY its moving part — {a.moving} — "
                     f"{'with NO ' + a.static + ' at all' if a.static else ''}, as {count} animation frames of {motion}, "
                     f"laid out as a sprite sheet of {a.cols} columns x {rows_n} rows of equal cells on a {canvas} canvas "
                     "(left-to-right, top-to-bottom). "
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
    sq_src, sq_off = _square(src)
    with cf.ThreadPoolExecutor(2) as ex:
        futs = {k: ex.submit(edit, _png_bytes(sq_src if k == "static" else src), p, sz, "transparent", a.model)
                for k, (p, sz) in jobs.items()}
        for k, f in futs.items():
            (out / f"{k}_raw.png").write_bytes(f.result()[0])
    crops = _slice_sheet(_remove_dividers(_rgba(out / "sheet_raw.png")), count, a.cols)
    if not crops:
        sys.exit("frames: no frames found in the sheet")
    static = ref_mask = None
    if a.static:
        sq_static = _register(_rgba(out / "static_raw.png").resize(sq_src.size, Image.LANCZOS), sq_src)
        static = sq_static.crop((sq_off[0], sq_off[1], sq_off[0] + src.width, sq_off[1] + src.height))
        static.save(out / "static.png")
        ref_mask = _moving_mask(static)
    qa = _frame_qa(_align_frames(crops, a.anchor), src, ref_mask)
    bad = _qa_report(qa)
    if a.reroll and bad:
        crops, qa = _reroll(a, crops, qa, bad, src, ref_mask, motion, out)
        _qa_report(qa, " after re-roll")
    frames = _align_frames(crops, a.anchor)
    if a.match_colors:
        frames = _match_colors(frames, src, ref_mask)
        qa = _frame_qa(frames, src, ref_mask)
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


def _fmtn(v, nd=2):
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
                          "paths": [([_ellipse(g["cx"], g["cy"], g["rx"], g["rx"])], {"fill": "glow"})]}
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
    """Per layer: 'move' = max mean point displacement from the t=0 pose / the layer's size; 'fade' = opacity range."""
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
        rep[L["name"]] = {"kind": L["kind"], "move": round(move, 4), "fade": round(max(ops) - min(ops), 3)}
    moving = [r["move"] for r in rep.values() if r["kind"] != "glow"]
    return {"layers": rep, "motion": round(sum(moving) / len(moving), 4) if moving else None}


def _normalize_vec(vec):
    """Drop a closing keyframe that repeats the first (it stalls the loop) and store canonical path data."""
    for al in vec.get("animated_layers", []):
        keys = _align_keys([_subpaths(k) for k in al["keyframes"]])
        if len(keys) > 2:
            a, b = _points(keys[0]), _points(keys[-1])
            x0, y0, x1, y1 = _bbox(a)
            if max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b)) < 0.005 * math.hypot(x1 - x0, y1 - y0):
                keys = keys[:-1]
        al["keyframes"] = [_d(k) for k in keys]
    return vec


def _validate_vec(vec):
    errs = []
    ids = set(re.findall(r'\bid="([^"]+)"', vec.get("defs", "")))
    names = []

    def paint(where, p):
        for k in ("fill", "stroke"):
            ref = re.match(r"url\(#([^)]+)\)", p.get(k) or "")
            if ref and ref.group(1) not in ids:
                errs.append(f"{where}: {k} references missing gradient #{ref.group(1)}")
            elif not ref and (p.get(k) or "none") != "none" and not re.fullmatch(r"#[0-9a-fA-F]{3,8}", p[k]):
                errs.append(f"{where}: {k} {p[k]!r} is not a hex colour, url(#id) or none")
    for sl in vec.get("static_layers", []):
        names.append(sl["name"])
        if not sl["paths"]:
            errs.append(f"static {sl['name']}: no paths")
        for p in sl["paths"]:
            try:
                _subpaths(p["d"])
            except (ValueError, IndexError, TypeError) as e:
                errs.append(f"static {sl['name']}: {e}")
            paint(f"static {sl['name']}", p)
    for al in vec.get("animated_layers", []):
        names.append(al["name"])
        try:
            if len(al["keyframes"]) < 2:
                errs.append(f"animated {al['name']}: needs at least 2 keyframes")
            _align_keys([_subpaths(k) for k in al["keyframes"]])
        except (ValueError, IndexError, TypeError) as e:
            errs.append(f"animated {al['name']}: {e}")
        paint(f"animated {al['name']}", al)
    for g in vec.get("glows", []):
        names.append(g["name"])
        if g["rx"] <= 0 or g["ry"] <= 0:
            errs.append(f"glow {g['name']}: rx/ry must be > 0")
        if not re.fullmatch(r"#[0-9a-fA-F]{3,8}", g["color"]):
            errs.append(f"glow {g['name']}: color must be a hex colour")
    if len(set(names)) != len(names):
        errs.append("layer names must be unique")
    return errs


def _vec_vb(vec):
    return [float(x) for x in re.split(r"[\s,]+", vec["viewBox"].strip())]


def _fit_viewbox(vec, margin=0.02):
    """viewBox grown symmetrically so every pose of the animation (plus stroke) stays inside; None if it already fits."""
    x, y, w, h = _vec_vb(vec)
    rest, pts = [(x, y), (x + w, y + h)], []
    for L in _layers(vec):
        if L["kind"] == "glow":
            continue
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


def _matrix(tf):
    """SVG transform list -> (a, b, c, d, e, f)."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    mul = lambda p, q: (p[0] * q[0] + p[2] * q[1], p[1] * q[0] + p[3] * q[1], p[0] * q[2] + p[2] * q[3],
                        p[1] * q[2] + p[3] * q[3], p[0] * q[4] + p[2] * q[5] + p[4], p[1] * q[4] + p[3] * q[5] + p[5])
    for fn, args in re.findall(r"(\w+)\s*\(([^)]*)\)", tf or ""):
        v = [float(n) for n in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        if fn == "matrix" and len(v) == 6:
            q = tuple(v)
        elif fn == "translate":
            q = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif fn == "scale":
            q = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif fn == "rotate":
            r = math.radians(v[0])
            cx, cy = (v[1], v[2]) if len(v) > 2 else (0, 0)
            q = mul(mul((1, 0, 0, 1, cx, cy), (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)), (1, 0, 0, 1, -cx, -cy))
        else:
            continue
        m = mul(m, q)
    return m


def _defs_root(vec):
    defs = vec.get("defs", "") or ""
    return ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{defs if "<defs" in defs else f"<defs>{defs}</defs>"}</svg>')


def _gradient(defs_root, gid):
    """(linear, start, end, [(offset, hex, alpha)]) with any gradientTransform baked in (radius scaled by sqrt|det|)."""
    gel = next((e for e in defs_root.iter() if e.get("id") == gid), None)
    if gel is None:
        raise ValueError(f"gradient {gid} not found in defs")
    stops = []
    for s in gel:
        if s.tag.replace(SVG_NS, "") != "stop":
            continue
        style = dict(kv.split(":", 1) for kv in (s.get("style") or "").replace(" ", "").split(";") if ":" in kv)
        off = s.get("offset", "0")
        off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
        stops.append((min(1.0, max(0.0, off)), s.get("stop-color") or style.get("stop-color", "#000000"),
                      _num(s.get("stop-opacity") or style.get("stop-opacity"), 1.0)))
    a, b, c, d, e, f = _matrix(gel.get("gradientTransform"))
    ap = lambda x, y: [a * x + c * y + e, b * x + d * y + f]
    if gel.tag.endswith("linearGradient"):
        return True, ap(_num(gel.get("x1")), _num(gel.get("y1"))), ap(_num(gel.get("x2")), _num(gel.get("y2"))), stops
    s = ap(_num(gel.get("cx")), _num(gel.get("cy")))
    r = _num(gel.get("r")) * math.sqrt(abs(a * d - b * c))
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
    """Lottie fill (or stroke when width is given) for a hex colour or url(#gradient)."""
    ref = re.match(r"url\(#([^)]+)\)", fill or "")
    stroke = width is not None
    if grad is None and not ref:
        item = {"ty": "st" if stroke else "fl", "c": {"a": 0, "k": _hex_rgb(fill) + [1]}, "o": {"a": 0, "k": opacity * 100}}
    else:
        linear, s, e, stops = grad or _gradient(defs_root, ref.group(1))
        colors = [x for o, c, _ in stops for x in [o] + _hex_rgb(c)]
        alphas = [x for o, _, al in stops for x in (o, al)] if any(al < 1 for *_, al in stops) else []
        item = {"ty": "gs" if stroke else "gf", "t": 1 if linear else 2, "s": {"a": 0, "k": s}, "e": {"a": 0, "k": e},
                "g": {"p": len(stops), "k": {"a": 0, "k": colors + alphas}}, "o": {"a": 0, "k": opacity * 100}}
    if stroke:
        item.update({"w": {"a": 0, "k": width}, "lc": LOTTIE_CAP.get(cap, 2), "lj": LOTTIE_JOIN.get(join, 2), "ml": 4})
    else:
        item["r"] = 1
    return item


def _lottie_paints(p, defs_root):
    """SVG paints stroke over fill; Lottie draws earlier items on top, so the stroke goes first."""
    items = []
    op = p.get("opacity", 1)
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        items.append(_lottie_fill(p["stroke"], op, defs_root, float(p["stroke_width"]), p.get("cap", "round"), p.get("join", "round")))
    if (p.get("fill") or "none") != "none":
        items.append(_lottie_fill(p["fill"], op, defs_root))
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
    op = round(seconds * fps)
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
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": op, "w": round(vb[2] * k), "h": round(vb[3] * k), "nm": "gptkit",
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
    fr = _frames(ks, lambda k: base * k["opacity"]) if ks else [(0.0, base, None), (1.0, base, None)]
    if not window:
        return fr
    a, b = window
    at = lambda t: base * _motion_at(ks, t)["opacity"] if ks else base
    inner = [(a, at(a), None)] + [f for f in fr if a < f[0] < b] + [(b, at(b), None)]
    out = [] if a == 0 else [(0.0, 0.0, None), (a - eps, 0.0, None)]
    out += inner
    out += [] if b == 1 else [(b + eps, 0.0, None), (1.0, 0.0, None)]
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
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        color("strokeColor", p["stroke"])
        attrs.append(f'android:strokeWidth="{_fmtn(float(p["stroke_width"]))}"')
        attrs.append(f'android:strokeLineCap="{p.get("cap", "round")}" android:strokeLineJoin="{p.get("join", "round")}"')
    return attrs, kids


def build_avd(vec, seconds=1.2, width_dp=48):
    """<animated-vector> with the vector inlined: group transforms, fillAlpha fades and pathData morphs, all looping."""
    vb = _vec_vb(vec)
    ms = round(seconds * 1000)
    defs_root = _defs_root(vec)
    safe = lambda s: re.sub(r"\W", "_", s)
    body, targets = [], []
    for L in _layers(vec):
        gname = safe(L["name"])
        ks, (ax, ay) = L["ks"], L["anchor"]
        m0 = _motion_at(ks, 0.0)
        paths = []
        if L["kind"] == "morph":
            keys, n = L["keys"], len(L["keys"])
            p = L["paint"]
            if n == 2:
                paths.append((f"{gname}_0", _d(keys[0]), p, None, [
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
                    paths.append((f"{gname}_{j}", _d(keys[j]), p, (a, b), [
                        f'            <objectAnimator android:propertyName="pathData" android:valueType="pathType"\n'
                        f'                android:valueFrom="{_d(keys[j])}"\n                android:valueTo="{_d(keys[(j + 1) % n])}"\n'
                        f'                android:duration="{ms}" android:repeatCount="infinite">\n'
                        f'{_avd_interp(None, "                ", ip)}\n            </objectAnimator>']))
        elif L["kind"] == "glow":
            g = L["glow"]
            p = {"fill": "glow", "_grad": _glow_grad(g), "opacity": g["opacity"]}
            paths.append((f"{gname}_0", _d([_ellipse(g["cx"], g["cy"], g["rx"], g["rx"])]), p, None, []))
        else:
            for i, (subs, p) in enumerate(L["paths"]):
                paths.append((f"{gname}_{i}", _d(subs), p, None, []))
        xml = []
        for pname, d, p, window, anims in paths:
            base = float(p.get("opacity", 1))
            attrs, kids = _avd_paint(p, defs_root, "            ")
            fades = []
            for prop, on in (("fillAlpha", (p.get("fill") or "none") != "none"),
                             ("strokeAlpha", (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0)):
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


@functools.cache
def _lottie_js():
    """lottie-web source (cached under ~/.cache/gptkit) for inlining; None if it can't be fetched."""
    p = _cache_dir() / "lottie-5.12.2.min.js"
    if not p.exists():
        try:
            r = requests.get(LOTTIE_JS, timeout=60)
            r.raise_for_status()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content)
        except Exception:
            return None
    return p.read_text()


def _chrome_shot(html, out, w, h):
    """Screenshot an HTML string with transparent background; returns the RGBA image or None."""
    import tempfile
    if not _chrome():
        return None
    out = Path(out)
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html)
        try:
            proc = subprocess.Popen([_chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp}/p",
                                     "--default-background-color=00000000", f"--window-size={w},{h}", "--virtual-time-budget=8000",
                                     "--use-mock-keychain", f"--screenshot={out}", page.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return None
        # Chrome can linger after writing the screenshot (lottie's web worker), so stop once the file is complete
        size, end = -1, time.time() + 120
        while proc.poll() is None and time.time() < end:
            cur = out.stat().st_size if out.exists() else -1
            if cur > 0 and cur == size:
                break
            size = cur
            time.sleep(0.5)
        proc.kill()
        proc.wait()
    return _rgba(out) if out.exists() else None


def render_lottie(lot, times, out_png, T=256):
    """Render the Lottie at loop fractions `times` with lottie-web; returns RGBA frames (transparent) or None."""
    js = _lottie_js()
    if not js:
        return None
    frames = [round(t * lot["op"], 3) for t in times]
    html = ("<!doctype html><meta charset=utf-8><style>body{margin:0;display:flex}div{width:%dpx;height:%dpx;flex:none}</style>"
            "<body>%s<script>%s</script><script>var D=%s;%s.forEach(function(f,i){var a=lottie.loadAnimation({container:"
            "document.getElementById('f'+i),renderer:'svg',loop:false,autoplay:false,animationData:JSON.parse(JSON.stringify(D))});"
            "a.addEventListener('DOMLoaded',function(){a.goToAndStop(f,true)});});</script>") % (
        T, T, "".join(f"<div id=f{i}></div>" for i in range(len(frames))), js, json.dumps(lot), json.dumps(frames))
    strip = _chrome_shot(html, out_png, T * len(frames), T)
    if strip is None or strip.width < T * len(frames):
        return None
    return [strip.crop((i * T, 0, (i + 1) * T, T)) for i in range(len(frames))]


def _pixel_motion(frames):
    """Largest share of the drawn area (t=0 union frame i) whose colour visibly changes from t=0."""
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
        s = f' fill="{p.get("fill") or "none"}" opacity="{p.get("opacity", 1)}"'
        if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
            s += (f' stroke="{p["stroke"]}" stroke-width="{p["stroke_width"]}" stroke-linecap="{p.get("cap", "round")}" '
                  f'stroke-linejoin="{p.get("join", "round")}"')
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


def _svg_layers(svg_text, stem):
    """Import an SVG icon as static layers: one per top-level element (a <g> keeps its children together)."""
    root = ET.fromstring(svg_text)
    vb = root.get("viewBox") or f"0 0 {_num(root.get('width'), 24):g} {_num(root.get('height'), 24):g}"
    ET.register_namespace("", SVG_NS[1:-1])
    defs = "".join(ET.tostring(el, encoding="unicode") for el in root.iter()
                   if el.tag.replace(SVG_NS, "") in ("linearGradient", "radialGradient"))
    keys = ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "opacity", "fill-opacity")

    def attrs(el, inh):
        style = dict(kv.split(":", 1) for kv in (el.get("style") or "").replace(" ", "").split(";") if ":" in kv)
        out = dict(inh)
        for k in keys:
            v = el.get(k) or style.get(k)
            if v is not None:
                out[k] = v
        return out

    def paths(el, inh):
        a = attrs(el, inh)
        tag = el.tag.replace(SVG_NS, "")
        if tag == "g":
            return [p for ch in el for p in paths(ch, a)]
        d = _shape_path(el) if tag in ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon") else None
        if not d:
            return []
        stroke = a.get("stroke", "none")
        return [{"d": _d(_subpaths(d)), "fill": a.get("fill", "#000000"), "stroke": stroke,
                 "stroke_width": _num(a.get("stroke-width"), 1.0) if stroke != "none" else 0,
                 "opacity": _num(a.get("opacity"), 1.0) * _num(a.get("fill-opacity"), 1.0),
                 "cap": a.get("stroke-linecap", "butt"), "join": a.get("stroke-linejoin", "miter")}]
    layers, base = [], attrs(root, {})
    for i, el in enumerate(ch for ch in root if ch.tag.replace(SVG_NS, "") not in ("defs", "title", "desc", "metadata")):
        ps = paths(el, base)
        if ps:
            layers.append({"name": re.sub(r"\W", "_", el.get("id") or (stem if i == 0 else f"{stem}_{i + 1}")),
                           "paths": ps, "motion": {"anchor_x": 0, "anchor_y": 0, "keys": []}})
    return vb, defs, layers


def _merge_imported(vec, imported):
    """Imported SVG layers are the source of truth for geometry; the model only adds motion and extra layers."""
    vb, defs, layers = imported
    vec["viewBox"] = vb
    if defs and defs not in vec.get("defs", ""):
        vec["defs"] = defs + vec.get("defs", "")
    got = {sl["name"]: sl for sl in vec.get("static_layers", [])}
    for L in layers:
        if L["name"] in got:
            got[L["name"]]["paths"] = L["paths"]
        else:
            vec.setdefault("static_layers", []).insert(0, dict(L))
    return vec


def _review_sheet(vec, lot, src, out, bg, n=8):
    """keys_review.png: the source, then n evenly spaced frames of the real Lottie (lottie-web) or SVG fallback."""
    times = [i / n for i in range(n)]
    frames = render_lottie(lot, times, out / ".strip.png")
    (out / ".strip.png").unlink(missing_ok=True)
    how = "lottie-web"
    if frames is None:
        how, frames = "svg", []
        for i, t in enumerate(times):
            p = out / f"key_{i + 1}.svg"
            p.write_text(_vec_svg(vec, t, bg=bg))
            prev = _render_svg(p, out)
            if prev:
                frames.append(_rgba(prev))
    px = _pixel_motion(frames) if how == "lottie-web" else None
    tiles = [("asset", src)] if src is not None else []
    for t, f in zip(times, frames):
        base = Image.new("RGBA", f.size, bg)
        base.alpha_composite(f)
        tiles.append((f"t={t:.2f}", base))
    rows = [tiles[i:i + 5] for i in range(0, len(tiles), 5)]
    if rows:
        sheet(rows, out / "keys_review.png", T=220)
    else:
        how = "not written: no SVG renderer"
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


def _ask_vector(a, mock, asset, src, imported, bg):
    thr = a.min_motion if a.min_motion is not None else MOTION_MIN[a.strength]
    parts = [VECTOR_PROMPT.format(moving=a.moving, static=a.static or "nothing", k=a.keyframes, seconds=a.seconds,
                                  motion=a.motion or "a gentle natural loop", strength=MOTION_STRENGTH[a.strength])]
    if imported:
        parts.append(f"The asset already exists as vector static_layers (viewBox \"{imported[0]}\"). Return each with exactly "
                     "its name and paths: [] (the geometry is restored from the source) and add the motion, glows and any "
                     "extra layers the animation needs:\n" + json.dumps(imported[2]))
    if src is not None:
        if imported:
            flat = Image.new("RGBA", src.size, bg)
            flat.alpha_composite(src)
            src = flat
        parts.append(src)
    if asset:
        parts += ["For reference, how it appears in the approved design:", ref_crop(mock, asset["bbox"], 0.3, 512)]
    best, vec = None, None
    for attempt in range(2):
        vec = ask(parts, VECTOR_SCHEMA, "vector_layers")
        if imported:
            _merge_imported(vec, imported)
        errs = _validate_vec(vec)
        rep = None if errs else motion_report(_normalize_vec(vec))
        weak = rep is not None and rep["motion"] is not None and rep["motion"] < thr
        if rep and (best is None or (rep["motion"] or 0) > (best[1]["motion"] or 0)):
            best = (vec, rep)
        if rep and not weak or attempt:
            break
        per = ", ".join(f"{n} {r['move']:.1%}" for n, r in rep["layers"].items() if r["kind"] != "glow") if rep else ""
        fb = (f"Your previous answer was invalid: {'; '.join(errs)}. Fix it and follow every rule." if errs else
              f"Your previous answer moves too little: its animated layers move {rep['motion']:.1%} of their size on "
              f"average ({per}), at least {thr:.0%} is needed. Keep the geometry but exaggerate the keyframes and motion keys.")
        print("  retrying:", fb, flush=True)
        parts = parts + [fb + "\nPrevious answer:\n" + json.dumps(vec)]
    return best[0] if best else vec


def cmd_vectorize(a):
    d, m, mock = _load(a.dir)
    out = d / "vector" / a.name
    out.mkdir(parents=True, exist_ok=True)
    asset = next((x for x in m["assets"] if x["name"] == a.name), None)
    png = d / "assets" / f"{a.name}.png"
    svg = Path(a.svg) if a.svg else d / "svg" / f"{a.name}.svg"
    src, imported = None, None
    if png.exists() and not a.svg:
        src = _rgba(png)
    elif svg.exists():
        imported = _svg_layers(svg.read_text(), a.name)
        src = _svg_image(svg, out / "source.png")
    elif not a.rebuild:
        sys.exit(f"vectorize: no source ({png} or {svg})")
    if a.rebuild:
        vec = json.loads((out / "layers.json").read_text())
    elif not a.moving:
        sys.exit("vectorize: --moving is required (or --rebuild to reuse an edited layers.json)")
    else:
        vec = _ask_vector(a, mock, asset, src, imported, a.preview_bg)
    errs = _validate_vec(vec)
    (out / "layers.json").write_text(json.dumps(vec if errs else _normalize_vec(vec), indent=2))
    if errs:
        print("! still invalid:", errs)
        return
    rep = motion_report(vec)
    thr = a.min_motion if a.min_motion is not None else MOTION_MIN[a.strength]
    fit = None if a.no_fit else _fit_viewbox(vec)
    built = dict(vec, viewBox=fit) if fit else vec
    lot = build_lottie(built, a.size, a.seconds, a.fps)
    (out / f"{a.name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    dp = a.dp or ((asset or {}).get("dp") or {}).get("w") or 48
    dp = round(dp * _vec_vb(built)[2] / _vec_vb(vec)[2], 1)
    (out / f"{a.name}_avd.xml").write_text(build_avd(built, a.seconds, dp))
    (out / "preview.html").write_text(
        "<!doctype html><meta charset=utf-8><body style='margin:0;background:" + a.preview_bg + "'>"
        "<div id=a style='width:512px;height:512px'></div>"
        f"<script src='{LOTTIE_JS}'></script>"
        f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{json.dumps(lot)}}})</script>")
    how, px = _review_sheet(built, lot, src, out, a.preview_bg, a.review_frames)
    for n, r in rep["layers"].items():
        print(f"  {n:24s} {r['kind']:6s} move={r['move']:.1%} fade={r['fade']:.2f}")
    mm = rep["motion"]
    print(f"  motion {'-' if mm is None else f'{mm:.1%}'} (min {thr:.0%} for --strength {a.strength})" +
          (" TOO SUBTLE: re-run with --strength strong or a stronger --motion" if mm is not None and mm < thr else "") +
          (f"; rendered frames change up to {px:.0%} of the drawn area" if px is not None else ""))
    if fit:
        print(f"  motion leaves the viewBox: canvas grown to \"{fit}\" (AVD {dp:g}dp wide) so nothing clips")
    print(f"-> {out}/layers.json, {a.name}.json (Lottie), {a.name}_avd.xml (AnimatedVectorDrawable), "
          f"keys_review.png ({how}), preview.html")


# ---- export

DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}


def _fit(img, w, h):
    im = img.copy()
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def cmd_export(a):
    img = _rgba(a.file)
    name = a.name or Path(a.file).stem
    wdp = a.dp
    hdp = a.hdp or round(wdp * img.height / img.width, 2)
    res = Path(a.res)
    if a.platform == "android":
        for dens, s in DENSITIES.items():
            out = res / f"drawable-{dens}"
            out.mkdir(parents=True, exist_ok=True)
            im = _fit(img, round(wdp * s), round(hdp * s))
            p = out / f"{name}.{a.format}"
            im.save(p, "WEBP", lossless=True) if a.format == "webp" else im.save(p)
            print(f"  {p} {im.size}")
    elif a.platform == "ios":
        out = res / f"{name}.imageset"
        out.mkdir(parents=True, exist_ok=True)
        images = []
        for s in (1, 2, 3):
            fn = f"{name}@{s}x.png"
            _fit(img, round(wdp * s), round(hdp * s)).save(out / fn)
            images.append({"idiom": "universal", "filename": fn, "scale": f"{s}x"})
        (out / "Contents.json").write_text(json.dumps({"images": images, "info": {"version": 1, "author": "xcode"}}, indent=2))
        print(f"  {out}")
    else:
        res.mkdir(parents=True, exist_ok=True)
        for s, suffix in ((1, ""), (2, "@2x")):
            p = res / f"{name}{suffix}.webp"
            _fit(img, round(wdp * s), round(hdp * s)).save(p, "WEBP", quality=92)
            print(f"  {p}")


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
    """The fitted background at `size` px; span > 1 extends it past the concept by clamping (edge continuation)."""
    rgb = np.clip(_bg_feats(size, span) @ coef, 0, 255).round().astype(np.uint8)
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


def cmd_appicon_concepts(a):
    out = Path(a.out)
    (out / "review").mkdir(parents=True, exist_ok=True)
    ref = _rgba(a.mockup) if a.mockup else None
    if ref is not None and a.crop:
        x, y, w, h = map(int, a.crop.split(","))
        ref = ref.crop((x, y, x + w, y + h))
    text = f'The only text is "{a.text}", large and legible.' if a.text else "No text, letters or numbers anywhere."
    dirs = [s.strip() for s in a.directions.split("|")] if a.directions else ICON_DIRECTIONS
    colors = a.colors or (", ".join(_palette(ref)) if ref is not None else "")

    def one(i):
        direction = dirs[i % len(dirs)]
        head = ICON_REF_PROMPT.format(colors=colors, brief=a.brief, direction=direction) if ref is not None else \
            ICON_TXT_PROMPT.format(brief=a.brief, direction=direction)
        prompt = head + ICON_RULES.format(text=text)
        for _ in range(2):
            if ref is not None:
                img, meta = edit(_png_bytes(_square_ref(ref)), prompt, "1024x1024", "opaque", a.model, a.fidelity)
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
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    rows = []
    with cf.ThreadPoolExecutor(min(a.jobs, a.variants)) as ex:
        for p, direction, prompt, meta in ex.map(one, range(a.variants)):
            img = _rgba(p)
            m = _icon_metrics(img, ref)
            report[p.name] = {"path": str(p), "direction": direction, "prompt": prompt, "metrics": m,
                              "reference": a.mockup, "quality": meta["quality"]}
            print(f"{p}  {direction}\n    {m}")
            rows.append(_icon_rows(f"{p.stem}: {direction}  r={m['subject_r']} bg_resid={m['bg_resid']} "
                                   f"palette_de={m.get('palette_de', '-')} {' '.join(m['flags'])}", img))
    report_path.write_text(json.dumps(report, indent=2))
    print("review sheet ->", _stack(rows, out / "review" / f"concepts_{time.strftime('%H%M%S')}.png"))


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
    p = Path(work) / "_glyph_probe.svg"
    p.write_text(black)
    png = _render_svg(p, work, size)
    p.unlink(missing_ok=True)
    if not png:
        return None
    im = Image.open(png).convert("RGBA")
    L = np.asarray(Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("L").resize((size, size)), float)
    png.unlink(missing_ok=True)
    return Image.fromarray((255 - L).astype(np.uint8), "L")


def _vd_group(xml, s, tx, ty):
    decl, tag, body = xml.split(">", 2)
    return (f'{decl}>{tag}>\n    <group android:scaleX="{s:.4f}" android:scaleY="{s:.4f}" android:translateX="{tx:.3f}" '
            f'android:translateY="{ty:.3f}">' + body.replace("</vector>", "").rstrip() + "\n    </group>\n</vector>\n")




def _aapt2_check(res, name, work):
    sdk = _android_sdk()
    if not sdk:
        return "skipped (aapt2 not found; set ANDROID_HOME)"
    aapt, jar = sdk
    work.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(aapt), "compile", "--dir", str(res), "-o", str(work / "res.zip")], capture_output=True, text=True)
    if r.returncode:
        return "compile FAILED: " + (r.stderr or r.stdout).strip()[:800]
    if not jar:
        return "compile ok (link skipped: no platforms/android.jar)"
    (work / "AndroidManifest.xml").write_text(
        '<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.gptkit.iconcheck">\n'
        '  <uses-sdk android:minSdkVersion="21" android:targetSdkVersion="34" />\n'
        f'  <application android:icon="@mipmap/{name}" android:roundIcon="@mipmap/{name}_round" />\n</manifest>\n')
    r = subprocess.run([str(aapt), "link", "-I", str(jar), "--manifest", str(work / "AndroidManifest.xml"),
                        "-o", str(work / "check.apk"), str(work / "res.zip")], capture_output=True, text=True)
    return f"compile+link ok ({Path(jar).parent.name})" if not r.returncode else "link FAILED: " + (r.stderr or r.stdout).strip()[:800]


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
        (res / "drawable" / f"{name}.xml").write_text(_vd_gradient_xml(grad))
        return _vd_gradient_render(grad, M), f"@drawable/{name}", info
    bgm = model
    if kind == "image":
        raw, _ = edit(_png_bytes(concept), ICON_BG_PROMPT.format(subject=a.subject), "1024x1024", "opaque", a.model)
        (work / "bg_api_raw.png").write_bytes(raw)
        plate = np.asarray(_rgba(work / "bg_api_raw.png").convert("RGB").resize((N, N), Image.LANCZOS))
        p = (M - N) // 2
        bgm = Image.fromarray(np.pad(plate, ((p, M - N - p), (p, M - N - p), (0, 0)), mode="reflect")).convert("RGBA")
    _save_densities(bgm, res, "mipmap", name, 108)
    return bgm, f"@mipmap/{name}", info


def _icon_glyph(a, concept, sil_mask, limit, res, work, M):
    """gpt-6 single-colour glyph, measured and fitted onto the silhouette's box inside the safe zone -> VectorDrawable."""
    svg = ask([ICON_GLYPH_PROMPT + f"\nThe subject is: {a.subject}.", concept, "Its foreground silhouette:",
               Image.merge("RGB", [sil_mask.point(lambda v: 255 - v)] * 3).resize((512, 512))], "json")["svg"]
    (work / "glyph_raw.svg").write_text(svg)
    ET.fromstring(svg)
    g_mask = _svg_raster(svg, work, 432)
    s, tx, ty, info, glyph = 1.0, 0.0, 0.0, {"note": "no SVG renderer: placement not verified"}, None
    if g_mask is not None and g_mask.getbbox():
        place = lambda s, tx, ty: g_mask.transform((432, 432), Image.AFFINE, (1 / s, 0, -tx * 4 / s, 0, 1 / s, -ty * 4 / s),
                                                   Image.BILINEAR)
        gb = [v / 4 for v in g_mask.point(lambda v: 255 if v > 128 else 0).getbbox()]
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
    (res / "drawable").mkdir(exist_ok=True)
    (res / "drawable" / f"{a.name}_monochrome.xml").write_text(_vd_group(svg_to_vd(svg, 108), s, tx, ty))
    (work / "glyph.svg").write_text(re.sub(r"(<svg\b[^>]*>)", rf'\1<g transform="translate({tx:.3f} {ty:.3f}) '
                                           rf'scale({s:.4f})">', svg, count=1).replace("</svg>", "</g></svg>"))
    return glyph, info


def cmd_appicon_build(a):
    out = Path(a.out)
    res, work = out / "android" / "res", out / "work"
    for p in (res, work, out / "review"):
        p.mkdir(parents=True, exist_ok=True)
    N, M = 1024, 1536
    vp = (M // 6, M // 6, M // 6 + N, M // 6 + N)
    concept = _rgba(a.concept).convert("RGB")
    side = min(concept.size)
    concept = concept.crop(((concept.width - side) // 2, (concept.height - side) // 2,
                            (concept.width + side) // 2, (concept.height + side) // 2)).resize((N, N), Image.LANCZOS)
    C = np.asarray(concept, float)
    _, resid, coef = _bg_fit(concept)
    report = {"concept": str(a.concept)}

    # background first: the foreground is matted against exactly the background layer that ships
    bgm, bg_ref, report["background"] = _icon_background(a, concept, coef, resid, res, work, N, M)
    report["background"]["resource"] = bg_ref
    B = np.asarray(bgm.crop(vp).convert("RGB"), float)
    fg = _unmix(C, B, Bz=np.asarray(_bg_render(coef, N).convert("RGB"), float))
    if resid > BUSY_RESID and report["background"]["kind"] != "image":
        report["warnings"] = [f"busy background (fit residual {resid:.1f} > {BUSY_RESID}): scenery ends up in the "
                              "foreground; use --bg image"]
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
        raw, _ = edit(_png_bytes(concept), ICON_FG_PROMPT.format(subject=a.subject), "1024x1024", "transparent", a.model)
        (work / "fg_api_raw.png").write_bytes(raw)
        target = Image.new("RGBA", (N, N), (0, 0, 0, 0))
        target.putalpha(fg.getchannel("A").point(lambda v: 255 if v > 128 else 0))
        fg = _register(_rgba(work / "fg_api_raw.png").resize((N, N), Image.LANCZOS), target)
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
        print("  ! no API key: monochrome glyph skipped, using the silhouette")
    elif a.mono in ("glyph", "both"):
        try:
            glyph, report["glyph"] = _icon_glyph(a, concept, sil_mask, limit, res, work, M)
            if glyph is not None:
                monos["glyph"] = glyph
            mono_ref = f"@drawable/{a.name}_monochrome"
        except (ValueError, KeyError, ET.ParseError, RuntimeError) as e:
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
    view.convert("RGB").resize((512, 512), Image.LANCZOS).save(out / "android" / "play_store_512.png")

    # iOS: one opaque 1024 (the concept itself unless --ios-from composite) + dark / tinted appearances
    ios = out / "ios" / "AppIcon.appiconset"
    ios.mkdir(parents=True, exist_ok=True)
    ios_img = concept if a.ios_from == "concept" else view.convert("RGB")
    ios_img.save(ios / "AppIcon-1024.png")
    fg.save(ios / "AppIcon-1024-dark.png")
    tinted = Image.new("RGBA", (N, N), (0, 0, 0, 255))
    tinted.alpha_composite(fg)
    tinted.convert("L").convert("RGB").save(ios / "AppIcon-1024-tinted.png")
    entry = lambda fn, **ap: {**({"appearances": [ap]} if ap else {}), "filename": fn, "idiom": "universal",
                              "platform": "ios", "size": "1024x1024"}
    (ios / "Contents.json").write_text(json.dumps({"images": [
        entry("AppIcon-1024.png"), entry("AppIcon-1024-dark.png", appearance="luminosity", value="dark"),
        entry("AppIcon-1024-tinted.png", appearance="luminosity", value="tinted")],
        "info": {"author": "xcode", "version": 1}}, indent=2))
    chk = Image.open(ios / "AppIcon-1024.png")
    report["ios"] = {"size": chk.size, "opaque": chk.mode == "RGB" and "transparency" not in chk.info,
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
                    ("iOS tinted", _masked(Image.open(ios / "AppIcon-1024-tinted.png"), "squircle", 180)),
                    ("maskable, 80% circle", maskable_prev), ("favicon 16/32/48", favs)], T=200)]
    sheet_path = _stack(rows, out / "review" / f"appicon_{time.strftime('%H%M%S')}.png")
    (out / "appicon.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {res}\n-> {ios}\n-> {web}\nreview sheet -> {sheet_path}")


def cmd_pick(a):
    d, m, mock = _load(a.dir)
    asset = next(x for x in m["assets"] if x["name"] == a.name)
    report = json.loads((d / "forge_report.json").read_text())
    src = Path(a.candidate) if Path(a.candidate).exists() else d / "raw" / f"{a.name}_c{a.candidate}.png"
    img = _rgba(src)
    sc = score(mock, asset["bbox"], img, None, asset["transparent"])
    report[a.name]["chosen"] = {"path": str(src), "meta": {}, "score": sc}
    report[a.name]["picked_manually"] = True
    report[a.name]["color_match"] = cm = _color_final(mock, asset, src, sc, a.color_match, d / "raw")
    if cm:
        print(f"colour match {cm['strength']}: fid {cm['fid_before']} -> {cm['fid_after']} ({'kept' if cm['applied'] else 'discarded'})")
    _finalize(asset, _rgba(_current(report[a.name]))).save(d / "assets" / f"{a.name}.png")
    (d / "forge_report.json").write_text(json.dumps(report, indent=2))
    print(f"{a.name} <- {src}")


# ---- raster -> vector tracing, part rigs, procedural keyframes

def chrome_shot(html, w, h, budget=3000):
    """Render an HTML string in headless Chrome at w x h CSS px on a transparent background -> RGBA image or None."""
    import tempfile
    exe = _chrome()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        page, png = Path(td) / "page.html", Path(td) / "shot.png"
        page.write_text(html)
        try:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                            "--force-device-scale-factor=1", f"--virtual-time-budget={budget}", "--use-mock-keychain",
                            "--default-background-color=00000000", f"--screenshot={png}", page.as_uri()],
                           capture_output=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"  ! headless Chrome failed ({type(e).__name__})")
        return Image.open(png).convert("RGBA") if png.exists() else None


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


def mask_to_path(mask, tol=0.8, corner_deg=70, sigma=0.7, min_area=4.0, field=None):
    """Binary mask (or a soft `field` in 0..1) -> smooth, correctly wound absolute path data."""
    f = field if field is not None else _blur(mask.astype(np.float32), sigma)
    loops = [L for L in _contours(f, 0.5) if abs(_area(L)) >= min_area]
    return _beziers_d([_fit_loop(L, tol, corner_deg) for L in _orient(loops)])


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
        if r < 0.8 * best["rmse"]:
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
        if r < 0.9 * best["rmse"] and R > 1:
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
    for i in sorted(range(n), key=lambda i: area[i]):
        if area[find(i)] < min_area and shared.get(i):
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
                           "color": _tr_hex(c_i), "mask": region})
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
    s = work / max(img.size)
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
    return {"preset": "vtracer", "params": {"work": work, "bands": bands, **opts}, "size": (w, h), "src_size": img.size,
            "layers": layers}


def trace_image(img, preset="illustration", **over):
    """RGBA image -> layered vector: glow bands (bottom) + gradient-filled regions. Returns layers + svg text."""
    P = dict(TRACE_PRESETS[preset])
    P.update({k: v for k, v in over.items() if v is not None})
    s = P["work"] / max(img.size)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    arr = np.asarray(img.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255
    core = alpha >= 0.5
    layers = _alpha_bands(alpha, rgb, core, P["bands"], P["tol"] * 1.5, P.get("band_kinds", ("flat",)),
                          P.get("band_gamma", 1.0))
    R = segment_regions(rgb, core, P["k"], P["min_frac"], P["edge_tol"], P["fit_tol"], P.get("soft_tol", 2.0)) \
        if core.any() else np.full(core.shape, -1, np.int32)
    ids = [r for r in range(int(R.max()) + 1)]
    masks = {r: R == r for r in ids}
    filled = {r: (_fill_holes(masks[r]) & core) | masks[r] for r in ids}
    order = sorted(ids, key=lambda r: -int(filled[r].sum()))
    later = np.zeros_like(core)
    draw = {}
    for r in reversed(order):
        draw[r] = filled[r] | (_dilate(masks[r], 2) & later)
        later |= masks[r]
    for r in order:
        ys, xs = np.nonzero(masks[r])
        if len(xs) < 4:
            continue
        fill = fit_fill(np.c_[xs + 0.5, ys + 0.5], rgb[ys, xs], P["kinds"], P["stops"], P["stop_tol"])
        d = mask_to_path(draw[r], P["tol"], P["corner"], 0.7, 4.0)
        if d:
            layers.append({"name": f"region_{len(layers) + 1}", "role": "region", "d": d, "fill": fill, "opacity": 1.0,
                           "area": int(len(xs)), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                           "color": _tr_hex(rgb[ys, xs].mean(0)), "mask": masks[r]})
    return {"preset": preset, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
            "size": (w, h), "src_size": img.size, "layers": layers}


def layers_svg(layers, vb, size=None, bg=None):
    """Traced layers -> SVG text (absolute M/L/C/Z paths, userSpaceOnUse gradients, no transforms)."""
    w, h = vb
    W, H = size or vb
    defs, body = [], []
    for i, L in enumerate(layers):
        fill, gd = _fill_svg(L["fill"], f"g{i}")
        defs.append(gd)
        op = f' opacity="{L["opacity"]:g}"' if L.get("opacity", 1) < 1 else ""
        body.append(f'<path d="{L["d"]}" fill="{fill}"{op}/>')
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {w} {h}">'
            f'<defs>{"".join(defs)}</defs>{rect}' + "\n".join(body) + "</svg>")


def layers_vec(res):
    """A trace as a build_lottie/animate-ready layers.json: static layers 'glow' (bands) and 'art' (regions)."""
    defs, static = [], []
    for role, name in (("glow", "glow"), ("region", "art")):
        paths = []
        for L in (x for x in res["layers"] if x["role"] == role):
            f, gd = _fill_svg(L["fill"], f"{name}_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": L.get("opacity", 1)})
        if paths:
            static.append({"name": name, "paths": paths})
    w, h = res["size"]
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static]}


def _vd_color(v):
    v = v.strip().lstrip("#")
    if len(v) in (3, 4):
        v = "".join(c * 2 for c in v)
    return ("#" + v[2:], int(v[:2], 16) / 255) if len(v) == 8 else ("#" + v, 1.0)


def _tr_vd_to_svg(xml):
    """Android VectorDrawable XML -> equivalent SVG, so the VD itself (not its source SVG) can be rendered and checked."""
    A = "{http://schemas.android.com/apk/res/android}"
    root = ET.fromstring(xml)
    vw, vh = root.get(A + "viewportWidth"), root.get(A + "viewportHeight")
    defs, body = [], []

    def grad(el, gid):
        g_ = next((x for x in el.iter() if x.tag == "gradient"), None)
        if g_ is None:
            return None
        items = "".join(f'<stop offset="{it.get(A + "offset")}" stop-color="{_vd_color(it.get(A + "color"))[0]}" '
                        f'stop-opacity="{_vd_color(it.get(A + "color"))[1]:.3f}"/>' for it in g_ if it.tag == "item")
        t = g_.get(A + "type", "linear")
        if t == "radial":
            defs.append(f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{g_.get(A + "centerX")}" '
                        f'cy="{g_.get(A + "centerY")}" r="{g_.get(A + "gradientRadius")}">{items}</radialGradient>')
        else:
            defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{g_.get(A + "startX")}" '
                        f'y1="{g_.get(A + "startY")}" x2="{g_.get(A + "endX")}" y2="{g_.get(A + "endY")}">{items}</linearGradient>')
        return f"url(#{gid})"

    def walk(el, out):
        for ch in el:
            if ch.tag == "group":
                tr = []
                tx, ty = ch.get(A + "translateX", "0"), ch.get(A + "translateY", "0")
                px, py = ch.get(A + "pivotX", "0"), ch.get(A + "pivotY", "0")
                tr.append(f"translate({tx} {ty}) translate({px} {py}) rotate({ch.get(A + 'rotation', '0')}) "
                          f"scale({ch.get(A + 'scaleX', '1')} {ch.get(A + 'scaleY', '1')}) translate(-{px} -{py})")
                inner = []
                walk(ch, inner)
                out.append(f'<g transform="{" ".join(tr)}">{"".join(inner)}</g>')
            elif ch.tag == "path":
                gid = f"v{len(defs)}"
                attr_grad = {x.get("name"): x for x in ch if x.tag.endswith("attr")}
                fill = grad(attr_grad["android:fillColor"], gid) if "android:fillColor" in attr_grad else None
                fo = float(ch.get(A + "fillAlpha", "1"))
                if fill is None and ch.get(A + "fillColor"):
                    fill, a_ = _vd_color(ch.get(A + "fillColor"))
                    fo *= a_
                s = f'<path d="{ch.get(A + "pathData")}" fill="{fill or "none"}" fill-opacity="{fo:.4f}"'
                if ch.get(A + "fillType") == "evenOdd":
                    s += ' fill-rule="evenodd"'
                if ch.get(A + "strokeColor"):
                    sc, sa = _vd_color(ch.get(A + "strokeColor"))
                    s += (f' stroke="{sc}" stroke-opacity="{sa * float(ch.get(A + "strokeAlpha", "1")):.4f}" '
                          f'stroke-width="{ch.get(A + "strokeWidth", "1")}" stroke-linecap="{ch.get(A + "strokeLineCap", "butt")}" '
                          f'stroke-linejoin="{ch.get(A + "strokeLineJoin", "miter")}"')
                out.append(s + "/>")
    walk(root, body)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw} {vh}"><defs>{"".join(defs)}</defs>'
            f'{"".join(body)}</svg>')


@functools.cache
def _aapt2():
    for base in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"), "~/Library/Android/sdk", "~/Android/Sdk"):
        if base:
            found = sorted(Path(base).expanduser().glob("build-tools/*/aapt2"),
                           key=lambda p: [int(x) if x.isdigit() else 0 for x in p.parent.name.split(".")])
            if found:
                return str(found[-1])
    return shutil.which("aapt2")


def _tr_aapt2_check(xml_text, name="traced"):
    """Compile a drawable with aapt2 (what the Android build does). Returns 'ok', an error string, or None."""
    import tempfile
    exe = _aapt2()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "res" / "drawable" / f"{name}.xml"
        f.parent.mkdir(parents=True)
        f.write_text(xml_text)
        r = subprocess.run([exe, "compile", str(f), "-o", td], capture_output=True, text=True)
        return "ok" if r.returncode == 0 else (r.stderr or r.stdout).strip().splitlines()[-1][:300]


def vector_advice(m, layers, dp=None):
    """Vector or raster? Fidelity bar plus Android guidance (VDs <=200dp, simple paths, lint VectorPath >3000 chars)."""
    disp = m.get("display", m)
    lens = [len(re.sub(r"\s+", " ", L["d"])) for L in layers]
    grads = sum(L["fill"]["type"] != "flat" for L in layers)
    fid = ("faithful" if disp["diff"] <= 2 else "close" if disp["diff"] <= 3.5 else
           "stylised" if disp["diff"] <= 5 else "lossy")
    if disp["iou"] < 0.95:
        fid = "lossy"
    reasons = [f"display-size diff {disp['diff']} / silhouette IoU {disp['iou']} -> {fid}"]
    vec_ok = fid in ("faithful", "close")
    vd_ok = vec_ok
    if dp and max(dp) > 200:
        vd_ok = False
        reasons.append(f"{max(dp):g}dp > 200dp: Android recommends raster above 200x200dp (draw time)")
    if len(layers) > 64 or sum(lens) > 40000:
        vd_ok = False
        reasons.append(f"{len(layers)} paths / {sum(lens)} pathData chars: heavy first draw, prefer WebP for a VD slot")
    long_ = sum(n > 3000 for n in lens)
    if long_:
        reasons.append(f"{long_} path(s) over 3000 chars (lint VectorPath warns when vectors are rasterised for minSdk<21)")
    if grads:
        reasons.append(f"{grads} gradient fill(s): VectorDrawable gradients need API 24+ or AndroidX VectorDrawableCompat")
    if fid == "stylised":
        reasons.append("visibly simplified vs the raster; acceptable only if a flatter look is fine")
    prefer = "vd" if vd_ok else ("svg/lottie" if vec_ok else "png/webp")
    return {"fidelity": fid, "vector_ok": vec_ok, "vd_ok": vd_ok, "prefer": prefer, "paths": len(layers),
            "pathdata_chars": sum(lens), "max_path_chars": max(lens, default=0), "gradients": grads, "reasons": reasons}


def _asset_dp(png):
    """dp size of a forged asset, looked up from the build dir's assets.json when the PNG lives in <dir>/assets/."""
    p = Path(png).resolve()
    mf = p.parent.parent / "assets.json"
    if p.parent.name == "assets" and mf.exists():
        a = next((x for x in json.loads(mf.read_text())["assets"] if x["name"] == p.stem), None)
        if a and a.get("dp"):
            return a["dp"]["w"], a["dp"]["h"]
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


def trace_asset(src, presets, dp=None, display=None, **over):
    """Trace with each preset, render all with Chrome, score, and pick the smallest near-best candidate."""
    cands = []
    for pr in presets:
        t = time.time()
        res = trace_vtracer(src, **{k: v for k, v in over.items() if k in ("work", "bands") and v is not None}) \
            if pr == "vtracer" else trace_image(src, pr, **over)
        svg = layers_svg(res["layers"], res["size"], src.size)
        cands.append({"preset": pr, "res": res, "svg": svg, "secs": round(time.time() - t, 2), "bytes": len(svg)})
    display = display or (round(max(dp) * 3) if dp else 192)
    rens = _tr_render_svgs([c["svg"] for c in cands], [src.size] * len(cands))
    for c, r in zip(cands, rens or [None] * len(cands)):
        c["render"] = r
        c["metrics"] = trace_metrics(src, r, display) if r is not None else None
    scored = [c for c in cands if c["metrics"]]
    if scored:
        best = min(c["metrics"]["display"]["diff"] for c in scored)
        near = [c for c in scored if c["metrics"]["display"]["diff"] <= best * 1.1 + 0.15]
        chosen = min(near, key=lambda c: c["bytes"])
    else:
        chosen = next((c for c in cands if c["preset"] == "illustration"), cands[0])
    return chosen, cands


def cmd_trace(a):
    out = Path(a.out)
    (out / "review").mkdir(parents=True, exist_ok=True)
    if a.preset == "auto":
        presets = ["flat", "illustration", "gradient"] + (["vtracer"] if _vtracer() else [])
    else:
        presets = [a.preset]
        if a.preset == "vtracer" and not _vtracer():
            sys.exit("vtracer not installed: pip install vtracer (or use --preset flat|illustration|gradient)")
    if not _chrome():
        print("  (no Chrome/Chromium found: fidelity is not measured; set GPTKIT_CHROME to enable it)")
    for f in a.files:
        src = _rgba(f)
        name = a.name or Path(f).stem
        dp = (a.dp, a.dp * src.height / src.width) if a.dp else _asset_dp(f)
        chosen, cands = trace_asset(src, presets, dp, a.display, work=a.work, k=a.colors, bands=a.bands, tol=a.tol)
        res = chosen["res"]
        svg_path = out / f"{name}.svg"
        svg_path.write_text(chosen["svg"])
        report = {"source": str(f), "src_size": src.size, "dp": dp, "chosen": chosen["preset"],
                  "candidates": [{k: c[k] for k in ("preset", "bytes", "secs", "metrics")} | {"paths": len(c["res"]["layers"])}
                                 for c in cands]}
        m = chosen["metrics"]
        report["advice"] = vector_advice(m, res["layers"], dp) if m else None
        try:
            vd_w = round(dp[0], 1) if dp else round(48 * src.width / max(src.size), 1)
            vd = svg_to_vd(chosen["svg"], vd_w)
            (out / f"{a.prefix}{name}.xml").write_text(vd)
            report["vd"] = {"file": str(out / f"{a.prefix}{name}.xml"), "bytes": len(vd), "width_dp": vd_w,
                            "width_from": "dp" if dp else "default 48dp long side (pass --dp)",
                            "aapt2": _tr_aapt2_check(vd, a.prefix + name)}
            vr = _tr_render_svgs([_tr_vd_to_svg(vd)], [src.size])
            if vr and chosen.get("render") is not None:
                report["vd"]["vs_svg_diff"] = trace_metrics(chosen["render"], vr[0])["diff"]
        except ValueError as e:
            report["vd"] = {"error": str(e)}
        (out / f"{name}.layers.json").write_text(json.dumps(layers_vec(res), indent=1))
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
        adv = report["advice"]
        print(f"{name}: {chosen['preset']} -> {svg_path} ({chosen['bytes'] // 1024} KB, {len(res['layers'])} paths)"
              + (f"  display diff={m['display']['diff']} iou={m['display']['iou']}" if m else ""))
        for c in cands:
            if c["metrics"] and len(cands) > 1:
                print(f"    {c['preset']:13s} d={c['metrics']['display']['diff']:5.2f} iou={c['metrics']['display']['iou']:.3f} "
                      f"{c['bytes'] // 1024:4d} KB {len(c['res']['layers']):3d} paths")
        if adv:
            print(f"  advice: prefer {adv['prefer']} ({adv['fidelity']}); " + "; ".join(adv["reasons"][1:]))
        if "vd" in report:
            print(f"  vd: {report['vd'].get('file', report['vd'].get('error'))}  aapt2={report['vd'].get('aapt2')}  "
                  f"vd-vs-svg diff={report['vd'].get('vs_svg_diff')}")


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
    x0, y0, x1, y1 = box
    W, H = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
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
    if box is None:
        pts = np.vstack([_path_points(d) for d in paths])
        box = (*pts.min(0), *pts.max(0))
    return [[map_path(d, motion_field(kind, box, j / n, amp, phase, seed)) for j in range(n)] for d in paths]


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
            pts = np.vstack([_path_points(p["d"]) for p in l["paths"]])
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
        for nm, p, k in zip(names, sl["paths"], keys):
            vec["animated_layers"].append({"name": nm, "fill": p["fill"], "opacity": p.get("opacity", 1), "keyframes": k})
        vec["static_layers"] = [l for l in vec["static_layers"] if l["name"] != name]
        i = order.index(name) if name in order else len(order)
        order = order[:i] + names + order[i + 1:]
    vec["layer_order"] = order
    return vec


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
    regs = [L for L in res["layers"] if L["role"] == "region"]
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
    regs = [L for L in res["layers"] if L["role"] == "region"]
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


def isolate_parts(src, parts, model=None, jobs=2):
    """Route (a): GPT redraws each named part alone on the asset's canvas (occluded areas completed)."""
    sq, off = _square(src)
    names = list(parts)

    def one(name):
        others = ", ".join(parts[o] for o in names if o != name) or "everything else"
        prompt = (f"The attached image is an APPROVED illustration. Keep ONLY {parts[name]}, exactly as drawn: same "
                  f"size, position, angle, colours and gradients on the same canvas. Remove {others} completely. Where "
                  f"the removed parts covered {parts[name]}, complete its hidden shape naturally. Transparent background, "
                  "no glow, no shadow, nothing else.")
        raw, _ = edit(_png_bytes(sq), prompt, "1024x1024", "transparent", model)
        return name, raw
    with cf.ThreadPoolExecutor(jobs) as ex:
        return dict(ex.map(one, names))


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
    regs = [L for L in res["layers"] if L["role"] == "region"]
    glow = [L for L in res["layers"] if L["role"] == "glow"]
    parts = {}
    for i, L in enumerate(regs):
        parts.setdefault(assign.get(i + 1, "other"), []).append(L)
    if "other" in parts and "other" not in order and len(parts) > 1:
        # unnamed slivers join the named part they share the most boundary with
        for L in parts.pop("other"):
            ring = _dilate(L["mask"], 2) & ~L["mask"]
            near = {p: sum(int((ring & R["mask"]).sum()) for R in rs) for p, rs in parts.items()}
            parts[max(near, key=near.get)].append(L)
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
        glows = {"glow": glow}
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
            m = np.logical_or.reduce([L["mask"] for L in glows[p]])
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
            "rig": {"parts": info, "boxes": boxes, "src_size": list(src.size), "work_scale": round(sc, 5)}}


def _vec_render(vec, size, k=0, only=None):
    """layers.json -> SVG text at keyframe k, optionally only some layers (exploded previews)."""
    v = json.loads(json.dumps(vec))
    if only is not None:
        v["static_layers"] = [l for l in v["static_layers"] if l["name"] in only]
        v["animated_layers"] = [l for l in v["animated_layers"] if l["name"].split("__")[0] in only]
    for al in v["animated_layers"]:
        al["keyframes"] = [al["keyframes"][k % len(al["keyframes"])]]
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
        out[k] = (bits[0], float(bits[1]) if len(bits) > 1 and bits[1] else 1.0,
                  float(bits[2]) if len(bits) > 2 and bits[2] else 0.0, anchor.strip() or None)
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
    (out / f"{name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    (out / "preview.html").write_text(
        f"<!doctype html><meta charset=utf-8><body style='margin:0;background:{bg}'>"
        f"<div id=a style='width:{lot['w']}px;height:{lot['h']}px'></div><script src='{TR_LOTTIE_JS}'></script>"
        f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{json.dumps(lot)}}})</script>")
    vb = [float(x) for x in vec["viewBox"].split()]
    sz = (round(300 * vb[2] / max(vb[2:])), round(300 * vb[3] / max(vb[2:])))
    nk = max((len(al["keyframes"]) for al in vec["animated_layers"]), default=1)
    keys = _tr_render_svgs([_vec_render(vec, sz, k) for k in range(nk)], [sz] * nk)
    report = {"lottie": str(out / f"{name}.json"), "bytes": len(json.dumps(lot, separators=(",", ":"))),
              "frames": lot["op"], "fps": fps, "motion": _motion_stats(vec)}
    if keys:
        sheet([[(f"key {k + 1}", _on(im, bg)) for k, im in enumerate(keys)]], out / "keys_review.png", T=220)
        # tears: transparent pixels enclosed by the art, beyond what the rest pose already has
        holes = []
        for im in keys:
            op = np.asarray(im.getchannel("A")) > 127
            holes.append(int((_fill_holes(op) & ~op).sum()))
        area = max(1, int((np.asarray(keys[0].getchannel("A")) > 127).sum()))
        report["tear_px"] = [h_ - holes[0] for h_ in holes]
        report["tear_frac_max"] = round(max(0, max(report["tear_px"])) / area, 5)
    frames = [round(lot["op"] * i / strip) for i in range(strip)]
    film = lottie_filmstrip(lot, frames, 220, bg)
    if film is not None:
        film.save(out / "lottie_frames.png")
        tiles = [np.asarray(film.crop((i * film.width // strip, 0, (i + 1) * film.width // strip, film.height)), np.float32)
                 for i in range(strip)]
        report["lottie_frame_delta"] = [round(float(np.abs(tiles[i] - tiles[i - 1]).mean()), 2) for i in range(1, strip)]
        report["lottie_rendered"] = bool(np.asarray(film)[..., :3].reshape(-1, 3).std(0).max() > 3)
    (out / "animation_report.json").write_text(json.dumps(report, indent=2))
    return report


def cmd_animate(a):
    src = Path(a.layers)
    vec = json.loads(src.read_text())
    specs = _parse_motions(a.part)
    if not specs:
        sys.exit("give --part 'layer=motion[:amp[:phase]]; ...' (motions: " + ", ".join(TR_MOTIONS) + ")")
    try:
        vec = animate_layers(vec, specs, a.keyframes)
    except ValueError as e:
        sys.exit(f"animate: {e}")
    stem = src.name[:-len(".layers.json")] if src.name.endswith(".layers.json") else ""
    out = Path(a.out) if a.out else src.parent / "anim" / stem
    rep = write_animation(vec, out, a.name or stem or src.parent.name, a.seconds, a.fps, a.size, a.ease, a.preview_bg)
    print(f"-> {out}/layers.json, {rep['lottie']} ({rep['bytes'] // 1024} KB, {rep['frames']} frames), keys_review.png, "
          f"lottie_frames.png, preview.html")
    for k, v in rep["motion"].items():
        print(f"  {k:24s} travel mean={v['mean']} max={v['max']} (viewBox units)")
    if "lottie_frame_delta" in rep:
        print(f"  lottie-web frame-to-frame delta: {rep['lottie_frame_delta']}  rendered={rep['lottie_rendered']}")


def cmd_rig(a):
    d, m, mock = _load(a.dir)
    src = _rgba(d / "assets" / f"{a.name}.png")
    out = d / "rig" / a.name
    out.mkdir(parents=True, exist_ok=True)
    parts = _parse_kv(a.parts)
    if not parts:
        sys.exit("give --parts 'name=description; ...' listed bottom to top")
    order = [p for p in (a.order.split(",") if a.order else parts)]
    res = trace_image(src, a.preset)
    region_overlay(src, res).save(out / "regions.png")
    amodal, gpt_imgs = None, {}
    if a.route in ("gpt", "hybrid"):
        (out / "parts_gpt").mkdir(exist_ok=True)
        cached = {p: out / "parts_gpt" / f"{p}_raw.png" for p in parts}
        todo = {p: v for p, v in parts.items() if a.force or not cached[p].exists()}
        if todo:
            print(f"isolating {len(todo)} part(s) with the image model ({len(todo)} image calls)...")
            for p, raw in isolate_parts(src, todo, a.model).items():
                cached[p].write_bytes(raw)
        reg_info = {}
        for p in parts:
            gpt_imgs[p], reg_info[p] = register_part(Image.open(cached[p]).convert("RGBA"), src)
            gpt_imgs[p].save(out / "parts_gpt" / f"{p}.png")
        print("  registered GPT parts (scale, shift px):", reg_info)
        amodal = gpt_imgs
    if a.route == "gpt":
        vec = _rig_from_parts(src, gpt_imgs, order, res, a.preset)
        assign = None
    else:
        if a.assign:
            assign = parse_assign(a.assign, res)
        elif a.route == "hybrid":
            assign = _assign_by_masks(res, gpt_imgs, order)
        elif a.namer == "gpt":
            print("naming regions with the vision model (1 call)...")
            assign = name_regions(src, res, parts, a.text_model)
        else:
            print(f"regions -> {out / 'regions.png'}: Read it, then re-run with --assign 'part=1,2,3; other_part=4,5'")
            return
        (out / "assign.json").write_text(json.dumps(assign, indent=1))
        vec = build_rig(src, res, assign, order, amodal if a.route == "hybrid" else None, min_detail=a.min_detail,
                        glow_mode=a.glow)
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
    if a.animate:
        try:
            vec = animate_layers(vec, _parse_motions(a.animate), a.keyframes)
        except ValueError as e:
            sys.exit(f"rig --animate: {e} (the rig itself was written)")
        rep["animation"] = write_animation(vec, out / "anim", a.name, a.seconds, a.fps, 512, "linear", a.preview_bg)
        print(f"  animated -> {out / 'anim'} (Lottie {a.name}.json, keys_review.png, lottie_frames.png, preview.html)")
        for k, v in rep["animation"]["motion"].items():
            print(f"    {k:24s} travel mean={v['mean']} max={v['max']}")
    (out / "rig_report.json").write_text(json.dumps(rep, indent=2, default=str))


def parse_assign(spec, res):
    """'drop=c1; inner_ring=c2; outer_ring=rest; logs=3,4,7' -> {region number: part} (cN = a whole component)."""
    comps = region_components(res)
    assign, rest = {}, None
    for part, ids in _parse_kv(spec).items():
        for tok in re.split(r"[,\s]+", ids):
            if tok == "rest":
                rest = part
            elif tok.startswith("c") and tok[1:].isdigit():
                assign.update({i: part for i, c in comps.items() if c == int(tok[1:])})
            elif tok.isdigit():
                assign[int(tok)] = part
    if rest:
        assign.update({i: rest for i in comps if i not in assign})
    return assign


def _assign_by_masks(res, masks, order):
    """Hybrid route: each traced region goes to the top-most GPT part that covers most of it."""
    w, h = res["size"]
    M = {p: np.asarray(im.resize((w, h), Image.LANCZOS).getchannel("A")) > 128 for p, im in masks.items()}
    assign = {}
    for i, L in enumerate(r for r in res["layers"] if r["role"] == "region"):
        cover = {p: (M[p] & L["mask"]).sum() / max(1, L["mask"].sum()) for p in M}
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


def _add_trace_rig_parsers(sub):
    p = sub.add_parser("trace", help="raster asset -> multi-colour SVG + VectorDrawable with gradient fits, glow "
                                     "bands, measured fidelity and a vector-or-raster recommendation")
    p.add_argument("files", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=["auto", "flat", "illustration", "gradient", "vtracer"], default="auto")
    p.add_argument("--dp", type=float, help="display width in dp (default: from assets.json when tracing DIR/assets/*.png)")
    p.add_argument("--display", type=int, help="px size (long side) the display-size fidelity is measured at")
    p.add_argument("--work", type=int, help="tracing resolution, long side (default 512)")
    p.add_argument("--colors", type=int, help="k-means colours before region merging")
    p.add_argument("--bands", type=int, help="alpha bands used to rebuild glow/soft edges (0 = none)")
    p.add_argument("--tol", type=float, help="curve-fit tolerance in work px")
    p.add_argument("--name"); p.add_argument("--prefix", default="img_", help="VectorDrawable file prefix")
    p.add_argument("--bg", default="#1b1d33", help="preview backdrop"); p.set_defaults(fn=cmd_trace)
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
    p.add_argument("--min-detail", type=float, default=0.0, help="drop detail regions smaller than this fraction of "
                                                                 "their part (smaller Lottie; the part base covers them)")
    p.add_argument("--animate", help="'part=motion[:amp[:phase]][@anchor_part]; ...' motions: " + ", ".join(TR_MOTIONS))
    p.add_argument("--keyframes", type=int, default=8); p.add_argument("--seconds", type=float, default=1.6)
    p.add_argument("--fps", type=int, default=30); p.add_argument("--force", action="store_true")
    p.add_argument("--model", default=None); p.add_argument("--text-model", default=None)
    p.add_argument("--glow", choices=["part", "shared"], default="part",
                   help="part: each part owns (and moves with) the glow nearest to it; shared: one glow layer")
    p.add_argument("--preview-bg", default="#1b1d33"); p.set_defaults(fn=cmd_rig)
    p = sub.add_parser("animate", help="procedural morph keyframes (sway/flicker/breathe/squash/bob/wave/ripple) "
                                       "for named layers of a layers.json -> Lottie; no API calls")
    p.add_argument("layers", help="layers.json from rig, vectorize or trace")
    p.add_argument("--part", required=True, help="'layer=motion[:amp[:phase]][@anchor_layer]; ...' (anchor: move "
                                                 "with that layer's field, e.g. glow=flicker@outer_flame)")
    p.add_argument("--keyframes", type=int, default=8); p.add_argument("--seconds", type=float, default=1.6)
    p.add_argument("--fps", type=int, default=30); p.add_argument("--size", type=int, default=512)
    p.add_argument("--ease", choices=["linear", "keys"], default="linear",
                   help="linear between procedural keys (smooth); keys = build_lottie's ease at every key")
    p.add_argument("--out"); p.add_argument("--name"); p.add_argument("--preview-bg", default="#1b1d33")
    p.set_defaults(fn=cmd_animate)


# ---- design-to-code: colour science + text measurement (shared by tokens, fonts, compare)

def _lab_arr(rgb):
    """sRGB 0-255 array (...x3) -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
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


def _hexi(h):
    h = h.strip().lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h[:6]
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


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
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "x", "y", "w", "h"],
                               "properties": {"text": {"type": "string"}, **{k: {"type": "integer"} for k in "xywh"}}}}}}


def _text_targets(d, spec, mock, vision=False):
    """Locate every typography sample in the mockup (spec components first, optional vision fallback)."""
    rgb = np.asarray(mock.convert("RGB"))
    area = mock.width * mock.height
    vpath = d / "text_boxes_vision.json"
    vboxes = json.loads(vpath.read_text()) if vpath.exists() else {}
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
        res = ask([f"This UI design is {mock.width}x{mock.height}px. For each string below return the TIGHT pixel bbox "
                   "of its first visible occurrence (one line; ignore other lines). Strings:\n" +
                   "\n".join(f"- {s}" for s in missing), mock.convert("RGB")], TEXT_BOX_SCHEMA, "text_boxes")
        for b in res["boxes"]:
            vboxes[b["text"]] = [b["x"], b["y"], b["w"], b["h"]]
        vpath.write_text(json.dumps(vboxes, indent=2))
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
            sb = [bx - int(0.3 * st["size_px"]), by + n * pitch - int(0.25 * st["size_px"]), bw + int(0.6 * st["size_px"]),
                  bh + int(0.5 * st["size_px"])]
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
               "icon", "icons", "label", "labels", "background"}


def _words(s):
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in _TOKEN_STOP}


def _rel(w, words):
    return any(w == y or (min(len(w), len(y)) >= 5 and (w.startswith(y) or y.startswith(w))) for y in words)


def _link_regions(query, regions):
    """Regions whose name (and, weaker, notes) share informative words with `query` (IDF-weighted)."""
    docs = [(_words(r["name"]), _words(r.get("notes", ""))) for r in regions]
    vocab = [nw | tw for nw, tw in docs]
    idf = lambda w: math.log((len(regions) + 1) / (sum(1 for v in vocab if _rel(w, v)) + 0.5))
    scores = []
    for (nw, tw), r in zip(docs, regions):
        sc = sum(max(idf(w) if _rel(w, nw) else 0, 0.5 * idf(w) if _rel(w, tw) else 0) for w in _words(query))
        scores.append((sc, r))
    best = max((sc for sc, _ in scores), default=0)
    return [r for sc, r in scores if best >= 1.0 and sc >= 0.8 * best]


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
        if t.get("found"):
            by_hex.setdefault(st["color"].upper(), []).append((st["role"], t))
    rng = np.random.default_rng(0)
    out = []
    for p in spec.get("palette", []):
        guess = p["hex"].upper()
        text = p["name"] + " " + p.get("usage", "")
        entry = {"name": p["name"], "usage": p.get("usage", ""), "hex_guess": guess}
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


def _r05(v):
    return round(v * 2) / 2


def _num_s(v):
    return f"{v:g}"


def _d2c_load_spec(dir_):
    d = Path(dir_)
    if not (d / "design_spec.json").exists():
        sys.exit(f"gptkit: no {d / 'design_spec.json'}; run `spec {d}` first")
    spec = json.loads((d / "design_spec.json").read_text())
    mpath = d / "mockup.png"
    if (d / "assets.json").exists():
        m = json.loads((d / "assets.json").read_text())
        mpath = d / m.get("mockup", "mockup.png")
        spec.setdefault("px_to_dp", round(m["screen_dp"] / m["width"], 5))
    return d, spec, _rgba(mpath)


def _token_model(spec, palette, texts, fonts, redline=None):
    k = spec.get("px_to_dp", 1.0)
    colors = [dict(p, kt=n, sw=_ident(p["name"], "camel"), css=_ident(p["name"], "kebab"))
              for p, n in zip(palette, _uniq([_ident(p["name"]) for p in palette]))]
    by_guess = {c["hex_guess"]: c for c in colors}
    fstyles = {s["role"]: s for s in (fonts or {}).get("styles", [])}
    families = {f["family"]: f for f in (fonts or {}).get("families", [])}
    types = []
    for st, n in zip(spec.get("typography", []), _uniq([_ident(s["role"]) for s in spec.get("typography", [])])):
        fs = fstyles.get(st["role"], {})
        size_px = fs.get("size_px") or st["size_px"]
        track = fs.get("tracking_em", st.get("letter_spacing_em", 0))
        t = texts.get(st["role"], {})
        lh_px = t.get("line_height_px") or round(size_px * 1.25)
        guess = st.get("family_guess", "")
        generic = "serif" if re.search(r"(?<!sans[- ])serif|baskerville|garamond|times|georgia|playfair|bodoni", guess, re.I) \
            and not re.search(r"sans", guess, re.I) else "sans"
        fam = families.get(fs.get("family"))
        col = by_guess.get(st["color"].upper())
        ink = t.get("ink") if t.get("found") else None
        types.append({"role": st["role"], "kt": n, "sw": n[:1].lower() + n[1:], "css": _ident(st["role"], "kebab"),
                      "family": fam["family"] if fam else None, "family_kt": fam and _ident(fam["family"]),
                      "family_css": fam and _ident(fam["family"], "kebab"), "generic": generic,
                      "weight": fs.get("weight", st.get("weight", 400)), "italic": fs.get("italic", False),
                      "size": _r05(size_px * k), "line_height": _r05(lh_px * k), "tracking": round(track, 3),
                      "color": col, "color_hex": col["hex"] if col else (ink or st["color"].upper()),
                      "postscript": (fam or {}).get("postscript", {}).get(str(fs.get("weight")) + ("i" if fs.get("italic") else "")),
                      "natural_lh": fs.get("natural_line_height_em"), "size_source": "fonts" if fs.get("size_px") else "spec"})
    measured = {r["element"]: r["measured_px"] for r in (redline or {}).get("radii", []) if r.get("measured_px") is not None}
    radii = [{"name": r["element"], "dp": _r05(measured.get(r["element"], r["px"]) * k),
              "source": "redline" if r["element"] in measured else "spec"} for r in spec.get("radii", [])]
    for r, n in zip(radii, _uniq([_ident(r["name"]) for r in radii])):
        r.update(kt=n, sw=n[:1].lower() + n[1:], css=_ident(r["name"], "kebab"))
    spacing = [{"name": s["between"], "dp": _r05(s["px"] * k)} for s in spec.get("spacing", [])
               if not re.search(r"pixels? per|\bpx ?per\b|per (compose )?dp|px_to_dp|ratio|density|scale factor", s["between"], re.I)]
    for s, n in zip(spacing, _uniq([_ident(s["name"]) for s in spacing])):
        s.update(kt=n, sw=n[:1].lower() + n[1:], css=_ident(s["name"], "kebab"))
    return {"colors": colors, "types": types, "radii": radii, "spacing": spacing, "families": list(families.values()),
            "px_to_dp": k}


_SCHEME_RULES = [("background", r"screen|background|canvas"), ("surface", r"surface|panel|sheet"),
                 ("primary", r"primary.*(fill|button)|accent fill|button fill|primary lavender"),
                 ("onPrimary", r"on[- ]accent|on[- ]primary|button label"), ("onBackground", r"primary text"),
                 ("onSurface", r"primary text"), ("onSurfaceVariant", r"secondary text"), ("outline", r"border|outline")]


def _kt_header(a, pkg):
    head = f"/**\n * @author {a.author}\n * @since {time.strftime('%d/%m/%y')}\n */\n" if a.author else ""
    return f"{head}package {pkg}\n\n"


def _emit_compose(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    pkg = a.package
    lum = lambda h: sum(v * w for v, w in zip(_hexi(h), (0.2126, 0.7152, 0.0722))) / 255
    scheme = {}
    for role, pat in _SCHEME_RULES:
        c = next((c for c in tm["colors"] if re.search(pat, f"{c['name']} {c['usage']}", re.I)), None)
        if c and role not in scheme:
            scheme[role] = c["kt"]
    bg = next((c for c in tm["colors"] if c["kt"] == scheme.get("background")), None)
    dark = bg is not None and lum(bg["hex"]) < 0.5
    fn = "darkColorScheme" if dark else "lightColorScheme"
    lines = [f"val {c['kt']} = Color(0xFF{c['hex'][1:]})" for c in tm["colors"]]
    args = "".join(f"    {r} = {v},\n" for r, v in scheme.items())
    (out / "Color.kt").write_text(_kt_header(a, pkg) + f"import androidx.compose.material3.ColorScheme\n"
                                  f"import androidx.compose.material3.{fn}\nimport androidx.compose.ui.graphics.Color\n\n"
                                  + "\n".join(lines) + f"\n\nval {a.prefix}ColorScheme: ColorScheme = {fn}(\n{args})\n")
    rpkg = a.r_package or re.sub(r"(\.ui)?(\.theme)?$", "", pkg)
    fam_lines = []
    for f in tm["families"]:
        fonts_ = ",\n".join(f"    Font(R.font.{r['android']}, FontWeight.W{r['weight']}"
                            + (", FontStyle.Italic" if r.get("italic") else "") + ")" for r in f["files"])
        fam_lines.append(f"val {_ident(f['family'])} = FontFamily(\n{fonts_},\n)")
    uses_italic = any(t["italic"] for t in tm["types"]) or any(r.get("italic") for f in tm["families"] for r in f["files"])
    body = []
    for t in tm["types"]:
        fam = t["family_kt"] or ("FontFamily.Serif" if t["generic"] == "serif" else "FontFamily.SansSerif")
        col = t["color"]["kt"] if t["color"] else f"Color(0xFF{t['color_hex'][1:]})"
        ls = f"({t['tracking']:g}).em" if t["tracking"] < 0 else f"{t['tracking']:g}.em"
        body.append(f"    val {t['kt']} = TextStyle(\n        fontFamily = {fam},\n        fontWeight = FontWeight.W{t['weight']},\n"
                    + ("        fontStyle = FontStyle.Italic,\n" if t["italic"] else "")
                    + f"        fontSize = {_num_s(t['size'])}.sp,\n        lineHeight = {_num_s(t['line_height'])}.sp,\n"
                    f"        letterSpacing = {ls},\n        color = {col},\n    )")
    imports = ["androidx.compose.ui.graphics.Color", "androidx.compose.ui.text.TextStyle",
               "androidx.compose.ui.text.font.FontFamily", "androidx.compose.ui.text.font.FontWeight",
               "androidx.compose.ui.unit.em", "androidx.compose.ui.unit.sp"]
    if tm["families"]:
        imports += ["androidx.compose.ui.text.font.Font", f"{rpkg}.R"]
    if uses_italic:
        imports.append("androidx.compose.ui.text.font.FontStyle")
    (out / "Type.kt").write_text(_kt_header(a, pkg) + "".join(f"import {i}\n" for i in sorted(imports)) + "\n"
                                 + "".join(f"{f}\n\n" for f in fam_lines)
                                 + f"object {a.prefix}Typography {{\n" + "\n\n".join(body) + "\n}\n")
    shapes = [f"val {r['kt']}Shape = RoundedCornerShape({_num_s(r['dp'])}.dp)" for r in tm["radii"]]
    (out / "Shape.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.foundation.shape.RoundedCornerShape\n"
                                  "import androidx.compose.ui.unit.dp\n\n" + "\n".join(shapes) + "\n")
    dims = [f"    val {s['kt']} = {_num_s(s['dp'])}.dp" for s in tm["spacing"]]
    (out / "Dimens.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.ui.unit.dp\n\n"
                                   f"object {a.prefix}Dimens {{\n" + "\n".join(dims) + "\n}\n")
    return [out / f for f in ("Color.kt", "Type.kt", "Shape.kt", "Dimens.kt")]


def _emit_swift(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"    static let {c['sw']} = Color(red: {_hexi(c['hex'])[0]} / 255, green: {_hexi(c['hex'])[1]} / 255, "
            f"blue: {_hexi(c['hex'])[2]} / 255)" for c in tm["colors"]]
    wmap = {100: "ultraLight", 200: "thin", 300: "light", 400: "regular", 500: "medium", 600: "semibold", 700: "bold",
            800: "heavy", 900: "black"}
    styles = []
    for t in tm["types"]:
        if t["postscript"]:
            font = f'.custom("{t["postscript"]}", size: {_num_s(t["size"])})'
            natural = (t["natural_lh"] or 1.2) * t["size"]
        else:
            design = ".serif" if t["generic"] == "serif" else ".default"
            font = f".system(size: {_num_s(t['size'])}, weight: .{wmap.get(round(t['weight'] / 100) * 100, 'regular')}, design: {design})"
            natural = 1.19 * t["size"]
            if t["italic"]:
                font += ".italic()"
        col = f".{t['color']['sw']}" if t["color"] else (f"Color(red: {_hexi(t['color_hex'])[0]} / 255, green: "
                                                          f"{_hexi(t['color_hex'])[1]} / 255, blue: {_hexi(t['color_hex'])[2]} / 255)")
        styles.append(f"    static let {t['sw']} = TextToken(font: {font}, tracking: {round(t['tracking'] * t['size'], 2):g}, "
                      f"lineSpacing: {max(0.0, round(t['line_height'] - natural, 1)):g}, color: {col})")
    radii = [f"    static let {r['sw']}: CGFloat = {_num_s(r['dp'])}" for r in tm["radii"]]
    space = [f"    static let {s['sw']}: CGFloat = {_num_s(s['dp'])}" for s in tm["spacing"]]
    p = a.prefix
    src = ("import SwiftUI\n\nextension Color {\n" + "\n".join(cols) + "\n}\n\n"
           f"struct {p}TextToken {{\n    let font: Font\n    let tracking: CGFloat\n    let lineSpacing: CGFloat\n    let color: Color\n}}\n\n"
           f"extension {p}TextToken {{\n" + "\n".join(styles).replace("TextToken(", f"{p}TextToken(") + "\n}\n\n"
           "@available(iOS 16.0, macOS 13.0, *)\nextension View {\n"
           f"    func textToken(_ t: {p}TextToken) -> some View {{\n"
           "        font(t.font).tracking(t.tracking).lineSpacing(t.lineSpacing).foregroundStyle(t.color)\n    }\n}\n\n"
           f"enum {p}Radius {{\n" + "\n".join(radii) + "\n}\n\n" f"enum {p}Spacing {{\n" + "\n".join(space) + "\n}\n")
    (out / f"{p}Tokens.swift").write_text(src)
    return [out / f"{p}Tokens.swift"]


def _css_family(t):
    fallback = "serif" if t["generic"] == "serif" else "sans-serif"
    return f'"{t["family"]}", {fallback}' if t["family"] else ("Georgia, serif" if t["generic"] == "serif" else
                                                              "system-ui, -apple-system, sans-serif")


def _emit_web(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    fams = {}
    for t in tm["types"]:
        fams.setdefault(t["family_css"] or t["generic"], _css_family(t))
    v = [f"  --color-{c['css']}: {c['hex']};" for c in tm["colors"]]
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
    (out / "tokens.css").write_text(":root {\n" + "\n".join(v) + "\n}\n\n" + "\n\n".join(cls) + "\n")
    js = lambda o: json.dumps(o, indent=6).replace("\n}", "\n    }")
    cfg = {"colors": {c["css"]: c["hex"] for c in tm["colors"]},
           "fontFamily": {k: [x.strip().strip('"') for x in f.split(",")] for k, f in fams.items()},
           "fontSize": {t["css"]: [f"{_num_s(t['size'])}px", {"lineHeight": f"{_num_s(t['line_height'])}px",
                                                               "letterSpacing": f"{t['tracking']:g}em",
                                                               "fontWeight": str(t["weight"])}] for t in tm["types"]},
           "borderRadius": {r["css"]: f"{_num_s(r['dp'])}px" for r in tm["radii"]},
           "spacing": {s["css"]: f"{_num_s(s['dp'])}px" for s in tm["spacing"]}}
    body = ",\n".join(f"    {k}: {js(val)}" for k, val in cfg.items())
    (out / "tailwind.config.js").write_text("/** @type {import('tailwindcss').Config} */\nmodule.exports = {\n"
                                            "  content: ['./src/**/*.{html,js,jsx,ts,tsx,vue,svelte}'],\n"
                                            f"  theme: {{\n    extend: {{\n{body}\n    }},\n  }},\n}};\n")
    th = [f"  --color-{c['css']}: {c['hex']};" for c in tm["colors"]]
    th += [f"  --font-{k}: {f};" for k, f in fams.items()]
    for t in tm["types"]:
        th += [f"  --text-{t['css']}: {_num_s(t['size'])}px;", f"  --text-{t['css']}--line-height: {_num_s(t['line_height'])}px;",
               f"  --text-{t['css']}--letter-spacing: {t['tracking']:g}em;", f"  --text-{t['css']}--font-weight: {t['weight']};"]
    th += [f"  --radius-{r['css']}: {_num_s(r['dp'])}px;" for r in tm["radii"]]
    th += [f"  --spacing-{s['css']}: {_num_s(s['dp'])}px;" for s in tm["spacing"]]
    (out / "tailwind.theme.css").write_text('@import "tailwindcss";\n\n@theme {\n' + "\n".join(th) + "\n}\n")
    return [out / f for f in ("tokens.css", "tailwind.config.js", "tailwind.theme.css")]


def _verify_tokens(tm, out, a):
    res = {}
    sw = out / "swiftui" / f"{a.prefix}Tokens.swift"
    if shutil.which("swiftc"):
        r = subprocess.run(["swiftc", "-typecheck", str(sw)], capture_output=True, text=True)
        res["swiftui"] = "ok (swiftc -typecheck)" if r.returncode == 0 else "FAILED: " + r.stderr.strip()[:600]
    else:
        res["swiftui"] = "skipped (no swiftc)"
    if shutil.which("node"):
        r = subprocess.run(["node", "-e", f"const c=require({json.dumps(str((out / 'web' / 'tailwind.config.js').resolve()))});"
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
        try:
            dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--use-mock-keychain",
                                  "--dump-dom", probe.resolve().as_uri()], capture_output=True, text=True, timeout=60).stdout
        except (subprocess.TimeoutExpired, OSError):
            dom = ""
        m_ = re.search(r'<pre id="r">(.*?)</pre>', dom, re.S)
        if m_:
            got = json.loads(m_.group(1).replace("&quot;", '"'))
            empty = [n for n, v in got["vars"].items() if not v]
            bad = [t["css"] for t in tm["types"] if got["text"].get(f"text-{t['css']}", [""])[0] != f"{_num_s(t['size'])}px"]
            res["css"] = ("ok (Chrome parsed %d vars, %d text classes)" % (len(got["vars"]), len(got["text"]))
                          if not empty and not bad else f"FAILED: empty vars {empty[:5]}, wrong text sizes {bad[:5]}")
        else:
            res["css"] = "FAILED: Chrome produced no probe output"
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


def cmd_tokens(a):
    d, spec, mock = _d2c_load_spec(a.dir)
    out = Path(a.out) if a.out else d / "tokens"
    out.mkdir(parents=True, exist_ok=True)
    texts = _text_targets(d, spec, mock, a.vision)
    _save_text_boxes(d, mock, texts, out / "text_boxes.png")
    assets = json.loads((d / "assets.json").read_text())["assets"] if (d / "assets.json").exists() else []
    palette = _snap_palette(spec, mock, texts, assets) if not a.no_snap else \
        [{"name": p["name"], "usage": p.get("usage", ""), "hex_guess": p["hex"].upper(), "hex": p["hex"].upper(),
          "de00": 0.0, "snapped": False} for p in spec.get("palette", [])]
    fonts_path = Path(a.fonts) if a.fonts else d / "fonts" / "fonts.json"
    fonts = json.loads(fonts_path.read_text()) if fonts_path.exists() else None
    rl_path = d / "redline" / "redline.json"
    redline = json.loads(rl_path.read_text()) if rl_path.exists() else None
    tm = _token_model(spec, palette, texts, fonts, redline)
    files = []
    if "compose" in a.platforms:
        files += _emit_compose(tm, a, out / "compose")
    if "swiftui" in a.platforms:
        files += _emit_swift(tm, a, out / "swiftui")
    if "web" in a.platforms:
        files += _emit_web(tm, a, out / "web")
    _palette_sheet(palette, mock, texts, spec, out / "palette_check.png")
    report = {"palette": palette, "text": {r: {k: v for k, v in t.items() if k not in ("alpha", "instances")}
                                           for r, t in texts.items()},
              "typography": [{k: t[k] for k in ("role", "family", "weight", "size", "line_height", "tracking", "color_hex",
                                                "size_source")} for t in tm["types"]],
              "radii": tm["radii"], "px_to_dp": tm["px_to_dp"], "fonts": str(fonts_path) if fonts else None}
    print(f"palette ({sum(p['snapped'] for p in palette)}/{len(palette)} snapped to mockup pixels):")
    for p in palette:
        mark = f"{p['hex_guess']} -> {p['hex']}  dE00 {p['de00']:>4}" if p["snapped"] else f"{p['hex_guess']}  (kept: not found)"
        print(f"  {p['name'][:30]:30s} {mark}   [{p.get('source', '')[:60]}]")
    miss = [r for r, t in texts.items() if not t.get("found")]
    print(f"text samples located: {len(texts) - len(miss)}/{len(texts)}" + (f"  missing: {', '.join(miss)}"
                                                                           " (add --vision)" if miss else ""))
    if not fonts:
        print("  (no fonts.json: generic serif/sans families; run `fonts` first for matched Google Fonts)")
    if a.verify:
        report["verify"] = _verify_tokens(tm, out, a)
        for k, v in report["verify"].items():
            print(f"verify {k:8s}: {v}")
    (out / "tokens.json").write_text(json.dumps(report, indent=2))
    print(*[f"  {f}" for f in files], sep="\n")
    print(f"-> {out}/tokens.json, palette_check.png, text_boxes.png")


# ---- fonts: match typography to Google Fonts by rendering and measuring

GF_METADATA = "https://fonts.google.com/metadata/fonts"
_GENERIC_WORDS = {"display", "text", "sans", "serif", "italic", "mono", "pro", "regular", "bold", "medium", "light", "condensed",
                  "similar", "alternatives", "transitional", "contrast", "high", "slanted", "lightly", "rounded", "humanist",
                  "geometric", "grotesque", "neo", "system", "font", "family", "style", "classic", "modern", "elegant"}


def _cache_dir():
    return Path(os.environ.get("GPTKIT_CACHE", "~/.cache/gptkit")).expanduser()


@functools.cache
def _gf_meta():
    p = _cache_dir() / "gf_metadata.json"
    if not p.exists() or time.time() - p.stat().st_mtime > 7 * 86400:
        try:
            r = requests.get(GF_METADATA, timeout=60)
            r.raise_for_status()
        except requests.RequestException:
            if not p.exists():
                raise
            print("  (Google Fonts unreachable: using the cached metadata)")
        else:
            t = r.text[4:] if r.text.startswith(")]}'") else r.text
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t)
    return json.loads(p.read_text())["familyMetadataList"]


def _gf_ttf(family, weight, italic=False):
    """Static TTF of one Google Fonts family/weight via the CSS API (cached)."""
    p = _cache_dir() / "fonts" / re.sub(r"\W+", "_", family) / f"{weight}{'i' if italic else ''}.ttf"
    if p.exists():
        return p
    url = f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:ital,wght@{int(italic)},{weight}"
    try:
        css = requests.get(url, headers={"User-Agent": "gptkit"}, timeout=30)
        m_ = re.search(r"url\((https://[^)]+\.ttf)\)", css.text) if css.ok else None
        r = requests.get(m_.group(1), timeout=60) if m_ else None
    except requests.RequestException:
        return None
    if r is None or not r.ok:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.{id(r)}.part")
    tmp.write_bytes(r.content)
    tmp.replace(p)
    return p


def _gf_license(family):
    """(license id, repo dir, license text) from the google/fonts repository METADATA.pb."""
    slug = re.sub(r"[^a-z0-9]", "", family.lower())
    cache = _cache_dir() / "licenses" / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    res = {"license": "unknown (see fonts.google.com)", "dir": None, "text": None}
    for lic_dir, fname in (("ofl", "OFL.txt"), ("apache", "LICENSE.txt"), ("ufl", "UFL.txt")):
        try:
            r = requests.get(f"https://raw.githubusercontent.com/google/fonts/main/{lic_dir}/{slug}/METADATA.pb", timeout=20)
        except requests.RequestException:
            break
        if r.ok:
            m_ = re.search(r'license:\s*"([^"]+)"', r.text)
            t = requests.get(f"https://raw.githubusercontent.com/google/fonts/main/{lic_dir}/{slug}/{fname}", timeout=20)
            res = {"license": m_.group(1) if m_ else lic_dir.upper(), "dir": f"{lic_dir}/{slug}", "text": t.text if t.ok else None}
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


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


def _render_line(font, text, track_px, kern=None, ss=4):
    """Coverage map of one text line (tight, supersampled) plus each character's [x0, x1) advance span in it."""
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
        dr.text((x, pad + asc), ch, font=big, fill=255, anchor="ls", stroke_width=FONT_EMBOLDEN, stroke_fill=255)
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
    r, spans = _render_line(font, text, track * size, kern)
    r = r ** FONT_GAMMA
    words = [(m_.start(), m_.end() - 1) for m_ in re.finditer(r"\S+", text)]
    sim, word_mode = _shape_similarity(target, r, text, metric, [(spans[i][0], spans[j][1]) for i, j in words])
    asc, desc = font.getmetrics()
    return {"similarity": round(sim, 4), "size_px": round(size, 1), "tracking_em": round(track, 3),
            "word_mode": word_mode, "natural_line_height_em": round((asc + desc) / size, 3), "render": r}


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


def cmd_fonts(a):
    d, spec, mock = _d2c_load_spec(a.dir)
    try:
        _gf_meta()
    except (requests.RequestException, ValueError) as e:
        sys.exit(f"fonts: can't load Google Fonts metadata from {GF_METADATA} ({e}); needs network once, then it is cached")
    out = Path(a.out) if a.out else d / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    rgb = mock.convert("RGB")
    texts = _text_targets(d, spec, mock, a.vision)
    styles = [st for st in spec.get("typography", []) if texts.get(st["role"], {}).get("found")]
    skipped = [st["role"] for st in spec.get("typography", []) if st not in styles]
    if skipped:
        print("  ! no text box for:", ", ".join(skipped), "(these keep their spec values; add --vision to locate them)")
    groups = {}
    for st in styles:
        groups.setdefault(_font_group(st), []).append(st)
    results, report_groups, sheet_rows = {}, [], []
    k = spec.get("px_to_dp", 1.0)
    for gkey, members in groups.items():
        italic = gkey.endswith("|italic")
        guess = members[0].get("family_guess", "")
        wts = sorted({st.get("weight", 400) for st in members})
        display_ok = any(st["size_px"] * k >= 20 for st in members)
        cands, cat = _font_shortlist(guess, wts[len(wts) // 2], italic, a.candidates, display_ok)
        if a.extra:
            extra = {x.strip().lower() for x in a.extra.split(",")}
            cands += [f for f in _gf_meta() if f["family"].lower() in extra and f not in cands]
        print(f"group '{guess[:50]}' ({len(members)} styles, {cat}): {len(cands)} candidate families")
        need = sorted({(f["family"], w) for f in cands for st in members
                       for w in sorted((int(x.rstrip("i")) for x in f["fonts"] if x.endswith("i") == italic),
                                       key=lambda x: abs(x - st.get("weight", 400)))[:a.weights]})
        with cf.ThreadPoolExecutor(8) as ex:
            paths = dict(zip(need, ex.map(lambda fw: _gf_ttf(fw[0], fw[1], italic), need)))
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
            continue
        named = [f["family"] for f in _named_families(guess)]
        best = ranking[0][1]
        prior = next((f for f, sc in ranking if f in named and sc >= best - a.tie), None)
        winner = a.family if a.family and any(f == a.family for f, _ in ranking) else (prior or ranking[0][0])
        why = "forced" if winner == a.family else ("spec's named family, within tie margin" if prior and prior != ranking[0][0]
                                                   else "best measured")
        report_groups.append({"guess": guess, "category": cat, "italic": italic, "winner": winner, "decided_by": why,
                              "ranking": [[f, round(sc, 4)] for f, sc in ranking[:8]], "styles": [m["role"] for m in members]})
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
            ok = rs[0][2]["similarity"] >= 0.45 and -0.15 < fit["tracking_em"] < 0.6
            avail = [int(x.rstrip("i")) for f in cands if f["family"] == winner for x in f["fonts"] if x.endswith("i") == italic]
            spec_w = min(avail, key=lambda x: abs(x - st.get("weight", w))) if avail else w
            results[st["role"]] = {"role": st["role"], "family": winner, "weight": w if ok else spec_w, "italic": italic,
                                   "size_px": fit["size_px"] if ok else None,
                                   "tracking_em": fit["tracking_em"] if ok else st.get("letter_spacing_em", 0),
                                   "reliable": ok,
                                   "similarity": fit["similarity"], "word_mode": fit["word_mode"],
                                   "natural_line_height_em": fit["natural_line_height_em"],
                                   "spec": {"size_px": st["size_px"], "weight": st.get("weight"),
                                            "letter_spacing_em": st.get("letter_spacing_em"), "family_guess": guess},
                                   "alternatives": [[r[0], r[1], r[2]["similarity"]] for r in rs[:6]]}
            if not ok:
                results[st["role"]]["warning"] = (f"unreliable measurement (best fit {rs[0][2]['similarity']:.2f}); kept the spec's "
                                                  "size/weight/tracking; check its box in text_boxes.png")
            elif rs[0][0] != winner and rs[0][2]["similarity"] > fit["similarity"] + 0.1:
                results[st["role"]]["warning"] = (f"fits {winner} poorly ({fit['similarity']:.2f}); {rs[0][0]} {rs[0][1]} fits this "
                                                  f"style much better ({rs[0][2]['similarity']:.2f})")
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
    _font_sheet(sheet_rows, out / "fonts_review.png")
    families = []
    ttf_dir, res_font, lic_dir = out / "ttf", out / "android" / "res" / "font", out / "licenses"
    if (out / "fonts.json").exists():
        for f in json.loads((out / "fonts.json").read_text()).get("families", []):
            slug = re.sub(r"[^a-z0-9]+", "_", f["family"].lower()).strip("_")
            stale = [Path(x["ttf"]) for x in f["files"]] + [res_font / f"{x['android']}.ttf" for x in f["files"]]
            stale += [res_font / f"{slug}.xml"] + list(lic_dir.glob(re.sub(r"\W+", "", f["family"]) + "-*.txt"))
            for x in stale:
                x.unlink(missing_ok=True)
    for fam in sorted({r["family"] for r in results.values()}):
        rs = [r for r in results.values() if r["family"] == fam]
        lic = _gf_license(fam)
        files, ps = [], {}
        for w, it in sorted({(r["weight"], r["italic"]) for r in rs}):
            src = _gf_ttf(fam, w, it)
            if not src:
                print(f"  ! could not download {fam} {w}{' italic' if it else ''}; skipped")
                continue
            psn, full = _ttf_names(src)
            fname = (psn or fam.replace(" ", "") + f"-{w}") + ".ttf"
            ttf_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, ttf_dir / fname)
            wname = {100: "thin", 200: "extralight", 300: "light", 400: "regular", 500: "medium", 600: "semibold", 700: "bold",
                     800: "extrabold", 900: "black"}.get(w, str(w))
            android = f"{re.sub(r'[^a-z0-9]+', '_', fam.lower()).strip('_')}_{wname}{'_italic' if it else ''}"
            res_font.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, res_font / f"{android}.ttf")
            files.append({"weight": w, "italic": it, "ttf": str(ttf_dir / fname), "android": android, "postscript": psn})
            ps[f"{w}{'i' if it else ''}"] = psn
        xml_name = re.sub(r"[^a-z0-9]+", "_", fam.lower()).strip("_")
        items = "".join(f'    <font\n        android:font="@font/{f["android"]}"\n        android:fontStyle="'
                        f'{"italic" if f["italic"] else "normal"}"\n        android:fontWeight="{f["weight"]}" />\n' for f in files)
        (res_font / f"{xml_name}.xml").write_text('<?xml version="1.0" encoding="utf-8"?>\n<font-family xmlns:android='
                                                  f'"http://schemas.android.com/apk/res/android">\n{items}</font-family>\n')
        if lic.get("text"):
            lic_dir.mkdir(parents=True, exist_ok=True)
            (lic_dir / (re.sub(r"\W+", "", fam) + f"-{lic['license']}.txt")).write_text(lic["text"])
        families.append({"family": fam, "license": lic["license"], "source": lic.get("dir"), "files": files, "postscript": ps,
                         "android_family": f"@font/{xml_name}",
                         "css": f"https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}:wght@"
                                + ";".join(str(w) for w in sorted({f['weight'] for f in files}))})
    web = out / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "fonts.css").write_text("".join(
        f"@font-face {{\n  font-family: \"{f['family']}\";\n  font-style: {'italic' if x['italic'] else 'normal'};\n"
        f"  font-weight: {x['weight']};\n  font-display: swap;\n  src: url(\"../ttf/{Path(x['ttf']).name}\") format(\"truetype\");\n}}\n"
        for f in families for x in f["files"]))
    ios = out / "ios"
    ios.mkdir(parents=True, exist_ok=True)
    (ios / "UIAppFonts.plist").write_text("<key>UIAppFonts</key>\n<array>\n" + "".join(
        f"    <string>{Path(x['ttf']).name}</string>\n" for f in families for x in f["files"]) + "</array>\n")
    manifest = {"families": families, "groups": report_groups, "styles": list(results.values()), "unmatched": skipped}
    (out / "fonts.json").write_text(json.dumps(manifest, indent=2))
    print("\nstyle                           family                 wt  size px (spec)   tracking em (spec)  sim")
    for r in results.values():
        print(f"  {r['role'][:30]:30s} {r['family'][:20]:20s} {r['weight']:>4} {r['size_px'] or '-':>6} ({r['spec']['size_px']:>4})"
              f"   {r['tracking_em']:>7} ({r['spec']['letter_spacing_em']:>6})   {r['similarity']:.3f}")
    for r in results.values():
        if r.get("warning"):
            print(f"  ! {r['role']}: {r['warning']}")
    for f in families:
        print(f"{f['family']}: {f['license']} ({f['source']}); files: " + ", ".join(Path(x["ttf"]).name for x in f["files"]))
    print(f"-> {out}/fonts.json, fonts_review.png, ttf/, android/res/font/, ios/UIAppFonts.plist, web/fonts.css, licenses/")
    print("iOS: add ttf/*.ttf to the app target and merge ios/UIAppFonts.plist into Info.plist; use Font.custom(<PostScript name>). "
          "Web: link web/fonts.css (or the Google Fonts css URL in fonts.json). Re-run `tokens` to use these families.")


# ---- redline: pixel-true geometry (edges, corner radii, gaps) for the components of an approved design

def _rise(dev, tau, raw=None):
    """Sub-pixel edge in a deviation profile: detected where the smoothed profile reaches tau, placed at the half-height
    crossing of the raw profile (None if the profile starts inside or never rises)."""
    idx = np.flatnonzero(dev >= tau)
    if not len(idx) or idx[0] == 0:
        return None
    i = idx[0]
    if raw is None:
        return i - 1 + (tau - dev[i - 1]) / max(dev[i] - dev[i - 1], 1e-9) + 0.5
    lo, hi = max(0, i - 3), min(len(raw), i + 4)
    half = (raw[lo] + raw[lo:hi].max()) / 2
    seg = raw[lo:hi]
    j = next((n for n in range(1, len(seg)) if seg[n - 1] < half <= seg[n]), None)
    if j is None:
        return i - 1 + (tau - dev[i - 1]) / max(dev[i] - dev[i - 1], 1e-9) + 0.5
    return lo + j - 1 + (half - seg[j - 1]) / max(seg[j] - seg[j - 1], 1e-9) + 0.5


def _surface_geometry(rgb, b):
    """Refined edges and corner radius of a surface (fill, border or image) around bbox b, or None if its outline
    doesn't stand out from the surroundings near the bbox."""
    H, W = rgb.shape[:2]
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    if w < 12 or h < 12:
        return None
    m = max(8, int(0.08 * min(w, h)))
    ring = np.concatenate([rgb[max(0, y - m):max(0, y - 2), max(0, x):x + w].reshape(-1, 3),
                           rgb[y + h + 2:min(H, y + h + m), max(0, x):x + w].reshape(-1, 3),
                           rgb[max(0, y):y + h, max(0, x - m):max(0, x - 2)].reshape(-1, 3),
                           rgb[max(0, y):y + h, x + w + 2:min(W, x + w + m)].reshape(-1, 3)])
    inner = rgb[y + h // 3:y + h - h // 3, x + w // 3:x + w - w // 3]
    if inner.size == 0 or ring.size == 0:
        return None
    c_out = _dominant(ring.astype(float))
    oy, ox = max(0, y - m), max(0, x - m)
    lab = _lab_arr(rgb[oy:min(H, y + h + m), ox:min(W, x + w + m)].astype(np.float64))
    raw = np.linalg.norm(lab - _lab_arr(c_out), axis=-1)
    dev = _box_mean(raw, 3)
    ring_dev = np.linalg.norm(_lab_arr(ring.astype(float)) - _lab_arr(c_out), axis=-1)
    tau = max(3.0, 2 * float(np.percentile(ring_dev, 90)))
    edges = {}
    for side in ("top", "bottom", "left", "right"):
        vals = []
        for f in np.linspace(0.4, 0.6, 9):
            if side in ("top", "bottom"):
                col = int(x + f * w) - ox
                prof, rprof = (dev[:, col], raw[:, col]) if side == "top" else (dev[::-1, col], raw[::-1, col])
            else:
                row = int(y + f * h) - oy
                prof, rprof = (dev[row, :], raw[row, :]) if side == "left" else (dev[row, ::-1], raw[row, ::-1])
            c = _rise(prof[:2 * m + 2], tau, rprof[:2 * m + 2])
            if c is not None:
                vals.append(c)
        if len(vals) < 5:
            return None
        v = float(np.median(vals))
        edges[side] = {"top": oy + v, "left": ox + v, "bottom": oy + dev.shape[0] - v, "right": ox + dev.shape[1] - v}[side]
    tol_x, tol_y = max(6, 0.05 * w), max(6, 0.05 * h)
    if (abs(edges["left"] - x) > tol_x or abs(edges["right"] - x - w) > tol_x or abs(edges["top"] - y) > tol_y
            or abs(edges["bottom"] - y - h) > tol_y):
        return None
    ew, eh = edges["right"] - edges["left"], edges["bottom"] - edges["top"]
    radii = []
    for flip_x, flip_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        dd, rr = (dev[:, ::-1], raw[:, ::-1]) if flip_x else (dev, raw)
        dd, rr = (dd[::-1], rr[::-1]) if flip_y else (dd, rr)
        top = (oy + dev.shape[0] - edges["bottom"]) if flip_y else edges["top"] - oy
        left = (ox + dev.shape[1] - edges["right"]) if flip_x else edges["left"] - ox
        pts = []
        for r_ in range(int(math.ceil(top)), int(top + min(ew, eh) / 2)):
            c = _rise(dd[r_, :int(left + min(ew, eh) / 2)], tau, rr[r_, :int(left + min(ew, eh) / 2)])
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
        radii.append(best[1])
    if not radii:
        return None
    r = float(np.median(radii))
    round_ = r > 0.45 * min(ew, eh)
    r = min(ew, eh) / 2 if round_ else r
    return {"box": [round(edges["left"], 1), round(edges["top"], 1), round(ew, 1), round(eh, 1)], "radius": round(r, 1),
            "corner_radii": [round(v, 1) for v in radii], "fill": _to_hex(_dominant(inner.astype(float))), "outside": _to_hex(c_out),
            "shape": "circle" if round_ and abs(ew - eh) < 0.03 * max(ew, eh) else ("pill" if round_ else "rect")}


def cmd_redline(a):
    d, spec, mock = _d2c_load_spec(a.dir)
    rgb = np.asarray(mock.convert("RGB"))
    k = spec.get("px_to_dp", 1.0)
    m = json.loads((d / "assets.json").read_text()) if (d / "assets.json").exists() else None
    texts = _text_targets(d, spec, mock) if spec.get("typography") else {}
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
                        shape=geo["shape"])
        else:
            pad = 4
            x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
            ob = _object_box(rgb[y0:b["y"] + b["h"] + pad, x0:b["x"] + b["w"] + pad].astype(np.float64))
            if not ob:
                continue
            item.update(kind="object", px=[round(x0 + ob[0], 1), round(y0 + ob[1], 1), round(ob[2], 1), round(ob[3], 1)])
        item["dp"] = [round(v * k, 1) for v in item["px"]]
        if "radius_px" in item:
            item["radius_dp"] = round(item["radius_px"] * k, 1)
        items.append(item)
    for role, t in texts.items():
        for inst in t.get("instances", []):
            bx = inst["box"]
            items.append({"name": f"text: {inst['sample'][:30]}", "role": role, "kind": "text", "px": bx,
                          "dp": [round(v * k, 1) for v in bx], "baseline_dp": round(inst["baseline"] * k, 1), "ink": inst["ink"]})
    for it in items:
        x, y, w, h = it["px"]
        par = [o for o in items if o is not it and o["kind"] == "surface" and o["px"][0] <= x + 0.5 and o["px"][1] <= y + 0.5
               and o["px"][0] + o["px"][2] >= x + w - 0.5 and o["px"][1] + o["px"][3] >= y + h - 0.5 and o["px"][2] * o["px"][3] > w * h]
        if par:
            p = min(par, key=lambda o: o["px"][2] * o["px"][3])
            it["parent"] = p["name"]
            it["inset_dp"] = {"left": round((x - p["px"][0]) * k, 1), "top": round((y - p["px"][1]) * k, 1),
                              "right": round((p["px"][0] + p["px"][2] - x - w) * k, 1),
                              "bottom": round((p["px"][1] + p["px"][3] - y - h) * k, 1)}
    col = sorted(items, key=lambda o: o["px"][1])
    for it in col:
        x, y, w, h = it["px"]
        below = [o for o in col if o["px"][1] >= y + h - 0.5 and min(x + w, o["px"][0] + o["px"][2]) - max(x, o["px"][0]) > 0.3 * min(w, o["px"][2])
                 and o.get("parent") == it.get("parent")]
        if below:
            nb = min(below, key=lambda o: o["px"][1])
            it["gap_below_dp"] = [nb["name"], round((nb["px"][1] - y - h) * k, 1)]
    radii_cmp = []
    notes = {c["name"]: c.get("notes", "") for c in spec.get("components", [])}
    for rr in spec.get("radii", []):
        hits = _link_regions(rr["element"], [{"name": i["name"], "notes": notes.get(i["name"], "")} for i in items
                                             if i["kind"] == "surface"])
        hit = next((i for h_ in hits for i in items if i["name"] == h_["name"] and 0.5 <= (i["radius_px"] + 1) / (rr["px"] + 1) <= 2), None)
        radii_cmp.append({"element": rr["element"], "spec_px": rr["px"], "measured_px": hit["radius_px"] if hit else None,
                          "component": hit["name"] if hit else None, "shape": hit["shape"] if hit else None})
    ov = mock.convert("RGB")
    dr = ImageDraw.Draw(ov)
    for it in items:
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
    print(f"{len(items)} items ({n_s} surfaces with measured edges + corner radius, "
          f"{sum(i['kind'] == 'object' for i in items)} objects, {sum(i['kind'] == 'text' for i in items)} text lines)")
    for rc in radii_cmp:
        got = f"measured {rc['measured_px']:>5}px ({rc['shape']}, {rc['component']})" if rc["component"] else "no matching surface measured"
        print(f"  radius {rc['element'][:30]:30s} spec {rc['spec_px']:>5}px  {got}")
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
    """Bounding box (x, y, w, h) of what stands out from the dominant background of a region, or None."""
    d = np.linalg.norm(_lab_arr(rgb) - _lab_arr(_dominant(rgb)), axis=-1)
    m = d > max(12.0, 0.25 * np.percentile(d, 99.5))
    ys, xs = np.nonzero(m)
    if len(ys) < 12:
        return None
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    return float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)


def _tolerant_de(ref, cur):
    """Per-pixel CIEDE2000 after a light blur, minimised over +-1px shifts (ignores anti-aliasing jitter)."""
    blur = lambda x: _lab_arr(np.asarray(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                                         dtype=np.float64))
    lm, lc = blur(ref), blur(cur)
    de = np.full(ref.shape[:2], np.inf)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            de = np.minimum(de, _de00(lm, np.roll(lc, (oy, ox), (0, 1))))
    return de


def _impl_font_check(fonts, role, sample, alpha, size_px):
    """Which font explains implemented text best: the one fonts.json chose, a neighbouring weight, or another family?

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

    def one(c):
        try:
            path = _gf_ttf(c[0], c[1], it)
        except requests.RequestException:
            return None
        fit = _fit_font(path, sample, alpha, size_px) if path else None
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
    d = Path(a.dir)
    spec = json.loads((d / "design_spec.json").read_text()) if (d / "design_spec.json").exists() else {"components": []}
    m = json.loads((d / "assets.json").read_text()) if (d / "assets.json").exists() else None
    mock = _rgba(a.mockup or d / (m or {}).get("mockup", "mockup.png")).convert("RGB")
    W, H = mock.size
    k = spec.get("px_to_dp") or (m["screen_dp"] / m["width"] if m else 1.0)
    if spec.get("effective_width_dp"):
        print(f"  (native scale: for a 1:1 check the screenshot should show the screen at ~{spec['effective_width_dp']}dp wide; "
              "a narrower device reflows the layout)")
    shot = Image.open(a.screenshot).convert("RGB")
    if a.crop_top or a.crop_bottom:
        shot = shot.crop((0, a.crop_top, shot.width, shot.height - a.crop_bottom))
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
    blur = lambda x: np.asarray(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                                dtype=np.float64)
    de = _tolerant_de(M, A)
    de[~valid] = 0
    ssim = _ssim_map(Mg, _gray(A))
    overall = {"ssim": round(float(ssim[valid].mean()), 4), "mean_de00": round(float(de[valid].mean()), 2),
               "pct_pixels_de_gt_10": round(float((de[valid] > 10).mean() * 100), 1),
               "vertical_offset_px": int(dy0), "status_bar_px": int(top_extra), "extra_bottom_px": int(bottom_extra)}
    overall["score"] = round(100 * overall["ssim"] * max(0.0, 1 - overall["mean_de00"] / 50), 1)
    texts = _text_targets(d, spec, _rgba(a.mockup or d / (m or {}).get("mockup", "mockup.png"))) if spec.get("typography") else {}
    role_of = {}
    for r_, t_ in texts.items():
        for inst in t_.get("instances", []):
            if inst.get("component"):
                role_of.setdefault(inst["component"], []).append((r_, inst))
    fpath = d / "fonts" / "fonts.json"
    fonts_json = json.loads(fpath.read_text()) if fpath.exists() else None
    styles = {st["role"]: st for st in spec.get("typography", [])}
    Sg32, Mg32 = Sg.astype(np.float32), Mg.astype(np.float32)
    R = max(24, int(0.04 * H))
    found = []
    for r in _regions(spec, m, W, H):
        b = r["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        if w < 4 or h < 4:
            continue
        hit = _find_region(Mg32, Sg32, b, dy0, R)
        if hit and hit[0] < 0.5 and bottom_extra + top_extra > R:
            far = _find_region(Mg32, Sg32, b, dy0 + (bottom_extra + top_extra) // 2, (bottom_extra + top_extra) // 2 + R)
            hit = far if far and far[0] > hit[0] + 0.1 else hit
        rep = {"name": r["name"], "asset": r.get("asset"), "bbox": b}
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
            judge = fonts_json and th_ >= 16
            fspec, fbest = _impl_font_check(fonts_json, role, t["sample"], got["alpha"], st["size_px"]) if judge else (None, None)
            sh = [round((gx - tx) * k, 1), round((got["baseline"] - dy0 - t["baseline"]) * k, 1)]
            rep.setdefault("text", []).append({"role": role, "sample": t["sample"], "height_ratio": round(hr, 3),
                                               "width_ratio": round(wr, 3), "ink": [t["ink"], got["ink"]], "ink_de00": dei,
                                               "glyph_similarity_vs_design": shape, "spec_font_fit": fspec,
                                               "best_font_fit": fbest, "shift_dp": sh})
            lab = f"'{t['sample'][:20]}'"
            if abs(hr - 1) >= 0.04 and abs(gh - th_) >= 2:
                hints.append(f"{lab} {abs(hr - 1) * 100:.0f}% {'larger' if hr > 1 else 'smaller'} (ink height {th_} -> {gh}px; "
                             f"try {st['size_px'] * k / hr:.1f}sp)")
            elif abs(wr - 1) >= 0.05 and abs(gw - tw_) >= 3:
                hints.append(f"{lab} {abs(wr - 1) * 100:.0f}% {'wider' if wr > 1 else 'narrower'} at the same height "
                             "(letter-spacing or font family/weight differs)")
            if fspec and fbest and fbest[2] > fspec[2] + (0.08 if fbest[0] != fspec[0] else 0.05):
                what = f"{fbest[0]} {fbest[1]}" if fbest[0] != fspec[0] else f"weight {fbest[1]}"
                hints.append(f"{lab} looks like {what} (glyph fit {fbest[2]:.2f}) rather than the specified {fspec[0]} {fspec[1]} "
                             f"({fspec[2]:.2f})")
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
        sx_, sy_ = rep["shift_dp"]
        if abs(sx_) >= 2 or abs(sy_) >= 2:
            hints.append(f"shifted {'%+.0fdp horizontally' % sx_ if abs(sx_) >= 2 else ''}"
                         f"{' and ' if abs(sx_) >= 2 and abs(sy_) >= 2 else ''}{'%+.0fdp vertically' % sy_ if abs(sy_) >= 2 else ''}")
        if "text" not in rep and not r["children"] and r.get("asset_kind") not in ("background", "photo"):
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
        if rep["match"] is not None and rep["match"] < 0.6:
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
                      + (1.0 if rep["diff_blob"]["what"].startswith(("missing", "extra")) else 0) if "diff_blob" in rep else 0))
        rep["badness"] = round(badness, 3)
        rep["hints"] = hints
        found.append(rep)
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
    if a.vision and found:
        top = found[:min(6, len(found))]
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
                if 1 <= row["index"] <= len(top):
                    top[row["index"] - 1]["vision"] = row["differences"]
        except Exception as e:
            print(f"  ! vision pass failed: {e}")
    heat = (_heat_rgb(de / 20) * 0.75 + M * 0.25).astype(np.uint8)
    hm = Image.fromarray(heat)
    dr = ImageDraw.Draw(hm)
    for i, r in enumerate(found[:a.top]):
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
    for i, r in enumerate(found[:a.top]):
        b = r["bbox"]
        pad = 10
        box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(W, b["x"] + b["w"] + pad), min(H, b["y"] + b["h"] + pad))
        rows.append([(f"#{i + 1} {r['name'][:30]} design", mock.crop(box)), ("implemented", shot.crop(_impl_box(box, r, dy0, k))),
                     ("dE heat", hm.crop(box))])
    if rows:
        sheet(rows, out / "regions.png", T=260)
    report = {"overall": overall, "regions": found, "unlabelled": unl, "mockup": str(a.mockup or d / (m or {}).get("mockup", "mockup.png")),
              "screenshot": str(a.screenshot), "px_to_dp": k}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"overall score {overall['score']}/100  (SSIM {overall['ssim']}, mean dE00 {overall['mean_de00']}, "
          f"{overall['pct_pixels_de_gt_10']}% px dE>10)")
    if top_extra or bottom_extra:
        print(f"  aligned with a {top_extra}px ({top_extra * k:.0f}dp) top offset (status bar / extra top inset); "
              f"{bottom_extra}px extra at the bottom (taller screen or nav bar) ignored")
    print(f"worst regions (of {len(found)}):")
    for i, r in enumerate(found[:a.top]):
        hint = "; ".join(r["hints"]) or f"minor (dE {r['de00_mean']}, SSIM {r['ssim']})"
        print(f"  {i + 1:2d}. {r['name'][:34]:34s} bad {r['badness']:.2f}  {hint}")
        for v in r.get("vision", []):
            print(f"        vision: {v}")
    if unl:
        print(f"  + {len(unl)} unlabelled 32px cells with dE>8 (white boxes on the heatmap)")
    print(f"-> {out}/report.json, heatmap.png (design | implementation aligned | dE heatmap), regions.png")


def _design_to_code_parsers(sub):
    p = sub.add_parser("tokens", help="design_spec.json -> Compose/SwiftUI/CSS/Tailwind theme code (palette snapped to pixels)")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/tokens")
    p.add_argument("--package", default="com.example.app.ui.theme", help="Kotlin package for the Compose files")
    p.add_argument("--r-package", help="package of the app's R class (default: --package minus .ui.theme)")
    p.add_argument("--author", help="adds an @author/@since header to Kotlin files"); p.add_argument("--prefix", default="App")
    p.add_argument("--platforms", default="compose,swiftui,web"); p.add_argument("--fonts", help="fonts.json from `fonts`")
    p.add_argument("--no-snap", action="store_true", help="keep the spec's hex guesses")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.add_argument("--verify", action="store_true", help="typecheck Swift, load Tailwind config in node, parse CSS in Chrome")
    p.set_defaults(fn=cmd_tokens)
    p = sub.add_parser("fonts", help="match typography to Google Fonts by rendering + measuring; download winners")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/fonts")
    p.add_argument("--candidates", type=int, default=24, help="similar families per group besides the guessed ones")
    p.add_argument("--weights", type=int, default=3, help="weights tried per family and style")
    p.add_argument("--extra", help="comma-separated Google Fonts families to add to every shortlist")
    p.add_argument("--family", help="force this family for its group when it was a candidate")
    p.add_argument("--tie", type=float, default=0.02, help="keep the spec's named family if it scores within this of the best")
    p.add_argument("--show", type=int, default=4, help="candidates per row on the review sheet")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.set_defaults(fn=cmd_fonts)
    p = sub.add_parser("compare", help="implemented-screen screenshot vs approved mockup: heatmap, score, ranked fixes")
    p.add_argument("dir", help="build dir with assets.json / design_spec.json"); p.add_argument("screenshot")
    p.add_argument("--mockup", help="default: the dir's mockup"); p.add_argument("--out", help="default DIR/compare")
    p.add_argument("--top", type=int, default=10); p.add_argument("--vision", action="store_true",
                                                                   help="one gpt vision call describing the top regions")
    p.add_argument("--crop-top", type=int, default=0, help="screenshot px to drop at the top (status bar)")
    p.add_argument("--crop-bottom", type=int, default=0, help="screenshot px to drop at the bottom (nav bar)")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("redline", help="pixel-true geometry: refined edges, corner radii, insets, gaps (dp) -> DIR/redline")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/redline"); p.set_defaults(fn=cmd_redline)


def cmd_doctor(a):
    c = _config()
    print(f"config file : {c['config_path']} ({'found' if Path(c['config_path']).exists() else 'not found'})")
    print(f"base url    : {c['base']}")
    print(f"api key     : {'set' if c['key'] else 'MISSING'}")
    print(f"image model : {c['image_model']}")
    print(f"text model  : {c['text_model']}")
    renderer = _renderer_name()
    print(f"svg render  : {renderer or 'none (install Chrome/Chromium or pip install cairosvg)'}"
          f"{' (no alpha: matted from two renders)' if renderer == 'qlmanage' else ''}")
    sdk = _android_sdk()
    print(f"aapt2       : {sdk[0] if sdk else 'not found (set ANDROID_HOME to validate VectorDrawables)'}")
    if not c["key"]:
        sys.exit(1)
    base, headers = _auth()
    try:
        r = requests.get(base + "/models", headers=headers, timeout=30)
        r.raise_for_status()
        ids = {m["id"] for m in r.json().get("data", [])}
    except Exception as e:
        sys.exit(f"models      : request failed ({e})")
    print(f"models      : {len(ids)} available")
    for label, mid in (("image", c["image_model"]), ("text", c["text_model"])):
        print(f"  {label:5s} {mid}: {'ok' if mid in ids else 'NOT LISTED (set GPTKIT_' + label.upper() + '_MODEL)'}")
    images = sorted(i for i in ids if "image" in i)
    if images:
        print("  image models on this endpoint:", ", ".join(images))


def cmd_score(a):
    d, m, mock = _load(a.dir)
    asset = next(x for x in m["assets"] if x["name"] == a.name)
    for f in a.files:
        if f.endswith(".svg"):
            r = render_svgs([Path(f).read_text()], 256)[0]
            sc = svg_fidelity(mock, asset["bbox"], r) if r else "not renderable"
            print(f, sc, *(f"\n  ! {i}" for i in (_svg_issues(sc) if r else [])))
        else:
            print(f, score(mock, asset["bbox"], _rgba(f), None, asset["transparent"]))


def _color_args(p):
    p.add_argument("--color-match", type=float, default=0.75, metavar="STRENGTH",
                   help="pull the final file's colours toward the design's (0-1, default 0.75); kept only if it scores better")
    p.add_argument("--no-color-match", dest="color_match", action="store_const", const=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("doctor", help="check config, API access, models and optional tools")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("mockup", help="generate UI mockup variants (or --edit an existing one)")
    p.add_argument("prompt"); p.add_argument("--out", required=True); p.add_argument("--name", default="mockup")
    p.add_argument("--size", default="1024x1536"); p.add_argument("--variants", type=int, default=1)
    p.add_argument("--edit", help="existing mockup PNG to revise with the prompt"); p.add_argument("--raw", action="store_true")
    p.add_argument("--style-ref", help="approved screen PNG: design a NEW screen in its exact design language")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform: native components, system UI, metrics")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_mockup)
    p = sub.add_parser("critique", help="vision design QA of a mockup; --apply edits the fixes in, kept only if better")
    p.add_argument("mockup"); p.add_argument("--out", help="output dir (default: the mockup's dir)")
    p.add_argument("--style-ref", help="approved screen of the same app: also check consistency with it")
    p.add_argument("--platform", choices=list(PLATFORMS), help="default: design.json next to the mockup, else android")
    p.add_argument("--screen-dp", type=float); p.add_argument("--note", help="extra focus for the review")
    p.add_argument("--apply", action="store_true", help="edit the issues' fixes into the mockup, verify, keep if better")
    p.add_argument("--min-severity", choices=["high", "medium", "low"], default="medium")
    p.add_argument("--rounds", type=int, default=1); p.add_argument("--model", default=None)
    p.set_defaults(fn=cmd_critique)
    p = sub.add_parser("manifest", help="inventory assets of an approved mockup -> DIR/assets.json")
    p.add_argument("mockup"); p.add_argument("--out", required=True)
    p.add_argument("--platform", choices=list(PLATFORMS), help="default: design.json written by mockup")
    p.add_argument("--screen-dp", type=float, help="logical screen width (default: the platform's, else 390)")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("spec", help="extract palette/typography/spacing/native widgets -> DIR/design_spec.json")
    p.add_argument("dir")
    p.add_argument("--scale", choices=["auto", "native", "design"], default="auto",
                   help="px->dp from native widget sizes (auto: when GPT drew them >12%% off) or the screen width")
    p.add_argument("--recalibrate", action="store_true", help="only redo the px->dp scale of an existing spec (no API)")
    p.set_defaults(fn=cmd_spec)
    p = sub.add_parser("forge", help="redraw png assets from scratch, 1:1, with candidates + scoring")
    p.add_argument("dir"); p.add_argument("--only"); p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--jobs", type=int, default=6); p.add_argument("--force", action="store_true")
    p.add_argument("--include-svg", action="store_true", help="also raster-forge svg-format assets")
    p.add_argument("--same-prompt", action="store_true", help="no per-candidate colour/shape emphasis")
    _color_args(p); p.add_argument("--model", default=None); p.set_defaults(fn=cmd_forge)
    p = sub.add_parser("fix", help="vision-diff the chosen asset against the design and apply one correction")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--note", help="extra user feedback to include"); p.add_argument("--accept", action="store_true",
                                                                                   help="keep the result even if the score is worse")
    _color_args(p); p.add_argument("--model", default=None); p.set_defaults(fn=cmd_fix)
    p = sub.add_parser("svg", help="write svg-format assets as SVG: scored vs the design, retried, 24-grid, previews")
    p.add_argument("dir"); p.add_argument("--only")
    p.add_argument("--no-retry", dest="retry", action="store_false", help="skip the vision-guided retry of weak icons")
    p.add_argument("--grid", type=float, default=24, help="normalise icon viewBoxes to this grid (0 = keep as drawn)")
    p.add_argument("--pad", type=float, default=2, help="grid padding around the icon's content")
    p.add_argument("--check", action="store_true", help="only score existing DIR/svg files (no API calls)")
    p.set_defaults(fn=cmd_svg)
    p = sub.add_parser("iconspec", help="extract the icon-set style spec + Material/SF symbol names -> DIR/icon_spec.json")
    p.add_argument("dir"); p.set_defaults(fn=cmd_iconspec)
    p = sub.add_parser("icons", help="draw NEW icons (not in the mockup) in the design's icon style as SVG")
    p.add_argument("dir"); p.add_argument("concepts", help='comma-separated, e.g. "notifications, sleep timer, download"')
    p.add_argument("--out", help="default DIR/icons")
    p.add_argument("--treatment", choices=["auto", "outlined", "filled"], default="auto")
    p.add_argument("--stroke", type=float, help="stroke width on the 24 grid (default: measured from the set)")
    p.add_argument("--color", help="hex colour (default: the set's most common)")
    p.add_argument("--snap", action="store_true", help="force every stroke-width to the set's stroke")
    p.add_argument("--vd", help="also write VectorDrawables to this res/drawable dir"); p.add_argument("--dp", type=float)
    p.set_defaults(fn=cmd_icons)
    p = sub.add_parser("vd", help="convert SVG files to Android VectorDrawable XML")
    p.add_argument("svgs", nargs="+"); p.add_argument("--out", required=True); p.add_argument("--dp", type=float)
    p.add_argument("--prefix", default="ic_")
    p.add_argument("--no-check", action="store_true", help="skip the render round-trip and aapt2 validation")
    p.set_defaults(fn=cmd_vd)
    p = sub.add_parser("frames", help="frame-by-frame animation from a forged asset (moving layer sprite sheet, "
                                      "or --procedural local motion)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving", help="part that moves (generated route)")
    p.add_argument("--static", help="part that stays still (generated once, composited)")
    p.add_argument("--static-order", choices=["front", "back"], default="front")
    p.add_argument("--motion"); p.add_argument("--count", type=int, help="frames (default 8; procedural: one loop)")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--anchor", choices=["bottom", "center"], default="bottom")
    p.add_argument("--fps", type=float, help="default 12 (procedural 24, or 12 for slow loops)")
    p.add_argument("--procedural", choices=sorted(MOTIONS), help="no API calls: eased affine motion of the asset")
    p.add_argument("--amount", type=float, default=1.0, help="procedural motion strength multiplier")
    p.add_argument("--reuse", choices=["all", "static"], help="reuse sheet_raw/static_raw.png from a previous run")
    p.add_argument("--reroll", action="store_true", help="redraw QA-flagged frames from a good neighbour (1 call each)")
    p.add_argument("--match-colors", action="store_true", help="shift frame colours onto the approved asset (Lab)")
    p.add_argument("--formats", default="gif,webp,apng,lottie", help="comma list of gif,webp,apng,lottie")
    p.add_argument("--px", type=int, default=480, help="long side of the animated WebP/APNG/Lottie/GIF")
    p.add_argument("--preview-bg", default="#1b1d33"); p.add_argument("--model", default=None); p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("vectorize", help="vector layers (morphs, transforms, glows) -> Lottie + AnimatedVectorDrawable")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving"); p.add_argument("--static")
    p.add_argument("--motion"); p.add_argument("--keyframes", type=int, default=4); p.add_argument("--size", type=int, default=512)
    p.add_argument("--seconds", type=float, default=1.2); p.add_argument("--fps", type=int, default=30)
    p.add_argument("--strength", choices=list(MOTION_STRENGTH), default="normal", help="how much the moving parts move")
    p.add_argument("--min-motion", type=float, help="retry once below this share of a layer's size (default per --strength)")
    p.add_argument("--svg", help="animate this SVG (default DIR/svg/NAME.svg when DIR/assets/NAME.png is missing)")
    p.add_argument("--dp", type=float, help="AnimatedVectorDrawable width in dp (default: the asset's dp, else 48)")
    p.add_argument("--rebuild", action="store_true", help="rebuild outputs from an edited layers.json, no API call")
    p.add_argument("--no-fit", action="store_true", help="keep the viewBox even if the motion leaves it (clips)")
    p.add_argument("--review-frames", type=int, default=8)
    p.add_argument("--preview-bg", default="#1b1d33"); p.set_defaults(fn=cmd_vectorize)
    p = sub.add_parser("export", help="export a PNG to android densities / ios imageset / web")
    p.add_argument("file"); p.add_argument("--dp", type=float, required=True, help="width in dp/pt/css px")
    p.add_argument("--hdp", type=float); p.add_argument("--name"); p.add_argument("--res", required=True)
    p.add_argument("--platform", choices=["android", "ios", "web"], default="android")
    p.add_argument("--format", choices=["webp", "png"], default="webp"); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("appicon", help="app launcher icons: concepts, then Android adaptive / iOS / web icon sets")
    isub = p.add_subparsers(dest="icmd", required=True)
    q = isub.add_parser("concepts", help="square full-bleed icon concepts + mask review sheet")
    q.add_argument("brief", help='brand brief, e.g. "Hush, a calm sleep app, moon motif"'); q.add_argument("--out", required=True)
    q.add_argument("--mockup", help="approved design to take the visual language from (sent as the single reference)")
    q.add_argument("--crop", help="x,y,w,h region of --mockup to use as the reference instead of the whole screen")
    q.add_argument("--variants", type=int, default=3); q.add_argument("--directions", help="'|'-separated concept directions")
    q.add_argument("--text", help="wordmark/letter to include (default: no text)"); q.add_argument("--colors")
    q.add_argument("--name", default="concept"); q.add_argument("--jobs", type=int, default=2)
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
    q.add_argument("--fg-scale", type=float, default=1.0, help="fraction of the 66dp safe circle the subject may fill")
    q.add_argument("--ios-from", choices=["concept", "composite"], default="concept")
    q.add_argument("--name", default="ic_launcher"); q.add_argument("--no-validate", dest="validate", action="store_false")
    q.add_argument("--model", default=None); q.set_defaults(fn=cmd_appicon_build)
    p = sub.add_parser("pick", help="override the chosen candidate (number from the review sheet, or a file path)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("candidate"); _color_args(p); p.set_defaults(fn=cmd_pick)
    _design_to_code_parsers(sub)
    p = sub.add_parser("score", help="score arbitrary PNG or SVG files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    _add_trace_rig_parsers(sub)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
