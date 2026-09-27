#!/usr/bin/env python3
"""gptkit: design-to-assets pipeline on any OpenAI-compatible API — UI mockups, asset inventory, 1:1 from-scratch
redraws, SVG icons, Android VectorDrawables, animation frames and Lottie. Run `gptkit.py -h` for commands.

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
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

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
    im = img.convert("RGBA").copy()
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


def score(mock, bbox, asset, ref=None, transparent=True):
    """Fidelity vs the mockup element plus copy/crop red flags. Lower 'diff' = closer to the design."""
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
    return {"diff": round(diff, 1), "flags": flags, "warn": warn, "sharp_ratio": round(sharp, 1),
            "transparent_frac": round(transp, 3), "aspect_err": round(abs((g.width / g.height) / (w / h) - 1), 3)}


def _rank(sc, meta):
    return (len(sc["flags"]), sc["diff"] + (6 if (meta or {}).get("quality") == "low" else 0))


# ---------------------------------------------------------------- prompts

MOCKUP_WRAP = ("High-fidelity production UI design render of a single app screen on a {aspect} canvas. {prompt} "
               "Flat, pixel-crisp screen render at 1:1, no device frame, no hands, no perspective, no watermark. "
               "Real, legible UI text. Consistent spacing grid and one cohesive icon style.")

ASSET_PROMPT = ("The attached image is a zoomed, low-resolution screenshot region from an APPROVED app design. "
                "Recreate ONLY the {kind} \"{name}\" ({description}) from scratch as a clean, high-resolution "
                "standalone production asset that matches the design 1:1: identical silhouette, proportions, "
                "number of layers/parts, colour values{colors}, gradient directions, highlights and glow. "
                "Do NOT upscale, trace or copy the screenshot pixels. Do NOT include the backdrop, container shape, "
                "text, or any neighbouring element. {bg} Centered with about 10% empty padding on every side.")

BACKGROUND_PROMPT = ("The attached image is an APPROVED app UI design ({W}x{H}px). Recreate ONLY the {kind} \"{name}\" "
                     "({description}) that occupies the region x={x}, y={y}, w={w}, h={h} as a standalone full-bleed "
                     "image, from scratch at full resolution. REMOVE every UI element drawn on top of it: all text, "
                     "buttons, chips, icons, badges, rounded corners and borders. Where UI covered it, continue the "
                     "artwork naturally. Keep the exact composition, framing, colours and lighting of the design.")

MANIFEST_PROMPT = """This is an APPROVED mobile/web UI design ({W}x{H} px). Inventory every visual ASSET a developer
needs as a separate file to build it in code: icons, illustrations, images/photos, logos, decorative graphics, and
artwork that sits behind UI (kind "background"). Rules:
- Exclude plain text, solid/gradient fills, simple shapes, dividers, shadows and standard containers code can draw.
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

VECTOR_PROMPT = """Vectorise this asset from an APPROVED design for a looping Lottie animation. The MOVING part is:
{moving}. The STATIC part is: {static}. Rules:
- Every path "d" uses ONLY absolute M, L, C and Z commands (no H/V/S/Q/A, no relative commands).
- For each animated layer, all {k} keyframes must have the IDENTICAL command sequence (same count and order of
  M/L/C/Z) so they can be morphed; together they form a seamless, natural loop ({motion}).
- fill is a hex colour or url(#id) referencing a gradient defined in "defs" (gradientUnits="userSpaceOnUse").
- Match the design's silhouette, proportions and colours; layer_order lists layer names bottom to top."""

VECTOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["viewBox", "defs", "static_layers", "animated_layers", "layer_order"],
    "properties": {
        "viewBox": {"type": "string"}, "defs": {"type": "string"},
        "layer_order": {"type": "array", "items": {"type": "string"}},
        "static_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "paths"],
            "properties": {"name": {"type": "string"}, "paths": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["d", "fill", "opacity"],
                "properties": {"d": {"type": "string"}, "fill": {"type": "string"}, "opacity": {"type": "number"}}}}}}},
        "animated_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "fill", "opacity", "keyframes"],
            "properties": {"name": {"type": "string"}, "fill": {"type": "string"}, "opacity": {"type": "number"},
                           "keyframes": {"type": "array", "items": {"type": "string"}}}}},
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
                                                  "required": ["role", "sample", "family_guess", "size_px", "weight", "letter_spacing_em", "color"],
                                                  "properties": {"role": {"type": "string"}, "sample": {"type": "string"},
                                                                 "family_guess": {"type": "string"}, "size_px": {"type": "number"},
                                                                 "weight": {"type": "integer"}, "letter_spacing_em": {"type": "number"},
                                                                 "color": {"type": "string"}}}},
        "radii": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["element", "px"],
                                             "properties": {"element": {"type": "string"}, "px": {"type": "number"}}}},
        "spacing": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["between", "px"],
                                               "properties": {"between": {"type": "string"}, "px": {"type": "number"}}}},
        "components": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["name", "bbox", "notes"],
                                                  "properties": {"name": {"type": "string"}, "notes": {"type": "string"},
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


def cmd_mockup(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    aspect = ASPECTS.get(a.size, a.size)
    prompt = a.prompt if a.raw else MOCKUP_WRAP.format(prompt=a.prompt, aspect=aspect)
    edit_prompt = f"{a.prompt} Keep the {aspect} canvas and everything not mentioned exactly as it is."

    def one(i):
        # the backend picks aspect from the wording, not from `size`, so verify and retry once
        for _ in range(2):
            if a.edit:
                img, meta = edit(Path(a.edit).read_bytes(), edit_prompt, a.size, model=a.model)
            else:
                img, meta = generate(prompt, a.size, model=a.model)
            if _aspect_ok(img, a.size):
                break
        p = out / f"{a.name}_v{i + 1}.png"
        p.write_bytes(img)
        return p, meta
    with cf.ThreadPoolExecutor(a.variants) as ex:
        for p, meta in ex.map(one, range(a.variants)):
            print(f"{p}  quality={meta['quality']} size={meta['size']}")


def cmd_manifest(a):
    mock_path = Path(a.mockup).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mock = _rgba(mock_path)
    local = out / "mockup.png"
    if mock_path != local.resolve():
        mock.save(local)
    res = ask([MANIFEST_PROMPT.format(W=mock.width, H=mock.height), mock], MANIFEST_SCHEMA, "asset_manifest")
    seen = set()
    for x in res["assets"]:
        n = re.sub(r"_(svg|png|icon|asset)$", "", re.sub(r"[^a-z0-9_]", "_", x["name"].lower()).strip("_"))
        while n in seen:
            n += "_2"
        seen.add(n)
        x["name"] = n
        x["material_symbol"], x["sf_symbol"] = _symbol_names(x.get("material_symbol", ""), x.get("sf_symbol", ""))
        x["dp"] = {"w": round(x["bbox"]["w"] * a.screen_dp / mock.width, 1),
                   "h": round(x["bbox"]["h"] * a.screen_dp / mock.width, 1)}
    manifest = {"mockup": "mockup.png", "width": mock.width, "height": mock.height,
                "screen_dp": a.screen_dp, "assets": res["assets"]}
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
    d, m, mock = _load(a.dir)
    res = ask([f"This is an APPROVED UI design ({mock.width}x{mock.height}px; the screen is {m['screen_dp']}dp wide). "
               "Extract the design spec a developer needs to rebuild it 1:1: palette (hex), every text style "
               "(role, a sample string, likely font family, size in px of THIS image, weight, letter spacing, colour), "
               "corner radii, key spacings, and the component list with pixel bboxes.", mock], SPEC_SCHEMA, "design_spec")
    res["px_to_dp"] = round(m["screen_dp"] / mock.width, 5)
    (d / "design_spec.json").write_text(json.dumps(res, indent=2))
    print(f"-> {d / 'design_spec.json'}  (multiply px values by px_to_dp={res['px_to_dp']} for dp/pt)")


def _forge_job(mock, asset, cand, model):
    b = asset["bbox"]
    colors = f" ({', '.join(asset['colors'])})" if asset.get("colors") else ""
    if asset["kind"] in ("background", "photo") and not asset["transparent"]:
        prompt = BACKGROUND_PROMPT.format(W=mock.width, H=mock.height, kind=asset["kind"], name=asset["name"],
                                          description=asset["description"], **b)
        ref = mock.convert("RGB")
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]), "opaque", model)
    else:
        ref = ref_crop(mock, b)
        bg = "Transparent background (real alpha)." if asset["transparent"] else "Opaque background matching the design."
        prompt = ASSET_PROMPT.format(kind=asset["kind"], name=asset["name"].replace("_", " "),
                                     description=asset["description"], colors=colors, bg=bg)
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", model)
    return asset["name"], cand, img, meta, ref


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
        futs = [ex.submit(_forge_job, mock, x, c, a.model) for x, c in jobs]
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
            print(f"  {name} c{cand + 1}: diff={sc['diff']} flags={sc['flags'] or '-'} warn={sc['warn'] or '-'} "
                  f"quality={meta['quality']}")
    rows = []
    for name, cands in results.items():
        asset = by_name[name]
        if not cands:
            continue
        cands.sort(key=lambda c: _rank(c["score"], c["meta"]))
        best = cands[0]
        final = _finalize(asset, _rgba(best["path"]))
        out = d / "assets" / f"{name}.png"
        final.save(out)
        report[name] = {"chosen": best, "candidates": cands, "final": str(out), "final_size": final.size,
                        "dp": asset.get("dp"), "needs_attention": bool(best["score"]["flags"])}
        b = asset["bbox"]
        pad = int(max(b["w"], b["h"]) * 0.25)
        target = mock.crop((max(0, b["x"] - pad), max(0, b["y"] - pad), b["x"] + b["w"] + pad, b["y"] + b["h"] + pad))
        rows.append([(f"{name} (design)", target)] +
                    [(f"{'BEST ' if c is best else ''}d={c['score']['diff']} {' '.join(c['score']['flags'])}", _rgba(c["path"]))
                     for c in cands])
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
    cur_path = Path(report[a.name]["chosen"]["path"])
    cur = _rgba(cur_path)
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
        better = _rank(sc, meta) < _rank(cur_score, None)
        print(f"round {rnd + 1}: diff {cur_score['diff']} -> {sc['diff']} flags={sc['flags'] or '-'} -> "
              f"{'KEPT' if better or a.accept else 'discarded (worse)'}  ({p})")
        if better or a.accept:
            cur, cur_score = new, sc
            report[a.name]["chosen"] = {"path": str(p), "meta": {"quality": meta["quality"], "size": meta["size"]}, "score": sc}
            _finalize(asset, new).save(d / "assets" / f"{a.name}.png")
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
def _chrome():
    names = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge")
    cands = [os.environ.get("GPTKIT_CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium",
             "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"] + [shutil.which(n) for n in names]
    return next((c for c in cands if c and Path(c).exists()), None)


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


def _lab(rgb):
    c = np.asarray(rgb, dtype=float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


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
    return None


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
             max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) >= size / 1.25]
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

def _longest_run(values):
    best = cur = 0
    for v in values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _remove_dividers(img):
    """Clear drawn grid lines: thin lines that run unbroken across the whole sheet."""
    a = img.getchannel("A")
    W, H = img.size
    px = img.load()
    for x in range(W):
        if _longest_run(a.getpixel((x, y)) > 180 for y in range(H)) >= 0.97 * H:
            for y in range(H):
                px[x, y] = (0, 0, 0, 0)
    for y in range(H):
        if _longest_run(a.getpixel((x, y)) > 180 for x in range(W)) >= 0.97 * W:
            for x in range(W):
                px[x, y] = (0, 0, 0, 0)
    return img


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
    fcrop = frame.crop(fb)
    ys, xs = np.nonzero(visible)
    if not len(xs):
        return None
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


def cmd_frames(a):
    d, m, mock = _load(a.dir)
    src = _rgba(d / "assets" / f"{a.name}.png")
    out = d / "frames" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rows_n = math.ceil(a.count / a.cols)
    motion = a.motion or "a gentle, natural looping motion"
    moving_prompt = (f"The attached image is an APPROVED asset. Produce ONLY its moving part — {a.moving} — "
                     f"{'with NO ' + a.static + ' at all' if a.static else ''}, as {a.count} animation frames of {motion}, "
                     f"laid out as a {a.cols} x {rows_n} sprite sheet of equal cells (left-to-right, top-to-bottom). "
                     "Every frame: same style, colours, gradients and glow as the attached asset; same overall size and "
                     "the SAME anchor point at the bottom centre of its cell; only the natural motion changes, and the "
                     "last frame flows back into the first. Transparent background everywhere. Absolutely no grid lines, "
                     "dividers, borders, numbers or labels. Leave empty transparent space between cells.")
    jobs = {"sheet": (moving_prompt, "1536x1024" if a.cols >= rows_n else "1024x1536")}
    if a.static:
        jobs["static"] = (f"The attached image is an APPROVED asset. Keep ONLY {a.static}, exactly as drawn: same size, "
                          f"position, angle and colours on the same canvas. Remove {a.moving} completely and repair "
                          "what it covered. Transparent background.", "1024x1024")
    with cf.ThreadPoolExecutor(2) as ex:
        sq_src, sq_off = _square(src)
        futs = {k: ex.submit(edit, _png_bytes(sq_src if k == "static" else src), p, sz, "transparent", a.model)
                for k, (p, sz) in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    (out / "sheet_raw.png").write_bytes(res["sheet"][0])
    sheet_img = _remove_dividers(_rgba(out / "sheet_raw.png"))
    W, H = sheet_img.size
    cw, ch = W / a.cols, H / rows_n
    crops = []
    for i in range(a.count):
        c, r = i % a.cols, i // a.cols
        cell = sheet_img.crop((round(c * cw), round(r * ch), round((c + 1) * cw), round((r + 1) * ch)))
        bb = content_box(cell, 24)
        if not bb:
            print(f"  ! frame {i + 1} empty")
            continue
        if bb[0] <= 1 or bb[1] <= 1 or bb[2] >= cell.width - 1 or bb[3] >= cell.height - 1:
            print(f"  ! frame {i + 1} touches its cell edge (may be cut off)")
        crops.append(cell.crop(bb))
    fw = max(c.width for c in crops)
    fh = max(c.height for c in crops)
    pad = int(max(fw, fh) * 0.06)
    FW, FH = fw + 2 * pad, fh + 2 * pad
    frames = []
    for i, c in enumerate(crops):
        f = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
        x = (FW - c.width) // 2
        y = FH - pad - c.height if a.anchor == "bottom" else (FH - c.height) // 2
        f.alpha_composite(c, (x, y))
        f.save(out / f"moving_f{i + 1:02d}.png")
        frames.append(f)
    comps = []
    if a.static:
        (out / "static_raw.png").write_bytes(res["static"][0])
        sq_static = _register(_rgba(out / "static_raw.png").resize(sq_src.size, Image.LANCZOS), sq_src)
        static = sq_static.crop((sq_off[0], sq_off[1], sq_off[0] + src.width, sq_off[1] + src.height))
        static.save(out / "static.png")
        mid = frames[len(frames) // 2]
        fit = _fit_moving(src, static, mid)
        if not fit:
            print("  ! could not locate the moving part in the asset; composites skipped")
        else:
            s, cx, by, _ = fit
            P = max(src.size)
            size = (src.width + 2 * P, src.height + 2 * P)
            st = Image.new("RGBA", size, (0, 0, 0, 0))
            st.alpha_composite(static, (P, P))
            mid_bottom = content_box(mid, 100)[3]
            for f in frames:
                # all frames share one scale and the anchor found for the middle frame
                fc = f.resize((max(1, round(f.width * s)), max(1, round(f.height * s))), Image.LANCZOS)
                layer = Image.new("RGBA", size, (0, 0, 0, 0))
                layer.alpha_composite(fc, (P + round(cx - fc.width / 2), P + round(by - mid_bottom * s)))
                comp = Image.new("RGBA", size, (0, 0, 0, 0))
                for lyr in ((layer, st) if a.static_order == "front" else (st, layer)):
                    comp.alpha_composite(lyr)
                comps.append(comp)
            # one shared crop box keeps every frame registered
            boxes = [content_box(c, 8) for c in comps]
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
            comps = [c.crop(box) for c in comps]
            for i, c in enumerate(comps):
                c.save(out / f"frame_{i + 1:02d}.png")
    seq = comps or frames
    bg = Image.new("RGBA", seq[0].size, a.preview_bg)
    gif = [Image.alpha_composite(bg, f.resize(seq[0].size)).convert("P", palette=Image.ADAPTIVE) for f in seq]
    gif[0].save(out / "preview.gif", save_all=True, append_images=gif[1:], duration=int(1000 / a.fps), loop=0, disposal=2)
    dur = int(1000 / a.fps)
    base = "frame" if comps else "moving_f"
    items = "\n".join(f'    <item android:drawable="@drawable/{a.name}_{base.strip("_")}_{i + 1:02d}" android:duration="{dur}" />'
                      for i in range(len(seq)))
    (out / f"{a.name}_anim.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<animation-list xmlns:android="http://schemas.android.com/apk/res/android"\n'
        f'    android:oneshot="false">\n{items}\n</animation-list>\n')
    sheet([[(f"f{i + 1}", f) for i, f in enumerate(seq)][j:j + 8] for j in range(0, len(seq), 8)],
          out / "frames_review.png", T=200)
    print(f"{len(seq)} frames -> {out}  (preview.gif, frames_review.png, {a.name}_anim.xml)")


# ---- vector keyframes + Lottie

_TOK = re.compile(r"[MLCZmlczHhVvSsQqTtAa]|-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?")


def _subpaths(d):
    toks = _TOK.findall(d)
    bad = {t for t in toks if t.isalpha() and t not in "MLCZ"}
    if bad:
        raise ValueError(f"unsupported path commands {sorted(bad)}")
    subs, cur, i, cmd = [], None, 0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd == "Z":
                if cur:
                    cur["closed"] = True
                continue
        if cmd == "M":
            cur = {"pts": [(float(toks[i]), float(toks[i + 1]))], "segs": [], "closed": False}
            subs.append(cur)
            i += 2
            cmd = "L"
        elif cmd == "L":
            cur["segs"].append(("L", (float(toks[i]), float(toks[i + 1]))))
            i += 2
        elif cmd == "C":
            v = [float(x) for x in toks[i:i + 6]]
            cur["segs"].append(("C", (v[0], v[1]), (v[2], v[3]), (v[4], v[5])))
            i += 6
    return subs


def _lottie_shape(sp):
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
    if sp["closed"] and len(v) > 1 and abs(v[-1][0] - v[0][0]) < 1e-3 and abs(v[-1][1] - v[0][1]) < 1e-3:
        ins[0] = ins[-1]
        v, ins, outs = v[:-1], ins[:-1], outs[:-1]
    return {"i": ins, "o": outs, "v": [list(p) for p in v], "c": sp["closed"]}


def _hex_rgb(h):
    h = h.lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h[:6]
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _lottie_fill(fill, opacity, defs_root):
    m_ = re.match(r"url\(#([^)]+)\)", fill or "")
    if not m_:
        return {"ty": "fl", "c": {"a": 0, "k": _hex_rgb(fill) + [1]}, "o": {"a": 0, "k": opacity * 100}, "r": 1}
    gel = next((e for e in defs_root.iter() if e.get("id") == m_.group(1)), None)
    if gel is None:
        raise ValueError(f"gradient {m_.group(1)} not found in defs")
    stops = []
    for s in gel:
        if s.tag.replace(SVG_NS, "") != "stop":
            continue
        off = s.get("offset", "0")
        off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
        stops.append((off, _hex_rgb(s.get("stop-color", "#000")), _num(s.get("stop-opacity"), 1.0)))
    colors = [x for o, c, _ in stops for x in [o] + c]
    alphas = [x for o, _, al in stops for x in (o, al)] if any(al < 1 for *_, al in stops) else []
    linear = gel.tag.endswith("linearGradient")
    s = [_num(gel.get("x1")), _num(gel.get("y1"))] if linear else [_num(gel.get("cx")), _num(gel.get("cy"))]
    e = [_num(gel.get("x2")), _num(gel.get("y2"))] if linear else [s[0] + _num(gel.get("r")), s[1]]
    return {"ty": "gf", "t": 1 if linear else 2, "s": {"a": 0, "k": s}, "e": {"a": 0, "k": e},
            "g": {"p": len(stops), "k": {"a": 0, "k": colors + alphas}}, "o": {"a": 0, "k": opacity * 100}, "r": 1}


def build_lottie(vec, size=512, seconds=1.2, fps=30):
    vb = [float(x) for x in re.split(r"[\s,]+", vec["viewBox"].strip())]
    scale = size / max(vb[2], vb[3]) * 100
    op = round(seconds * fps)
    defs_root = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg"><defs>{vec.get("defs", "")}</defs></svg>'
                              if "<defs" not in vec.get("defs", "") else f'<svg xmlns="http://www.w3.org/2000/svg">{vec["defs"]}</svg>')
    named = {}
    for sl in vec["static_layers"]:
        items = []
        for p in sl["paths"]:
            shapes = [{"ty": "sh", "ks": {"a": 0, "k": _lottie_shape(sp)}} for sp in _subpaths(p["d"])]
            items.append({"ty": "gr", "it": shapes + [_lottie_fill(p["fill"], p.get("opacity", 1), defs_root), _tr()]})
        # Lottie draws earlier shapes on top; SVG paths are listed bottom-to-top
        named[sl["name"]] = items[::-1]
    for al in vec["animated_layers"]:
        kfs = [[_lottie_shape(sp) for sp in _subpaths(k)] for k in al["keyframes"]]
        n = len(kfs)
        shapes = []
        for si in range(len(kfs[0])):
            ks = [{"t": round(op * j / n), "s": [kfs[j][si]], "i": {"x": [0.45], "y": [1]}, "o": {"x": [0.55], "y": [0]}}
                  for j in range(n)] + [{"t": op, "s": [kfs[0][si]]}]
            shapes.append({"ty": "sh", "ks": {"a": 1, "k": ks}})
        named[al["name"]] = [{"ty": "gr", "it": shapes + [_lottie_fill(al["fill"], al.get("opacity", 1), defs_root), _tr()]}]
    order = [n for n in vec.get("layer_order", []) if n in named] + [n for n in named if n not in vec.get("layer_order", [])]
    layers = []
    for idx, name in enumerate(reversed(order)):
        layers.append({"ddd": 0, "ind": idx + 1, "ty": 4, "nm": name, "sr": 1, "ip": 0, "op": op, "st": 0, "bm": 0,
                       "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0},
                              "p": {"a": 0, "k": [-vb[0] * scale / 100, -vb[1] * scale / 100, 0]},
                              "a": {"a": 0, "k": [0, 0, 0]}, "s": {"a": 0, "k": [scale, scale, 100]}},
                       "shapes": named[name]})
    w = round(vb[2] * scale / 100)
    h = round(vb[3] * scale / 100)
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": op, "w": w, "h": h, "nm": "gptkit", "ddd": 0, "assets": [], "layers": layers}


def _tr():
    return {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]},
            "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}, "sk": {"a": 0, "k": 0}, "sa": {"a": 0, "k": 0}}


def _validate_vec(vec):
    errs = []
    for sl in vec["static_layers"]:
        for p in sl["paths"]:
            try:
                _subpaths(p["d"])
            except ValueError as e:
                errs.append(f"static {sl['name']}: {e}")
    for al in vec["animated_layers"]:
        try:
            sigs = {"".join(t for t in _TOK.findall(k) if t.isalpha()) for k in al["keyframes"]}
            [_subpaths(k) for k in al["keyframes"]]
            if len(sigs) != 1:
                errs.append(f"animated {al['name']}: keyframes differ in command sequence")
        except ValueError as e:
            errs.append(f"animated {al['name']}: {e}")
    return errs


def _vec_svg(vec, k, size=256, bg=None):
    defs = vec.get("defs", "")
    defs = defs if "<defs" in defs else f"<defs>{defs}</defs>"
    body = {}
    for sl in vec["static_layers"]:
        body[sl["name"]] = "".join(f'<path d="{p["d"]}" fill="{p["fill"]}" opacity="{p.get("opacity", 1)}"/>' for p in sl["paths"])
    for al in vec["animated_layers"]:
        body[al["name"]] = f'<path d="{al["keyframes"][k]}" fill="{al["fill"]}" opacity="{al.get("opacity", 1)}"/>'
    order = [n for n in vec.get("layer_order", []) if n in body] + [n for n in body if n not in vec.get("layer_order", [])]
    rect = f'<rect x="-10000" y="-10000" width="20000" height="20000" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="{vec["viewBox"]}">'
            f'{defs}{rect}{"".join(body[n] for n in order)}</svg>')


def cmd_vectorize(a):
    d, m, mock = _load(a.dir)
    src_path = d / "assets" / f"{a.name}.png"
    src = _rgba(src_path)
    asset = next((x for x in m["assets"] if x["name"] == a.name), None)
    parts = [VECTOR_PROMPT.format(moving=a.moving, static=a.static or "nothing", k=a.keyframes,
                                  motion=a.motion or "a gentle natural loop"), src]
    if asset:
        parts += ["For reference, how it appears in the approved design:", ref_crop(mock, asset["bbox"], 0.3, 512)]
    vec, errs = None, ["not run"]
    for attempt in range(2):
        vec = ask(parts, VECTOR_SCHEMA, "vector_layers")
        errs = _validate_vec(vec)
        if not errs:
            break
        print("  retrying, validation failed:", *errs, sep="\n    ")
        parts = parts + [f"Your previous answer was invalid: {'; '.join(errs)}. Fix it and follow every rule."]
    out = d / "vector" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "layers.json").write_text(json.dumps(vec, indent=2))
    if errs:
        print("! still invalid:", errs)
        return
    rows = [("asset", src)]
    for k in range(len(vec["animated_layers"][0]["keyframes"]) if vec["animated_layers"] else 1):
        p = out / f"key_{k + 1}.svg"
        p.write_text(_vec_svg(vec, k, bg=a.preview_bg))
        prev = _render_svg(p, out)
        if prev:
            rows.append((f"key {k + 1}", _rgba(prev)))
    sheet([rows], out / "keys_review.png", T=220)
    lot = build_lottie(vec, a.size, a.seconds, a.fps)
    (out / f"{a.name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    (out / "preview.html").write_text(
        "<!doctype html><meta charset=utf-8><body style='margin:0;background:" + a.preview_bg + "'>"
        "<div id=a style='width:512px;height:512px'></div>"
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js'></script>"
        f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{json.dumps(lot)}}})</script>")
    print(f"-> {out}/layers.json, keys_review.png, {a.name}.json (Lottie), preview.html")


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


def cmd_pick(a):
    d, m, mock = _load(a.dir)
    asset = next(x for x in m["assets"] if x["name"] == a.name)
    report = json.loads((d / "forge_report.json").read_text())
    src = Path(a.candidate) if Path(a.candidate).exists() else d / "raw" / f"{a.name}_c{a.candidate}.png"
    img = _rgba(src)
    report[a.name]["chosen"] = {"path": str(src), "meta": {}, "score": score(mock, asset["bbox"], img, None, asset["transparent"])}
    report[a.name]["picked_manually"] = True
    _finalize(asset, img).save(d / "assets" / f"{a.name}.png")
    (d / "forge_report.json").write_text(json.dumps(report, indent=2))
    print(f"{a.name} <- {src}")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("doctor", help="check config, API access, models and optional tools")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("mockup", help="generate UI mockup variants (or --edit an existing one)")
    p.add_argument("prompt"); p.add_argument("--out", required=True); p.add_argument("--name", default="mockup")
    p.add_argument("--size", default="1024x1536"); p.add_argument("--variants", type=int, default=1)
    p.add_argument("--edit", help="existing mockup PNG to revise with the prompt"); p.add_argument("--raw", action="store_true")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_mockup)
    p = sub.add_parser("manifest", help="inventory assets of an approved mockup -> DIR/assets.json")
    p.add_argument("mockup"); p.add_argument("--out", required=True)
    p.add_argument("--screen-dp", type=float, default=390, help="logical screen width the mockup represents")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("spec", help="extract palette/typography/spacing -> DIR/design_spec.json")
    p.add_argument("dir"); p.set_defaults(fn=cmd_spec)
    p = sub.add_parser("forge", help="redraw png assets from scratch, 1:1, with candidates + scoring")
    p.add_argument("dir"); p.add_argument("--only"); p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--jobs", type=int, default=6); p.add_argument("--force", action="store_true")
    p.add_argument("--include-svg", action="store_true", help="also raster-forge svg-format assets")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_forge)
    p = sub.add_parser("fix", help="vision-diff the chosen asset against the design and apply one correction")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--note", help="extra user feedback to include"); p.add_argument("--accept", action="store_true",
                                                                                   help="keep the result even if the score is worse")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_fix)
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
    p = sub.add_parser("frames", help="frame-by-frame animation from a forged asset (moving layer sprite sheet)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving", required=True)
    p.add_argument("--static", help="part that stays still (generated once, composited)")
    p.add_argument("--static-order", choices=["front", "back"], default="front")
    p.add_argument("--motion"); p.add_argument("--count", type=int, default=8); p.add_argument("--cols", type=int, default=4)
    p.add_argument("--anchor", choices=["bottom", "center"], default="bottom"); p.add_argument("--fps", type=float, default=12)
    p.add_argument("--preview-bg", default="#1b1d33"); p.add_argument("--model", default=None); p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("vectorize", help="morphable vector keyframes -> Lottie JSON + preview")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving", required=True); p.add_argument("--static")
    p.add_argument("--motion"); p.add_argument("--keyframes", type=int, default=4); p.add_argument("--size", type=int, default=512)
    p.add_argument("--seconds", type=float, default=1.2); p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preview-bg", default="#1b1d33"); p.set_defaults(fn=cmd_vectorize)
    p = sub.add_parser("export", help="export a PNG to android densities / ios imageset / web")
    p.add_argument("file"); p.add_argument("--dp", type=float, required=True, help="width in dp/pt/css px")
    p.add_argument("--hdp", type=float); p.add_argument("--name"); p.add_argument("--res", required=True)
    p.add_argument("--platform", choices=["android", "ios", "web"], default="android")
    p.add_argument("--format", choices=["webp", "png"], default="webp"); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("pick", help="override the chosen candidate (number from the review sheet, or a file path)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("candidate"); p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("score", help="score arbitrary PNG or SVG files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
