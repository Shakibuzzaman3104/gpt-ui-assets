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
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

SVG_NS = "{http://www.w3.org/2000/svg}"


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
- colors: main hex colours sampled from the design."""

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

_NUM = {"type": "number"}
_MOTION = {"type": "object", "additionalProperties": False, "required": ["anchor_x", "anchor_y", "keys"],
           "properties": {"anchor_x": _NUM, "anchor_y": _NUM, "keys": {"type": "array", "items": {
               "type": "object", "additionalProperties": False,
               "required": ["t", "dx", "dy", "sx", "sy", "rotate", "opacity", "ease"],
               "properties": {**{k: _NUM for k in ("t", "dx", "dy", "sx", "sy", "rotate", "opacity")},
                              "ease": {"type": "string", "enum": list(EASES)}}}}}}
_PAINT = {"fill": {"type": "string"}, "opacity": _NUM, "stroke": {"type": "string"}, "stroke_width": _NUM}

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
            "properties": {"name": {"type": "string"}, **{k: _NUM for k in ("cx", "cy", "rx", "ry", "opacity")},
                           "color": {"type": "string"}, "motion": _MOTION}}},
    },
}

MANIFEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["assets"],
    "properties": {"assets": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["name", "kind", "bbox", "format", "transparent", "animated", "description", "colors"],
        "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["icon", "illustration", "photo", "logo", "decoration", "background"]},
            "bbox": {"type": "object", "additionalProperties": False, "required": ["x", "y", "w", "h"],
                     "properties": {k: {"type": "integer"} for k in "xywh"}},
            "format": {"type": "string", "enum": ["svg", "png"]},
            "transparent": {"type": "boolean"}, "animated": {"type": "boolean"},
            "description": {"type": "string"}, "colors": {"type": "array", "items": {"type": "string"}}}}}},
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


def _render_svg(svg_path, out_dir, size=256):
    """Rasterise an SVG preview with cairosvg, rsvg-convert or macOS Quick Look, whichever exists."""
    global _warned_render
    svg_path, out_dir = Path(svg_path), Path(out_dir)
    final = out_dir / f"{svg_path.stem}.png"
    txt = svg_path.read_text()
    m_ = re.search(r"<svg\b[^>]*>", txt)
    root_tag = re.sub(r'\s(width|height)="[^"]*"', "", m_.group(0)).replace("<svg", f'<svg width="{size}" height="{size}"', 1)
    sized = txt[:m_.start()] + root_tag + txt[m_.end():]
    if _cairosvg():
        try:
            _cairosvg().svg2png(bytestring=sized.encode(), write_to=str(final), output_width=size, output_height=size)
            return final
        except Exception as e:
            print(f"  ! cairosvg failed on {svg_path.name}: {str(e).splitlines()[0]}")
    tmp = out_dir / f".{svg_path.stem}.svg"
    tmp.write_text(sized)
    try:
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-w", str(size), "-h", str(size), "-o", str(final), str(tmp)], capture_output=True)
        elif shutil.which("qlmanage"):
            subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", str(out_dir), str(tmp)], capture_output=True)
            ql = out_dir / f"{tmp.name}.png"
            if ql.exists():
                ql.rename(final)
        elif not _warned_render:
            print("  (no SVG renderer found: pip install cairosvg for previews)")
            _warned_render = True
    finally:
        tmp.unlink(missing_ok=True)
    return final if final.exists() else None


def cmd_svg(a):
    d, m, mock = _load(a.dir)
    todo = [x for x in _select(m, a.only) if x["format"] == "svg" or a.only]
    (d / "svg" / "preview").mkdir(parents=True, exist_ok=True)
    schema = {"type": "object", "additionalProperties": False, "required": ["files"], "properties": {"files": {
        "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "svg"],
                                   "properties": {"name": {"type": "string"}, "svg": {"type": "string"}}}}}}
    batches = [todo[i:i + 6] for i in range(0, len(todo), 6)]

    def run(batch):
        parts = [f"The first image is an APPROVED UI design. After it come zoomed crops of specific graphics. {SVG_RULES}\n"
                 "Return one file per requested asset, name exactly as given.", mock.convert("RGB")]
        for x in batch:
            b = x["bbox"]
            parts += [f"Asset \"{x['name']}\" at bbox x={b['x']},y={b['y']},w={b['w']},h={b['h']}: {x['description']} "
                      f"Colours: {', '.join(x.get('colors', []))}", ref_crop(mock, b, margin=0.25, long_side=512)]
        return ask(parts, schema, "svg_files")["files"]
    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        for files in ex.map(run, batches):
            for f in files:
                name = re.sub(r"[^a-z0-9_]", "_", f["name"].lower().removesuffix(".svg"))
                p = d / "svg" / f"{name}.svg"
                p.write_text(f["svg"])
                try:
                    ET.fromstring(f["svg"])
                except ET.ParseError as e:
                    print(f"  ! {name}: invalid XML ({e})")
                    continue
                prev = _render_svg(p, d / "svg" / "preview")
                asset = next((x for x in m["assets"] if x["name"] == name), None)
                if asset and prev:
                    b = asset["bbox"]
                    pad = int(max(b["w"], b["h"]) * 0.2)
                    rows.append([(f"{name} (design)", mock.crop((b["x"] - pad, b["y"] - pad, b["x"] + b["w"] + pad, b["y"] + b["h"] + pad))),
                                 ("svg render", _rgba(prev))])
                print(f"  {p}")
    if rows:
        (d / "review").mkdir(exist_ok=True)
        stamp = time.strftime("%H%M%S")
        for i in range(0, len(rows), 8):
            print("review sheet ->", sheet(rows[i:i + 8], d / "review" / f"svg_{stamp}_{i // 8 + 1}.png", T=180))


# ---- SVG -> Android VectorDrawable

def _num(v, default=0.0):
    if v is None:
        return default
    return float(re.sub(r"[a-z%]+$", "", str(v).strip()) or default)


def _color(v, opacity=1.0):
    v = (v or "").strip().lower()
    named = {"white": "#ffffff", "black": "#000000"}
    v = named.get(v, v)
    if not v.startswith("#"):
        raise ValueError(f"unsupported colour {v!r}")
    h = v[1:]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    alpha = int(round(opacity * 255)) if len(h) == 6 else int(h[6:8], 16)
    return f"#{alpha:02X}{h[:6].upper()}" if alpha < 255 else f"#{h[:6].upper()}"


def _shape_path(el):
    tag = el.tag.replace(SVG_NS, "")
    g = lambda k, dflt=0.0: _num(el.get(k), dflt)
    if tag == "path":
        return el.get("d")
    if tag == "circle" or tag == "ellipse":
        cx, cy = g("cx"), g("cy")
        rx = g("r") if tag == "circle" else g("rx")
        ry = g("r") if tag == "circle" else g("ry")
        return f"M{cx - rx},{cy}a{rx},{ry} 0 1,0 {2 * rx},0a{rx},{ry} 0 1,0 {-2 * rx},0Z"
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        rx = g("rx", g("ry"))
        ry = g("ry", rx)
        if not rx:
            return f"M{x},{y}h{w}v{h}h{-w}Z"
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        return (f"M{x + rx},{y}h{w - 2 * rx}a{rx},{ry} 0 0,1 {rx},{ry}v{h - 2 * ry}a{rx},{ry} 0 0,1 {-rx},{ry}"
                f"h{-(w - 2 * rx)}a{rx},{ry} 0 0,1 {-rx},{-ry}v{-(h - 2 * ry)}a{rx},{ry} 0 0,1 {rx},{-ry}Z")
    if tag == "line":
        return f"M{g('x1')},{g('y1')}L{g('x2')},{g('y2')}"
    if tag in ("polyline", "polygon"):
        pts = re.findall(r"-?[\d.]+(?:e-?\d+)?", el.get("points", ""))
        pairs = [f"{pts[i]},{pts[i + 1]}" for i in range(0, len(pts) - 1, 2)]
        return "M" + "L".join(pairs) + ("Z" if tag == "polygon" else "")
    return None


def svg_to_vd(svg_text, width_dp=None):
    root = ET.fromstring(svg_text)
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if v] or \
         [0, 0, _num(root.get("width"), 24), _num(root.get("height"), 24)]
    vw, vh = vb[2], vb[3]
    wdp = width_dp or vw
    hdp = round(wdp * vh / vw, 2)
    grads = {}
    for gel in root.iter():
        tag = gel.tag.replace(SVG_NS, "")
        if tag in ("linearGradient", "radialGradient"):
            if gel.get("gradientUnits") != "userSpaceOnUse":
                raise ValueError(f"gradient #{gel.get('id')} must use gradientUnits=userSpaceOnUse")
            if gel.get("gradientTransform"):
                raise ValueError(f"gradient #{gel.get('id')} uses gradientTransform (unsupported)")
            stops = []
            for s in gel:
                if s.tag.replace(SVG_NS, "") == "stop":
                    style = dict(kv.split(":", 1) for kv in (s.get("style") or "").replace(" ", "").split(";") if ":" in kv)
                    col = s.get("stop-color") or style.get("stop-color", "#000000")
                    op = _num(s.get("stop-opacity") or style.get("stop-opacity"), 1.0)
                    off = s.get("offset", "0")
                    off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
                    stops.append((off, _color(col, op)))
            grads[gel.get("id")] = (tag, gel, stops)
    used_grad = False
    lines = []

    def grad_xml(ref, attr, indent):
        nonlocal used_grad
        used_grad = True
        tag, gel, stops = grads[ref]
        if tag == "linearGradient":
            head = (f'android:type="linear" android:startX="{_num(gel.get("x1"))}" android:startY="{_num(gel.get("y1"))}" '
                    f'android:endX="{_num(gel.get("x2"))}" android:endY="{_num(gel.get("y2"))}"')
        else:
            head = (f'android:type="radial" android:centerX="{_num(gel.get("cx"))}" android:centerY="{_num(gel.get("cy"))}" '
                    f'android:gradientRadius="{_num(gel.get("r"))}"')
        items = "".join(f'\n{indent}      <item android:offset="{o:g}" android:color="{c}"/>' for o, c in stops)
        return (f'\n{indent}  <aapt:attr name="android:{attr}">\n{indent}    <gradient {head}>{items}\n'
                f'{indent}    </gradient>\n{indent}  </aapt:attr>')

    def walk(el, inherited):
        for ch in el:
            tag = ch.tag.replace(SVG_NS, "")
            if tag in ("defs", "linearGradient", "radialGradient", "title", "desc", "metadata"):
                continue
            if ch.get("transform"):
                raise ValueError(f"<{tag}> uses transform (unsupported) — ask for flattened coordinates")
            if tag in ("text", "image", "use", "mask", "clipPath", "filter", "style"):
                raise ValueError(f"<{tag}> is not convertible")
            attrs = dict(inherited)
            style = dict(kv.split(":", 1) for kv in (ch.get("style") or "").replace(" ", "").split(";") if ":" in kv)
            for k in ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "fill-rule",
                      "fill-opacity", "stroke-opacity", "opacity", "stroke-miterlimit"):
                v = ch.get(k) or style.get(k)
                if v is not None:
                    attrs[k] = v
            if tag == "g":
                walk(ch, attrs)
                continue
            d = _shape_path(ch)
            if not d:
                continue
            op = _num(attrs.get("opacity"), 1.0)
            fill = attrs.get("fill", "#000000")
            stroke = attrs.get("stroke", "none")
            out = [f'    <path\n        android:pathData="{d.strip()}"']
            children = ""
            if fill != "none":
                m_ = re.match(r"url\(#([^)]+)\)", fill)
                if m_:
                    children += grad_xml(m_.group(1), "fillColor", "    ")
                else:
                    out.append(f'        android:fillColor="{_color(fill)}"')
                fo = _num(attrs.get("fill-opacity"), 1.0) * op
                if fo < 1:
                    out.append(f'        android:fillAlpha="{fo:g}"')
                if attrs.get("fill-rule") == "evenodd":
                    out.append('        android:fillType="evenOdd"')
            if stroke != "none":
                m_ = re.match(r"url\(#([^)]+)\)", stroke)
                if m_:
                    children += grad_xml(m_.group(1), "strokeColor", "    ")
                else:
                    out.append(f'        android:strokeColor="{_color(stroke)}"')
                out.append(f'        android:strokeWidth="{_num(attrs.get("stroke-width"), 1.0):g}"')
                so = _num(attrs.get("stroke-opacity"), 1.0) * op
                if so < 1:
                    out.append(f'        android:strokeAlpha="{so:g}"')
                if attrs.get("stroke-linecap") in ("round", "square"):
                    out.append(f'        android:strokeLineCap="{attrs["stroke-linecap"]}"')
                if attrs.get("stroke-linejoin") in ("round", "bevel"):
                    out.append(f'        android:strokeLineJoin="{attrs["stroke-linejoin"]}"')
            lines.append("\n".join(out) + (f">{children}\n    </path>" if children else " />"))
    walk(root, {})
    ns = ' xmlns:aapt="http://schemas.android.com/aapt"' if used_grad else ""
    return (f'<?xml version="1.0" encoding="utf-8"?>\n<vector xmlns:android="http://schemas.android.com/apk/res/android"{ns}\n'
            f'    android:width="{wdp:g}dp"\n    android:height="{hdp:g}dp"\n'
            f'    android:viewportWidth="{vw:g}"\n    android:viewportHeight="{vh:g}">\n' + "\n".join(lines) + "\n</vector>\n")


def cmd_vd(a):
    for f in a.svgs:
        p = Path(f)
        name = a.prefix + p.stem
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            xml = svg_to_vd(p.read_text(), a.dp)
        except ValueError as e:
            print(f"  ! {p.name}: {e}")
            continue
        (out_dir / f"{name}.xml").write_text(xml)
        print(f"  {out_dir / (name + '.xml')}")


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


def _fmt(v, nd=2):
    return f"{round(v, nd) + 0:g}"


def _d(subs):
    out = []
    for sp in subs:
        out.append(f"M{_fmt(sp['pts'][0][0])},{_fmt(sp['pts'][0][1])}")
        out += [s[0] + " ".join(f"{_fmt(p[0])},{_fmt(p[1])}" for p in s[1:]) for s in sp["segs"]]
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


def _vb(vec):
    return [float(x) for x in re.split(r"[\s,]+", vec["viewBox"].strip())]


def _fit_viewbox(vec, margin=0.02):
    """viewBox grown symmetrically so every pose of the animation (plus stroke) stays inside; None if it already fits."""
    x, y, w, h = _vb(vec)
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
    return f"{_fmt(cx - hw)} {_fmt(cy - hh)} {_fmt(2 * hw)} {_fmt(2 * hh)}"


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
    vb = _vb(vec)
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
        head = f'{ind}    <keyframe android:fraction="{_fmt(f, 4)}" android:value="{_fmt(v, 4)}"'
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
        head = (f'android:type="linear" android:startX="{_fmt(s[0])}" android:startY="{_fmt(s[1])}" '
                f'android:endX="{_fmt(e[0])}" android:endY="{_fmt(e[1])}"' if linear else
                f'android:type="radial" android:centerX="{_fmt(s[0])}" android:centerY="{_fmt(s[1])}" '
                f'android:gradientRadius="{_fmt(max(math.dist(s, e), 0.01))}"')
        items = "".join(f'\n{ind}        <item android:offset="{o:g}" android:color="{_color(c, al)}" />' for o, c, al in stops)
        kids.append(f'{ind}<aapt:attr name="android:{attr}">\n{ind}    <gradient {head}>{items}\n{ind}    </gradient>\n{ind}</aapt:attr>')
    if p.get("fill") == "glow":
        color("fillColor", "url(#glow)")
    elif (p.get("fill") or "none") != "none":
        color("fillColor", p["fill"])
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        color("strokeColor", p["stroke"])
        attrs.append(f'android:strokeWidth="{_fmt(float(p["stroke_width"]))}"')
        attrs.append(f'android:strokeLineCap="{p.get("cap", "round")}" android:strokeLineJoin="{p.get("join", "round")}"')
    return attrs, kids


def build_avd(vec, seconds=1.2, width_dp=48):
    """<animated-vector> with the vector inlined: group transforms, fillAlpha fades and pathData morphs, all looping."""
    vb = _vb(vec)
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
                    attrs.append(f'android:{prop}="{_fmt(v0, 4)}"')
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
            xml = [f'        <group android:pivotX="{_fmt(g["cx"])}" android:pivotY="{_fmt(g["cy"])}" '
                   f'android:scaleY="{g["ry"] / g["rx"]:.4g}">'] + xml + ["        </group>"]
        if ks:
            chans = {"translateX": ("dx", 0), "translateY": ("dy", 0), "scaleX": ("sx", 1), "scaleY": ("sy", 1), "rotation": ("rotate", 0)}
            pvhs, gattrs = [], [f'android:name="{gname}"', f'android:pivotX="{_fmt(ax)}"', f'android:pivotY="{_fmt(ay)}"']
            for prop, (c, rest) in chans.items():
                if abs(m0[c] - rest) > 1e-6:
                    gattrs.append(f'android:{prop}="{_fmt(m0[c], 4)}"')
                if len({round(k[c], 5) for k in ks}) > 1:
                    pvhs.append(_avd_pvh(prop, _frames(ks, lambda k, c=c: k[c]), "                "))
            if pvhs:
                targets.append(_avd_target(gname, [_avd_animator(pvhs, ms, "            ")]))
            xml = [f'        <group {" ".join(gattrs)}>'] + ["    " + ln for x in xml for ln in x.split("\n")] + ["        </group>"]
        body += xml
    if vb[0] or vb[1]:
        body = [f'        <group android:translateX="{_fmt(-vb[0])}" android:translateY="{_fmt(-vb[1])}">'] + \
               ["    " + ln for x in body for ln in x.split("\n")] + ["        </group>"]
    hdp = width_dp * vb[3] / vb[2]
    vector = (f'    <aapt:attr name="android:drawable">\n        <vector android:width="{_fmt(width_dp)}dp" android:height="{_fmt(hdp)}dp"\n'
              f'            android:viewportWidth="{_fmt(vb[2])}" android:viewportHeight="{_fmt(vb[3])}">\n' +
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
              *(shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"))):
        if c and Path(c).exists():
            return c
    return None


@functools.cache
def _lottie_js():
    """lottie-web source (cached under ~/.cache/gptkit) for inlining; None if it can't be fetched."""
    p = Path("~/.cache/gptkit/lottie-5.12.2.min.js").expanduser()
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
                                     f"--screenshot={out}", page.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def lottie_frames(lot, times, out_png, T=256):
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
    frames = lottie_frames(lot, times, out / ".strip.png")
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
    sheet(rows, out / "keys_review.png", T=220)
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
    dp = round(dp * _vb(built)[2] / _vb(vec)[2], 1)
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
    renderer = "cairosvg" if _cairosvg() else next((t for t in ("rsvg-convert", "qlmanage") if shutil.which(t)), None)
    print(f"svg preview : {renderer or 'none (pip install cairosvg)'}")
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
    p = sub.add_parser("svg", help="write svg-format assets as SVG with previews")
    p.add_argument("dir"); p.add_argument("--only"); p.set_defaults(fn=cmd_svg)
    p = sub.add_parser("vd", help="convert SVG files to Android VectorDrawable XML")
    p.add_argument("svgs", nargs="+"); p.add_argument("--out", required=True); p.add_argument("--dp", type=float)
    p.add_argument("--prefix", default="ic_"); p.set_defaults(fn=cmd_vd)
    p = sub.add_parser("frames", help="frame-by-frame animation from a forged asset (moving layer sprite sheet)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving", required=True)
    p.add_argument("--static", help="part that stays still (generated once, composited)")
    p.add_argument("--static-order", choices=["front", "back"], default="front")
    p.add_argument("--motion"); p.add_argument("--count", type=int, default=8); p.add_argument("--cols", type=int, default=4)
    p.add_argument("--anchor", choices=["bottom", "center"], default="bottom"); p.add_argument("--fps", type=float, default=12)
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
    p = sub.add_parser("pick", help="override the chosen candidate (number from the review sheet, or a file path)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("candidate"); p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("score", help="score arbitrary files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
