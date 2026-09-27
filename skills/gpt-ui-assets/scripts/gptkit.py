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


def _runs(flags, merge=1):
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
    runs = _runs(keep.any(1), merge=max(2, int(0.12 * size)))
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
    segs = _runs(keep[ry0:ry1].any(0), merge=max(3, int(0.6 * lh)))
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
    (d / "text_boxes.json").write_text(json.dumps({r: strip(t) for r, t in texts.items()}, indent=2))


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


def _load_spec(dir_):
    d = Path(dir_)
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
               if not re.search(r"pixels? per|per dp|px_to_dp", s["between"], re.I)]
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


def _chrome_bin():
    for c in (os.environ.get("GPTKIT_CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("chrome")):
        if c and Path(c).exists():
            return c
    return None


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
    chrome = _chrome_bin()
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
        r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--dump-dom",
                            probe.resolve().as_uri()], capture_output=True, text=True, timeout=60)
        m_ = re.search(r'<pre id="r">(.*?)</pre>', r.stdout, re.S)
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
    d, spec, mock = _load_spec(a.dir)
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
        r = requests.get(GF_METADATA, timeout=60)
        r.raise_for_status()
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
        segs = _runs(ink, merge)
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
    d, spec, mock = _load_spec(a.dir)
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
            results[st["role"]] = {"role": st["role"], "family": winner, "weight": w if ok else st.get("weight", w), "italic": italic,
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
    d, spec, mock = _load_spec(a.dir)
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
    _design_to_code_parsers(sub)
    p = sub.add_parser("score", help="score arbitrary files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
