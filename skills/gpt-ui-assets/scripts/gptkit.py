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


def _align(crops, anchor):
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


def _placement(src, static, frames):
    """One scale + origin for every frame: silhouette fit of the middle frame or extent fit of any frame, by IoU."""
    target = (np.asarray(src.getchannel("A")) > 60) & _moving_mask(static)
    if not target.any():
        return None
    ys, xs = np.nonzero(target)
    tb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    cands = []
    mid = len(frames) // 2
    fit = _fit_moving(src, static, frames[mid])
    if fit:
        s, cx, by, fb = fit
        cands.append(("silhouette", mid, s, (cx - (fb[0] + fb[2]) / 2 * s, by - fb[3] * s)))
    for i, f in enumerate(frames):
        b = content_box(f, 60)
        if b:
            s = math.sqrt((tb[2] - tb[0]) / (b[2] - b[0]) * (tb[3] - tb[1]) / (b[3] - b[1]))
            org = ((tb[0] + tb[2] - (b[0] + b[2]) * s) / 2, (tb[1] + tb[3] - (b[1] + b[3]) * s) / 2)
            cands.append(("extent", i, s, org))
    if not cands:
        return None

    def iou(c):
        m = _placed_alpha(src.size, frames[c[1]], c[2], c[3]) > 60
        return (m & target).sum() / max(1, (m | target).sum())
    scored = [(iou(c), c) for c in cands]
    v, (how, i, s, org) = max(scored, key=lambda t: t[0])
    print(f"  registered by {how} fit of f{i + 1} (IoU {v:.2f} with the asset's moving part)")
    return s, org


def _composite(src, static, frames, order):
    """Place every aligned moving frame with one shared transform onto the asset and stack the static layer."""
    fit = _placement(src, static, frames)
    if not fit:
        print("  ! could not locate the moving part in the asset; composites skipped")
        return []
    s, org = fit
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
    return [c.crop(box) for c in comps]


# -- frame QA

_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def _lab(rgb):
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    t = c @ _XYZ.T / _D65
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


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

def _lottie_ks(w, h):
    return {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [w / 2, h / 2, 0]},
            "a": {"a": 0, "k": [w / 2, h / 2, 0]}, "s": {"a": 0, "k": [100, 100, 100]}}


def _lottie_image(img, i):
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return {"id": f"img_{i}", "w": img.width, "h": img.height, "u": "", "e": 1,
            "p": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def lottie_frames(frames, fps, name="gptkit"):
    """Lottie image sequence: each frame is an embedded PNG shown for exactly one Lottie frame."""
    w, h = frames[0].size
    layers = [{"ddd": 0, "ind": i + 1, "ty": 2, "nm": f"frame {i + 1}", "refId": f"img_{i}", "sr": 1, "ip": i,
               "op": i + 1, "st": 0, "bm": 0, "ks": _lottie_ks(w, h)} for i in range(len(frames))]
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
        lot = lottie or lottie_frames(small, fps, anim)
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
        tq = _frame_qa(_align(trial, a.anchor), ref, ref_mask)
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
    qa = _frame_qa(_align(crops, a.anchor), src, ref_mask)
    bad = _qa_report(qa)
    if a.reroll and bad:
        crops, qa = _reroll(a, crops, qa, bad, src, ref_mask, motion, out)
        _qa_report(qa, " after re-roll")
    frames = _align(crops, a.anchor)
    if a.match_colors:
        frames = _match_colors(frames, src, ref_mask)
        qa = _frame_qa(frames, src, ref_mask)
        _qa_report(qa, " after colour match")
    for i, f in enumerate(frames):
        f.save(out / f"moving_f{i + 1:02d}.png")
    (out / "qa.json").write_text(json.dumps(qa, indent=1))
    seq = (_composite(src, static, frames, a.static_order) if static is not None else []) or frames
    for i, f in enumerate(seq):
        f.save(out / f"frame_{i + 1:02d}.png")
    labels = [f"f{r['frame']} {' '.join(r['flags'])}".strip() for r in qa["frames"]]
    _write_anim(out, a.name, seq, fps, a, labels)


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
    p = sub.add_parser("score", help="score arbitrary files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
