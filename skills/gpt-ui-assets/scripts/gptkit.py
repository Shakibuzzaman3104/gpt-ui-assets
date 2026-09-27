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


# ---- raster -> vector tracing, part rigs, procedural keyframes

CHROME_PATHS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium")


@functools.cache
def _chrome():
    if os.environ.get("GPTKIT_CHROME") and Path(os.environ["GPTKIT_CHROME"]).exists():
        return os.environ["GPTKIT_CHROME"]
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    return next((shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
                 if shutil.which(n)), None)


def chrome_shot(html, w, h, budget=3000):
    """Render an HTML string in headless Chrome at w x h CSS px on a transparent background -> RGBA image or None."""
    import tempfile
    exe = _chrome()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        page, png = Path(td) / "page.html", Path(td) / "shot.png"
        page.write_text(html)
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                        "--force-device-scale-factor=1", f"--virtual-time-budget={budget}",
                        "--default-background-color=00000000", f"--screenshot={png}", page.as_uri()],
                       capture_output=True, timeout=180)
        return Image.open(png).convert("RGBA") if png.exists() else None


def render_svgs(svgs, sizes):
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


def _fmt(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else ("-0" if s == "-0.0" else s)


def _beziers_d(loops_bez, scale=1.0, dx=0.0, dy=0.0):
    """Beziers -> absolute M/L/C/Z path data (the command set build_lottie and svg_to_vd accept)."""
    out = []
    for bz in loops_bez:
        if not bz:
            continue
        P = lambda p: f"{_fmt(p[0] * scale + dx)} {_fmt(p[1] * scale + dy)}"
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


def _hex(c, alpha=None):
    c = np.clip(np.round(c), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c[:3])


def _fill_svg(fill, gid, scale=1.0):
    """fill spec -> (fill attribute, <defs> entry)."""
    if fill["type"] == "flat":
        return _hex(fill["color"]), ""
    stops = "".join(f'<stop offset="{o:.3f}" stop-color="{_hex(c)}"/>' for o, c in fill["stops"])
    if fill["type"] == "linear":
        (x1, y1), (x2, y2) = fill["p0"] * scale, fill["p1"] * scale
        return f"url(#{gid})", (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{_fmt(x1)}" '
                                f'y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}">{stops}</linearGradient>')
    cx, cy = fill["c"] * scale
    return f"url(#{gid})", (f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{_fmt(cx)}" cy="{_fmt(cy)}" '
                            f'r="{_fmt(fill["r"] * scale)}">{stops}</radialGradient>')


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
                           "color": _hex(c_i), "mask": region})
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
        out.append(_fmt(float(t) + (dx if k % 2 == 0 else dy)))
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
                           "color": _hex(rgb[ys, xs].mean(0)), "mask": masks[r]})
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


def vd_to_svg(xml):
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


def aapt2_check(xml_text, name="traced"):
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
    rens = render_svgs([c["svg"] for c in cands], [src.size] * len(cands))
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
                            "aapt2": aapt2_check(vd, a.prefix + name)}
            vr = render_svgs([vd_to_svg(vd)], [src.size])
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
                out.append(f"{_fmt(x)} {_fmt(y)}")
            nums = []
            out.append(t)
        else:
            nums.append(float(t))
    return " ".join(out[:-1]).replace(" Z", "Z")


def _path_points(d):
    v = [float(t) for t in _TOK.findall(d) if not t.isalpha()]
    return np.array(v[:len(v) // 2 * 2]).reshape(-1, 2)


MOTIONS = {  # default amplitude per motion (fraction of the part's width/height)
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
    A = MOTIONS[kind] * amp
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
    raise ValueError(f"unknown motion {kind!r} (choose {', '.join(MOTIONS)})")


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


LOTTIE_JS = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"


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
    html = (f"<!doctype html><body style='margin:0;background:{bg}'>{divs}<script src='{LOTTIE_JS}'></script>"
            f"<script>{js}</script></body>")
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
        f"<div id=a style='width:{lot['w']}px;height:{lot['h']}px'></div><script src='{LOTTIE_JS}'></script>"
        f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{json.dumps(lot)}}})</script>")
    vb = [float(x) for x in vec["viewBox"].split()]
    sz = (round(300 * vb[2] / max(vb[2:])), round(300 * vb[3] / max(vb[2:])))
    nk = max((len(al["keyframes"]) for al in vec["animated_layers"]), default=1)
    keys = render_svgs([_vec_render(vec, sz, k) for k in range(nk)], [sz] * nk)
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
        report["lottie_rendered"] = bool(np.asarray(film)[..., :3].std() > 3)
    (out / "animation_report.json").write_text(json.dumps(report, indent=2))
    return report


def cmd_animate(a):
    src = Path(a.layers)
    vec = json.loads(src.read_text())
    specs = _parse_motions(a.part)
    if not specs:
        sys.exit("give --part 'layer=motion[:amp[:phase]]; ...' (motions: " + ", ".join(MOTIONS) + ")")
    vec = animate_layers(vec, specs, a.keyframes)
    out = Path(a.out) if a.out else src.parent / "anim"
    rep = write_animation(vec, out, a.name or src.parent.name, a.seconds, a.fps, a.size, a.ease, a.preview_bg)
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
    rens = render_svgs(svgs, [src.size] * len(svgs))
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
        vec = animate_layers(vec, _parse_motions(a.animate), a.keyframes)
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
    p.add_argument("--animate", help="'part=motion[:amp[:phase]][@anchor_part]; ...' motions: " + ", ".join(MOTIONS))
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
    p = sub.add_parser("score", help="score arbitrary files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    _add_trace_rig_parsers(sub)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
