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


def _lab(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= [0.9505, 1.0, 1.089]
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


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
    L = np.asarray(Image.open(png).convert("RGBA").convert("L").resize((size, size)), float)
    png.unlink(missing_ok=True)
    return Image.fromarray((255 - L).astype(np.uint8), "L")


def _vd_group(xml, s, tx, ty):
    decl, tag, body = xml.split(">", 2)
    return (f'{decl}>{tag}>\n    <group android:scaleX="{s:.4f}" android:scaleY="{s:.4f}" android:translateX="{tx:.3f}" '
            f'android:translateY="{ty:.3f}">' + body.replace("</vector>", "").rstrip() + "\n    </group>\n</vector>\n")


def _android_sdk():
    sdk = Path(os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or "~/Library/Android/sdk").expanduser()
    ver = lambda p: [int(x) for x in re.findall(r"\d+", p.parent.name)]
    aapt = sorted((sdk / "build-tools").glob("*/aapt2"), key=ver)
    jars = sorted((sdk / "platforms").glob("android-*/android.jar"), key=ver)
    return (aapt[-1] if aapt else shutil.which("aapt2")), (jars[-1] if jars else None)


def _aapt2_check(res, name, work):
    aapt, jar = _android_sdk()
    if not aapt:
        return "skipped (aapt2 not found; set ANDROID_HOME)"
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
    return f"compile+link ok ({jar.parent.name})" if not r.returncode else "link FAILED: " + (r.stderr or r.stdout).strip()[:800]


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
    q.add_argument("--separate", choices=["local", "api"], default="local",
                   help="foreground: local matte against the shipped background (exact), or an image-edit redraw (1 call)")
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
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("candidate"); p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("score", help="score arbitrary files against a manifest asset")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
