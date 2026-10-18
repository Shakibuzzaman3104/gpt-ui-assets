"""Appicon concepts / build."""
import concurrent.futures as cf
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

from .core import ApiError, _clip_bbox, _config, _open_image, _plan, _png_bytes, _res_name, ask, edit, generate
from .imaging import _lab, _mean_diff, _rgba, _unusable, checker, score, trim
from .design import _aspect_ok, _asset, _load
from .forge import (
    _color_final, _current, _entry, _finalize, _read_report, _render_svg, _renderer_name, _rep_abs, _rep_rel,
    _update_report,
)
from .vectordrawable import _aapt2_tree, _vd_lint, svg_to_vd
from .frames import _register
from .export import DENSITIES, _fit

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
