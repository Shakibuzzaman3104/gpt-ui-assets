"""Lottie-web / SVG previews in headless Chrome."""
import base64
import functools
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
from PIL import Image

from .core import SVG_NS, _chrome_err, _chrome_flags, _need, _open_image, _res_name, ask
from .imaging import _rgba, ref_crop, sheet
from .prompts import MOTION_MIN, MOTION_STRENGTH, VECTOR_PROMPT, VECTOR_SCHEMA
from .design import _load
from .forge import _render_svg
from .vectordrawable import (
    _IDENTITY, _aapt2_line, _inherit, _mapply, _mmul, _num, _shape_path, _style_dict, aapt2_check,
)
from .vectorize import (
    GLOW_STOPS, _bbox, _canon_vec, _cap, _curve_pts, _d, _fit_viewbox, _fmtn, _grad_chain, _gradient, _join,
    _layer_name, _layers, _matrix, _morph_at, _motion_at, _normalize_vec, _paint_alphas, _paint_color, _subpaths,
    _validate_vec, _vec_vb, _xform_subs, build_avd, build_lottie, motion_report,
)

# ---- previews: lottie-web in headless Chrome, SVG fallback

LOTTIE_JS = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"


def _codex_seatbelt():
    """True inside the Codex CLI's macOS sandbox. Chrome cannot start there: Seatbelt blocks its registration with the
    window server, so it aborts in TransformProcessType (SIGABRT, and a crash report every launch)."""
    return sys.platform == "darwin" and bool(os.environ.get("CODEX_SANDBOX"))


def _chrome_missing_reason():
    """Why _chrome() returned None, in one line."""
    if os.environ.get("GPTKIT_CHROME") == "none":
        return "Chrome disabled (GPTKIT_CHROME=none)"
    if _codex_seatbelt():
        return "Chrome cannot start inside the Codex sandbox (CODEX_SANDBOX is set); run this outside it"
    return "no Chrome/Chromium found (set GPTKIT_CHROME)"


@functools.cache
def _chrome():
    if os.environ.get("GPTKIT_CHROME") == "none" or _codex_seatbelt():
        return None
    for c in (os.environ.get("GPTKIT_CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
              *(shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"))):
        if c and Path(c).exists():
            return c
    return None


_LOTTIE_JS_REASON = ""  # why _lottie_js() returned None (set on every call)
_LOTTIE_REASON = ""  # why the last render_lottie() returned None
_SHOT_REASON = ""  # why the last _chrome_shot() returned None
_warned_shot = False


def _lottie_js_ok(txt):
    return len(txt) > 100_000 and (txt.startswith("/*!") or "lottie" in txt)


def _lottie_js_cache():
    from .fonts import _cache_dir  # deferred: fonts imports this module
    return _cache_dir() / "lottie-5.12.2.min.js"


@functools.cache
def _lottie_js():
    """lottie-web source (cached under ~/.cache/gptkit) for inlining; None with _LOTTIE_JS_REASON set when unavailable."""
    global _LOTTIE_JS_REASON
    p = _lottie_js_cache()
    try:
        txt = p.read_text() if p.exists() else ""
    except OSError as e:
        txt, _LOTTIE_JS_REASON = "", f"cannot read {p}: {e}"
    if txt and _lottie_js_ok(txt):
        _LOTTIE_JS_REASON = ""
        return txt
    if txt:
        print(f"  ! cached {p} is not lottie-web ({len(txt)} bytes); fetching again")
    try:
        r = requests.get(LOTTIE_JS, timeout=60)
        r.raise_for_status()
        txt = r.text
    except Exception as e:  # requests.RequestException and friends; the preview is optional
        _LOTTIE_JS_REASON = f"could not fetch {LOTTIE_JS} ({type(e).__name__}: {str(e)[:120]})"
        return None
    if not _lottie_js_ok(txt):
        _LOTTIE_JS_REASON = f"{LOTTIE_JS} returned {len(txt)} bytes that are not lottie-web"
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(txt)
        os.replace(tmp, p)
    except OSError as e:
        print(f"  ! could not cache lottie-web at {p}: {e}")
    _LOTTIE_JS_REASON = ""
    return txt


def _chrome_shot(html, out, w, h, budget=8000, timeout=120):
    """Screenshot an HTML string at w x h with a transparent background; the RGBA image, or None (_SHOT_REASON says why)."""
    global _SHOT_REASON, _warned_shot
    exe = _chrome()
    if not exe:
        _SHOT_REASON = _chrome_missing_reason()
        return None
    out = Path(out)
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page, errf = Path(tmp) / "page.html", Path(tmp) / "stderr.txt"
        page.write_text(html)
        try:
            with errf.open("w") as ef:
                proc = subprocess.Popen([exe, *_chrome_flags(w, h), f"--user-data-dir={tmp}/p", f"--virtual-time-budget={budget}",
                                         f"--screenshot={out}", page.as_uri()], stdout=subprocess.DEVNULL, stderr=ef)
        except OSError as e:
            _SHOT_REASON = f"{exe}: {e}"
            return None
        # Chrome can linger after writing the screenshot (lottie's web worker), so stop once the file is complete
        size, end = -1, time.time() + timeout
        while proc.poll() is None and time.time() < end:
            cur = out.stat().st_size if out.exists() else -1
            if cur > 0 and cur == size:
                break
            size = cur
            time.sleep(0.5)
        proc.kill()
        proc.wait()
        if not out.exists():
            r = subprocess.CompletedProcess(exe, proc.returncode, "", errf.read_text(errors="replace"))
            _SHOT_REASON = f"{exe}: {_chrome_err(r)}"
            if not _warned_shot:
                print(f"  ! headless Chrome produced no screenshot: {_SHOT_REASON}")
                _warned_shot = True
            return None
    try:
        img = _rgba(out)
    except (OSError, ValueError) as e:
        _SHOT_REASON = f"{exe} wrote an unreadable screenshot: {e}"
        return None
    _SHOT_REASON = ""
    return img.crop((0, 0, w, h))


def render_lottie(lot, times, out_png, T=256):
    """Render the Lottie at loop fractions `times` with lottie-web; RGBA frames (transparent), or None (_LOTTIE_REASON)."""
    global _LOTTIE_REASON
    js = _lottie_js()
    if not js:
        _LOTTIE_REASON = f"lottie-web unavailable: {_LOTTIE_JS_REASON}"
        return None
    frames = [round(t * lot["op"], 3) for t in times]
    html = ("<!doctype html><meta charset=utf-8><style>body{margin:0;display:flex}div{width:%dpx;height:%dpx;flex:none}</style>"
            "<body>%s<script>%s</script><script>var D=%s;%s.forEach(function(f,i){var a=lottie.loadAnimation({container:"
            "document.getElementById('f'+i),renderer:'svg',loop:false,autoplay:false,animationData:JSON.parse(JSON.stringify(D))});"
            "a.addEventListener('DOMLoaded',function(){a.goToAndStop(f,true)});});</script>") % (
        T, T, "".join(f"<div id=f{i}></div>" for i in range(len(frames))), js, _js_json(lot), json.dumps(frames))
    strip = _chrome_shot(html, out_png, T * len(frames), T)
    if strip is None:
        _LOTTIE_REASON = f"lottie-web unavailable: headless Chrome failed ({_SHOT_REASON})"
        return None
    if strip.width < T * len(frames):
        _LOTTIE_REASON = f"lottie-web unavailable: screenshot is {strip.width}x{strip.height}, expected {T * len(frames)}x{T}"
        return None
    _LOTTIE_REASON = ""
    return [strip.crop((i * T, 0, (i + 1) * T, T)) for i in range(len(frames))]


def _js_json(obj):
    """json.dumps for inlining inside <script>: a layer name containing '</script>' must not end the element."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _pixel_motion(frames):
    """Largest share of the drawn area (t=0 union frame i) whose colour visibly changes from t=0."""
    if len(frames) < 2:
        return 0.0
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
        fa, sa = _paint_alphas(p)
        s = f' fill="{p.get("fill") or "none"}" fill-opacity="{fa:g}"'
        if p.get("fill_rule") == "evenodd":
            s += ' fill-rule="evenodd"'
        if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
            s += (f' stroke="{p["stroke"]}" stroke-opacity="{sa:g}" stroke-width="{p["stroke_width"]}" '
                  f'stroke-linecap="{_cap(p.get("cap"))}" stroke-linejoin="{_join(p.get("join"))}"')
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


_SVG_SHAPES = ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon")
_SVG_SKIP = ("defs", "title", "desc", "metadata", "linearGradient", "radialGradient", "stop", "clipPath", "mask", "filter",
             "pattern", "marker", "script")
_SVG_REJECT = ("use", "symbol", "style", "text", "tspan", "textPath", "image", "foreignObject", "switch")


def _svg_layers(svg_text, stem, file="svg"):
    """Import an SVG as static layers, one per top-level element (a <g> keeps its children together).
    Paint is inherited (style beats attributes, opacity multiplies), transforms are baked into the coordinates
    (stroke width scaled by sqrt|det|), fill-rule/fill-opacity/stroke-opacity/caps/joins are kept, display:none and
    visibility:hidden elements are skipped, every colour becomes '#RRGGBB' and gradients become explicit
    userSpaceOnUse defs. Raises ValueError (one line, naming the file) for <use>/<symbol>/CSS/clip-path/mask/filter/
    text/image and for malformed transforms, path data or colours."""
    root = ET.fromstring(svg_text)
    tag = lambda el: el.tag.replace(SVG_NS, "") if isinstance(el.tag, str) else ""
    if tag(root) != "svg":
        raise ValueError(f"{file}: root element is <{tag(root)}>, not <svg>")
    hint = f"flatten it first (svgo: inlineStyles, convertShapeToPath, removeUselessDefs) or re-run svg --only {stem}"
    for el in root.iter():
        if tag(el) in _SVG_REJECT:
            raise ValueError(f"{file}: <{tag(el)}> is not supported; {hint}")
        for k in ("clip-path", "mask", "filter"):
            if (el.get(k) or _style_dict(el).get(k) or "none") != "none" and tag(el) not in _SVG_SKIP:
                raise ValueError(f"{file}: {k} on <{tag(el)}> is not supported; {hint}")
    vb_s = root.get("viewBox")
    if vb_s:
        try:
            vbl = [float(x) for x in re.split(r"[\s,]+", vb_s.strip())]
            if len(vbl) != 4 or vbl[2] <= 0 or vbl[3] <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"{file}: viewBox {vb_s!r} must be 4 numbers \"x y w h\" with w and h > 0")
    else:
        vbl = [0.0, 0.0, _num(root.get("width"), 24), _num(root.get("height"), 24)]
        if vbl[2] <= 0 or vbl[3] <= 0:
            raise ValueError(f"{file}: no viewBox and no positive width/height")
    vb = " ".join(_fmtn(v, 4) for v in vbl)
    defs_out, grad_ids, notes, used = [], {}, [], set()

    def grad_ref(v, m, subs, what):
        """url(#id) -> the id of an explicit userSpaceOnUse copy resolved for this shape (bbox units, baked transform)."""
        ref = re.fullmatch(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)(\s+\S+)?", v.strip())
        if not ref:
            return None
        gid = ref.group(1)
        bbox = _bbox(_curve_pts(subs))
        try:
            linear, s0, e0, stops = _gradient(root, gid, bbox, vbl)
        except ValueError as e:
            raise ValueError(f"{file}: {what}: {e}")
        if not linear and abs((bbox[2] - bbox[0]) - (bbox[3] - bbox[1])) > 0.05 * max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1e-9) \
                and (next((el.get("gradientUnits") for el in _grad_chain(root, gid) if el.get("gradientUnits")), None) or "objectBoundingBox") != "userSpaceOnUse":
            notes.append(f"gradient #{gid} uses objectBoundingBox units on a non-square shape; approximated as a circle")
        s1, e1 = _mapply(m, *s0), _mapply(m, *e0)
        if not linear:
            e1 = (s1[0] + math.dist(s0, e0) * math.sqrt(abs(m[0] * m[3] - m[1] * m[2])), s1[1])
        key = (linear, tuple(round(x, 3) for x in s1 + e1), tuple(stops))
        if key not in grad_ids:
            base = re.sub(r"[^\w-]", "_", gid) or "g"
            nid = base if all(k[0] != base for k in grad_ids.values()) else f"{base}_{len(grad_ids) + 1}"
            grad_ids[key] = (nid,)
            geo = (f'x1="{_fmtn(s1[0], 3)}" y1="{_fmtn(s1[1], 3)}" x2="{_fmtn(e1[0], 3)}" y2="{_fmtn(e1[1], 3)}"' if linear else
                   f'cx="{_fmtn(s1[0], 3)}" cy="{_fmtn(s1[1], 3)}" r="{_fmtn(e1[0] - s1[0], 3)}"')
            kind = "linearGradient" if linear else "radialGradient"
            defs_out.append(f'<{kind} id="{nid}" gradientUnits="userSpaceOnUse" {geo}>' + "".join(
                f'<stop offset="{o:g}" stop-color="{c}"' + (f' stop-opacity="{al:g}"' if al < 1 else "") + "/>"
                for o, c, al in stops) + f"</{kind}>")
        return f"url(#{grad_ids[key][0]})"

    def pct(v, dflt=1.0):
        sv = str(v).strip() if v is not None else ""
        if not sv:
            return dflt
        n = _num(sv) / 100 if sv.endswith("%") else _num(sv, dflt)
        return min(1.0, max(0.0, n))

    def walk(el, inh, m):
        t = tag(el)
        if not t or t in _SVG_SKIP:
            return []
        a = _inherit(el, inh)
        if a.get("display") == "none":
            return []
        try:
            mm = _mmul(m, _matrix(el.get("transform")))
        except ValueError as e:
            raise ValueError(f"{file}: <{t}>: {e}")
        if t in ("g", "a"):
            return [p for ch in el for p in walk(ch, a, mm)]
        if t == "svg":
            raise ValueError(f"{file}: nested <svg> is not supported; {hint}")
        if t not in _SVG_SHAPES:
            notes.append(f"<{t}> ignored")
            return []
        if a.get("visibility") in ("hidden", "collapse"):
            return []
        d = _shape_path(el)
        if not d:
            return []
        what = f"<{t}{' id=' + el.get('id') if el.get('id') else ''}>"
        try:
            subs = _subpaths(d)
        except (ValueError, IndexError) as e:
            raise ValueError(f"{file}: {what}: {e}")
        if not subs or not any(sp["segs"] for sp in subs):
            return []
        cur = a.get("color")
        fill, stroke = (a.get("fill") or "").strip() or "#000000", (a.get("stroke") or "none").strip() or "none"
        fo, so = pct(a.get("fill-opacity")), pct(a.get("stroke-opacity"))
        try:
            if fill != "none":
                g = grad_ref(fill, mm, subs, what)
                if g:
                    fill = g
                else:
                    if fill.lower() == "currentcolor" and not cur:
                        notes.append(f"{what}: currentColor with no color attribute; using #000000")
                    fill, al = _paint_color(fill, cur)
                    fo *= al
            if stroke != "none":
                g = grad_ref(stroke, mm, subs, what)
                if g:
                    stroke = g
                else:
                    if stroke.lower() == "currentcolor" and not cur:
                        notes.append(f"{what}: currentColor with no color attribute; using #000000")
                    stroke, al = _paint_color(stroke, cur)
                    so *= al
        except ValueError as e:
            raise ValueError(f"{file}: {what}: {e}" if not str(e).startswith(file) else str(e))
        det = math.sqrt(abs(mm[0] * mm[3] - mm[1] * mm[2]))
        sw = _num(a.get("stroke-width"), 1.0) * det if stroke != "none" else 0.0
        if sw <= 0:
            stroke, sw = "none", 0.0
        if fill == "none" and stroke == "none":
            return []
        if stroke != "none" and (a.get("stroke-dasharray") or "none") != "none":
            notes.append(f"{what}: dashes are drawn solid")
        p = {"d": _d(_xform_subs(subs, mm)), "fill": fill, "opacity": round(min(1.0, max(0.0, _num(a.get("opacity"), 1.0))), 4),
             "stroke": stroke, "stroke_width": round(sw, 4), "cap": _cap(a.get("stroke-linecap"), "butt"),
             "join": _join(a.get("stroke-linejoin"), "miter")}
        if fo < 1:
            p["fill_opacity"] = round(fo, 4)
        if so < 1:
            p["stroke_opacity"] = round(so, 4)
        if fill != "none" and (a.get("fill-rule") or "nonzero") == "evenodd":
            p["fill_rule"] = "evenodd"
        return [p]

    layers = []
    base = _inherit(root, {})
    for el in (ch for ch in root if tag(ch) and tag(ch) not in _SVG_SKIP):
        ps = walk(el, base, _IDENTITY)
        if ps:
            layers.append({"name": _layer_name(el.get("id") or (stem if not layers else f"{stem}_{len(layers) + 1}"), used),
                           "paths": ps, "motion": {"anchor_x": 0, "anchor_y": 0, "keys": []}})
    for n in dict.fromkeys(notes):
        print(f"  ~ {file}: {n}")
    return vb, "".join(defs_out), layers


def _merge_imported(vec, imported):
    """Imported SVG layers are the source of truth for geometry; the model only adds motion and extra layers."""
    vb, defs, layers = imported
    vec["viewBox"] = vb
    if defs and defs not in (vec.get("defs") or ""):
        vec["defs"] = defs + (vec.get("defs") or "")
    got = {}
    for sl in vec.get("static_layers") or []:
        if isinstance(sl, dict):
            got.setdefault(_res_name(str(sl.get("name") or ""), "layer_"), sl)
    for L in layers:
        if L["name"] in got:
            got[L["name"]]["paths"] = json.loads(json.dumps(L["paths"]))
            got[L["name"]]["name"] = L["name"]
        else:
            vec.setdefault("static_layers", []).insert(0, json.loads(json.dumps(L)))
    return vec


def _review_sheet(vec, lot, src, out, bg, n=8):
    """keys_review.png: the source, then n evenly spaced frames of the real Lottie (lottie-web) or an SVG fallback.
    Returns (how, pixel_motion): how is 'lottie-web', 'svg fallback (...)' or 'NOT RENDERED (...)' with the reason."""
    out = Path(out)
    for stale in out.glob("key_*.svg"):  # written by older versions; the fallback now renders from a temp dir
        stale.unlink()
    times = [i / n for i in range(n)]
    frames = render_lottie(lot, times, out / ".strip.png")
    (out / ".strip.png").unlink(missing_ok=True)
    how, px = "lottie-web", None
    if frames is not None:
        px = _pixel_motion(frames)
    else:
        reason, frames = _LOTTIE_REASON, []
        with tempfile.TemporaryDirectory() as td:
            for i, t in enumerate(times):
                p = Path(td) / f"key_{i + 1}.svg"
                p.write_text(_vec_svg(vec, t, bg=bg))
                prev = _svg_image(p, Path(td) / f"key_{i + 1}.png", 256)
                if prev is not None:
                    frames.append(prev)
        how = f"svg fallback ({reason})" if frames else (f"NOT RENDERED ({reason}): fix Chrome (GPTKIT_CHROME, "
                                                         f"GPTKIT_CHROME_NO_SANDBOX=1) or pip install cairosvg")
    tiles = [("asset", src)] if src is not None else []
    for t, f in zip(times, frames):
        base = Image.new("RGBA", f.size, bg)
        base.alpha_composite(f)
        tiles.append((f"t={t:.2f}", base))
    if frames:
        sheet([tiles[i:i + 5] for i in range(0, len(tiles), 5)], out / "keys_review.png", T=220)
    else:
        (out / "keys_review.png").unlink(missing_ok=True)
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


def _import_vec(imported):
    vb, defs, layers = imported
    return {"viewBox": vb, "defs": defs, "static_layers": layers, "animated_layers": [], "glows": [],
            "layer_order": [L["name"] for L in layers]}


def _contrast_bg(img, dark="#1b1d33"):
    """White when the drawn pixels are dark (alpha-weighted mean luminance < 0.5), else `dark`: a black outline icon
    must not be shown to the model on navy at 1.3:1 contrast."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    w = arr[..., 3] / 255
    if w.sum() < 1:
        return dark
    lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) / 255
    return "#FFFFFF" if float((lum * w).sum() / w.sum()) < 0.5 else dark


def _check_vec(vec):
    """_canon_vec + _validate_vec without repeating a colour error in two wordings."""
    cerrs = _canon_vec(vec)
    heads = {e.split(" is not")[0] for e in cerrs}
    return cerrs + [e for e in _validate_vec(vec) if e.split(" is not")[0] not in heads]


def _ask_vector(a, mock, asset, src, imported, bg):
    thr = a.min_motion if a.min_motion is not None else MOTION_MIN[a.strength]
    parts = [VECTOR_PROMPT.format(moving=a.moving, static=a.static or "nothing", k=a.keyframes, seconds=a.seconds,
                                  motion=a.motion or "a gentle natural loop", strength=MOTION_STRENGTH[a.strength])]
    if imported:
        parts.append(f"The asset already exists as vector static_layers (viewBox \"{imported[0]}\"). Return each with exactly "
                     "its name and paths: [] (the geometry is restored from the source) and add the motion, glows and any "
                     "extra layers the animation needs:\n" + json.dumps(imported[2]))
    if src is not None:
        flat = Image.new("RGBA", src.size, _contrast_bg(src, bg))
        flat.alpha_composite(src)
        parts.append(flat)
    if asset and mock is not None:
        parts += ["For reference, how it appears in the approved design:", ref_crop(mock, asset["bbox"], 0.3, 512)]
    kit_errs = set(_validate_vec(_import_vec(imported))) if imported else set()  # never blame the model for the import
    imported_names = {L["name"] for L in imported[2]} if imported else set()
    best, vec = None, None
    for attempt in range(2):
        vec = ask(parts, VECTOR_SCHEMA, "vector_layers")
        raw = json.loads(json.dumps(vec))
        if imported:
            _merge_imported(vec, imported)
        errs = [e for e in _check_vec(vec) if e not in kit_errs]
        rep = None if errs else motion_report(_normalize_vec(vec))
        weak = rep is not None and (rep["motion"] is None or rep["motion"] < thr)
        if rep and (best is None or (rep["motion"] or 0) > (best[1]["motion"] or 0)):
            best = (vec, rep)
        if rep and not weak or attempt:
            break
        per = ", ".join(f"{n} {r['move']:.1%} move / {r['fade']:.0%} fade" for n, r in rep["layers"].items() if r["kind"] != "glow") if rep else ""
        fb = (f"Your previous answer was invalid: {'; '.join(errs[:6])}. Fix it and follow every rule." if errs else
              "Your previous answer animates nothing: no layer has motion keys and there are no animated_layers. Add the motion "
              f"that was asked for ({a.moving})." if rep["motion"] is None else
              f"Your previous answer moves too little: its animated layers move {rep['motion']:.1%} of their size on "
              f"average ({per}), at least {thr:.0%} is needed. Keep the geometry but exaggerate the keyframes and motion keys.")
        print("  retrying:", fb, flush=True)
        for sl in raw.get("static_layers") or []:  # the model's own answer, imported geometry elided as it was asked
            if isinstance(sl, dict) and _res_name(str(sl.get("name") or ""), "layer_") in imported_names:
                sl["paths"] = []
        parts = parts + [fb + "\nPrevious answer:\n" + json.dumps(raw)]
    return best[0] if best else vec


def _lottie_lint(lot):
    """Portability checks on a built Lottie (a failure is a kit bug): no merge paths (Android/Windows only), no
    expressions (web only), every group ends with its transform, shape layers span the whole composition (image layers may take a window of it)."""
    errs = []

    def walk(items, where):
        for it in items:
            if it.get("ty") == "mm":
                errs.append(f"{where}: merge path (mm) is not rendered by lottie-ios or web players")
            if it.get("ty") == "gr":
                if not it.get("it") or it["it"][-1].get("ty") != "tr":
                    errs.append(f"{where}: group does not end with a transform")
                walk(it["it"], where)
            for k, v in it.items():
                if isinstance(v, dict) and isinstance(v.get("x"), str):
                    errs.append(f"{where}: expression in {k} (web only)")
    if not (isinstance(lot.get("op"), (int, float)) and lot["op"] > lot.get("ip", 0)):
        errs.append("op must be after ip")
    for lyr in lot.get("layers", []):
        walk(lyr.get("shapes", []), lyr.get("nm", "?"))
        for k, v in (lyr.get("ks") or {}).items():
            if isinstance(v, dict) and isinstance(v.get("x"), str):
                errs.append(f"{lyr.get('nm')}: expression in ks.{k} (web only)")
        if lyr.get("ty") == 2 and 0 <= lyr.get("ip", -1) < (lyr.get("op") or 0) <= lot["op"]:
            continue  # an image layer may show for a window of the composition (frames' image-sequence Lottie)
        if lyr.get("ip", 0) != 0 or lyr.get("op") != lot["op"]:
            errs.append(f"{lyr.get('nm')}: layer ip/op {lyr.get('ip')}/{lyr.get('op')} differ from the composition 0/{lot['op']}")
    return errs


def _aapt2_report(xml_path, what="AVD"):
    """Run aapt2 on one drawable XML and print the verdict per contract C4 (never 'valid' for compile-only)."""
    res = aapt2_check([xml_path])
    if res:
        for f, e in res.items():
            print(f"  ! aapt2: {f}: {e}")
    else:  # None (no aapt2) and {} (every available step passed) are both one honest line
        print(f"  {_aapt2_line()}")
    return res


def cmd_vectorize(a):
    d = Path(a.dir)
    if (d / "assets.json").exists():
        d, m, mock = _load(a.dir)
    elif a.svg or a.rebuild:
        m, mock = {"assets": []}, None
    else:
        sys.exit(f"vectorize: {d / 'assets.json'} not found: run manifest first, or pass --svg FILE (no design dir needed)")
    if not 2 <= a.keyframes <= 24:
        sys.exit(f"vectorize: --keyframes {a.keyframes} must be between 2 and 24 (the model writes every keyframe)")
    out = d / "vector" / a.name
    asset = next((x for x in m.get("assets", []) if isinstance(x, dict) and x.get("name") == a.name), None)
    png = d / "assets" / f"{a.name}.png"
    svg = _need(a.svg, "SVG", "pass --svg FILE") if a.svg else d / "svg" / f"{a.name}.svg"
    src, imported = None, None
    if png.exists() and not a.svg:
        src = _open_image(png, "asset")
    elif svg.exists():
        try:
            imported = _svg_layers(svg.read_text(), a.name, svg.name)
        except ET.ParseError as e:
            sys.exit(f"vectorize: {svg}: not valid SVG XML ({e})")
        except UnicodeDecodeError as e:  # a PNG passed as --svg
            sys.exit(f"vectorize: {svg} is not a text SVG file ({e.reason}); pass the .svg, or the PNG as DIR/assets/NAME.png")
        except (ValueError, IndexError) as e:
            sys.exit(f"vectorize: {e}" + ("" if a.rebuild else " (no API call was made)"))
        if not a.rebuild:
            if not imported[2]:
                sys.exit(f"vectorize: {svg}: no drawable shapes (path, circle, rect, ...) found; nothing to animate (no API call was made)")
            errs = _validate_vec(_import_vec(imported))
            if errs:
                sys.exit(f"vectorize: {svg}: {errs[0]} (fix the SVG; no API call was made)")
        out.mkdir(parents=True, exist_ok=True)
        src = _svg_image(svg, out / "source.png")
    elif not a.rebuild:
        sys.exit(f"vectorize: no source: neither {png} nor {svg} exists (pass --svg FILE)")
    out.mkdir(parents=True, exist_ok=True)
    lp = out / "layers.json"
    if a.rebuild:
        if not lp.exists():
            sys.exit(f"vectorize: {lp} not found: run without --rebuild first")
        try:
            vec = json.loads(lp.read_text())
        except json.JSONDecodeError as e:
            sys.exit(f"vectorize: {lp}: invalid JSON at line {e.lineno}: {e.msg}")
        errs = _check_vec(vec)
        if errs:
            sys.exit(f"vectorize: {lp} invalid: {errs[0]}; edit it and run --rebuild again")
    elif not a.moving:
        sys.exit("vectorize: --moving is required (or --rebuild to reuse an edited layers.json)")
    else:
        vec = _ask_vector(a, mock, asset, src, imported, a.preview_bg)
        errs = _check_vec(vec)
        lp.write_text(json.dumps(vec, indent=2))  # keep the answer for hand editing either way
        if errs:
            sys.exit(f"vectorize: the model's answer is still invalid after a retry: {errs[0]}; edit {lp} and run --rebuild")
    notes = []
    before = json.dumps(vec, sort_keys=True)
    _normalize_vec(vec, notes)
    if a.rebuild and json.dumps(vec, sort_keys=True) != before and not (out / "layers.orig.json").exists():
        (out / "layers.orig.json").write_text(lp.read_text())
        notes.append(f"layers.json normalised (paths as absolute M/L/C/Z, rounded); the file you edited is kept as layers.orig.json")
    for n in notes:
        print(f"  ({n})")
    lp.write_text(json.dumps(vec, indent=2))
    rep = motion_report(vec)
    if rep["motion"] is None and not rep["layers"]:
        if a.rebuild:
            print("  ! nothing moves: no layer has motion keys and there are no animated layers (the outputs are static)")
        else:
            sys.exit(f"vectorize: the model animated nothing (no motion keys, no animated layers) even after a retry; edit {lp} and "
                     "run --rebuild, or describe --moving/--motion more concretely")
    thr = a.min_motion if a.min_motion is not None else MOTION_MIN[a.strength]
    fit = None if a.no_fit else _fit_viewbox(vec)
    built = dict(vec, viewBox=fit) if fit else vec
    lot = build_lottie(built, a.size, a.seconds, a.fps)
    lint = _lottie_lint(lot)
    if lint:
        sys.exit(f"vectorize: internal error, the Lottie failed its portability check: {lint[0]} (please report)")
    (out / f"{a.name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    dp = a.dp or ((asset or {}).get("dp") or {}).get("w") or 48
    dp = round(dp * _vec_vb(built)[2] / _vec_vb(vec)[2], 1)
    avd = out / f"{_res_name(a.name)}_avd.xml"
    if avd.stem != f"{a.name}_avd":
        print(f"  (Android resource name: {avd.name})")
    avd.write_text(build_avd(built, a.seconds, dp))
    _aapt2_report(avd)
    js = _lottie_js()  # inline the cached player so the preview works offline; the CDN is the fallback
    (out / "preview.html").write_text(
        "<!doctype html><meta charset=utf-8><body style='margin:0;background:" + a.preview_bg + "'>"
        "<div id=a style='width:512px;height:512px'></div>"
        + (f"<script>{js}</script>" if js else f"<script src='{LOTTIE_JS}'></script>")
        + f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{_js_json(lot)}}})</script>")
    how, px = _review_sheet(built, lot, src, out, a.preview_bg, a.review_frames)
    for n, r in rep["layers"].items():
        print(f"  {n:24s} {r['kind']:6s} move={r['move']:.1%} fade={r['fade']:.2f} score={r['score']:.1%}")
    mm = rep["motion"]
    print(f"  motion {'-' if mm is None else f'{mm:.1%}'} (min {thr:.0%} for --strength {a.strength}; area-weighted, a full fade counts 25%)" +
          (" TOO SUBTLE: re-run with --strength strong or a stronger --motion" if mm is not None and mm < thr else "") +
          (f"; rendered frames change up to {px:.0%} of the drawn area" if px is not None else ""))
    for L in _layers(vec):
        if L["kind"] == "static" and len(L["paths"]) > 1 and rep["layers"].get(L["name"], {}).get("fade", 0) > 0:
            print(f"  ~ {L['name']} fades with {len(L['paths'])} overlapping paths: the AVD fades each path separately (no group alpha)")
    if abs(lot["op"] / a.fps - a.seconds) > 1e-6:
        print(f"  (Lottie: {lot['op']} frames at {lot['fr']:g} fps so the loop is exactly {a.seconds:g} s like the AVD's {round(a.seconds * 1000)} ms)")
    if a.seconds > 5:
        print("  ! loop longer than 5 s: the RenderThread AVD subsamples keyframed animators past ~300 frames "
              "(\"Duration for the animation is too long\"); prefer the Lottie or a shorter loop")
    if fit:
        print(f"  motion leaves the viewBox: canvas grown to \"{fit}\" (AVD {dp:g}dp wide; show the Lottie at {dp:g}dp too, "
              f"the {a.size}px canvas now holds the grown viewBox) so nothing clips")
    print(f"-> {out}/layers.json, {a.name}.json (Lottie), {avd.name} (AnimatedVectorDrawable), keys_review.png ({how}), preview.html")
