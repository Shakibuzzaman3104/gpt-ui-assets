"""Fonts: Google Fonts matching."""
import concurrent.futures as cf
import functools
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

from .core import _res_name
from .vectordrawable import _aapt2_tree
from .measure import _d2c_runs, _hexi, _text_targets
from .tokens import _d2c_load_spec

# ---- fonts: match typography to Google Fonts by rendering and measuring

GF_METADATA = "https://fonts.google.com/metadata/fonts"
GF_WEBFONTS = "https://www.googleapis.com/webfonts/v1/webfonts"
GF_HOSTS = ("fonts.google.com (family list; or the Developer API at www.googleapis.com with GOOGLE_FONTS_API_KEY), "
            "fonts.googleapis.com + fonts.gstatic.com (TTF downloads), raw.githubusercontent.com (licence texts)")
_GENERIC_WORDS = {"display", "text", "sans", "serif", "italic", "mono", "pro", "regular", "bold", "medium", "light", "condensed",
                  "similar", "alternatives", "transitional", "contrast", "high", "slanted", "lightly", "rounded", "humanist",
                  "geometric", "grotesque", "neo", "system", "font", "family", "style", "classic", "modern", "elegant"}
_GF_TTF_ERRORS = {}  # (family, weight, italic) -> why the last download attempt failed


def _cache_dir():
    return Path(os.environ.get("GPTKIT_CACHE", "~/.cache/gptkit")).expanduser()


def _redact_key(s):
    return re.sub(r"(key=)[^&\s]+", r"\1***", str(s))


def _gf_fetch_meta():
    """Fresh family metadata as (list, source): the fonts.google.com list, else the Developer API (GOOGLE_FONTS_API_KEY)
    mapped to the same shape (family, category, subsets, fonts {"400": {}, "700i": {}}, popularity)."""
    errors = []
    try:
        r = requests.get(GF_METADATA, timeout=60)
        r.raise_for_status()
        t = r.text[4:] if r.text.startswith(")]}'") else r.text
        fams = json.loads(t)["familyMetadataList"]
        if not isinstance(fams, list) or not fams or not all(isinstance(f, dict) and f.get("family") for f in fams[:20]):
            raise ValueError("familyMetadataList is not a family list")
        return fams, "fonts.google.com"
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        errors.append(f"{GF_METADATA}: {_redact_key(e)[:160]}")
    key = os.environ.get("GOOGLE_FONTS_API_KEY")
    if key:
        cat = {"sans-serif": "Sans Serif", "serif": "Serif", "display": "Display", "handwriting": "Handwriting", "monospace": "Monospace"}
        try:
            r = requests.get(GF_WEBFONTS, params={"key": key, "sort": "popularity"}, timeout=60)
            r.raise_for_status()
            fams = []
            for i, it in enumerate(r.json()["items"]):
                fonts = {}
                for v in it.get("variants", []):
                    m_ = re.fullmatch(r"(\d{3})?(italic)?", str(v).replace("regular", ""))
                    if m_:
                        fonts[(m_.group(1) or "400") + ("i" if m_.group(2) else "")] = {}
                fams.append({"family": it["family"], "category": cat.get(it.get("category"), it.get("category")),
                             "subsets": it.get("subsets", []), "fonts": fonts, "popularity": i + 1,
                             "isNoto": str(it["family"]).startswith("Noto"), "isOpenSource": True, "classifications": [], "stroke": None})
            if not fams:
                raise ValueError("no items")
            return fams, "webfonts/v1"
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            errors.append(f"{GF_WEBFONTS}: {_redact_key(e)[:160]}")
    else:
        errors.append("Developer API not tried (set GOOGLE_FONTS_API_KEY)")
    raise RuntimeError("could not load the Google Fonts family list (" + "; ".join(errors) + f"); needs network to {GF_HOSTS} once, "
                       "then it is cached for 7 days")


@functools.cache
def _gf_meta():
    """Google Fonts family metadata: the validated cache (refreshed after 7 days when reachable, kept when not), else a
    fresh fetch written atomically. A corrupt cache is deleted and refetched; offline, RuntimeError names the file."""
    p = _cache_dir() / "gf_metadata.json"
    cached, corrupt = None, None
    if p.exists():
        try:
            data = json.loads(p.read_text())
            cached = data["familyMetadataList"]
            if not isinstance(cached, list) or not cached or not all(isinstance(f, dict) and f.get("family") for f in cached[:20]):
                raise ValueError("familyMetadataList is not a family list")
        except (OSError, ValueError, KeyError, TypeError) as e:
            corrupt, cached = f"{type(e).__name__}: {str(e)[:80]}", None
    if cached and time.time() - p.stat().st_mtime <= 7 * 86400:
        return cached
    try:
        fams, src = _gf_fetch_meta()
    except RuntimeError as e:
        if cached:
            print("  (Google Fonts unreachable: using the cached metadata)")
            return cached
        if corrupt:
            p.unlink(missing_ok=True)
            raise RuntimeError(f"cached Google Fonts metadata at {p} was corrupt ({corrupt}; file deleted) and the network is "
                               f"unavailable ({e}); re-run online")
        raise
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps({"familyMetadataList": fams, "source": src, "fetched": time.strftime("%Y-%m-%d")}))
    tmp.replace(p)
    return fams


def _gf_ttf(family, weight, italic=False):
    """Static TTF of one Google Fonts family/weight via the CSS API (cached); None when it cannot be downloaded, with the
    reason kept in _GF_TTF_ERRORS[(family, weight, italic)]."""
    p = _cache_dir() / "fonts" / re.sub(r"\W+", "_", family) / f"{weight}{'i' if italic else ''}.ttf"
    if p.exists():
        return p
    key = (family, weight, italic)
    url = f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:ital,wght@{int(italic)},{weight}"
    try:
        css = requests.get(url, headers={"User-Agent": "gptkit"}, timeout=30)
        if not css.ok:
            _GF_TTF_ERRORS[key] = f"HTTP {css.status_code} from fonts.googleapis.com"
            return None
        m_ = re.search(r"url\((https://[^)]+\.ttf)\)", css.text)
        if not m_:
            _GF_TTF_ERRORS[key] = "no TTF url in the css2 response (weight not available?)"
            return None
        r = requests.get(m_.group(1), timeout=60)
        if not r.ok:
            _GF_TTF_ERRORS[key] = f"HTTP {r.status_code} from fonts.gstatic.com"
            return None
    except requests.RequestException as e:
        _GF_TTF_ERRORS[key] = f"{type(e).__name__}: {str(e)[:120]}"
        return None
    if not r.content[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
        _GF_TTF_ERRORS[key] = "downloaded file is not a TrueType/OpenType font"
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.{id(r)}.part")
    tmp.write_bytes(r.content)
    tmp.replace(p)
    _GF_TTF_ERRORS.pop(key, None)
    return p


def _gf_license(family):
    """{license, dir, text} from the google/fonts repository METADATA.pb (+ the licence file). Only complete lookups are
    cached, so an offline run never pins 'unknown' for a family."""
    slug = re.sub(r"[^a-z0-9]", "", family.lower())
    cache = _cache_dir() / "licenses" / f"{slug}.json"
    if cache.exists():
        try:
            got = json.loads(cache.read_text())
            if got.get("dir") and got.get("text"):
                return got
        except (OSError, ValueError):
            pass
    res = {"license": "unknown (see fonts.google.com)", "dir": None, "text": None}
    for lic_dir, fname in (("ofl", "OFL.txt"), ("apache", "LICENSE.txt"), ("ufl", "UFL.txt")):
        base = f"https://raw.githubusercontent.com/google/fonts/main/{lic_dir}/{slug}/"
        try:
            r = requests.get(base + "METADATA.pb", timeout=20)
            if not r.ok:
                continue
            m_ = re.search(r'license:\s*"([^"]+)"', r.text)
            t = requests.get(base + fname, timeout=20)
            res = {"license": m_.group(1) if m_ else lic_dir.upper(), "dir": f"{lic_dir}/{slug}", "text": t.text if t.ok else None}
        except requests.RequestException as e:
            res["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        break
    if res.get("dir") and res.get("text"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.part")
        tmp.write_text(json.dumps(res))
        tmp.replace(cache)
    return res


def _gf_family(name, what):
    """The metadata entry whose family name matches `name` case-insensitively; exits with close matches otherwise."""
    import difflib
    meta = _gf_meta()
    hit = next((f for f in meta if f["family"].lower() == name.strip().lower()), None)
    if hit:
        return hit
    close = difflib.get_close_matches(name.strip(), [f["family"] for f in meta], n=4, cutoff=0.6)
    sys.exit(f"fonts: {what} '{name}' is not a Google Fonts family" + (f" (close: {', '.join(close)})" if close else ""))


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


def _render_line(font, text, track_px, kern=None, ss=4, stroke=None):
    """Coverage map of one text line (tight, supersampled) plus each character's [x0, x1) advance span in it.
    `stroke` is the stem darkening in quarter pixels (default FONT_EMBOLDEN, calibrated against Chrome/Skia text)."""
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
        dr.text((x, pad + asc), ch, font=big, fill=255, anchor="ls", stroke_width=FONT_EMBOLDEN if stroke is None else stroke,
                stroke_fill=255)
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
    words = [(m_.start(), m_.end() - 1) for m_ in re.finditer(r"\S+", text)]
    best = None
    for stroke in sorted({FONT_EMBOLDEN, 0}):  # the mockup's renderer may or may not darken stems: keep the better fit
        r, spans = _render_line(font, text, track * size, kern, stroke=stroke)
        r = r ** FONT_GAMMA
        sim, word_mode = _shape_similarity(target, r, text, metric, [(spans[i][0], spans[j][1]) for i, j in words])
        if best is None or sim > best[0]:
            best = (sim, word_mode, r, stroke)
    sim, word_mode, r, stroke = best
    asc, desc = font.getmetrics()
    return {"similarity": round(sim, 4), "size_px": round(size, 1), "tracking_em": round(track, 3), "embolden": stroke,
            "word_mode": word_mode, "natural_line_height_em": round((asc + desc) / size, 3), "ascent_em": round(asc / size, 3),
            "render": r}


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


_CSS_FALLBACK = {"Serif": "Georgia, 'Times New Roman', serif", "Sans Serif": "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
                 "Display": "system-ui, sans-serif", "Handwriting": "cursive", "Monospace": "ui-monospace, Menlo, Consolas, monospace"}


_WOFF2_WHY = "not attempted"


def _woff2(ttf):
    """Write <name>.woff2 next to a TTF when fontTools and brotli are importable; None (with the reason in _WOFF2_WHY) otherwise."""
    global _WOFF2_WHY
    try:
        import brotli  # noqa: F401
        from fontTools.ttLib import TTFont
    except ImportError as e:
        _WOFF2_WHY = f"{e.name or 'fonttools/brotli'} not installed (pip install fonttools brotli)"
        return None
    out = Path(ttf).with_suffix(".woff2")
    try:
        f = TTFont(str(ttf))
        f.flavor = "woff2"
        f.save(str(out))
    except Exception as e:  # any fontTools/brotli failure: keep the TTF and say why
        _WOFF2_WHY = f"{type(e).__name__}: {str(e)[:80]}"
        return None
    return out


def _fonts_collect_group(gkey, members, extras, forced, force_all, a, texts, k):
    """Collect a group's font candidates and download its renderable TTFs."""
    italic = gkey.endswith("|italic")
    guess = members[0].get("family_guess", "")
    wts = sorted({st.get("weight", 400) for st in members})
    display_ok = any(st["size_px"] * k >= 20 for st in members)
    cands, cat = _font_shortlist(guess, wts[len(wts) // 2], italic, a.candidates, display_ok)
    cands += [f for f in extras if f not in cands]
    if forced and forced not in cands and (forced["category"] == cat or force_all):
        cands.append(forced)
    print(f"group '{guess[:50]}' ({len(members)} styles, {cat}): {len(cands)} candidate families")
    need = sorted({(f["family"], w) for f in cands for st in members
                   for w in sorted((int(x.rstrip("i")) for x in f["fonts"] if x.endswith("i") == italic),
                                   key=lambda x: abs(x - st.get("weight", 400)))[:a.weights]})
    with cf.ThreadPoolExecutor(8) as ex:
        paths = dict(zip(need, ex.map(lambda fw: _gf_ttf(fw[0], fw[1], italic), need)))
    failed = [fw for fw in need if paths[fw] is None]
    return {"italic": italic, "guess": guess, "cands": cands, "cat": cat, "need": need,
            "paths": paths, "failed": failed,
            "last_err": _GF_TTF_ERRORS.get((failed[-1][0], failed[-1][1], italic), "") if failed else ""}


def _fonts_match_group(members, texts, cands, italic, paths, a):
    """Render and score each style against the downloaded candidate fonts."""
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(zip([m["role"] for m in members],
                        ex.map(lambda st: _match_style(st, texts[st["role"]], cands, italic, a.weights, paths), members)))


def _fonts_rank_group(members, texts, per):
    """Rank candidate families using reliable, area-weighted style matches."""
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
    return reliable, ranking


def _fonts_store_style(st, t, rs, winner, italic, cands, guess, rgb, show, results, sheet_rows):
    """Store one measured typography style and add its visual review row."""
    mine = [r for r in rs if r[0] == winner]
    if not mine:
        return
    fam, w, fit = mine[0]
    tie = None
    spec_w = st.get("weight")
    alt = next((r for r in mine if r[1] == spec_w), None)
    steps = sorted({r[1] for r in mine})
    if alt and alt[1] != w and abs(steps.index(alt[1]) - steps.index(w)) == 1 and alt[2]["similarity"] >= fit["similarity"] - 0.07:
        tie = {"measured": w, "chosen": spec_w, "similarity": [fit["similarity"], alt[2]["similarity"]]}
        fam, w, fit = alt
    ok = rs[0][2]["similarity"] >= 0.45 and -0.15 < fit["tracking_em"] < 0.6
    avail = [int(x.rstrip("i")) for f in cands if f["family"] == winner for x in f["fonts"] if x.endswith("i") == italic]
    spec_w = min(avail, key=lambda x: abs(x - st.get("weight", w))) if avail else w
    results[st["role"]] = {"role": st["role"], "family": winner, "weight": w if ok else spec_w, "italic": italic,
                           "size_px": fit["size_px"] if ok else None,
                           "tracking_em": fit["tracking_em"] if ok else st.get("letter_spacing_em", 0),
                           "reliable": ok, "weight_tie": tie,
                           "similarity": fit["similarity"], "word_mode": fit["word_mode"],
                           "natural_line_height_em": fit["natural_line_height_em"], "ascent_em": fit["ascent_em"],
                           "spec": {"size_px": st["size_px"], "weight": st.get("weight"),
                                    "letter_spacing_em": st.get("letter_spacing_em"), "family_guess": guess},
                           "alternatives": [[r[0], r[1], r[2]["similarity"]] for r in rs[:6]]}
    if not ok:
        results[st["role"]]["warning"] = (f"unreliable measurement (best fit {rs[0][2]['similarity']:.2f}); kept the spec's "
                                           "size/weight/tracking; check its box in text_boxes.png")
    elif rs[0][0] != winner and rs[0][2]["similarity"] > fit["similarity"] + 0.1:
        results[st["role"]]["warning"] = (f"fits {winner} poorly ({fit['similarity']:.2f}); {rs[0][0]} {rs[0][1]} fits this "
                                           f"style much better ({rs[0][2]['similarity']:.2f})")
    if tie:
        results[st["role"]]["note"] = (f"weights {tie['measured']} and {tie['chosen']} tie within 0.07 "
                                        f"({tie['similarity'][0]:.3f} vs {tie['similarity'][1]:.3f}); kept the spec's {tie['chosen']}")
    bg, ink = np.array(_hexi(t["bg"]), float), np.array(_hexi(t["ink"]), float)
    x, y, w_, h_ = t["box"]
    tiles = [("DESIGN (mockup pixels)", rgb.crop((x - 4, y - 4, x + w_ + 4, y + h_ + 4)))]
    seen = set()
    for fam_, wt, ft in [mine[0]] + [r for r in rs if r is not mine[0]]:
        if (fam_, wt) in seen or len(tiles) > show:
            continue
        seen.add((fam_, wt))
        col = (bg + ft["render"][..., None] * (ink - bg)).astype(np.uint8)
        tile = Image.new("RGB", (col.shape[1] + 8, col.shape[0] + 8), tuple(int(v) for v in bg))
        tile.paste(Image.fromarray(col), (4, 4))
        star = "*" if fam_ == winner and wt == w else ""
        tiles.append((f"{star}{fam_} {wt} sim {ft['similarity']:.3f}", tile))
    sheet_rows.append((f"{st['role']} ({t['sample'][:18]})", tiles))


def _fonts_write_family(fam, rs, out, meta):
    """Download one selected family and write its Android font resources and licence."""
    ttf_dir, res_font, lic_dir = out / "ttf", out / "android" / "res" / "font", out / "licenses"
    lic = _gf_license(fam)
    files, ps, fetch_failed = [], {}, []
    for w, it in sorted({(r["weight"], r["italic"]) for r in rs}):
        src = _gf_ttf(fam, w, it)
        if not src:
            fetch_failed.append(f"{fam} {w}{' italic' if it else ''}")
            print(f"  ! could not download {fam} {w}{' italic' if it else ''} ({_GF_TTF_ERRORS.get((fam, w, it), '')}); skipped")
            continue
        psn, full = _ttf_names(src)
        fname = (psn or fam.replace(" ", "") + f"-{w}") + ".ttf"
        ttf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, ttf_dir / fname)
        wname = {100: "thin", 200: "extralight", 300: "light", 400: "regular", 500: "medium", 600: "semibold", 700: "bold",
                 800: "extrabold", 900: "black"}.get(w, str(w))
        android = _res_name(f"{fam}_{wname}{'_italic' if it else ''}", "f_")
        res_font.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, res_font / f"{android}.ttf")
        entry = {"weight": w, "italic": it, "ttf": str(Path("ttf") / fname), "android": android, "postscript": psn}
        woff2 = _woff2(ttf_dir / fname)
        if woff2:
            entry["woff2"] = str(Path("ttf") / woff2.name)
        files.append(entry)
        ps[f"{w}{'i' if it else ''}"] = psn
    if not files:
        return None, fetch_failed
    xml_name = _res_name(fam, "f_")
    if xml_name != re.sub(r"[^a-z0-9]+", "_", fam.lower()).strip("_"):
        print(f"  (Android resource name for {fam}: {xml_name})")
    items = "".join(f'    <font\n        android:font="@font/{f["android"]}"\n        android:fontStyle="'
                    f'{"italic" if f["italic"] else "normal"}"\n        android:fontWeight="{f["weight"]}"\n'
                    f'        app:font="@font/{f["android"]}"\n        app:fontStyle="{"italic" if f["italic"] else "normal"}"\n'
                    f'        app:fontWeight="{f["weight"]}" />\n' for f in files)
    (res_font / f"{xml_name}.xml").write_text('<?xml version="1.0" encoding="utf-8"?>\n<font-family xmlns:android='
                                               '"http://schemas.android.com/apk/res/android"\n    xmlns:app='
                                               f'"http://schemas.android.com/apk/res-auto">\n{items}</font-family>\n')
    if lic.get("text"):
        lic_dir.mkdir(parents=True, exist_ok=True)
        (lic_dir / (re.sub(r"\W+", "", fam) + f"-{lic['license']}.txt")).write_text(lic["text"])
    else:
        print(f"  ! license text for {fam} not fetched ({lic.get('error', 'not found on github.com/google/fonts')}); "
              "add it before shipping (the OFL requires the licence and copyright notice to accompany the fonts)")
    cat = next((f.get("category") for f in meta if f["family"] == fam), None)
    wts = sorted({f["weight"] for f in files})
    has_italic = any(f["italic"] for f in files)
    axis = ("ital,wght@" + ";".join(f"{int(f['italic'])},{f['weight']}" for f in sorted(files, key=lambda f: (f["italic"], f["weight"])))
            if has_italic else "wght@" + ";".join(map(str, wts)))
    family = {"family": fam, "category": cat, "license": lic["license"], "source": lic.get("dir"), "files": files,
              "postscript": ps, "android_family": f"@font/{xml_name}",
              "css": f"https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}:{axis}&display=swap",
              "css_fallback": _CSS_FALLBACK.get(cat, "sans-serif")}
    return family, fetch_failed


def _fonts_write_web(families, out):
    """Write web font CSS, CDN link markup, the iOS plist, and Android resource checks."""
    web = out / "web"
    web.mkdir(parents=True, exist_ok=True)
    stack = "".join(f"/* {f['family']}: font-family: \"{f['family']}\", {f['css_fallback']}; */\n" for f in families)
    src_of = lambda x: (f"url(\"../{x['woff2']}\") format(\"woff2\"), " if x.get("woff2") else "") + \
        f"url(\"../{x['ttf']}\") format(\"truetype\")"
    (web / "fonts.css").write_text(stack + "".join(
        f"@font-face {{\n  font-family: \"{f['family']}\";\n  font-style: {'italic' if x['italic'] else 'normal'};\n"
        f"  font-weight: {x['weight']};\n  font-display: swap;\n  src: {src_of(x)};\n}}\n"
        for f in families for x in f["files"]))
    if families and not any(x.get("woff2") for f in families for x in f["files"]):
        print(f"  (WOFF2 skipped: {_WOFF2_WHY}; web/fonts.css serves the TTFs)")
    link_html = ""
    if families:
        href = "https://fonts.googleapis.com/css2?" + "&".join(
            f["css"].split("?", 1)[1].replace("&display=swap", "") for f in families) + "&display=swap"
        link_html = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" '
                     f'crossorigin>\n<link href="{href}" rel="stylesheet">')
        (web / "fonts-link.html").write_text(link_html + "\n")
    ios = out / "ios"
    ios.mkdir(parents=True, exist_ok=True)
    import plistlib
    (ios / "UIAppFonts.plist").write_bytes(plistlib.dumps({"UIAppFonts": [Path(x["ttf"]).name for f in families for x in f["files"]]}))
    aapt2_msg = _aapt2_tree(out / "android" / "res", extra_res={"values/gptkit_fontfamily_attrs.xml": (
        '<resources><declare-styleable name="FontFamilyFont"><attr name="font" format="reference"/>'
        '<attr name="fontStyle"><enum name="normal" value="0"/><enum name="italic" value="1"/></attr>'
        '<attr name="fontWeight" format="integer"/></declare-styleable></resources>')}) if families else None
    if aapt2_msg and aapt2_msg.startswith("aapt2 compile+link ok"):
        aapt2_msg += " (app: attributes against androidx.core's FontFamilyFont declaration)"
    return link_html, aapt2_msg, ios


def _fonts_save_manifest(out, families, report_groups, results, skipped, fetch_failed, link_html, aapt2_msg):
    """Write the complete fonts.json manifest atomically."""
    manifest = {"families": families, "groups": report_groups, "styles": list(results.values()), "unmatched": skipped,
                "downloads_failed": fetch_failed, "link_html": link_html, "aapt2": aapt2_msg, "paths_relative_to": str(out)}
    tmp = (out / "fonts.json").with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out / "fonts.json")


def _fonts_print_report(out, ios, families, results, aapt2_msg):
    """Print the measured styles, selected files, and setup instructions."""
    print("\nstyle                           family                 wt  size px (spec)   tracking em (spec)  sim")
    for r in results.values():
        sp = r.get("spec", {})
        print(f"  {r['role'][:30]:30s} {r['family'][:20]:20s} {r['weight']:>4} {r['size_px'] or '-':>6} ({sp.get('size_px', '-')!s:>4})"
              f"   {r['tracking_em']:>7} ({sp.get('letter_spacing_em', '-')!s:>6})   {r['similarity']:.3f}")
    for r in results.values():
        for key in ("warning", "note"):
            if r.get(key):
                print(f"  {'!' if key == 'warning' else '~'} {r['role']}: {r[key]}")
    for f in families:
        print(f"{f['family']}: {f['license']} ({f['source']}); files: " + ", ".join(Path(x["ttf"]).name for x in f["files"]))
    if aapt2_msg:
        print(f"  android/res/font: {aapt2_msg}")
    print(f"-> {out}/fonts.json, fonts_review.png, ttf/, android/res/font/, ios/UIAppFonts.plist, web/fonts.css, web/fonts-link.html, licenses/")
    print(f"iOS: add ttf/*.ttf to the app target (Build Phases > Copy Bundle Resources, or the SwiftPM target's resources), then\n"
          f"     /usr/libexec/PlistBuddy -c \"Merge {ios / 'UIAppFonts.plist'}\" App/Info.plist   (use Font.custom(<PostScript name>))\n"
          "Web: link web/fonts.css (self-hosted TTF), or paste web/fonts-link.html for the Google Fonts CDN. Re-run `tokens` to use these families.")


def _fonts_score_one_group(gkey, members, a, state):
    """Collect candidates, rank one typography group, and store its selected styles."""
    group = _fonts_collect_group(gkey, members, state["extras"], state["forced"], state["force_all"],
                                a, state["texts"], state["k"])
    italic, guess = group["italic"], group["guess"]
    cands, cat = group["cands"], group["cat"]
    need, paths, failed = group["need"], group["paths"], group["failed"]
    state["n_need"] += len(need)
    state["n_failed"] += len(failed)
    if failed:
        state["last_err"] = group["last_err"]
    if need and len(failed) == len(need):
        print(f"  ! none of the {len(need)} candidate TTFs for this group could be downloaded ({state['last_err']}); group skipped")
        state["skipped"] += [m["role"] for m in members]
        return
    if failed:
        print(f"  ! {len(failed)}/{len(need)} candidate fonts could not be downloaded: "
              + ", ".join(f"{f} {w}" for f, w in failed[:6]) + (" ..." if len(failed) > 6 else "") + f" ({state['last_err']})")
    per = _fonts_match_group(members, state["texts"], cands, italic, paths, a)
    reliable, ranking = _fonts_rank_group(members, state["texts"], per)
    if not ranking:
        best = max((per[m["role"]][0][2]["similarity"] for m in members if per[m["role"]]), default=0)
        print(f"  ! no reliable match (best similarity {best:.2f} < 0.45): likely custom lettering or a family outside "
              "the shortlist; ship it as an SVG/VectorDrawable, or pass --family/--extra. Styles: "
              + ", ".join(m["role"] for m in members))
        state["skipped"] += [m["role"] for m in members]
        return
    named = [f["family"] for f in _named_families(guess)]
    best = ranking[0][1]
    prior = next((f for f, sc in ranking if f in named and sc >= best - a.tie), None)
    forced = state["forced"]
    if forced and forced in cands and not any(f == forced["family"] for f, _ in ranking):
        print(f"  ! --family {forced['family']}: could not be measured for this group (download failed or no reliable fit); "
              "using the best measured family")
    winner = forced["family"] if forced and any(f == forced["family"] for f, _ in ranking) else (prior or ranking[0][0])
    why = "forced" if forced and winner == forced["family"] else ("spec's named family, within tie margin" if prior and prior != ranking[0][0]
                                                                  else "best measured")
    state["forced_used"] |= why == "forced"
    state["report_groups"].append({"guess": guess, "category": cat, "italic": italic, "winner": winner, "decided_by": why,
                                   "ranking": [[f, round(sc, 4)] for f, sc in ranking[:8]], "styles": [m["role"] for m in members],
                                   "downloads_failed": [f"{f} {w}" for f, w in failed]})
    print(f"  -> {winner} [{why}]   (top: " + ", ".join(f"{f} {sc:.3f}" for f, sc in ranking[:4]) + ")")
    ties = [f for f, sc in ranking[1:6] if sc >= ranking[0][1] - 0.005]
    if ties:
        state["report_groups"][-1]["tied_with"] = ties
        print(f"     measured tie with {', '.join(ties)}: the mockup can't separate them; choose by taste with --family")
    for st in members:
        _fonts_store_style(st, state["texts"][st["role"]], per[st["role"]], winner, italic, cands, guess,
                           state["rgb"], a.show, state["results"], state["sheet_rows"])


def _fonts_score_groups(a, spec, styles, extras, forced, texts, rgb, skipped):
    """Group typography styles, score each group, and enforce download-result checks."""
    groups = {}
    for st in styles:
        groups.setdefault(_font_group(st), []).append(st)
    state = {"texts": texts, "rgb": rgb, "extras": extras, "forced": forced, "skipped": skipped,
             "results": {}, "report_groups": [], "sheet_rows": [], "k": spec.get("px_to_dp", 1.0),
             "n_need": 0, "n_failed": 0, "last_err": "", "forced_used": False}
    cats = {gkey: _font_shortlist(members[0].get("family_guess", ""), 400, False, 0)[1]
            for gkey, members in groups.items()}
    state["force_all"] = bool(forced) and forced["category"] not in cats.values()
    if state["force_all"] and groups:
        print(f"  (--family {forced['family']} is {forced['category']}; no group has that category, so it is tried for every group)")
    for gkey, members in groups.items():
        _fonts_score_one_group(gkey, members, a, state)
    if state["n_need"] and state["n_failed"] == state["n_need"]:
        sys.exit(f"fonts: could not download any candidate TTF from fonts.googleapis.com/fonts.gstatic.com ({state['last_err']}); "
                 "check network/proxy, then re-run (fonts.json left untouched)")
    return state


def _fonts_prepare(a):
    """Load the design, font metadata, and selected typography styles."""
    d, spec, mock = _d2c_load_spec(a.dir)
    try:
        meta = _gf_meta()
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
        sys.exit(f"fonts: {e}")
    forced = _gf_family(a.family, "--family") if a.family else None
    extras = [_gf_family(x, "--extra") for x in (a.extra or "").split(",") if x.strip()]
    out = Path(a.out) if a.out else d / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    rgb = mock.convert("RGB")
    texts = _text_targets(d, spec, mock, a.vision)
    styles = [st for st in spec.get("typography", []) if texts.get(st["role"], {}).get("found")]
    skipped = [st["role"] for st in spec.get("typography", []) if st not in styles]
    if skipped:
        print("  ! no text box for:", ", ".join(skipped),
              "(these keep their spec values; " + ("vision couldn't find them: check the sample text)" if a.vision else "add --vision to locate them)"))
    if any(ord(ch) > 0x024F for st in styles for ch in st.get("sample", "")):
        print("  ! samples contain non-Latin characters: the matcher renders Latin glyphs only (no shaping, ligatures or RTL); "
              "results for those samples are unreliable")
    return d, spec, mock, meta, forced, extras, out, rgb, texts, styles, skipped


def cmd_fonts(a):
    d, spec, mock, meta, forced, extras, out, rgb, texts, styles, skipped = _fonts_prepare(a)
    state = _fonts_score_groups(a, spec, styles, extras, forced, texts, rgb, skipped)
    results, report_groups, sheet_rows = state["results"], state["report_groups"], state["sheet_rows"]
    skipped, forced_used = state["skipped"], state["forced_used"]
    if forced and not forced_used and results:
        print(f"  ! --family {forced['family']} was not used by any group")
    _font_sheet(sheet_rows, out / "fonts_review.png")
    families = []
    ttf_dir, res_font, lic_dir = out / "ttf", out / "android" / "res" / "font", out / "licenses"
    if (out / "fonts.json").exists():
        try:
            old = json.loads((out / "fonts.json").read_text()).get("families", [])
        except (OSError, ValueError):
            old = []
        for f in old:
            stale = [(out / x["ttf"]) if not Path(x["ttf"]).is_absolute() else Path(x["ttf"]) for x in f.get("files", []) if x.get("ttf")]
            stale += [res_font / f"{x['android']}.ttf" for x in f.get("files", []) if x.get("android")]
            stale += [res_font / f"{_res_name(f['family'], 'f_')}.xml"] + list(lic_dir.glob(re.sub(r"\W+", "", f["family"]) + "-*.txt"))
            for x in stale:
                x.unlink(missing_ok=True)
    fetch_failed, families = [], []
    for fam in sorted({r["family"] for r in results.values()}):
        rs = [r for r in results.values() if r["family"] == fam]
        family, failed = _fonts_write_family(fam, rs, out, meta)
        fetch_failed.extend(failed)
        if family:
            families.append(family)
    link_html, aapt2_msg, ios = _fonts_write_web(families, out)
    _fonts_save_manifest(out, families, report_groups, results, skipped, fetch_failed, link_html, aapt2_msg)
    _fonts_print_report(out, ios, families, results, aapt2_msg)
