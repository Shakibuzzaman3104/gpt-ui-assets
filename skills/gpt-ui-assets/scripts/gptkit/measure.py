"""Colour science and text measurement shared by tokens, fonts, compare."""
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from .core import ApiError, _clip_bbox, ask

# ---- design-to-code: colour science + text measurement (shared by tokens, fonts, compare)

_SRGB_LIN = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92, ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4)


def _lab_arr(rgb):
    """sRGB 0-255 array (...x3) -> CIELAB (D65). Integer-valued input goes through a 256-entry linearisation table."""
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        c = _SRGB_LIN[a]
    else:
        a = a.astype(np.float64, copy=False)
        r = np.rint(a)
        if a.size and np.array_equal(a, r) and 0 <= a.min() and a.max() <= 255:
            c = _SRGB_LIN[r.astype(np.uint8)]
        else:
            c = np.clip(a, 0, 255) / 255.0
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


def _norm_hex(s):
    """'#rgb', '#rgba', '#rrggbb', '#rrggbbaa', 'rgb(...)' or 'rgba(...)' -> ('#RRGGBB', alpha 0..1); ValueError otherwise."""
    t = str(s).strip()
    m_ = re.fullmatch(r"rgba?\(\s*([^)]*?)\s*\)", t, re.I)
    if m_:
        parts = [p for p in re.split(r"[,\s/]+", m_.group(1)) if p]
        if len(parts) not in (3, 4):
            raise ValueError(f"unreadable colour {s!r}; use #RRGGBB")
        try:
            chan = lambda v: float(v[:-1]) * 2.55 if v.endswith("%") else float(v)
            rgb = [int(round(min(255.0, max(0.0, chan(v))))) for v in parts[:3]]
            a = (float(parts[3][:-1]) / 100 if parts[3].endswith("%") else float(parts[3])) if len(parts) == 4 else 1.0
        except ValueError:
            raise ValueError(f"unreadable colour {s!r}; use #RRGGBB")
        return "#" + "".join(f"{v:02X}" for v in rgb), min(1.0, max(0.0, a))
    h = t[1:] if t.startswith("#") else t
    if len(h) not in (3, 4, 6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", h):
        raise ValueError(f"unreadable colour {s!r}; use #RRGGBB")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    return "#" + h[:6].upper(), (int(h[6:8], 16) / 255 if len(h) == 8 else 1.0)


def _hexi(h):
    return tuple(int(_norm_hex(h)[0][i:i + 2], 16) for i in (1, 3, 5))


def _to_hex(rgb):
    return "#" + "".join(f"{int(round(min(255, max(0, float(v))))):02X}" for v in rgb)


def _de_hex(a, b):
    return round(float(_de00(_lab_arr(_hexi(a)), _lab_arr(_hexi(b)))), 1)


def _dominant(rgb):
    px = rgb.reshape(-1, 3).astype(int)
    key = (px[:, 0] >> 4) * 256 + (px[:, 1] >> 4) * 16 + (px[:, 2] >> 4)
    return px[key == np.bincount(key).argmax()].mean(0)


def _screen_background(img):
    """Hex of a UI screen's background: the dominant colour among flat pixels (photos and gradients don't count)."""
    rgb = np.asarray(img.convert("RGB"), np.float64)
    H = rgb.shape[0]
    rgb = rgb[int(0.06 * H):int(0.9 * H)]  # status bar and navigation stay out
    g = rgb.mean(2)
    ii, ii2 = (np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1) for x in (g, g * g))
    r = 3
    box = lambda s: (s[2 * r + 1:, 2 * r + 1:] - s[:-2 * r - 1, 2 * r + 1:] - s[2 * r + 1:, :-2 * r - 1]
                     + s[:-2 * r - 1, :-2 * r - 1]) / (2 * r + 1) ** 2
    flat = np.zeros(g.shape, bool)
    flat[r:-r, r:-r] = box(ii2) - box(ii) ** 2 < 20
    px = rgb[::3, ::3][flat[::3, ::3]]
    return _to_hex(_dominant(px)) if len(px) else None


def _fill_colour(img, boxes):
    """Hex of the most common saturated colour inside the given pixel boxes, or None when they hold too little colour."""
    px = [np.asarray(img.convert("RGB").crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])), int).reshape(-1, 3)
          for b in boxes if b["w"] > 0 and b["h"] > 0]
    px = np.concatenate(px) if px else np.zeros((0, 3), int)
    lab = _lab_arr(px.astype(np.uint8))
    px = px[np.hypot(lab[:, 1], lab[:, 2]) > 20]
    if len(px) < 50:
        return None
    key = (px[:, 0] >> 3) * 1024 + (px[:, 1] >> 3) * 32 + (px[:, 2] >> 3)
    return _to_hex(np.median(px[key == np.bincount(key).argmax()], 0))


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
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["index", "text", "x", "y", "w", "h"],
                               "properties": {"index": {"type": "integer"}, "text": {"type": "string"},
                                              **{k: {"type": "integer"} for k in "xywh"}}}}}}


def _text_targets(d, spec, mock, vision=False):
    """Locate every typography sample in the mockup (spec components first, optional vision fallback)."""
    rgb = np.asarray(mock.convert("RGB"))
    area = mock.width * mock.height
    vpath = d / "text_boxes_vision.json"
    vboxes = {}
    if vpath.exists():
        try:
            for key, v in json.loads(vpath.read_text()).items():
                b = _clip_bbox(dict(zip("xywh", v)), mock.width, mock.height) if isinstance(v, (list, tuple)) and len(v) == 4 else None
                if b:
                    vboxes[key] = [b["x"], b["y"], b["w"], b["h"]]
        except (OSError, ValueError, AttributeError):
            print(f"  ! {vpath} is unreadable; ignoring the cached vision boxes")
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
        norm = lambda t: re.sub(r"\s+", " ", str(t)).strip().lower()
        by_text = {norm(s): s for s in missing}
        try:
            res = ask([f"This UI design is {mock.width}x{mock.height}px. For each numbered string below return its number as "
                       "`index` and the TIGHT pixel bbox of its first visible occurrence (one line; ignore other lines). Strings:\n" +
                       "\n".join(f"{i + 1}. {s}" for i, s in enumerate(missing)), mock.convert("RGB")], TEXT_BOX_SCHEMA, "text_boxes")
            boxes = res["boxes"] if isinstance(res, dict) and isinstance(res.get("boxes"), list) else None
            if boxes is None:
                raise ValueError("no 'boxes' in the response")
            dropped = []
            for b in boxes:
                idx = b.get("index") if isinstance(b, dict) else None
                key = missing[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(missing) else by_text.get(norm(b.get("text", "")))
                clipped = _clip_bbox(b, mock.width, mock.height) if isinstance(b, dict) else None
                if key is None or clipped is None or clipped["w"] * clipped["h"] < 0.25 * max(1, b.get("w", 0) * b.get("h", 0)):
                    dropped.append(str(b.get("text", "?"))[:30] if isinstance(b, dict) else "?")
                    continue
                vboxes[key] = [clipped["x"], clipped["y"], clipped["w"], clipped["h"]]
            if dropped:
                print(f"  ! vision returned {len(dropped)} box(es) outside the mockup or for unknown strings; ignored: {', '.join(dropped)}")
            vpath.write_text(json.dumps(vboxes, indent=2))
        except (ApiError, KeyError, ValueError, TypeError) as e:  # the call, or a box the parse above cannot read
            print(f"  ! vision text location failed ({e}); continuing without it")
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
            e_w = _text_expect(line, st["size_px"], st.get("letter_spacing_em", 0))[1]
            sx0 = max(0, min(box[0], bx - int(0.3 * st["size_px"])))
            sx1 = min(rgb.shape[1], max(box[0] + box[2], bx + int(max(bw, e_w * 1.4)) + int(0.3 * st["size_px"])))
            sb = [sx0, by + n * pitch - int(0.25 * st["size_px"]), sx1 - sx0, bh + int(0.5 * st["size_px"])]
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
