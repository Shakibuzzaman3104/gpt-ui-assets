"""Vectorize: keyframes, Lottie, AnimatedVectorDrawable."""
import math
import re
import xml.etree.ElementTree as ET

from .core import SVG_NS, _res_name
from .prompts import EASES
from .vectordrawable import _IDENTITY, _color, _mapply, _mmul, _num, _style_dict

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


def _fmtn(v, nd=None):
    """Compact number: 2 decimals, 3 below 1 (a viewBox of 0 0 1 1 must not quantise to 1 %)."""
    if nd is None:
        nd = 3 if abs(v) < 1 else 2
    return f"{round(v, nd) + 0:g}"


def _d(subs):
    out = []
    for sp in subs:
        out.append(f"M{_fmtn(sp['pts'][0][0])},{_fmtn(sp['pts'][0][1])}")
        out += [s[0] + " ".join(f"{_fmtn(p[0])},{_fmtn(p[1])}" for p in s[1:]) for s in sp["segs"]]
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
                          "paths": [([_ellipse(g["cx"], g["cy"], g["rx"], g["ry"])], {"fill": "glow"})]}
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
    """Per layer: 'move' = max mean point displacement from the t=0 pose / the layer's size; 'fade' = opacity range;
    'score' = max(move, fade / 4) capped at 1. 'motion' = area-weighted mean score of the non-glow layers (a tiny spark
    travelling five times its size must not carry a static flame; a pure fade counts as 25 %)."""
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
        fade = max(ops) - min(ops)
        rep[L["name"]] = {"kind": L["kind"], "move": round(move, 4), "fade": round(fade, 3), "size": round(size, 3),
                          "score": round(min(1.0, max(move, fade / 4)), 4)}
    moving = [(r["score"], r["size"] ** 2) for r in rep.values() if r["kind"] != "glow"]
    wsum = sum(w for _, w in moving)
    return {"layers": rep, "motion": round(sum(sc * w for sc, w in moving) / wsum, 4) if wsum else None}


def _normalize_vec(vec, notes=None):
    """Drop a closing keyframe that repeats the first (it stalls the loop) and store canonical path data.
    notes (a list) collects what changed, for the caller to print."""
    for al in vec.get("animated_layers", []):
        keys = _align_keys([_subpaths(k) for k in al["keyframes"]])
        if len(keys) > 2:
            a, b = _points(keys[0]), _points(keys[-1])
            x0, y0, x1, y1 = _bbox(a)
            if max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b)) < 0.005 * math.hypot(x1 - x0, y1 - y0):
                keys = keys[:-1]
                if notes is not None:
                    notes.append(f"dropped the closing keyframe of {al['name']} (it repeated the first one)")
        al["keyframes"] = [_d(k) for k in keys]
    return vec


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX6_RE = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
_CAPS = {"butt": "butt", "round": "round", "square": "square"}
_JOINS = {"miter": "miter", "round": "round", "bevel": "bevel", "miter-clip": "miter", "arcs": "miter"}


def _cap(v, dflt="round"):
    return _CAPS.get(str(v or dflt).strip().lower(), dflt)


def _join(v, dflt="round"):
    return _JOINS.get(str(v or dflt).strip().lower(), dflt)


def _finite(v):
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    except TypeError:
        return False


def _validate_vec(vec):
    """Problems with a vec (the layers.json shape) as one-line strings; [] when it builds. Pure: never modifies vec.
    Colours must already be '#RRGGBB' (see _canon_vec); the schema enforces this for model answers."""
    if not isinstance(vec, dict):
        return ["layers.json must be a JSON object"]
    errs = []
    try:
        vb = _vec_vb(vec)
        if len(vb) != 4 or not all(math.isfinite(v) for v in vb) or vb[2] <= 0 or vb[3] <= 0:
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError):
        errs.append(f"viewBox {vec.get('viewBox')!r} must be 4 numbers \"x y w h\" with w and h > 0")
    try:
        defs_root = _defs_root(vec)
    except ET.ParseError as e:
        errs.append(f"defs is not valid SVG XML ({e})")
        defs_root = None
    names = []

    def num(where, p, k, lo=None, hi=None):
        if k not in p:
            return
        v = p[k]
        if not _finite(v):
            errs.append(f"{where}: {k} {v!r} must be a number")
        elif (lo is not None and v < lo) or (hi is not None and v > hi):
            errs.append(f"{where}: {k} {v} must be between {lo} and {hi}" if hi is not None else f"{where}: {k} {v} must be >= {lo}")

    def paint(where, p):
        for k in ("fill", "stroke"):
            v = str(p.get(k) or "none")
            ref = re.fullmatch(r"url\(#([^)]+)\)", v)
            if ref:
                if defs_root is not None:
                    try:
                        _gradient(defs_root, ref.group(1))
                    except ValueError as e:
                        errs.append(f"{where}: {k} {e}")
            elif v != "none" and not _HEX6_RE.fullmatch(v):
                errs.append(f"{where}: {k} {v!r} is not a hex colour (#RRGGBB), url(#id) or none")
        for k in ("opacity", "fill_opacity", "stroke_opacity"):
            num(where, p, k, 0, 1)
        num(where, p, "stroke_width", 0)
        if p.get("fill_rule") not in (None, "nonzero", "evenodd"):
            errs.append(f"{where}: fill_rule {p.get('fill_rule')!r} must be nonzero or evenodd")
        for k, table in (("cap", _CAPS), ("join", _JOINS)):
            if p.get(k) is not None and str(p[k]).lower() not in table:
                errs.append(f"{where}: {k} {p[k]!r} must be one of {', '.join(sorted(set(table.values())))}")

    def pathdata(where, d):
        if not isinstance(d, str) or not d.strip():
            errs.append(f"{where}: empty path data")
            return None
        junk = _TOK.sub("", d).strip(" ,\t\r\n")
        if junk:
            errs.append(f"{where}: path data has unexpected characters {junk[:20]!r}")
            return None
        try:
            subs = _subpaths(d)
        except (ValueError, IndexError, TypeError) as e:
            errs.append(f"{where}: {e}")
            return None
        if not subs or not any(sp["segs"] for sp in subs):
            errs.append(f"{where}: path data draws nothing")
            return None
        return subs

    def motion(where, mo):
        if mo is None:
            return
        if not isinstance(mo, dict):
            errs.append(f"{where}: motion must be an object with anchor_x, anchor_y, keys")
            return
        for k in ("anchor_x", "anchor_y"):
            if k in mo and not _finite(mo[k]):
                errs.append(f"{where}: motion {k} {mo[k]!r} must be a number")
        keys = mo.get("keys") or []
        if not isinstance(keys, list):
            errs.append(f"{where}: motion keys must be a list")
            return
        for j, k in enumerate(keys, 1):
            if not isinstance(k, dict) or "t" not in k:
                errs.append(f"{where}: motion key {j} has no t")
                continue
            num(f"{where} key {j}", k, "t", 0, 1)
            num(f"{where} key {j}", k, "opacity", 0, 1)
            for c in ("dx", "dy", "sx", "sy", "rotate"):
                num(f"{where} key {j}", k, c)
            if "ease" in k and k["ease"] not in EASES:
                errs.append(f"{where} key {j}: ease {k['ease']!r} must be one of {', '.join(EASES)}")

    def name_of(where, L):
        n = L.get("name") if isinstance(L, dict) else None
        if not isinstance(n, str) or not n.strip():
            errs.append(f"{where}: missing name")
            return f"?{len(names)}"
        names.append(n)
        return n

    drawn = 0
    for i, sl in enumerate(vec.get("static_layers") or [], 1):
        n = name_of(f"static layer {i}", sl)
        where = f"static {n}"
        paths = sl.get("paths") if isinstance(sl, dict) else None
        if not isinstance(paths, list) or not paths:
            errs.append(f"{where}: no paths")
        else:
            drawn += 1
            for p in paths:
                if not isinstance(p, dict):
                    errs.append(f"{where}: each path must be an object with d, fill, stroke")
                    continue
                pathdata(where, p.get("d"))
                paint(where, p)
        motion(where, sl.get("motion") if isinstance(sl, dict) else None)
    for i, al in enumerate(vec.get("animated_layers") or [], 1):
        n = name_of(f"animated layer {i}", al)
        where = f"animated {n}"
        kf = al.get("keyframes") if isinstance(al, dict) else None
        if not isinstance(kf, list) or len(kf) < 2:
            errs.append(f"{where}: needs at least 2 keyframes")
        else:
            drawn += 1
            keys = [pathdata(f"{where} keyframe {j}", k) for j, k in enumerate(kf, 1)]
            if all(k is not None for k in keys):
                try:
                    keys = _align_keys(keys)
                    a = _points(keys[0])
                    x0, y0, x1, y1 = _bbox(a)
                    size = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
                    if all(max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, _points(k))) < 0.005 * size for k in keys[1:]):
                        errs.append(f"{where}: all keyframes are identical (nothing morphs): vary them or make it a static layer")
                except (ValueError, IndexError, TypeError) as e:
                    errs.append(f"{where}: {e}")
        if isinstance(al, dict):
            paint(where, al)
            motion(where, al.get("motion"))
    for i, g in enumerate(vec.get("glows") or [], 1):
        n = name_of(f"glow {i}", g)
        where = f"glow {n}"
        if not isinstance(g, dict):
            continue
        for k in ("cx", "cy", "rx", "ry"):
            if not _finite(g.get(k)):
                errs.append(f"{where}: {k} {g.get(k)!r} must be a number")
        if _finite(g.get("rx")) and _finite(g.get("ry")) and (g["rx"] <= 0 or g["ry"] <= 0):
            errs.append(f"{where}: rx/ry must be > 0")
        if not _HEX6_RE.fullmatch(str(g.get("color") or "")):
            errs.append(f"{where}: color {g.get('color')!r} must be a hex colour (#RRGGBB)")
        num(where, g, "opacity", 0, 1)
        motion(where, g.get("motion"))
    if not drawn and not any(e.endswith(": no paths") or "keyframes" in e for e in errs):
        errs.append("no static or animated layers: nothing to draw")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        errs.append(f"layer names must be unique (repeated: {', '.join(dup)})")
    safe = {}
    for n in names:
        safe.setdefault(_res_name(n, "layer_"), []).append(n)
    clash = [v for v in safe.values() if len(set(v)) > 1]
    if clash:
        errs.append("layer names collide after sanitising: " + "; ".join(" and ".join(repr(x) for x in v) for v in clash))
    unknown = [str(n) for n in (vec.get("layer_order") or []) if n not in names]
    if unknown:
        errs.append(f"layer_order names unknown layer(s): {', '.join(unknown)} (known: {', '.join(names)})")
    return errs


def _paint_color(v, current=None):
    """('#RRGGBB', alpha) for any colour the kit reads: hex 3/4/6/8 digits, CSS names, rgb()/rgba(), currentColor."""
    c = _color(v, 1.0, current)
    return (f"#{c[3:]}", int(c[1:3], 16) / 255) if len(c) == 9 else (c, 1.0)


def _layer_name(raw, used):
    """snake_case layer name unique within `used` (Android resource rules: [a-z0-9_], starts with a letter)."""
    base = _res_name(str(raw or "layer"), "layer_")
    n, k = base, 2
    while n in used:
        n, k = f"{base}_{k}", k + 1
    used.add(n)
    return n


def _canon_vec(vec, current=None):
    """Canonicalise a vec in place before validation: snake_case unique layer names (layer_order follows; renames are
    printed) and every colour as '#RRGGBB' with its alpha folded into fill_opacity / stroke_opacity (glows: opacity).
    Returns the colours that could not be read, as error strings."""
    if not isinstance(vec, dict):
        return []
    used, ren = set(), {}
    for key in ("static_layers", "animated_layers", "glows"):
        for L in vec.get(key) or []:
            if not isinstance(L, dict):
                continue
            raw = str(L.get("name") or "")
            L["name"] = _layer_name(raw, used)
            ren.setdefault(raw, L["name"])
    if isinstance(vec.get("layer_order"), list):
        vec["layer_order"] = [ren.get(str(n), str(n)) for n in vec["layer_order"]]
    renamed = [f"{a!r} -> {b}" for a, b in ren.items() if a != b]
    if renamed:
        print(f"  (layer names sanitised: {', '.join(renamed[:6])})")
    errs, notes = [], []

    def fix(where, p, key, okey):
        v = str(p.get(key) or "none").strip()
        if v == "none" or v.startswith("url("):
            p[key] = v
            return
        if v.lower() == "currentcolor" and not current:
            notes.append(where)
        try:
            c, al = _paint_color(v, current)
        except ValueError:
            errs.append(f"{where}: {key} {v!r} is not a colour (use #RRGGBB, url(#id) or none)")
            return
        p[key] = c
        if al < 1:
            p[okey] = round(min(1.0, max(0.0, _num(p.get(okey), 1.0))) * al, 4)
    for sl in vec.get("static_layers") or []:
        for p in (sl.get("paths") or []) if isinstance(sl, dict) else []:
            if isinstance(p, dict):
                fix(f"static {sl['name']}", p, "fill", "fill_opacity")
                fix(f"static {sl['name']}", p, "stroke", "stroke_opacity")
    for al in vec.get("animated_layers") or []:
        if isinstance(al, dict):
            fix(f"animated {al['name']}", al, "fill", "fill_opacity")
            fix(f"animated {al['name']}", al, "stroke", "stroke_opacity")
    for g in vec.get("glows") or []:
        if isinstance(g, dict):
            fix(f"glow {g['name']}", g, "color", "opacity")
    if notes:
        print(f"  ~ currentColor with no color attribute ({', '.join(notes[:4])}): using #000000")
    return errs


def _vec_vb(vec):
    return [float(x) for x in re.split(r"[\s,]+", vec["viewBox"].strip())]


def _fit_viewbox(vec, margin=0.02):
    """viewBox grown symmetrically so every pose of the animation (plus stroke) stays inside; None if it already fits."""
    x, y, w, h = _vec_vb(vec)
    rest, pts = [(x, y), (x + w, y + h)], []
    for L in _layers(vec):
        if L["kind"] == "glow":  # the halo is visible out to about 70 % of its radius (GLOW_STOPS)
            g = L["glow"]
            L = dict(L, glow={**g, "rx": g["rx"] * 0.7, "ry": g["ry"] * 0.7})
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
    return f"{_fmtn(cx - hw)} {_fmtn(cy - hh)} {_fmtn(2 * hw)} {_fmtn(2 * hh)}"


def _xform_subs(subs, m):
    """Subpaths with the affine matrix applied to every anchor and control point (cubics are affine-invariant)."""
    if tuple(m) == _IDENTITY:
        return subs
    return [{"pts": [_mapply(m, *sp["pts"][0])], "closed": sp["closed"],
             "segs": [(s[0],) + tuple(_mapply(m, *p) for p in s[1:]) for s in sp["segs"]]} for sp in subs]


def _matrix(tf):
    """SVG transform list -> (a, b, c, d, e, f). ValueError names a function with the wrong argument count."""
    m = _IDENTITY
    for fn, args in re.findall(r"(\w+)\s*\(([^)]*)\)", tf or ""):
        v = [float(n) for n in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        if fn == "matrix" and len(v) == 6:
            q = tuple(v)
        elif fn == "translate" and len(v) in (1, 2):
            q = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif fn == "scale" and len(v) in (1, 2):
            q = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif fn == "rotate" and len(v) in (1, 3):
            r = math.radians(v[0])
            cx, cy = (v[1], v[2]) if len(v) > 2 else (0, 0)
            q = _mmul(_mmul((1, 0, 0, 1, cx, cy), (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)), (1, 0, 0, 1, -cx, -cy))
        elif fn == "skewX" and len(v) == 1:
            q = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif fn == "skewY" and len(v) == 1:
            q = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        elif fn in ("matrix", "translate", "scale", "rotate", "skewX", "skewY"):
            raise ValueError(f"transform {fn}({args.strip()}): wrong number of arguments")
        else:
            raise ValueError(f"transform {fn}(...) is not supported")
        m = _mmul(m, q)
    return m


def _defs_root(vec):
    defs = vec.get("defs", "") or ""
    vb = str(vec.get("viewBox") or "").strip()
    return ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
                         f'{f" viewBox={vb!r}" if vb else ""}>{defs if "<defs" in defs else f"<defs>{defs}</defs>"}</svg>')


def _grad_chain(defs_root, gid):
    """The gradient element with id gid followed by its href chain (nearest first)."""
    chain, seen = [], set()
    while gid and gid not in seen:
        seen.add(gid)
        gel = next((e for e in defs_root.iter() if e.get("id") == gid), None)
        if gel is None:
            if chain:
                break
            raise ValueError(f"gradient #{gid} not found in defs")
        tag = gel.tag.replace(SVG_NS, "") if isinstance(gel.tag, str) else "?"
        if tag not in ("linearGradient", "radialGradient"):
            raise ValueError(f"#{gid} is a <{tag}>, not a gradient (patterns, masks and filters are not supported)")
        chain.append(gel)
        gid = (gel.get("href") or gel.get("{http://www.w3.org/1999/xlink}href") or "").strip().lstrip("#")
    return chain


def _gradient(defs_root, gid, bbox=None, vb=None):
    """(linear, start, end, [(offset, '#RRGGBB', alpha)]) in user units with any gradientTransform baked in (radius
    scaled by sqrt|det|). href chains supply missing attributes and stops; SVG defaults apply (x1=y1=y2=0% x2=100%,
    cx=cy=r=50%). Percentages resolve against the viewBox (vb, or defs_root's viewBox); the default objectBoundingBox
    units need the referencing shape's bbox (x0, y0, x1, y1), else ValueError tells the author what to set."""
    chain = _grad_chain(defs_root, gid)
    gel = chain[0]
    attr = lambda k: next((e.get(k) for e in chain if e.get(k) is not None), None)
    stops_el = next((e for e in chain if any(isinstance(c.tag, str) and c.tag.replace(SVG_NS, "") == "stop" for c in e)), None)
    stops = []
    for st in (stops_el if stops_el is not None else []):
        if not isinstance(st.tag, str) or st.tag.replace(SVG_NS, "") != "stop":
            continue
        style = _style_dict(st)
        off = str(st.get("offset", "0"))
        off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
        try:
            col, al = _paint_color(style.get("stop-color", st.get("stop-color")) or "#000000")
        except ValueError as e:
            raise ValueError(f"gradient #{gid}: {e}")
        so = str(style.get("stop-opacity", st.get("stop-opacity")) or "1")
        so = _num(so) / 100 if so.strip().endswith("%") else _num(so, 1.0)
        off = min(1.0, max(0.0, off))
        if stops and off < stops[-1][0]:
            off = stops[-1][0]
        stops.append((off, col, min(1.0, max(0.0, al * so))))
    if len(stops) < 2:
        raise ValueError(f"gradient #{gid}: needs at least 2 stops (has {len(stops)})")
    linear = gel.tag.endswith("linearGradient")
    obb = (attr("gradientUnits") or "objectBoundingBox").strip() != "userSpaceOnUse"
    if obb and bbox is None:
        raise ValueError(f'gradient #{gid} uses objectBoundingBox units (the SVG default): set gradientUnits="userSpaceOnUse" '
                         f"and coordinates in viewBox units")
    if vb is None and defs_root.get("viewBox"):
        vb = [float(x) for x in re.split(r"[\s,]+", defs_root.get("viewBox").strip())]

    def coord(k, dflt, axis):
        v = attr(k)
        sv = str(v).strip() if v is not None else dflt
        pct = sv.endswith("%")
        n = _num(sv) / 100 if pct else _num(sv)
        if obb:
            x0, y0, x1, y1 = bbox
            return (x0 + n * (x1 - x0) if axis == "x" else y0 + n * (y1 - y0) if axis == "y"
                    else n * math.sqrt(((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2))
        if pct:
            if vb is None or len(vb) != 4:
                raise ValueError(f"gradient #{gid}: {k}={sv} needs a viewBox to resolve the percentage")
            vx, vy, vw, vh = vb
            return vx + n * vw if axis == "x" else vy + n * vh if axis == "y" else n * math.sqrt((vw ** 2 + vh ** 2) / 2)
        return n
    a, b, c, d, e, f = _matrix(attr("gradientTransform"))
    ap = lambda x, y: [a * x + c * y + e, b * x + d * y + f]
    if linear:
        return True, ap(coord("x1", "0%", "x"), coord("y1", "0%", "y")), ap(coord("x2", "100%", "x"), coord("y2", "0%", "y")), stops
    s = ap(coord("cx", "50%", "x"), coord("cy", "50%", "y"))
    r = coord("r", "50%", "r") * math.sqrt(abs(a * d - b * c))
    if not r > 0:
        raise ValueError(f"gradient #{gid}: radius must be > 0")
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
    """Lottie fill (or stroke when width is given) for a '#RRGGBB' colour or url(#gradient)."""
    ref = re.match(r"url\(#([^)]+)\)", fill or "")
    stroke = width is not None
    if grad is None and not ref:
        item = {"ty": "st" if stroke else "fl", "c": {"a": 0, "k": _hex_rgb(fill) + [1]}, "o": {"a": 0, "k": round(opacity * 100, 3)}}
    else:
        linear, s, e, stops = grad or _gradient(defs_root, ref.group(1))
        colors = [x for o, c, _ in stops for x in [o] + _hex_rgb(c)]
        alphas = [x for o, _, al in stops for x in (o, al)] if any(al < 1 for *_, al in stops) else []
        item = {"ty": "gs" if stroke else "gf", "t": 1 if linear else 2, "s": {"a": 0, "k": s}, "e": {"a": 0, "k": e},
                "g": {"p": len(stops), "k": {"a": 0, "k": colors + alphas}}, "o": {"a": 0, "k": round(opacity * 100, 3)}}
    if stroke:
        item.update({"w": {"a": 0, "k": width}, "lc": LOTTIE_CAP[_cap(cap)], "lj": LOTTIE_JOIN[_join(join)], "ml": 4})
    else:
        item["r"] = 1
    return item


def _paint_alphas(p):
    """(fill alpha, stroke alpha) of a paint dict: opacity times fill_opacity / stroke_opacity, clamped to 0..1."""
    op = min(1.0, max(0.0, _num(p.get("opacity"), 1.0)))
    return (min(1.0, max(0.0, op * _num(p.get("fill_opacity"), 1.0))), min(1.0, max(0.0, op * _num(p.get("stroke_opacity"), 1.0))))


def _lottie_paints(p, defs_root):
    """SVG paints stroke over fill; Lottie draws earlier items on top, so the stroke goes first."""
    items = []
    fa, sa = _paint_alphas(p)
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        items.append(_lottie_fill(p["stroke"], sa, defs_root, float(p["stroke_width"]), p.get("cap", "round"), p.get("join", "round")))
    if (p.get("fill") or "none") != "none":
        fl = _lottie_fill(p["fill"], fa, defs_root)
        fl["r"] = 2 if p.get("fill_rule") == "evenodd" else 1
        items.append(fl)
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
    vb = _vec_vb(vec)
    k = size / max(vb[2], vb[3])
    op = max(1, round(seconds * fps))
    rate = round(op / seconds, 4)  # a fractional frame rate keeps the loop exactly `seconds` long (the AVD uses ms)
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
    return {"v": "5.7.0", "fr": rate, "ip": 0, "op": op, "w": round(vb[2] * k), "h": round(vb[3] * k), "nm": "gptkit",
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
        head = f'{ind}    <keyframe android:fraction="{_fmtn(f, 4)}" android:value="{_fmtn(v, 4)}"'
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
    base = min(1.0, max(0.0, base))
    cl = lambda v: min(1.0, max(0.0, v))
    fr = _frames(ks, lambda k: cl(base * k["opacity"])) if ks else [(0.0, base, None), (1.0, base, None)]
    if not window:
        return fr
    a, b = window
    at = lambda t: cl(base * _motion_at(ks, t)["opacity"]) if ks else base
    b_in = b if b == 1 else b - eps  # the outgoing copy fades over [b - eps, b] while the next one fades in: no double alpha
    inner = [(a, at(a), None)] + [f for f in fr if a < f[0] < b_in] + [(b_in, at(b), None)]
    out = [] if a == 0 else [(0.0, 0.0, None), (a - eps, 0.0, None)]
    out += inner
    out += [] if b == 1 else [(b, 0.0, None), (1.0, 0.0, None)]
    return out


def _avd_paint(p, defs_root, ind):
    attrs, kids = [], []

    def color(attr, val):
        ref = re.match(r"url\(#([^)]+)\)", val or "")
        if not ref:
            attrs.append(f'android:{attr}="{_color(val)}"')
            return
        linear, s, e, stops = p["_grad"] if "_grad" in p else _gradient(defs_root, ref.group(1))
        head = (f'android:type="linear" android:startX="{_fmtn(s[0])}" android:startY="{_fmtn(s[1])}" '
                f'android:endX="{_fmtn(e[0])}" android:endY="{_fmtn(e[1])}"' if linear else
                f'android:type="radial" android:centerX="{_fmtn(s[0])}" android:centerY="{_fmtn(s[1])}" '
                f'android:gradientRadius="{_fmtn(max(math.dist(s, e), 0.01))}"')
        items = "".join(f'\n{ind}        <item android:offset="{o:g}" android:color="{_color(c, al)}" />' for o, c, al in stops)
        kids.append(f'{ind}<aapt:attr name="android:{attr}">\n{ind}    <gradient {head}>{items}\n{ind}    </gradient>\n{ind}</aapt:attr>')
    if p.get("fill") == "glow":
        color("fillColor", "url(#glow)")
    elif (p.get("fill") or "none") != "none":
        color("fillColor", p["fill"])
        if p.get("fill_rule") == "evenodd":
            attrs.append('android:fillType="evenOdd"')
    if (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0:
        color("strokeColor", p["stroke"])
        attrs.append(f'android:strokeWidth="{_fmtn(float(p["stroke_width"]))}"')
        attrs.append(f'android:strokeLineCap="{_cap(p.get("cap"))}" android:strokeLineJoin="{_join(p.get("join"))}"')
    return attrs, kids


def build_avd(vec, seconds=1.2, width_dp=48):
    """<animated-vector> with the vector inlined: group transforms, fillAlpha fades and pathData morphs, all looping."""
    vb = _vec_vb(vec)
    ms = round(seconds * 1000)
    defs_root = _defs_root(vec)
    used = set()  # every android:name (groups and paths) is unique: targets resolve by name
    body, targets = [], []
    for L in _layers(vec):
        gname = _layer_name(L["name"], used)
        ks, (ax, ay) = L["ks"], L["anchor"]
        m0 = _motion_at(ks, 0.0)
        paths = []
        if L["kind"] == "morph":
            keys, n = L["keys"], len(L["keys"])
            p = L["paint"]
            if n == 2:
                paths.append((_layer_name(f"{gname}_p0", used), _d(keys[0]), p, None, [
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
                    paths.append((_layer_name(f"{gname}_p{j}", used), _d(keys[j]), p, (a, b), [
                        f'            <objectAnimator android:propertyName="pathData" android:valueType="pathType"\n'
                        f'                android:valueFrom="{_d(keys[j])}"\n                android:valueTo="{_d(keys[(j + 1) % n])}"\n'
                        f'                android:duration="{ms}" android:repeatCount="infinite">\n'
                        f'{_avd_interp(None, "                ", ip)}\n            </objectAnimator>']))
        elif L["kind"] == "glow":
            g = L["glow"]
            p = {"fill": "glow", "_grad": _glow_grad(g), "opacity": g["opacity"]}
            paths.append((_layer_name(f"{gname}_p0", used), _d([_ellipse(g["cx"], g["cy"], g["rx"], g["rx"])]), p, None, []))
        else:
            for i, (subs, p) in enumerate(L["paths"]):
                paths.append((_layer_name(f"{gname}_p{i}", used), _d(subs), p, None, []))
        xml = []
        for pname, d, p, window, anims in paths:
            fa, sa = _paint_alphas(p)
            attrs, kids = _avd_paint(p, defs_root, "            ")
            fades = []
            for prop, on, base in (("fillAlpha", (p.get("fill") or "none") != "none", fa),
                                   ("strokeAlpha", (p.get("stroke") or "none") != "none" and float(p.get("stroke_width") or 0) > 0, sa)):
                if not on:
                    continue
                fr = _alpha_frames(ks, base, window)
                v0 = fr[0][1]
                if v0 < 1:
                    attrs.append(f'android:{prop}="{_fmtn(v0, 4)}"')
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
            xml = [f'        <group android:pivotX="{_fmtn(g["cx"])}" android:pivotY="{_fmtn(g["cy"])}" '
                   f'android:scaleY="{g["ry"] / g["rx"]:.4g}">'] + xml + ["        </group>"]
        if ks:
            chans = {"translateX": ("dx", 0), "translateY": ("dy", 0), "scaleX": ("sx", 1), "scaleY": ("sy", 1), "rotation": ("rotate", 0)}
            pvhs, gattrs = [], [f'android:name="{gname}"', f'android:pivotX="{_fmtn(ax)}"', f'android:pivotY="{_fmtn(ay)}"']
            for prop, (c, rest) in chans.items():
                if abs(m0[c] - rest) > 1e-6:
                    gattrs.append(f'android:{prop}="{_fmtn(m0[c], 4)}"')
                if len({round(k[c], 5) for k in ks}) > 1:
                    pvhs.append(_avd_pvh(prop, _frames(ks, lambda k, c=c: k[c]), "                "))
            if pvhs:
                targets.append(_avd_target(gname, [_avd_animator(pvhs, ms, "            ")]))
            xml = [f'        <group {" ".join(gattrs)}>'] + ["    " + ln for x in xml for ln in x.split("\n")] + ["        </group>"]
        body += xml
    if vb[0] or vb[1]:
        body = [f'        <group android:translateX="{_fmtn(-vb[0])}" android:translateY="{_fmtn(-vb[1])}">'] + \
               ["    " + ln for x in body for ln in x.split("\n")] + ["        </group>"]
    assert len(used) == len(set(used)) and all(_NAME_RE.match(n) for n in used), used
    hdp = width_dp * vb[3] / vb[2]
    vector = (f'    <aapt:attr name="android:drawable">\n        <vector android:width="{_fmtn(width_dp)}dp" android:height="{_fmtn(hdp)}dp"\n'
              f'            android:viewportWidth="{_fmtn(vb[2])}" android:viewportHeight="{_fmtn(vb[3])}">\n' +
              "\n".join("    " + ln for x in body for ln in x.split("\n")) + "\n        </vector>\n    </aapt:attr>")
    return ('<?xml version="1.0" encoding="utf-8"?>\n<animated-vector xmlns:android="http://schemas.android.com/apk/res/android"\n'
            '    xmlns:aapt="http://schemas.android.com/aapt">\n' + vector + "\n" + "\n".join(targets) + "\n</animated-vector>\n")
