"""SVG -> VectorDrawable conversion and lint."""
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .core import SVG_NS, _chrome_probe, _res_name
from .forge import _renderer_name, render_svgs

# ---- SVG -> Android VectorDrawable

_PARAMS = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7, "Z": 0}
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
PRES_ATTRS = ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "fill-rule",
              "fill-opacity", "stroke-opacity", "stroke-dasharray", "color", "display", "visibility", "clip-rule",
              "marker-start", "marker-mid", "marker-end", "vector-effect", "paint-order")
NAMED_COLORS = {"transparent": "#00000000", **{k: "#" + v for k, v in (kv.split("=") for kv in (
    "aliceblue=f0f8ff antiquewhite=faebd7 aqua=00ffff aquamarine=7fffd4 azure=f0ffff beige=f5f5dc bisque=ffe4c4 black=000000 "
    "blanchedalmond=ffebcd blue=0000ff blueviolet=8a2be2 brown=a52a2a burlywood=deb887 cadetblue=5f9ea0 chartreuse=7fff00 "
    "chocolate=d2691e coral=ff7f50 cornflowerblue=6495ed cornsilk=fff8dc crimson=dc143c cyan=00ffff darkblue=00008b "
    "darkcyan=008b8b darkgoldenrod=b8860b darkgray=a9a9a9 darkgreen=006400 darkgrey=a9a9a9 darkkhaki=bdb76b darkmagenta=8b008b "
    "darkolivegreen=556b2f darkorange=ff8c00 darkorchid=9932cc darkred=8b0000 darksalmon=e9967a darkseagreen=8fbc8f "
    "darkslateblue=483d8b darkslategray=2f4f4f darkslategrey=2f4f4f darkturquoise=00ced1 darkviolet=9400d3 deeppink=ff1493 "
    "deepskyblue=00bfff dimgray=696969 dimgrey=696969 dodgerblue=1e90ff firebrick=b22222 floralwhite=fffaf0 forestgreen=228b22 "
    "fuchsia=ff00ff gainsboro=dcdcdc ghostwhite=f8f8ff gold=ffd700 goldenrod=daa520 gray=808080 green=008000 greenyellow=adff2f "
    "grey=808080 honeydew=f0fff0 hotpink=ff69b4 indianred=cd5c5c indigo=4b0082 ivory=fffff0 khaki=f0e68c lavender=e6e6fa "
    "lavenderblush=fff0f5 lawngreen=7cfc00 lemonchiffon=fffacd lightblue=add8e6 lightcoral=f08080 lightcyan=e0ffff "
    "lightgoldenrodyellow=fafad2 lightgray=d3d3d3 lightgreen=90ee90 lightgrey=d3d3d3 lightpink=ffb6c1 lightsalmon=ffa07a "
    "lightseagreen=20b2aa lightskyblue=87cefa lightslategray=778899 lightslategrey=778899 lightsteelblue=b0c4de lightyellow=ffffe0 "
    "lime=00ff00 limegreen=32cd32 linen=faf0e6 magenta=ff00ff maroon=800000 mediumaquamarine=66cdaa mediumblue=0000cd "
    "mediumorchid=ba55d3 mediumpurple=9370db mediumseagreen=3cb371 mediumslateblue=7b68ee mediumspringgreen=00fa9a "
    "mediumturquoise=48d1cc mediumvioletred=c71585 midnightblue=191970 mintcream=f5fffa mistyrose=ffe4e1 moccasin=ffe4b5 "
    "navajowhite=ffdead navy=000080 oldlace=fdf5e6 olive=808000 olivedrab=6b8e23 orange=ffa500 orangered=ff4500 orchid=da70d6 "
    "palegoldenrod=eee8aa palegreen=98fb98 paleturquoise=afeeee palevioletred=db7093 papayawhip=ffefd5 peachpuff=ffdab9 "
    "peru=cd853f pink=ffc0cb plum=dda0dd powderblue=b0e0e6 purple=800080 rebeccapurple=663399 red=ff0000 rosybrown=bc8f8f "
    "royalblue=4169e1 saddlebrown=8b4513 salmon=fa8072 sandybrown=f4a460 seagreen=2e8b57 seashell=fff5ee sienna=a0522d "
    "silver=c0c0c0 skyblue=87ceeb slateblue=6a5acd slategray=708090 slategrey=708090 snow=fffafa springgreen=00ff7f "
    "steelblue=4682b4 tan=d2b48c teal=008080 thistle=d8bfd8 tomato=ff6347 turquoise=40e0d0 violet=ee82ee wheat=f5deb3 "
    "white=ffffff whitesmoke=f5f5f5 yellow=ffff00 yellowgreen=9acd32").split())}}


def parse_path(d):
    """SVG path data -> [(cmd, [floats])], one entry per segment (implicit repeats and compact arc flags expanded)."""
    out, i, cmd, n = [], 0, None, len(d or "")
    while True:
        while i < n and (d[i].isspace() or d[i] == ","):
            i += 1
        if i >= n:
            return out
        if d[i].isalpha():
            cmd = d[i]
            i += 1
            if cmd.upper() not in _PARAMS:
                raise ValueError(f"unknown path command {cmd!r}")
            if cmd in "Zz":
                out.append((cmd, []))
                continue
        elif cmd is None or cmd in "Zz":
            raise ValueError(f"path data has numbers without a command near {d[i:i + 12]!r}")
        vals = []
        for j in range(_PARAMS[cmd.upper()]):
            while i < n and (d[i].isspace() or d[i] == ","):
                i += 1
            if cmd in "Aa" and j in (3, 4):
                if i >= n or d[i] not in "01":
                    raise ValueError(f"bad arc flag near {d[i:i + 12]!r}")
                vals.append(float(d[i]))
                i += 1
                continue
            m_ = _NUM.match(d, i)
            if not m_:
                raise ValueError(f"expected a number near {d[i:i + 12]!r}")
            vals.append(float(m_.group()))
            i = m_.end()
        out.append((cmd, vals))
        cmd = {"M": "L", "m": "l"}.get(cmd, cmd)


def _fmt(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path_str(segs):
    return " ".join(c + " ".join(_fmt(v) for v in vals) for c, vals in segs)


def transform_path(segs, s, tx=0.0, ty=0.0):
    """Scale by s and translate every coordinate of parsed path segments."""
    out = []
    for idx, (c, v) in enumerate(segs):
        rel = c.islower()
        if idx == 0 and c == "m":
            c, rel = "M", False
        U = c.upper()
        if U == "A":
            v = [v[0] * s, v[1] * s, v[2], v[3], v[4]] + ([v[5] * s, v[6] * s] if rel else [v[5] * s + tx, v[6] * s + ty])
        elif U == "H":
            v = [v[0] * s + (0 if rel else tx)]
        elif U == "V":
            v = [v[0] * s + (0 if rel else ty)]
        else:
            v = [x * s + (0 if rel else (tx if j % 2 == 0 else ty)) for j, x in enumerate(v)]
        out.append((c, v))
    return out


def _num(v, default=0.0):
    if v is None:
        return default
    return float(re.sub(r"[a-z%]+$", "", str(v).strip()) or default)


def _style_dict(el):
    return {k.strip(): v.strip() for k, _, v in (kv.partition(":") for kv in (el.get("style") or "").split(";")) if v}


def _css_rules(root):
    """Simple rules from every <style>: [(tag, class, id, {prop: value})] for selectors like `.st0`, `path`, `#a`,
    `circle.b` and comma lists; anything more complex (combinators, pseudo-classes) is ignored."""
    css = "".join(el.text or "" for el in root.iter() if el.tag.replace(SVG_NS, "") == "style")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = []
    for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        decl = {k.strip(): v.strip() for k, _, v in (kv.partition(":") for kv in body.split(";")) if v.strip()}
        for one in sel.split(","):
            m_ = re.fullmatch(r"\s*([a-zA-Z][\w-]*)?(?:\.([\w-]+))?(?:#([\w-]+))?\s*", one)
            if m_ and decl and any(m_.groups()):
                rules.append((m_.group(1), m_.group(2), m_.group(3), decl))
    return rules


def _css_props(el, rules):
    """Declarations that apply to el, lowest specificity first (tag < .class < #id)."""
    tag, classes, eid = el.tag.replace(SVG_NS, ""), set((el.get("class") or "").split()), el.get("id")
    out = {}
    # key= only: two rules with the same selector must not fall through to comparing their dicts (TypeError); the stable
    # sort keeps document order within a specificity, so the later rule wins as in CSS
    for t, c, i, decl in sorted(rules, key=lambda r: (bool(r[2]), bool(r[1]), bool(r[0]))):
        if (t is None or t == tag) and (c is None or c in classes) and (i is None or i == eid):
            out.update(decl)
    return out


def _opacity(v, dflt=1.0):
    """An SVG opacity value ("0.5", "50%") as a float."""
    if v is None:
        return dflt
    v = str(v).strip()
    return _num(v) / 100 if v.endswith("%") else _num(v, dflt)


def _inherit(el, parent, css=None):
    """Effective presentation attributes of el: attribute < <style> rule < inline style; opacity multiplies down the
    tree; `inherit` and empty values keep the parent's; `!important` is stripped."""
    attrs = dict(parent)
    rule = _css_props(el, css) if css else {}
    style = {**rule, **_style_dict(el)}
    for k in PRES_ATTRS:
        v = style.get(k, el.get(k))
        if v is not None:
            v = re.sub(r"\s*!important\s*$", "", v.strip())
            if v and v != "inherit":
                attrs[k] = v
    op = style.get("opacity", el.get("opacity"))
    attrs["opacity"] = _opacity(parent.get("opacity"), 1.0) * (_opacity(op, 1.0) if op is not None else 1.0)
    return attrs


def _color(v, opacity=1.0, current=None):
    """CSS colour -> #RRGGBB / #AARRGGBB (hex, rgb()/rgba(), hsl()/hsla(), the 148 named colours, currentColor)."""
    v = re.sub(r"\s*!important\s*$", "", (v or "").strip().lower())
    if v == "currentcolor":
        v = (current or "#000000").strip().lower()
    v = NAMED_COLORS.get(v, v)
    m_ = re.match(r"(rgba?|hsla?)\(([^)]*)\)", v)
    if m_:
        parts = [p.strip() for p in re.split(r"[\s,/]+", m_.group(2).strip()) if p.strip()]
        if len(parts) < 3:
            raise ValueError(f"unsupported colour {v!r}")
        a = (_num(parts[3]) / 100 if parts[3].endswith("%") else _num(parts[3])) if len(parts) > 3 else 1.0
        if m_.group(1).startswith("hsl"):
            unit = re.sub(r"^[-+0-9.eE]+", "", parts[0])  # CSS hue units: deg (default), turn, rad, grad
            h = _num(parts[0]) * {"turn": 360.0, "rad": 180 / math.pi, "grad": 0.9}.get(unit, 1.0)
            h = (h % 360) / 360
            sat, lig = min(1.0, max(0.0, _num(parts[1]) / 100)), min(1.0, max(0.0, _num(parts[2]) / 100))
            import colorsys
            rgb = [int(c * 255 + 0.5 + 1e-6) for c in colorsys.hls_to_rgb(h, lig, sat)]  # CSS rounds half up
        else:
            rgb = [round(_num(p) * 2.55) if p.endswith("%") else round(_num(p)) for p in parts[:3]]
        v = "#" + "".join(f"{max(0, min(255, c)):02x}" for c in rgb) + f"{round(max(0.0, min(1.0, a)) * 255):02x}"
    if not v.startswith("#") or len(v) not in (4, 5, 7, 9) or not re.fullmatch(r"#[0-9a-f]+", v):
        raise ValueError(f"unsupported colour {v!r}")
    h = v[1:]
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    alpha = round((int(h[6:8], 16) if len(h) == 8 else 255) * opacity)
    return f"#{alpha:02X}{h[:6].upper()}" if alpha < 255 else f"#{h[:6].upper()}"


def _shape_path(el):
    tag = el.tag.replace(SVG_NS, "")
    g = lambda k, dflt=0.0: _num(el.get(k), dflt)
    if tag == "path":
        return el.get("d")
    if tag in ("circle", "ellipse"):
        cx, cy = g("cx"), g("cy")
        rx = g("r") if tag == "circle" else g("rx", g("ry"))
        ry = g("r") if tag == "circle" else g("ry", rx)
        if rx <= 0 or ry <= 0:
            return None
        return f"M{cx - rx},{cy}a{rx},{ry} 0 1,0 {2 * rx},0a{rx},{ry} 0 1,0 {-2 * rx},0Z"
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        rx = g("rx", g("ry"))
        ry = g("ry", rx)
        if not rx or not ry:
            return f"M{x},{y}h{w}v{h}h{-w}Z"
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        return (f"M{x + rx},{y}h{w - 2 * rx}a{rx},{ry} 0 0,1 {rx},{ry}v{h - 2 * ry}a{rx},{ry} 0 0,1 {-rx},{ry}"
                f"h{-(w - 2 * rx)}a{rx},{ry} 0 0,1 {-rx},{-ry}v{-(h - 2 * ry)}a{rx},{ry} 0 0,1 {rx},{-ry}Z")
    if tag == "line":
        return f"M{g('x1')},{g('y1')}L{g('x2')},{g('y2')}"
    if tag in ("polyline", "polygon"):
        pts = _NUM.findall(el.get("points", ""))
        pairs = [f"{pts[i]},{pts[i + 1]}" for i in range(0, len(pts) - 1, 2)]
        return ("M" + "L".join(pairs) + ("Z" if tag == "polygon" else "")) if pairs else None
    return None


def _transforms(s):
    """SVG transform list -> VectorDrawable <group> attribute dicts, outermost first."""
    groups = []
    counts = {"translate": (1, 2), "scale": (1, 2), "rotate": (1, 3), "matrix": (6, 6)}
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM.findall(args)]
        lo, hi = counts.get(name, (0, 0))
        if name in counts and not lo <= len(v) <= hi:
            raise ValueError(f"bad {name}() arguments: {args.strip()!r}")
        if name == "translate":
            groups.append({"translateX": v[0], "translateY": v[1] if len(v) > 1 else 0})
        elif name == "scale":
            groups.append({"scaleX": v[0], "scaleY": v[1] if len(v) > 1 else v[0]})
        elif name == "rotate":
            groups.append({"rotation": v[0], "pivotX": v[1] if len(v) > 2 else 0, "pivotY": v[2] if len(v) > 2 else 0})
        elif name == "matrix":
            a, b, c, d, e, f = v
            if abs(a * c + b * d) > 1e-6:
                raise ValueError("transform matrix with skew is not expressible in VectorDrawable")
            sx = math.hypot(a, b)
            if not sx:
                raise ValueError("transform matrix with zero scale")
            groups.append({"translateX": e, "translateY": f, "rotation": math.degrees(math.atan2(b, a)),
                           "scaleX": sx, "scaleY": (a * d - b * c) / sx})
        else:
            raise ValueError(f"transform {name}() is not expressible in VectorDrawable")
    return groups


def _viewbox(root):
    """(x, y, w, h) of an <svg> root: its viewBox, else width/height; ValueError when malformed or without area."""
    raw = root.get("viewBox", "").strip()
    if raw:
        vb = [float(v) for v in re.split(r"[\s,]+", raw) if v]
        if len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0:
            raise ValueError(f"viewBox must be 4 numbers with positive size, got {raw!r}")
        return vb
    w, h = _num(root.get("width"), 24), _num(root.get("height"), 24)
    if w <= 0 or h <= 0:
        raise ValueError(f"width/height must be positive, got {root.get('width')!r} x {root.get('height')!r}")
    return [0, 0, w, h]


_UNIT_PX = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}


def _length_px(v):
    """An SVG length ("24", "24px", "18pt", "24mm") in CSS px; None for missing, percentage or unknown units."""
    m_ = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z%]*)\s*", str(v or ""))
    if not m_ or m_.group(2) not in _UNIT_PX:
        return None
    return float(m_.group(1)) * _UNIT_PX[m_.group(2)]


def _pct(v, ref, dflt):
    """A gradient coordinate: percentages against ref (viewBox width/height/diagonal), else user units."""
    if v is None:
        v = dflt
    v = str(v).strip()
    return _num(v) / 100 * ref if v.endswith("%") else _num(v)


_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _tf_affine(tf):
    """SVG transform list -> (a, b, c, d, e, f); _matrix parses skewX/skewY itself, so this is only the name the
    converter uses for it."""
    from .vectorize import _matrix  # deferred: vectorize imports this module
    return _matrix(tf)


def _mmul(p, q):
    """Affine (a, b, c, d, e, f) product p . q (q is applied first)."""
    return (p[0] * q[0] + p[2] * q[1], p[1] * q[0] + p[3] * q[1], p[0] * q[2] + p[2] * q[3],
            p[1] * q[2] + p[3] * q[3], p[0] * q[4] + p[2] * q[5] + p[4], p[1] * q[4] + p[3] * q[5] + p[5])


def _mapply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _abs_path(d, m=_IDENTITY):
    """Path data as absolute L/C segments with matrix m applied (arcs become cubics)."""
    from .vectorize import _d, _subpaths  # deferred: vectorize imports this module
    subs = _subpaths(d)
    if m != _IDENTITY:
        for sp in subs:
            sp["pts"] = [_mapply(m, *pt) for pt in sp["pts"]]
            sp["segs"] = [(sg[0], *[_mapply(m, *pt) for pt in sg[1:]]) for sg in sp["segs"]]
    return _d(subs)


def _path_bbox(d):
    """(x0, y0, x1, y1) hull of the path's anchor and control points (exact for straight edges)."""
    from .vectorize import _points, _subpaths  # deferred: vectorize imports this module
    pts = _points(_subpaths(d))
    if not pts:
        return None
    return min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts)


def _inline_uses(root, ids):
    """Replace every <use> with a copy of what it references (a <symbol> becomes a <g> scaled to the use's size)."""
    import copy
    for _ in range(8):
        uses = [(par, ch) for par in root.iter() for ch in par if ch.tag.replace(SVG_NS, "") == "use"]
        if not uses:
            return
        for par, use in uses:
            href = (use.get("href") or use.get("{http://www.w3.org/1999/xlink}href") or "").strip()
            if not href.startswith("#") or href[1:] not in ids:
                raise ValueError(f"<use> references {href!r}, which is not in this file")
            src = copy.deepcopy(ids[href[1:]])
            src.attrib.pop("id", None)
            tf = f"translate({_num(use.get('x')):g} {_num(use.get('y')):g})"
            if src.tag.replace(SVG_NS, "") in ("symbol", "svg"):
                vb = src.get("viewBox")
                if vb and use.get("width") and use.get("height"):
                    x, y, w, h = [float(v) for v in re.split(r"[\s,]+", vb.strip()) if v]
                    tf += f" scale({_num(use.get('width')) / w:g} {_num(use.get('height')) / h:g}) translate({-x:g} {-y:g})"
                src.tag = SVG_NS + "g"
                for k in ("viewBox", "width", "height", "preserveAspectRatio"):
                    src.attrib.pop(k, None)
            g = ET.Element(SVG_NS + "g", {k: v for k, v in use.attrib.items()
                                          if k not in ("x", "y", "width", "height", "href", "transform", "id")
                                          and not k.startswith("{http://www.w3.org/1999/xlink}")})
            g.set("transform", (use.get("transform", "") + " " + tf).strip())
            g.append(src)
            par[list(par).index(use)] = g
    raise ValueError("<use> chain nested more than 8 levels deep")


def _rewind_evenodd(d):
    """Path data with every subpath wound by nesting depth (a hole runs against its outer loop) so the nonzero rule
    keeps what evenodd cut out: VectorDrawable <clip-path> has no fillType. Exact for nested, non-crossing subpaths."""
    from .vectorize import _curve_pts, _d, _subpaths  # deferred: vectorize imports this module
    from .trace_rig import _area, _inside  # deferred: trace_rig imports this module
    subs = _subpaths(d)
    loops = [np.array(_curve_pts([sp])) for sp in subs]
    out = []
    for i, (sp, P) in enumerate(zip(subs, loops)):
        depth = sum(_inside(P[0], Q) for j, Q in enumerate(loops) if j != i and len(Q) >= 3 and abs(_area(Q)) > abs(_area(P)))
        if len(P) < 3 or (_area(P) > 0) == (depth % 2 == 0):
            out.append(sp)
            continue
        ends = [sp["pts"][0]] + [s[-1] for s in sp["segs"]]
        segs = [("L", ends[k]) if s[0] == "L" else ("C", s[2], s[1], ends[k]) for k, s in reversed(list(enumerate(sp["segs"])))]
        out.append({"pts": [ends[-1]], "segs": segs, "closed": sp["closed"]})
    return _d(out)


def svg_to_vd(svg_text, width_dp=None, warnings=None):
    """Convert an SVG to VectorDrawable XML. Unsupported features raise ValueError; lossy ones append to warnings.
    Handles shapes, groups and transforms (skews baked into coordinates), <style> classes, <use>/<symbol>, clip-path,
    linear/radial gradients (userSpaceOnUse and objectBoundingBox, gradientTransform), CSS colours.
    Default size: the root width/height (px; pt/mm/cm/in converted), else the viewBox, at the viewBox's aspect."""
    from .svg_icons import _SHAPES, _bake_transforms  # deferred: svg_icons imports this module
    warns = warnings if warnings is not None else []
    root = ET.fromstring(svg_text)
    vx, vy, vw, vh = _viewbox(root)
    if width_dp:
        wdp = float(width_dp)
    else:
        w, h = _length_px(root.get("width")), _length_px(root.get("height"))
        wdp = w if w and w > 0 else (h * vw / vh if h and h > 0 else vw)
        if wdp > 200:
            warns.append(f"{wdp:g}dp > 200dp (Android vectors should stay <= 200dp: pass --dp)")
    hdp = round(wdp * vh / vw, 2)
    ids = {el.get("id"): el for el in root.iter() if el.get("id")}
    _inline_uses(root, ids)
    if not root.get("viewBox", "").strip() and (root.get("width") is None or root.get("height") is None):
        warns.append("no viewBox and no width/height: a 24x24 user-unit viewport is assumed (anything past 24 is clipped)")
    pct_ref = {"x": vw, "cx": vw, "x1": vw, "x2": vw, "width": vw, "rx": vw, "y": vh, "cy": vh, "y1": vh, "y2": vh, "height": vh, "ry": vh,
               "r": math.sqrt((vw * vw + vh * vh) / 2)}  # percentage lengths on shapes resolve against the viewBox (SVG 1.1 7.10)
    for el in root.iter():
        if el.tag.replace(SVG_NS, "") in _SHAPES:
            for k, ref in pct_ref.items():
                v = (el.get(k) or "").strip()
                if v.endswith("%"):
                    el.set(k, f"{_num(v) / 100 * ref:g}")
    css = _css_rules(root)
    tag_of = lambda el: el.tag.replace(SVG_NS, "") if el.tag.startswith(SVG_NS) or "{" not in el.tag else None
    # skews (and matrices with skew) are not expressible as <group>: bake every transform into coordinates first
    for el in root.iter():
        if el.get("transform"):
            try:
                _transforms(el.get("transform"))
            except ValueError as e:
                if "skew" not in str(e) and "not expressible" not in str(e):
                    raise
                baked = _bake_transforms(ET.tostring(root, encoding="unicode"))
                if baked is None:
                    raise ValueError(f"{e}; could not bake it into path coordinates (non-uniform scale on a stroked shape?)")
                root = ET.fromstring(baked)
                ids = {el.get("id"): el for el in root.iter() if el.get("id")}
                warns.append("skew transform baked into path coordinates")
                break
    diag = math.sqrt((vw * vw + vh * vh) / 2)
    grads = {}
    for gel in root.iter():
        tag = tag_of(gel)
        if tag not in ("linearGradient", "radialGradient"):
            continue
        src = gel
        href = gel.get("href") or gel.get("{http://www.w3.org/1999/xlink}href")
        if not any(tag_of(s) == "stop" for s in gel) and href and href[1:] in ids:
            src = ids[href[1:]]
        get = lambda k, dflt=None: gel.get(k) if gel.get(k) is not None else src.get(k, dflt)
        gid = gel.get("id")
        obb = get("gradientUnits") != "userSpaceOnUse"
        stops, prev = [], 0.0
        for s in src:
            if tag_of(s) == "stop":
                st = {**_css_props(s, css), **_style_dict(s)}
                col = st.get("stop-color", s.get("stop-color", "#000000"))
                op = _opacity(st.get("stop-opacity", s.get("stop-opacity")), 1.0)
                off = s.get("offset", "0")
                off = _num(off) / 100 if off.strip().endswith("%") else _num(off)
                prev = min(1.0, max(0.0, off, prev))
                stops.append((prev, _color(col, op)))
        gt = _tf_affine(get("gradientTransform")) if get("gradientTransform") else _IDENTITY
        if tag == "linearGradient":
            rw, rh = (1.0, 1.0) if obb else (vw, vh)
            geo = {"startX": _pct(get("x1"), rw, "0%"), "startY": _pct(get("y1"), rh, "0%"),
                   "endX": _pct(get("x2"), rw, "100%"), "endY": _pct(get("y2"), rh, "0%")}
        else:
            # objectBoundingBox: r% is against the unit box's normalised diagonal sqrt((1+1)/2) = 1 (SVG 1.1 7.10), so
            # the default 50% is 0.5 bbox units; the bbox matrix scales it by sqrt(bw*bh) in geometry()
            rw, rh, rd = (1.0, 1.0, 1.0) if obb else (vw, vh, diag)
            geo = {"centerX": _pct(get("cx"), rw, "50%"), "centerY": _pct(get("cy"), rh, "50%"),
                   "gradientRadius": _pct(get("r"), rd, "50%")}
            fx, fy = get("fx"), get("fy")
            if (fx is not None and abs(_pct(fx, rw, 0) - geo["centerX"]) > 1e-6) or \
               (fy is not None and abs(_pct(fy, rh, 0) - geo["centerY"]) > 1e-6):
                warns.append(f"gradient #{gid}: focal point fx/fy ignored (VectorDrawable radial gradients are concentric)")
        grads[gid] = (tag, geo, stops, {"reflect": "mirror", "repeat": "repeat"}.get(get("spreadMethod")), obb, gt, gid)
    used_grad = False

    def geometry(ref, d):
        """Gradient geometry in the path's user space: objectBoundingBox mapped through the path's bbox, then gradientTransform."""
        from .vectorize import _subpaths  # deferred: vectorize imports this module
        tag, geo, _, _, obb, gt, gid = grads[ref]
        m = gt
        if obb:
            bb = _path_bbox(d)
            if not bb:
                raise ValueError(f"gradient #{gid}: objectBoundingBox needs a path with area")
            bw, bh = max(bb[2] - bb[0], 1e-9), max(bb[3] - bb[1], 1e-9)
            m = _mmul((bw, 0.0, 0.0, bh, bb[0], bb[1]), gt)
            if any(sg[0] == "C" for sp in _subpaths(d) for sg in sp["segs"]):
                warns.append(f"gradient #{gid}: objectBoundingBox mapped through the path's control-point box (curves: approximate)")
        if m == _IDENTITY:
            return dict(geo)
        a, b, c, d_ = m[:4]
        sx, sy = math.hypot(a, b), math.hypot(c, d_)
        similar = abs(sx - sy) < 1e-6 * max(sx, 1) and abs(a * c + b * d_) < 1e-6
        if tag == "linearGradient":
            p1, p2 = (geo["startX"], geo["startY"]), (geo["endX"], geo["endY"])
            q1, q2 = _mapply(m, *p1), _mapply(m, *p2)
            if not similar:  # iso-lines map to lines: keep the mapped direction perpendicular to them (exact)
                iso = (-(p2[1] - p1[1]), p2[0] - p1[0])
                iso = (a * iso[0] + c * iso[1], b * iso[0] + d_ * iso[1])
                nrm = math.hypot(*iso) or 1.0
                n_ = (iso[1] / nrm, -iso[0] / nrm)
                t = (q2[0] - q1[0]) * n_[0] + (q2[1] - q1[1]) * n_[1]
                q2 = (q1[0] + n_[0] * t, q1[1] + n_[1] * t)
            return {"startX": q1[0], "startY": q1[1], "endX": q2[0], "endY": q2[1]}
        cx, cy = _mapply(m, geo["centerX"], geo["centerY"])
        if not similar:
            warns.append(f"gradient #{gid}: elliptical radial gradient approximated by a circle")
        r = geo["gradientRadius"] * math.sqrt(abs(a * d_ - b * c)) if not similar else geo["gradientRadius"] * sx
        return {"centerX": cx, "centerY": cy, "gradientRadius": r}

    def paint(v):
        """SVG paint -> ("none", None) | ("solid", colour) | ("grad", id)."""
        if v in ("none", "transparent", ""):
            return "none", None
        m_ = re.match(r"""url\(\s*['"]?#([^)'"]+)['"]?\s*\)""", v)
        if not m_:
            return "solid", v
        ref = m_.group(1)
        if ref not in grads:
            raise ValueError(f"gradient #{ref} not found")
        stops = grads[ref][2]
        if not stops:
            return "none", None
        if len(stops) == 1:
            return "solid", stops[0][1]
        return "grad", ref

    def grad_xml(ref, attr, indent, d):
        nonlocal used_grad
        used_grad = True
        tag, _, stops, tile, *_ = grads[ref]
        geo = geometry(ref, d)
        if tag == "radialGradient" and geo["gradientRadius"] <= 0:
            raise ValueError(f"gradient #{ref}: radius 0")
        head = f'android:type="{"linear" if tag == "linearGradient" else "radial"}" ' + \
            " ".join(f'android:{k}="{_fmt(v)}"' for k, v in geo.items())
        if tile:
            head += f' android:tileMode="{tile}"'
        items = "".join(f'\n{indent}      <item android:offset="{o:g}" android:color="{c}"/>' for o, c in stops)
        return (f'\n{indent}  <aapt:attr name="android:{attr}">\n{indent}    <gradient {head}>{items}\n'
                f'{indent}    </gradient>\n{indent}  </aapt:attr>')

    def path_xml(d, attrs, ind):
        fk, fv = paint(attrs.get("fill", "#000000"))
        sk, sv = paint(attrs.get("stroke", "none"))
        if any(attrs.get(k) not in (None, "none") for k in ("marker-start", "marker-mid", "marker-end")):
            warns.append("marker-start/mid/end dropped (VectorDrawable has no markers: draw arrowheads as paths)")
        if fk == "none" and sk == "none":
            return ""
        op = attrs.get("opacity", 1.0)
        cur = attrs.get("color")
        out = [f'{ind}<path', f'{ind}    android:pathData="{path_str(parse_path(d))}"']
        children = ""
        if fk != "none":
            if fk == "grad":
                children += grad_xml(fv, "fillColor", ind, d)
            else:
                out.append(f'{ind}    android:fillColor="{_color(fv, 1.0, cur)}"')
            fo = _opacity(attrs.get("fill-opacity"), 1.0) * op
            if fo < 1:
                out.append(f'{ind}    android:fillAlpha="{fo:g}"')
            if attrs.get("fill-rule") == "evenodd":
                out.append(f'{ind}    android:fillType="evenOdd"')
        if sk != "none":
            if sk == "grad":
                children += grad_xml(sv, "strokeColor", ind, d)
            else:
                out.append(f'{ind}    android:strokeColor="{_color(sv, 1.0, cur)}"')
            out.append(f'{ind}    android:strokeWidth="{_num(attrs.get("stroke-width"), 1.0):g}"')
            so = _opacity(attrs.get("stroke-opacity"), 1.0) * op
            if so < 1:
                out.append(f'{ind}    android:strokeAlpha="{so:g}"')
            if attrs.get("stroke-linecap") in ("round", "square"):
                out.append(f'{ind}    android:strokeLineCap="{attrs["stroke-linecap"]}"')
            if attrs.get("stroke-linejoin") in ("round", "bevel"):
                out.append(f'{ind}    android:strokeLineJoin="{attrs["stroke-linejoin"]}"')
            if attrs.get("stroke-miterlimit"):
                out.append(f'{ind}    android:strokeMiterLimit="{_num(attrs["stroke-miterlimit"]):g}"')
            if attrs.get("stroke-dasharray") not in (None, "none", "0"):
                warns.append("stroke-dasharray dropped (VectorDrawable has no dashes; stroke drawn solid)")
            if attrs.get("vector-effect") == "non-scaling-stroke":
                warns.append("vector-effect=non-scaling-stroke ignored (the stroke scales with its group)")
            if "stroke" in (attrs.get("paint-order") or "") and (attrs.get("paint-order") or "").split()[0] == "stroke":
                warns.append("paint-order=stroke ignored (VectorDrawable always paints the stroke over the fill)")
        return "\n".join(out) + (f">{children}\n{ind}</path>" if children else " />")

    def wrap(body, tf, ind):
        groups = _transforms(tf)
        if "strokeWidth" in body and any(abs(g.get("scaleX", 1) - g.get("scaleY", 1)) > 1e-6 for g in groups):
            warns.append("non-uniform scale on a stroked path: Android scales the stroke width approximately")
        for g in reversed(groups):
            attrs = " ".join(f'android:{k}="{v:g}"' for k, v in g.items() if v or k.startswith("scale"))
            body = f"{ind}<group {attrs}>\n" + "\n".join("    " + ln for ln in body.split("\n")) + f"\n{ind}</group>"
        return body

    def clip_of(ref, el_paths, ind, body):
        """<group> with a <clip-path> around body; None when the clip is a rect covering the whole viewBox."""
        from .svg_icons import _SHAPES  # deferred: svg_icons imports this module
        from .vectorize import _subpaths  # deferred: vectorize imports this module
        cel = ids.get(ref)
        if cel is None or tag_of(cel) != "clipPath":
            raise ValueError(f"clip-path #{ref} not found")
        obb = cel.get("clipPathUnits") == "objectBoundingBox"
        base = _IDENTITY
        if obb:
            boxes = [b for b in (_path_bbox(d) for d in el_paths) if b]
            if not boxes:
                raise ValueError(f"clip-path #{ref}: objectBoundingBox needs content with area")
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            base = (x1 - x0, 0.0, 0.0, y1 - y0, x0, y0)
            warns.append(f"clip-path #{ref}: objectBoundingBox mapped through the content's control-point box (approximate)")
        parts = []
        for ch in cel:
            ctag = tag_of(ch)
            if ctag not in _SHAPES:
                if ctag and ctag not in ("title", "desc"):
                    warns.append(f"clip-path #{ref}: <{ctag}> ignored")
                continue
            cattrs = _inherit(ch, _inherit(cel, {}, css), css)
            evenodd = cattrs.get("clip-rule") == "evenodd" or cattrs.get("fill-rule") == "evenodd"
            if ch.get("clip-path") or cel.get("clip-path"):
                warns.append(f"clip-path #{ref}: nested clip-path ignored")
            m = _mmul(base, _tf_affine(ch.get("transform"))) if ch.get("transform") else base
            d = _shape_path(ch)
            if not d:
                continue
            if ctag == "rect" and m == _IDENTITY and not ch.get("rx") and not ch.get("ry") and len(list(cel)) == 1:
                x, y = _num(ch.get("x")), _num(ch.get("y"))
                w, h = _num(ch.get("width")), _num(ch.get("height"))
                if x <= vx + 1e-6 and y <= vy + 1e-6 and x + w >= vx + vw - 1e-6 and y + h >= vy + vh - 1e-6:
                    return None
            part = _abs_path(d, m) if m != _IDENTITY or ctag != "path" else path_str(parse_path(d))
            if evenodd and len(_subpaths(part)) > 1:  # <clip-path> is always nonzero: rewind the holes instead
                part = _rewind_evenodd(part)
                warns.append(f"clip-path #{ref}: evenodd clip-rule: subpaths rewound by nesting for the nonzero clip "
                             "(exact unless subpaths cross)")
            parts.append(part)
        if not parts:
            warns.append(f"clip-path #{ref} has no shapes: everything it clips is invisible")
            return ""
        inner = "\n".join("    " + ln for ln in body.split("\n"))
        return f'{ind}<group>\n{ind}    <clip-path android:pathData="{" ".join(parts)}"/>\n{inner}\n{ind}</group>'

    def clip_ref(el):
        v = _style_dict(el).get("clip-path") or _css_props(el, css).get("clip-path") or el.get("clip-path")
        m_ = re.match(r"""url\(\s*['"]?#([^)'"]+)['"]?\s*\)""", (v or "").strip())
        return m_.group(1) if m_ else None

    def walk(el, inherited, ind):
        from .svg_icons import _SHAPES  # deferred: svg_icons imports this module
        lines = []
        children = list(el)
        if tag_of(el) == "switch":
            first = [ch for ch in children if tag_of(ch) in _SHAPES + ("g", "a", "svg", "use", "text", "image")][:1]
            if len(children) > len(first):
                warns.append("<switch>: only its first renderable child is converted")
            children = first
        for ch in children:
            tag = tag_of(ch)
            if tag is None or tag in ("defs", "linearGradient", "radialGradient", "title", "desc", "metadata", "style",
                                      "clipPath", "symbol", "marker", "script"):
                continue
            if tag in ("text", "image", "mask", "filter", "pattern", "foreignObject"):
                raise ValueError(f"<{tag}> is not convertible")
            attrs = _inherit(ch, inherited, css)
            if attrs.get("display") == "none" or _style_dict(ch).get("display") == "none":
                continue
            if _style_dict(ch).get("mask") or _css_props(ch, css).get("mask") or ch.get("mask") or ch.get("filter") or \
               _style_dict(ch).get("filter"):
                raise ValueError(f"<{tag}> uses mask/filter (not convertible)")
            paths = []
            if tag in ("g", "a", "switch"):
                body = walk(ch, attrs, ind)
                paths = [x.get("d") or _shape_path(x) for x in ch.iter() if tag_of(x) in _SHAPES]
            elif tag in _SHAPES:
                if attrs.get("visibility") in ("hidden", "collapse"):
                    continue
                d = _shape_path(ch)
                paths = [d] if d else []
                body = path_xml(d, attrs, ind) if d else ""
            else:
                warns.append(f"<{tag}> ignored (not convertible: nested svg, foreignObject, markers, ...)"
                             if tag in ("svg",) else f"<{tag}> ignored")
                continue
            if body and clip_ref(ch):
                clipped = clip_of(clip_ref(ch), [p_ for p_ in paths if p_], ind, body)
                body = body if clipped is None else clipped
            if body:
                lines.append(wrap(body, ch.get("transform"), ind) if ch.get("transform") else body)
        return "\n".join(lines)
    body = walk(root, _inherit(root, {}, css), "    ")
    if clip_ref(root):
        clipped = clip_of(clip_ref(root), [x.get("d") or _shape_path(x) for x in root.iter() if tag_of(x) in _SHAPES], "    ", body)
        body = body if clipped is None else clipped
    if root.get("transform"):
        body = wrap(body, root.get("transform"), "    ")
    if vx or vy:
        body = wrap(body, f"translate({-vx:g} {-vy:g})", "    ")
    ns = ' xmlns:aapt="http://schemas.android.com/aapt"' if used_grad else ""
    api24 = [k for k, v in (("evenOdd fill", 'android:fillType="evenOdd"' in body), ("gradient", used_grad)) if v]
    if api24:
        warns.append(f"needs API 24+ or AppCompat srcCompat: {' / '.join(api24)}")
    return (f'<?xml version="1.0" encoding="utf-8"?>\n<vector xmlns:android="http://schemas.android.com/apk/res/android"{ns}\n'
            f'    android:width="{wdp:g}dp"\n    android:height="{hdp:g}dp"\n'
            f'    android:viewportWidth="{vw:g}"\n    android:viewportHeight="{vh:g}">\n' + body + "\n</vector>\n")


AND = "{http://schemas.android.com/apk/res/android}"


def vd_to_svg(xml):
    """Render-equivalent SVG of a VectorDrawable (used to verify conversions)."""
    root = ET.fromstring(xml)
    g = lambda el, k, dflt=None: el.get(AND + k, dflt)
    defs, n = [], [0]

    def colour(v, alpha):
        v = v.lstrip("#")
        a = int(v[:2], 16) / 255 if len(v) == 8 else 1.0
        return "#" + v[-6:], a * alpha

    def paint(el, attr, alpha):
        grad = next((ga for ga in el if ga.tag.endswith("attr") and ga.get("name") == f"android:{attr}"), None)
        if grad is not None:
            gr = grad[0]
            n[0] += 1
            stops = "".join(f'<stop offset="{g(it, "offset")}" stop-color="{colour(g(it, "color"), 1)[0]}" '
                            f'stop-opacity="{colour(g(it, "color"), 1)[1]:g}"/>' for it in gr)
            if g(gr, "type") == "radial":
                head = f'<radialGradient id="p{n[0]}" gradientUnits="userSpaceOnUse" cx="{g(gr, "centerX")}" cy="{g(gr, "centerY")}" r="{g(gr, "gradientRadius")}"'
                tagn = "radialGradient"
            else:
                head = f'<linearGradient id="p{n[0]}" gradientUnits="userSpaceOnUse" x1="{g(gr, "startX")}" y1="{g(gr, "startY")}" x2="{g(gr, "endX")}" y2="{g(gr, "endY")}"'
                tagn = "linearGradient"
            spread = {"mirror": "reflect", "repeat": "repeat"}.get(g(gr, "tileMode"), "pad")
            defs.append(f'{head} spreadMethod="{spread}">{stops}</{tagn}>')
            return f"url(#p{n[0]})", alpha
        v = g(el, attr)
        return colour(v, alpha) if v else ("none", 1.0)

    def walk(el):
        out, open_clips = [], 0
        for ch in el:
            if ch.tag == "group":
                tx, ty = float(g(ch, "translateX", 0)), float(g(ch, "translateY", 0))
                px, py = float(g(ch, "pivotX", 0)), float(g(ch, "pivotY", 0))
                tf = (f"translate({tx + px:g} {ty + py:g}) rotate({g(ch, 'rotation', 0)}) "
                      f"scale({g(ch, 'scaleX', 1)} {g(ch, 'scaleY', 1)}) translate({-px:g} {-py:g})")
                out.append(f'<g transform="{tf}">{walk(ch)}</g>')
            elif ch.tag == "clip-path":  # clips the siblings that follow it, like VectorDrawable does
                n[0] += 1
                defs.append(f'<clipPath id="c{n[0]}"><path d="{g(ch, "pathData")}"/></clipPath>')
                out.append(f'<g clip-path="url(#c{n[0]})">')
                open_clips += 1
            elif ch.tag == "path":
                fill, fa = paint(ch, "fillColor", float(g(ch, "fillAlpha", 1)))
                stroke, sa = paint(ch, "strokeColor", float(g(ch, "strokeAlpha", 1)))
                out.append(f'<path d="{g(ch, "pathData")}" fill="{fill}" fill-opacity="{fa:g}" stroke="{stroke}" '
                           f'stroke-opacity="{sa:g}" stroke-width="{g(ch, "strokeWidth", 0)}" '
                           f'stroke-linecap="{g(ch, "strokeLineCap", "butt")}" stroke-linejoin="{g(ch, "strokeLineJoin", "miter")}" '
                           f'stroke-miterlimit="{g(ch, "strokeMiterLimit", 4)}" '
                           f'fill-rule="{"evenodd" if g(ch, "fillType") == "evenOdd" else "nonzero"}"/>')
        return "".join(out) + "</g>" * open_clips
    body = walk(root)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {g(root, "viewportWidth")} {g(root, "viewportHeight")}">'
            f'<defs>{"".join(defs)}</defs>{body}</svg>')


def vd_roundtrip(pairs, size=128):
    """[(svg, vd xml)] -> per pair {"mean": mean premultiplied RGBA difference (0-255), "strong": pixels whose largest
    channel difference exceeds 64} between the SVG and its VectorDrawable rendered, or None without a renderer.
    Correct conversions measure strong == 0 at 128 px; dropped or moved content measures 90+ even when mean < 2."""
    ims = render_svgs([s for s, _ in pairs] + [vd_to_svg(x) for _, x in pairs], size)
    out = []
    for a, b in zip(ims[:len(pairs)], ims[len(pairs):]):
        if a is None or b is None:
            out.append(None)
            continue
        pa, pb = (np.asarray(im, dtype=float) for im in (a, b))
        pa[..., :3] *= pa[..., 3:] / 255
        pb[..., :3] *= pb[..., 3:] / 255
        diff = np.abs(pa - pb)
        out.append({"mean": round(float(diff.mean()), 2), "strong": int((diff.max(2) > 64).sum())})
    return out


@functools.cache
def _android_sdk():
    homes = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"), "~/Library/Android/sdk", "~/Android/Sdk"]
    for h in homes:
        if h and Path(h).expanduser().is_dir():
            sdk = Path(h).expanduser()
            ver = lambda p: [int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", p.parent.name.replace("android-", ""))]
            tools = sorted(sdk.glob("build-tools/*/aapt2"), key=ver)
            jars = sorted(sdk.glob("platforms/android-*/android.jar"), key=ver)
            if tools:
                return str(tools[-1]), str(jars[-1]) if jars else None
    return (shutil.which("aapt2"), None) if shutil.which("aapt2") else None


def aapt2_level():
    """"link" when aapt2 and a platforms/android-*/android.jar are available, "compile" with aapt2 alone, else None."""
    sdk = _android_sdk()
    return None if not sdk else ("link" if sdk[1] else "compile")


def _aapt2_line(n=None):
    """The one-line aapt2 verdict for `n` files that passed every available step (plan contract C4 wording)."""
    lvl, files = aapt2_level(), f": {n} file(s)" if n is not None else ""
    if lvl == "link":
        return f"aapt2 compile+link ok ({Path(_android_sdk()[1]).parent.name}){files}"
    if lvl == "compile":
        return ("aapt2 compile ok; link skipped (no $ANDROID_HOME/platforms/android-*/android.jar: attribute names, enums and "
                'dimensions NOT checked; sdkmanager "platforms;android-35")' + files)
    return "aapt2 not found (set ANDROID_HOME): not validated"


_VD_ATTRS = {
    "vector": {"width", "height", "viewportWidth", "viewportHeight", "tint", "tintMode", "autoMirrored", "alpha", "name",
               "opticalInsetLeft", "opticalInsetTop", "opticalInsetRight", "opticalInsetBottom"},
    "group": {"name", "rotation", "pivotX", "pivotY", "scaleX", "scaleY", "translateX", "translateY"},
    "path": {"name", "pathData", "fillColor", "strokeColor", "strokeWidth", "strokeAlpha", "fillAlpha", "trimPathStart",
             "trimPathEnd", "trimPathOffset", "strokeLineCap", "strokeLineJoin", "strokeMiterLimit", "fillType"},
    "clip-path": {"name", "pathData"},
    "gradient": {"type", "startX", "startY", "endX", "endY", "centerX", "centerY", "gradientRadius", "startColor",
                 "centerColor", "endColor", "tileMode"},
    "item": {"offset", "color"},
}
_VD_ENUMS = {"fillType": {"evenOdd", "nonZero"}, "strokeLineCap": {"butt", "round", "square"},
             "strokeLineJoin": {"miter", "round", "bevel"}, "tileMode": {"clamp", "repeat", "mirror", "disabled"},
             "type": {"linear", "radial", "sweep"}, "autoMirrored": {"true", "false"},
             "tintMode": {"add", "multiply", "screen", "src_atop", "src_in", "src_over"}}
_VD_COLORS = {"fillColor", "strokeColor", "startColor", "centerColor", "endColor", "color", "tint"}
_VD_FLOATS = {"viewportWidth", "viewportHeight", "rotation", "pivotX", "pivotY", "scaleX", "scaleY", "translateX", "translateY",
              "strokeWidth", "strokeAlpha", "fillAlpha", "trimPathStart", "trimPathEnd", "trimPathOffset", "strokeMiterLimit",
              "alpha", "startX", "startY", "endX", "endY", "centerX", "centerY", "gradientRadius", "offset"}
AAPT_NS = "{http://schemas.android.com/aapt}"


def _vd_lint(xml_text):
    """Offline VectorDrawable check (what `aapt2 link` would reject, plus runtime crashes aapt2 never sees): tag and
    attribute whitelist, enums, dimensions with units, colour formats, numbers, positive viewport, nan/inf and syntax
    in pathData, radial radius > 0, at least two gradient stops. [] = clean."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return [f"not well-formed XML: {e}"]
    errs = []
    if root.tag != "vector":
        return [f"root element must be <vector>, got <{root.tag}>"]
    dim = re.compile(r"^-?(\d+\.?\d*|\.\d+)(dp|dip|px|sp|pt|mm|in)$|^@(android:)?dimen/|^\?")
    col = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$|^@(android:)?color/|^\?")

    def check(el, parent):
        tag = el.tag.replace(AAPT_NS, "aapt:")
        allowed = _VD_ATTRS.get(tag)
        if tag == "aapt:attr":
            if set(el.attrib) - {"name"} or not re.fullmatch(r"android:(fillColor|strokeColor)", el.get("name", "")):
                errs.append(f'<aapt:attr> needs exactly name="android:fillColor|strokeColor"')
            if len(el) != 1 or el[0].tag != "gradient":
                errs.append("<aapt:attr> must contain exactly one <gradient>")
        elif allowed is None:
            errs.append(f"<{tag}> is not a VectorDrawable element" + (f" (inside <{parent}>)" if parent else ""))
            return
        else:
            for k, v in el.attrib.items():
                if not k.startswith(AND):
                    if not k.startswith("{"):
                        errs.append(f"<{tag}> {k}: attribute needs the android: namespace")
                    continue
                k = k[len(AND):]
                if k not in allowed:
                    errs.append(f"<{tag}> android:{k}: unknown attribute")
                    continue
                if k in _VD_ENUMS and v not in _VD_ENUMS[k]:
                    errs.append(f'<{tag}> {k}="{v}": expected {"|".join(sorted(_VD_ENUMS[k]))}')
                elif k in ("width", "height") and not dim.match(v):
                    errs.append(f'<{tag}> {k}="{v}": needs a dimension with unit (e.g. {v}dp)')
                elif k in _VD_COLORS and not col.match(v):
                    errs.append(f'<{tag}> {k}="{v}": not a colour (#RGB, #ARGB, #RRGGBB, #AARRGGBB or @color/)')
                elif k in _VD_FLOATS:
                    try:
                        f = float(v)
                        if math.isnan(f) or math.isinf(f):
                            raise ValueError
                    except ValueError:
                        errs.append(f'<{tag}> {k}="{v}": not a number')
                        continue
                    if k in ("viewportWidth", "viewportHeight") and f <= 0:
                        errs.append(f'<{tag}> {k}="{v}": must be > 0')
                    if k == "gradientRadius" and f <= 0:
                        errs.append(f'<gradient> gradientRadius="{v}": must be > 0 (radial gradients crash at inflation otherwise)')
                elif k == "pathData":
                    if re.search(r"\b(nan|inf)", v, re.I):
                        errs.append(f"<{tag}> pathData contains nan/inf")
                    else:
                        try:
                            parse_path(v)
                        except ValueError as e:
                            errs.append(f"<{tag}> pathData: {e}")
        if tag == "gradient":
            stops = [c for c in el if c.tag == "item"]
            if len(stops) < 2 and not (el.get(AND + "startColor") and el.get(AND + "endColor")):
                errs.append(f"<gradient> has {len(stops)} stop(s): Android needs at least two colours")
            if el.get(AND + "type") == "radial" and el.get(AND + "gradientRadius") is None:
                errs.append("<gradient android:type=\"radial\"> without gradientRadius")
        if tag == "vector":
            for k in ("width", "height", "viewportWidth", "viewportHeight"):
                if el.get(AND + k) is None:
                    errs.append(f"<vector> is missing android:{k}")
        for ch in el:
            if tag in ("gradient",) and ch.tag != "item":
                errs.append(f"<{ch.tag}> inside <gradient>")
                continue
            check(ch, tag)
    check(root, None)
    return errs


def aapt2_check(xml_files):
    """Compile (+ link when an android.jar exists: see aapt2_level) VectorDrawable files with aapt2.
    {file name: first real error line}; {} = every available step passed; None = no aapt2."""
    sdk = _android_sdk()
    if not sdk:
        return None
    aapt2, jar = sdk
    errs = {}
    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "res" / "drawable"
        res.mkdir(parents=True)
        for f in xml_files:
            shutil.copy(f, res / Path(f).name)
        (Path(td) / "AndroidManifest.xml").write_text(
            '<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="gptkit.check"/>')
        steps = [[aapt2, "compile", "-o", f"{td}/c.zip", "--dir", str(res.parent)]]
        if jar:
            steps.append([aapt2, "link", "-o", f"{td}/o.apk", "-I", jar, "--manifest", f"{td}/AndroidManifest.xml", f"{td}/c.zip"])
        generic = ("file failed to compile", "failed linking", "failed processing")
        for cmd in steps:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                return {"*": f"aapt2 did not run: {e}"}
            for line in (r.stderr + r.stdout).splitlines():
                m_ = re.search(r"drawable/([^/:]+\.xml)(?::\d+)?: error: (.*)", line)
                if m_ and (m_.group(1) not in errs or errs[m_.group(1)].startswith(generic)):
                    errs[m_.group(1)] = m_.group(2).strip()
            if r.returncode and not errs:
                errs["*"] = (r.stderr or r.stdout).strip()[:300] or f"aapt2 {cmd[1]} exited {r.returncode}"
            if errs:
                break
    return errs


def _aapt2_tree(res_dir, manifest=None, work=None, extra_res=None):
    """aapt2 over a whole res/ tree (drawables next to their PNGs, mipmaps, fonts): compile always, link when a platform
    android.jar exists (see aapt2_level). One line per contract C4: _aapt2_line() when every available step passed,
    'aapt2 compile|link FAILED: <first error text>' otherwise. `manifest` replaces the minimal AndroidManifest.xml used
    for link; `work` keeps the compiled zip/apk instead of a temp dir; `extra_res` ({"values/x.xml": text}) is added to a
    temporary copy of the tree (what a library such as androidx.core contributes at build time), never to res_dir."""
    sdk = _android_sdk()
    if not sdk:
        return _aapt2_line()
    aapt2, jar = sdk
    with tempfile.TemporaryDirectory() as td:
        w = Path(work) if work else Path(td)
        w.mkdir(parents=True, exist_ok=True)
        if extra_res:
            shutil.copytree(res_dir, Path(td) / "res")
            res_dir = Path(td) / "res"
            for rel, text in extra_res.items():
                (res_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                (res_dir / rel).write_text(text)
        steps = [("compile", [aapt2, "compile", "-o", str(w / "res.zip"), "--dir", str(res_dir)])]
        if jar:
            (w / "AndroidManifest.xml").write_text(
                manifest or '<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="gptkit.check"/>')
            steps.append(("link", [aapt2, "link", "-o", str(w / "check.apk"), "-I", jar, "--manifest",
                                   str(w / "AndroidManifest.xml"), str(w / "res.zip")]))
        for step, cmd in steps:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                return f"aapt2 {step} FAILED: aapt2 did not run ({e})"
            if r.returncode:
                return f"aapt2 {step} FAILED: " + (r.stderr or r.stdout).strip()[:300]
    return _aapt2_line()


def _report_dp(p):
    """Per-icon dp written by `svg` into svg_report.json next to (or one level above) the SVG, or None."""
    for d in (p.parent, p.parent.parent):
        rp = d / "svg_report.json"
        if rp.exists():
            try:
                dp = json.loads(rp.read_text()).get(p.stem, {}).get("dp")
                dp = float(dp) if dp else None
                if dp is not None and not 0 < dp < 1e5:  # also nan
                    raise ValueError(dp)
                return dp
            except (OSError, ValueError, AttributeError, TypeError):
                print(f"  ! {rp}: no usable dp for {p.stem} (hand-edited?); pass --dp")
                return None
    return None


def convert_vd(paths, out_dir, dp=None, prefix="ic_", check=True):
    """Convert SVG files to out_dir/<res name>.xml with the offline lint, render round-trip and aapt2; returns the number
    of files that failed (not written, lint or aapt2 error) after printing one line per file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, names, failed = [], {}, 0
    for p in paths:
        p, warns = Path(p), []
        try:
            svg = p.read_text(encoding="utf-8-sig")
            xml = svg_to_vd(svg, dp or _report_dp(p), warns)
        except OSError as e:
            print(f"  ! {p}: cannot read ({e.strerror or e})")
            failed += 1
            continue
        except (ValueError, ET.ParseError) as e:
            print(f"  ! {p.name}: {e}")
            failed += 1
            continue
        name = _res_name(prefix + p.stem, "ic_")
        if name in names:
            base, k = name, 2
            while name in names:
                name, k = f"{base}_{k}", k + 1
            print(f"  ! {p}: resource name {base} is already used by {names[base]}; writing {name}.xml")
        elif name != prefix + p.stem:
            print(f"  renamed {p.name} -> {name}.xml (Android resource names: [a-z0-9_], starting with a letter)")
        names[name] = p
        out = out_dir / f"{name}.xml"
        out.write_text(xml)
        done.append((out, svg, xml, warns))
    rts = vd_roundtrip([(s, x) for _, s, x, _ in done]) if done and check else [None] * len(done)
    for (out, _, xml, warns), rt in zip(done, rts):
        notes = [f"  ! {w}" for w in dict.fromkeys(warns) if not w.startswith("needs API")]
        notes += [f"  ({w})" for w in dict.fromkeys(warns) if w.startswith("needs API")]
        lint = _vd_lint(xml)
        if lint:
            failed += 1
        if rt is not None and (rt["mean"] > 2 or rt["strong"] > 16):
            notes.append(f"  ! render differs from the SVG (mean diff {rt['mean']}, {rt['strong']} px far off)")
        match = f"  (render match {rt['mean']})" if rt is not None and rt["mean"] <= 2 and rt["strong"] <= 16 else ""
        print(f"  {out}{match}{''.join(notes)}")
        for e in lint:
            print(f"      ! lint: {e}")
    if done and check and all(rt is None for rt in rts):
        why = "no SVG renderer" if not _renderer_name() else "renderer failed"
        print(f"  (render check skipped: {why}{': ' + _chrome_probe()[1] if not _renderer_name() else ''})")
    written = [x[0] for x in done]
    if written and check:
        errs = aapt2_check(written)
        if errs is None:
            print(f"  {_aapt2_line()}")
        elif errs:
            for k, v in errs.items():
                print(f"  ! aapt2: {k}: {v}")
            failed += len(errs)
        else:
            print(f"  {_aapt2_line(len(written))}")
    return failed


def cmd_vd(a):
    paths = []
    for f in a.svgs:
        p = Path(f)
        if p.is_dir():
            found = sorted(p.glob("*.svg"))
            if not found:
                print(f"  ! {p}: no .svg files in this directory")
            paths += found
        else:
            paths.append(p)
    if not paths:
        sys.exit("vd: no SVG files to convert")
    if convert_vd(paths, a.out, a.dp, a.prefix, not a.no_check):
        sys.exit(1)
