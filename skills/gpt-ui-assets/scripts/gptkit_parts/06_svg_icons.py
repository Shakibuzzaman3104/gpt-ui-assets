# gptkit part 06_svg_icons: svg, icons, iconspec.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---- SVG icons: fidelity retry, 24-grid normalisation, icon-set spec + consistency, new icons

SVG_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["files"], "properties": {"files": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "svg"],
                               "properties": {"name": {"type": "string"}, "svg": {"type": "string"}}}}}}

ICON_SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["style", "stroke_px", "stroke_24", "linecap", "linejoin", "corner_radius_24", "traits", "states", "icons"],
    "properties": {
        "style": {"type": "string", "enum": ["outlined", "filled", "mixed", "duotone"]},
        "stroke_px": {"type": "number"}, "stroke_24": {"type": "number"},
        "linecap": {"type": "string", "enum": ["round", "butt", "square"]},
        "linejoin": {"type": "string", "enum": ["round", "miter", "bevel"]},
        "corner_radius_24": {"type": "number"}, "traits": {"type": "string"},
        "states": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["state", "treatment", "color"],
            "properties": {"state": {"type": "string"}, "treatment": {"type": "string", "enum": ["outlined", "filled"]},
                           "color": {"type": "string"}}}},
        "icons": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "state", "treatment", "material_symbol", "sf_symbol"],
            "properties": {"name": {"type": "string"}, "state": {"type": "string"},
                           "treatment": {"type": "string", "enum": ["outlined", "filled"]},
                           "material_symbol": {"type": "string"}, "sf_symbol": {"type": "string"}}}}},
}

ICON_SPEC_PROMPT = """This is an APPROVED UI design ({W}x{H} px). After it come zoomed crops of its icons, each labelled with
its name. Describe the icon SET's shared style precisely enough that new icons can be drawn to match:
- style, stroke width in design px for the most common icon size, and the same stroke on a 24-unit grid whose live
  area is 20 units (stroke_px * 20 / icon size in px), line caps and joins, typical corner radius on that 24 grid;
- traits: geometry (geometric/organic), terminal shapes, level of detail, how active/selected icons differ;
- states: each visual state (e.g. "tab inactive", "tab active", "on accent button") with its treatment and hex colour;
- icons: for every listed icon, its state and treatment, plus the closest standard glyph name in Google Material
  Symbols (snake_case, e.g. home, settings, favorite, person, explore, play_arrow, chevron_right, graphic_eq, air) and
  Apple SF Symbols (e.g. house.fill, gearshape, heart, person, safari, play.fill, chevron.right, waveform, wind).
  Use "" when no standard glyph is a close match."""

SVG_FIX_PROMPT = """You wrote these SVGs for assets of an APPROVED design; each one either does not match the design closely
enough or breaks a VectorDrawable rule listed below. For each asset the image (when present) shows LEFT the design element
(zoomed screenshot on its backdrop) and RIGHT your SVG rendered at the same scale on the same backdrop; then come the
measured problems and your current SVG. Rewrite each SVG so RIGHT matches LEFT 1:1 and every rule holds: fix the listed
problems, keep what already matches, do not simplify, return the same names."""

ICONS_PROMPT = """Design NEW icons for the app in the first image (an APPROVED design). They must be members of the SAME icon
set as its existing icons (the second image shows them; their SVG sources follow): same grid, stroke weight, caps/joins,
corner rounding, level of detail and optical size, so nobody can tell which icons were added later.
Concepts: {concepts}.
Draw each on a 24x24 grid: viewBox="0 0 24 24", artwork inside the 2..22 live area and optically centred, {treatment},
stroke-width="{sw}", stroke-linecap="{cap}", stroke-linejoin="{join}", colour {color}. Use the simplest recognisable
metaphor (the one Material Symbols / SF Symbols use for it). No text, no background shape. Name each file after its
concept in snake_case."""

_SHAPES = ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon")


def _vb(root):
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if v]
    return vb if len(vb) == 4 else [0, 0, _num(root.get("width"), 24), _num(root.get("height"), 24)]


def _units_box(render, vb, par="xMidYMid meet"):
    """Alpha bounding box of a render (made by render_svgs: the SVG sized to a square) in viewBox units, honouring the
    root's preserveAspectRatio for non-square viewBoxes."""
    bb = content_box(render, 25)
    if not bb:
        return None
    R = render.width
    words = (par or "xMidYMid meet").split()
    align, slice_ = words[0], "slice" in words
    if align == "none":
        sx, sy, ox, oy = R / vb[2], R / vb[3], 0.0, 0.0
    else:
        sx = sy = R / (min(vb[2], vb[3]) if slice_ else max(vb[2], vb[3]))
        pos = lambda key, span: 0.0 if key + "min" in align.lower() else (R - span) if key + "max" in align.lower() else (R - span) / 2
        ox, oy = pos("x", vb[2] * sx), pos("y", vb[3] * sy)
    return (vb[0] + (bb[0] - ox) / sx, vb[1] + (bb[1] - oy) / sy, vb[0] + (bb[2] - ox) / sx, vb[1] + (bb[3] - oy) / sy)


def _bake_transforms(txt):
    """The SVG with every transform multiplied into absolute path coordinates (arcs become cubics), so the grid
    normaliser can rescale it. None when it cannot be done exactly: non-uniform scale or skew on stroked shapes,
    gradientTransform, nested <svg>, <use>, or percent units."""
    root = ET.fromstring(txt)
    if not any(el.get("transform") for el in root.iter()):
        return txt
    if any(el.get("gradientTransform") for el in root.iter()) or \
       any(el.tag.replace(SVG_NS, "") in ("svg", "use", "symbol") for el in list(root.iter())[1:]):
        return None
    mul, apply = _mmul, lambda m, pt: _mapply(m, *pt)  # the one affine helper set (no private copies)
    ok = [True]

    def walk(el, m):
        for ch in list(el):
            tag = ch.tag.replace(SVG_NS, "")
            cm = mul(m, _tf_affine(ch.get("transform"))) if ch.get("transform") else m
            if ch.get("transform"):
                del ch.attrib["transform"]
            if tag in ("g", "a", "switch", "defs", "clipPath", "mask", "marker", "pattern"):
                walk(ch, cm)
                continue
            if tag.endswith("Gradient"):
                if cm != _IDENTITY:
                    ok[0] = False
                continue
            if tag not in _SHAPES or cm == _IDENTITY:
                continue
            a, b, c, d = cm[:4]
            uniform = abs(math.hypot(a, b) - math.hypot(c, d)) < 1e-6 and abs(a * c + b * d) < 1e-6
            st = _style_dict(ch)
            stroked = (st.get("stroke") or ch.get("stroke") or (el.get("stroke") if tag != "g" else None) or "none") != "none"
            if not uniform and (stroked or root.get("stroke") not in (None, "none")):
                ok[0] = False
                return
            dpath = _shape_path(ch)
            if not dpath or any("%" in (ch.get(k) or "") for k in ch.attrib):
                ok[0] = False
                return
            subs = _subpaths(dpath)
            for sp in subs:
                sp["pts"] = [apply(cm, pt) for pt in sp["pts"]]
                sp["segs"] = [(sg[0], *[apply(cm, pt) for pt in sg[1:]]) for sg in sp["segs"]]
            k = math.hypot(a, b)
            for key in ("stroke-width", "stroke-dashoffset"):
                if st.get(key):
                    st[key] = _fmt(_num(st[key]) * k)
                elif ch.get(key):
                    ch.set(key, _fmt(_num(ch.get(key)) * k))
                elif key == "stroke-width" and stroked and abs(k - 1) > 1e-6:
                    ch.set(key, _fmt(_num(el.get(key) or root.get(key), 1.0) * k))
            if st.get("stroke-dasharray") not in (None, "none"):
                st["stroke-dasharray"] = " ".join(_fmt(float(v) * k) for v in _NUM.findall(st["stroke-dasharray"]))
            elif ch.get("stroke-dasharray") not in (None, "none"):
                ch.set("stroke-dasharray", " ".join(_fmt(float(v) * k) for v in _NUM.findall(ch.get("stroke-dasharray"))))
            if st:
                ch.set("style", ";".join(f"{k_}:{v}" for k_, v in st.items()))
            for key in ("cx", "cy", "r", "rx", "ry", "x", "y", "width", "height", "x1", "y1", "x2", "y2", "points"):
                ch.attrib.pop(key, None)
            ch.tag = SVG_NS + "path" if ch.tag.startswith(SVG_NS) else "path"
            ch.set("d", _d(subs))
    walk(root, _IDENTITY)
    return ET.tostring(root, encoding="unicode") if ok[0] else None


def normalize_svg(txt, box, grid=24, pad=2.0, scale=None):
    """Scale coordinates (never adding transforms) so content box `box` fits a grid x grid viewBox inside `pad`;
    `scale` overrides the fit factor (set-level normalisation: one scale per size group), the box is still centred."""
    baked = _bake_transforms(txt)
    if baked is None:
        return None
    root = ET.fromstring(baked)
    s = scale or (grid - 2 * pad) / max(box[2] - box[0], box[3] - box[1], 1e-6)
    tx, ty = grid / 2 - s * (box[0] + box[2]) / 2, grid / 2 - s * (box[1] + box[3]) / 2
    fx = {"x": lambda v: _fmt(_num(v) * s + tx), "y": lambda v: _fmt(_num(v) * s + ty), "l": lambda v: _fmt(_num(v) * s)}
    attrs = {"circle": {"cx": "x", "cy": "y", "r": "l"}, "ellipse": {"cx": "x", "cy": "y", "rx": "l", "ry": "l"},
             "rect": {"x": "x", "y": "y", "width": "l", "height": "l", "rx": "l", "ry": "l"},
             "line": {"x1": "x", "y1": "y", "x2": "x", "y2": "y"},
             "linearGradient": {"x1": "x", "y1": "y", "x2": "x", "y2": "y"},
             "radialGradient": {"cx": "x", "cy": "y", "r": "l", "fx": "x", "fy": "y", "fr": "l"}}
    stroked = False
    for el in root.iter():
        tag = el.tag.replace(SVG_NS, "")
        if tag.endswith("Gradient") and el.get("gradientUnits") != "userSpaceOnUse":
            continue
        for k, kind in attrs.get(tag, {}).items():
            if el.get(k) is not None:
                if "%" in el.get(k):
                    return None
                el.set(k, fx[kind](el.get(k)))
        if tag == "path" and el.get("d"):
            el.set("d", path_str(transform_path(parse_path(el.get("d")), s, tx, ty)))
        if tag in ("polyline", "polygon"):
            pts = [float(v) for v in _NUM.findall(el.get("points", ""))]
            el.set("points", " ".join(f"{_fmt(pts[i] * s + tx)},{_fmt(pts[i + 1] * s + ty)}" for i in range(0, len(pts) - 1, 2)))
        st = _style_dict(el)
        for k in ("stroke-width", "stroke-dashoffset"):
            if el.get(k) is not None:
                el.set(k, fx["l"](el.get(k)))
            if k in st:
                st[k] = fx["l"](st[k])
        if el.get("stroke-dasharray") not in (None, "none"):
            el.set("stroke-dasharray", " ".join(_fmt(float(v) * s) for v in _NUM.findall(el.get("stroke-dasharray"))))
        if st.get("stroke-dasharray") not in (None, "none"):
            st["stroke-dasharray"] = " ".join(_fmt(float(v) * s) for v in _NUM.findall(st["stroke-dasharray"]))
        if st:
            el.set("style", ";".join(f"{k}:{v}" for k, v in st.items()))
        stroked |= (st.get("stroke") or el.get("stroke") or "none") != "none"
    if stroked and root.get("stroke-width") is None and "stroke-width" not in _style_dict(root):
        root.set("stroke-width", _fmt(s))
    for k in ("width", "height"):
        root.set(k, _fmt(grid))
    root.set("viewBox", f"0 0 {_fmt(grid)} {_fmt(grid)}")
    return ET.tostring(root, encoding="unicode")


def svg_style(txt):
    """Effective style facts of an SVG: viewBox size, stroke widths, caps/joins, fill/stroke use, colours."""
    root = ET.fromstring(txt)
    vb = _vb(root)
    css = _css_rules(root)
    info = {"viewBox": [vb[2], vb[3]], "strokes": [], "caps": set(), "joins": set(), "filled": False,
            "stroked": False, "colors": set(), "transform": False}

    def walk(el, attrs):
        for ch in el:
            tag = ch.tag.replace(SVG_NS, "")
            at = _inherit(ch, attrs, css)
            info["transform"] |= bool(ch.get("transform"))
            if tag in ("g", "a", "switch"):
                walk(ch, at)
            if tag not in _SHAPES:
                continue
            fill, stroke = at.get("fill", "#000000"), at.get("stroke", "none")
            if stroke not in ("none", "transparent"):
                info["stroked"] = True
                info["strokes"].append(_num(at.get("stroke-width"), 1.0))
                segs = parse_path(ch.get("d")) if tag == "path" else []
                open_ = tag in ("line", "polyline") or (tag == "path" and segs and segs[-1][0] not in "Zz")
                if open_:
                    info["caps"].add(at.get("stroke-linecap", "butt"))
                if tag in ("path", "polyline", "polygon") or (tag == "rect" and not (ch.get("rx") or ch.get("ry"))):
                    info["joins"].add(at.get("stroke-linejoin", "miter"))
                info["colors"].add(stroke.lower())
            if fill not in ("none", "transparent") and tag != "line":
                info["filled"] = True
                info["colors"].add(fill.lower())
    walk(root, _inherit(root, {}, css))
    return info


def _stroke24(st):
    return round(float(np.median(st["strokes"])) * 24 / max(st["viewBox"]), 2) if st["strokes"] else None


def icon_set_report(entries, tol=0.15):
    """entries: [{name, svg, size (design px or None)}] -> consistency notes (stroke outliers per size group, caps, viewBox)."""
    rows, notes = [], []
    for e in entries:
        try:
            st = svg_style(e["svg"])
        except ET.ParseError as ex:
            notes.append(f"{e['name']}: invalid XML ({ex})")
            continue
        except ValueError as ex:
            notes.append(f"{e['name']}: unreadable path data ({ex})")
            continue
        rows.append({**e, "style": st, "sw": _stroke24(st)})
    vbs = {tuple(r["style"]["viewBox"]) for r in rows}
    if len(vbs) > 1:
        notes.append("mixed viewBoxes: " + ", ".join(f"{r['name']} {r['style']['viewBox'][0]:g}x{r['style']['viewBox'][1]:g}"
                                                     for r in rows))
    groups = []
    for r in sorted(rows, key=lambda r: r["size"] or 0):
        g = next((g for g in groups if (g[0]["size"] is None) == (r["size"] is None) and
                  (r["size"] is None or r["size"] <= g[0]["size"] * 1.25)), None)
        (g.append(r) if g is not None else groups.append([r]))
    for g in groups:
        strokes = [r for r in g if r["sw"]]
        if len(strokes) >= 2:
            med = float(np.median([r["sw"] for r in strokes]))
            for r in strokes:
                if abs(r["sw"] - med) / med > tol:
                    notes.append(f"{r['name']}: stroke {r['sw']} on the 24 grid vs {med:.2f} for its size group "
                                 f"({', '.join(x['name'] for x in strokes)})")
        for key in ("caps", "joins"):
            vals = [v for r in strokes for v in r["style"][key]]
            if vals:
                top = max(set(vals), key=vals.count)
                for r in strokes:
                    if r["style"][key] - {top}:
                        notes.append(f"{r['name']}: {key[:-1]} {'/'.join(sorted(r['style'][key]))} vs {top} in its group")
    for r in rows:
        if r["style"]["transform"]:
            notes.append(f"{r['name']}: uses transform attributes")
    return rows, notes


def _load_spec(d):
    p = Path(d) / "icon_spec.json"
    spec = _read_json(p, "icon_spec.json") if p.exists() else None
    if p.exists() and not isinstance(spec, dict):
        sys.exit(f"gptkit: icon_spec.json {p} is not a JSON object (hand-edited?): delete it or re-run `iconspec {d}`")
    return spec


def _spec_text(spec):
    if not spec:
        return ""
    states = "; ".join(f"{s['state']}: {s['treatment']} {s['color']}" for s in spec.get("states", []))
    return (f"\nICON SET SPEC (every icon must follow it so the set stays uniform): {spec['style']} style; stroke about "
            f"{spec['stroke_px']:g} px in the design ({spec['stroke_24']:g} on a 24-unit grid); {spec['linecap']} caps, "
            f"{spec['linejoin']} joins; corner radius about {spec['corner_radius_24']:g} on the 24 grid. {spec['traits']} "
            f"States: {states}.")


def _svg_issues(sc):
    out = []
    if sc["fit"] < 0.9 or sc["rel"] > 0.45:
        out.append(f"the silhouette differs from the design (overlap {sc['iou']:.2f}): check parts, proportions, positions")
    if sc["ink"] > 1.3:
        out.append(f"about {round((sc['ink'] - 1) * 100)}% heavier than the design: thinner strokes or smaller filled areas")
    elif sc["ink"] < 0.77:
        out.append(f"about {round((1 - sc['ink']) * 100)}% lighter than the design: thicker strokes or larger/filled areas")
    if sc["dE"] > 12:
        out.append(f"colour is off: design {sc['design_hex']}, yours {sc['svg_hex']}")
    if sc["aspect_err"] > 0.15:
        out.append("the width/height ratio differs from the design")
    return out


def _evaluate_svgs(mock, assets, svgs):
    """{name: svg} -> {name: {svg, score, issues, image}} with VectorDrawable-compatibility and fidelity checks."""
    res, texts = {}, {}
    for name, svg in svgs.items():
        issues = []
        try:
            ET.fromstring(svg)
            warns = []
            svg_to_vd(svg, 24, warns)
            issues += [f"not VectorDrawable-safe: {w}" for w in dict.fromkeys(warns) if not w.startswith("needs API")]
            texts[name] = svg
        except ET.ParseError as e:
            issues.append(f"invalid XML: {e}")
        except ValueError as e:
            issues.append(f"not convertible to VectorDrawable: {e}")
            texts[name] = svg
        res[name] = {"svg": svg, "score": None, "issues": issues, "image": None}
    names = list(texts)
    for name, r in zip(names, render_svgs([texts[n] for n in names], 256)):
        if r is None:
            continue
        a = assets.get(name)
        if a:
            sc, img = svg_fidelity(mock, a["bbox"], r, visual=True)
            res[name].update(score=sc, image=img, render=r)
            res[name]["issues"] += _svg_issues(sc)
        else:
            res[name]["render"] = r
    return res


def _svg_rank(e):
    hard = sum(1 for i in e["issues"] if i.startswith(("invalid", "not convertible")))
    return hard, e["score"]["rel"] if e["score"] else 9.0


def _norm_name(n):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", n.lower().removesuffix(".svg"))).strip("_")


def _size_groups(sizes):
    """Indices grouped like icon_set_report: sorted by size, sizes within 1.25x of the group's smallest; None sizes alone."""
    groups = []
    for i in sorted(range(len(sizes)), key=lambda i: sizes[i] or 0):
        g = next((g for g in groups if (sizes[g[0]] is None) == (sizes[i] is None) and
                  (sizes[i] is None or sizes[i] <= sizes[g[0]] * 1.25)), None)
        (g.append(i) if g is not None and sizes[i] is not None else groups.append([i]))
    return groups


def _grid_normalize(svgs, grid=24, pad=2.0, sizes=None, dps=None, with_dp=False):
    """[svg] -> [svg rescaled onto the grid, or None], from one batch render that measures each content box.
    With `sizes` (each icon's long side in design px) icons are normalised per size group: the group's largest icon
    fills the live area and the others keep their relative size, so one stroke and one display dp serve the group.
    `dps` (design dp per icon) gives that dp; with_dp=True returns (svg, dp) pairs."""
    n = len(svgs)
    boxes, vbs = [None] * n, [None] * n
    for i, (s, r) in enumerate(zip(svgs, render_svgs(svgs, 512))):
        try:
            root = ET.fromstring(s)
            vbs[i] = _vb(root)
            boxes[i] = r and _units_box(r, vbs[i], root.get("preserveAspectRatio"))
        except (ValueError, ET.ParseError):
            pass
    scales, dp_out = [None] * n, [None] * n
    long = lambda b: max(b[2] - b[0], b[3] - b[1], 1e-6)
    for g in (_size_groups(sizes) if sizes else [[i] for i in range(n)]):
        g = [i for i in g if boxes[i]]
        if not g:
            continue
        ref = max(g, key=lambda i: sizes[i] if sizes else 0)
        ref_px = sizes[ref] if sizes and sizes[ref] else None
        for i in g:
            if ref_px and sizes[i]:
                scales[i] = (grid - 2 * pad) / (long(boxes[i]) * ref_px / sizes[i])
            else:
                scales[i] = (grid - 2 * pad) / long(boxes[i])
            if dps and dps[ref]:
                dp_out[i] = round(dps[ref] * grid / (grid - 2 * pad), 1)
    out = []
    for i, s in enumerate(svgs):
        norm = None
        if boxes[i]:
            try:
                norm = normalize_svg(s, boxes[i], grid, pad, scales[i])
            except (ValueError, ET.ParseError):
                norm = None
        out.append((norm, dp_out[i]) if with_dp else norm)
    return out


def _svg_schema(names):
    """SVG_SCHEMA with `name` restricted to the requested names (strict mode then stops renamed and extra files)."""
    sch = json.loads(json.dumps(SVG_SCHEMA))
    if names:
        sch["properties"]["files"]["items"]["properties"]["name"]["enum"] = list(names)
    return sch


_LAST_BATCH_ERROR = None  # the last ApiError an _ask_batches() request raised (cmd_svg names it when icons stay missing)


def _ask_batches(fn, batches, what, fatal=True):
    """Run fn(batch) -> files in a pool; {returned name: svg} plus the batches whose request failed (printed). When every
    batch failed and `fatal`, the last ApiError is raised (the first request of a run); retries pass fatal=False."""
    got, failed = {}, []
    global _LAST_BATCH_ERROR
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(fn, b): b for b in batches}
        for fut in cf.as_completed(futs):
            b = futs[fut]
            try:
                for f in fut.result():
                    got[_norm_name(f["name"])] = _ensure_xmlns(f["svg"])
            except ApiError as e:
                _LAST_BATCH_ERROR = e
                if len(batches) > 1:
                    print(f"  ! {what} failed for {', '.join(n for n, *_ in b) if b and isinstance(b[0], tuple) else ', '.join(x['name'] for x in b)}: {e}")
                failed.append(b)
    if fatal and failed and len(failed) == len(batches):
        raise _LAST_BATCH_ERROR   # nothing came back at all: the error itself is the one-line exit (main() formats it)
    return got, failed


def cmd_svg(a):
    if not (math.isfinite(a.grid) and 0 <= a.grid <= 1024):
        sys.exit(f"svg: --grid {a.grid:g} must be a number of viewBox units, 0-1024 (24 = Material/SF grid, 0 = keep as drawn)")
    if not (math.isfinite(a.pad) and 0 <= a.pad and (not a.grid or a.pad < a.grid / 2)):
        sys.exit(f"svg: --pad {a.pad:g} must be >= 0 and less than half of --grid {a.grid:g}")
    d, m, mock = _load(a.dir)
    if a.only:
        known = {x["name"] for x in m["assets"]}
        unknown = [n for n in a.only.split(",") if n and n not in known]
        if unknown:
            sys.exit(f"svg: unknown asset(s): {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    todo = [x for x in _select(m, a.only) if x["format"] == "svg" or a.only]
    if not todo:
        sys.exit(f"svg: no svg-format assets in {d / 'assets.json'} (set \"format\": \"svg\" on the icons, or pass --only NAME)")
    for sub in ("preview", "raw"):
        (d / "svg" / sub).mkdir(parents=True, exist_ok=True)
    spec = _load_spec(d)
    per_icon = {i["name"]: i for i in (spec or {}).get("icons", [])}
    by_name = {x["name"]: x for x in todo}
    aliases = {_norm_name(n): n for n in by_name}  # model names come back normalised; assets.json names may not be
    batches = [todo[i:i + 6] for i in range(0, len(todo), 6)]
    if spec:
        print(f"using icon set spec {d / 'icon_spec.json'}")
    if a.check:
        return _svg_check(d, m, mock, by_name)

    def run(batch):
        parts = [f"The first image is an APPROVED UI design. After it come zoomed crops of specific graphics. {SVG_RULES}"
                 f"{_spec_text(spec)}\nReturn one file per requested asset, name exactly as given.", mock.convert("RGB")]
        for x in batch:
            b = x["bbox"]
            st = per_icon.get(x["name"])
            state = f" State: {st['state']}, {st['treatment']}." if st else ""
            parts += [f"Asset \"{x['name']}\" at bbox x={b['x']},y={b['y']},w={b['w']},h={b['h']}: {x['description']} "
                      f"Colours: {', '.join(x.get('colors', []))}.{state}", ref_crop(mock, b, margin=0.25, long_side=512)]
        return ask(parts, _svg_schema([x["name"] for x in batch]), "svg_files")["files"]

    def fix(batch):
        parts = [SVG_FIX_PROMPT + "\n" + SVG_RULES + _spec_text(spec)]
        for name, e in batch:
            parts += [f"Asset \"{name}\": {by_name[name]['description']}"]
            if e["image"] is not None:
                parts.append(e["image"])
            parts.append("Problems: " + "; ".join(e["issues"]) + f"\nCurrent SVG:\n{e['svg']}")
        return ask(parts, _svg_schema([n for n, _ in batch]), "svg_files")["files"]

    def keep(got):
        out = {}
        for n, svg in got.items():
            if n in aliases:
                out[aliases[n]] = svg
            else:
                print(f"  ! model returned an unrequested file {n!r}: not written")
        return out
    got, _ = _ask_batches(run, batches, "svg request")
    first = keep(got)
    missing = [n for n in by_name if n not in first]
    if missing:
        print(f"  ! not returned: {', '.join(missing)}; asking once more for them")
        got, _ = _ask_batches(run, [[by_name[n] for n in missing[i:i + 6]] for i in range(0, len(missing), 6)], "svg request", fatal=False)
        first.update(keep(got))
        missing = [n for n in by_name if n not in first]
    evals = _evaluate_svgs(mock, by_name, first)
    history = {n: [e] for n, e in evals.items()}
    todo_fix = [(n, e) for n, e in evals.items() if e["issues"] and n in by_name] if a.retry else []
    if todo_fix:
        print(f"retrying {len(todo_fix)} below threshold: {', '.join(n for n, _ in todo_fix)}")
        fixed, _ = _ask_batches(fix, [todo_fix[i:i + 4] for i in range(0, len(todo_fix), 4)], "svg retry", fatal=False)
        fixed = keep(fixed)
        for n, e in _evaluate_svgs(mock, by_name, {k: v for k, v in fixed.items() if k in history}).items():
            history[n].append(e)
    report_path = d / "svg_report.json"
    try:  # read after the API calls: a damaged old report must not throw this run's icons away
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
    except (OSError, ValueError) as e:
        print(f"  ! {report_path} is unreadable ({e}); rewritten with this run's icons")
        report = {}
    report = report if isinstance(report, dict) else {}
    rows = []
    best = {n: min(t, key=_svg_rank) for n, t in history.items()}
    final = {n: b["svg"] for n, b in best.items()}
    grid_ok = [n for n, b in best.items() if a.grid and by_name.get(n, {}).get("kind") == "icon" and
               not any(i.startswith("invalid") for i in b["issues"])]
    dps = {}
    px_per_dp = m["width"] / _screen_dp(m.get("platform"), m.get("screen_dp"))
    long_px = lambda x: max(x["bbox"]["w"], x["bbox"]["h"])
    long_dp = lambda x: max(x["dp"]["w"], x["dp"]["h"]) if isinstance(x.get("dp"), dict) else long_px(x) / px_per_dp
    for n, (norm, dp) in zip(grid_ok, _grid_normalize([final[n] for n in grid_ok], a.grid, a.pad,
                                                       sizes=[long_px(by_name[n]) for n in grid_ok],
                                                       dps=[long_dp(by_name[n]) for n in grid_ok], with_dp=True)):
        if norm:
            final[n] = norm
        if dp:
            dps[n] = dp
    previews = dict(zip(final, render_svgs(list(final.values()), 256)))
    for n in grid_ok:
        sc0 = best[n]["score"]
        if final[n] != best[n]["svg"] and sc0 and previews[n] is not None:
            if svg_fidelity(mock, by_name[n]["bbox"], previews[n])["rel"] > sc0["rel"] + 0.05:
                print(f"  ! {n}: grid normalisation changed the render, kept the original viewBox")
                final[n] = best[n]["svg"]
                previews[n] = render_svgs([final[n]], 256)[0]
    for name, tries in history.items():
        for i, e in enumerate(tries):
            (d / "svg" / "raw" / f"{name}_try{i + 1}.svg").write_text(e["svg"])
        b, asset, svg, prev = best[name], by_name.get(name, {}), final[name], previews[name]
        grid = a.grid if svg != b["svg"] else None
        p = d / "svg" / f"{name}.svg"
        p.write_text(svg)
        if prev:
            prev.save(d / "svg" / "preview" / f"{name}.png")
        sc = b["score"]
        line = f"  {p}"
        if sc:
            line += f"  rel={sc['rel']} fit={sc['fit']} ink={sc['ink']} dE={sc['dE']}"
        if len(tries) > 1:
            r0, r1 = (t["score"]["rel"] if t["score"] else "-" for t in tries[:2])
            line += f"  retry rel {r0} -> {r1}: {'KEPT' if b is tries[-1] else 'discarded'}"
        dp = dps.get(name)
        if grid and dp:
            line += f"  (24-grid; use at {dp:g}dp)"
        if asset.get("material_symbol") or asset.get("sf_symbol"):
            line += f"  [material: {asset.get('material_symbol') or '-'}, sf: {asset.get('sf_symbol') or '-'}]"
        print(line)
        for i in b["issues"]:
            print(f"      ! {i}")
        report[name] = {"svg": str(p), "grid": grid, "dp": dp if grid else None, "score": sc, "issues": b["issues"],
                        "tries": len(tries), "kept_try": tries.index(b) + 1}
        if sc and prev:
            rows.append([(f"{name}: design | svg", b["image"])] +
                        [(f"try {i + 1}: rel={t['score']['rel']}" + (" KEPT" if t is b else ""), t["render"])
                         for i, t in enumerate(tries) if t["score"]] +
                        [(f"final ink={sc['ink']} dE={sc['dE']}", prev)])
    for n in missing:
        print(f"  ! {n}: model returned no SVG (re-run: svg {d} --only {n})")
        report[n] = {"error": "missing"}
    for n in list(report):
        if n not in {x["name"] for x in m["assets"]}:
            del report[n]  # stale entries for assets no longer in assets.json
    report_path.write_text(json.dumps(report, indent=2))
    if rows:
        (d / "review").mkdir(exist_ok=True)
        stamp = time.strftime("%H%M%S")
        for i in range(0, len(rows), 8):
            print("review sheet ->", sheet(rows[i:i + 8], d / "review" / f"svg_{stamp}_{i // 8 + 1}.png", T=240))
    entries = [{"name": x["name"], "svg": (d / "svg" / f"{x['name']}.svg").read_text(),
                "size": max(x["bbox"]["w"], x["bbox"]["h"])}
               for x in m["assets"] if x["format"] == "svg" and x.get("kind") == "icon" and (d / "svg" / f"{x['name']}.svg").exists()]
    _, notes = icon_set_report(entries)
    print("icon set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")
    if missing:
        sys.exit(f"svg: {len(missing)} icon(s) missing ({', '.join(missing)}): re-run svg {d} --only {','.join(missing)}"
                 + (f"; last request error: {_LAST_BATCH_ERROR}" if _LAST_BATCH_ERROR else ""))


def _svg_check(d, m, mock, by_name):
    have = {n: (d / "svg" / f"{n}.svg").read_text() for n in by_name if (d / "svg" / f"{n}.svg").exists()}
    if not have:
        sys.exit(f"svg --check: no svg files to check in {d / 'svg'} (run svg first)")
    rows = []
    for n, e in _evaluate_svgs(mock, by_name, have).items():
        sc = e["score"]
        print(f"  {n:22s} " + (f"rel={sc['rel']} fit={sc['fit']} ink={sc['ink']} dE={sc['dE']}" if sc else "not scored"))
        for i in e["issues"]:
            print(f"      ! {i}")
        if sc:
            rows.append([(f"{n}: design | svg", e["image"]), (f"rel={sc['rel']} ink={sc['ink']}", e["render"])])
    if rows:
        (d / "review").mkdir(exist_ok=True)
        print("review sheet ->", sheet(rows, d / "review" / f"svgcheck_{time.strftime('%H%M%S')}.png", T=240))
    missing = [n for n in by_name if n not in have]
    if missing:
        print(f"  ! not in {d / 'svg'}: {', '.join(missing)}")
    _, notes = icon_set_report([{"name": n, "svg": s, "size": max(by_name[n]["bbox"]["w"], by_name[n]["bbox"]["h"])}
                                for n, s in have.items() if by_name[n].get("kind") == "icon"])
    print("icon set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")


def cmd_iconspec(a):
    d, m, mock = _load(a.dir)
    icons = [x for x in m["assets"] if x.get("format") == "svg"]
    if not icons:
        sys.exit(f"gptkit: iconspec: no format \"svg\" assets in {d / 'assets.json'} (run `manifest` first, or set "
                 "\"format\": \"svg\" on the icon entries)")
    parts = [ICON_SPEC_PROMPT.format(W=mock.width, H=mock.height), mock.convert("RGB")]
    for x in icons:
        parts += [f"Icon \"{x['name']}\" ({str(x.get('description') or '')[:160]})",
                  ref_crop(mock, x["bbox"], margin=0.25, long_side=256)]
    spec = ask(parts, ICON_SPEC_SCHEMA, "icon_spec", model=a.model)
    if not isinstance(spec, dict) or not isinstance(spec.get("icons"), list):
        sys.exit(f"gptkit: the model's icon-spec response has no 'icons' list (non-conforming gateway?): {str(spec)[:200]}")
    spec["icons"] = [i for i in spec["icons"] if isinstance(i, dict) and i.get("name")]
    for i in spec["icons"]:
        i["name"] = _norm_name(str(i["name"]))
        i["material_symbol"], i["sf_symbol"] = _symbol_names(str(i.get("material_symbol") or ""),
                                                             str(i.get("sf_symbol") or ""))
    (d / "icon_spec.json").write_text(json.dumps(spec, indent=2))
    names = {i["name"]: i for i in spec["icons"]}
    for x in m["assets"]:
        if x["name"] in names:
            x["material_symbol"], x["sf_symbol"] = names[x["name"]]["material_symbol"], names[x["name"]]["sf_symbol"]
    (d / "assets.json").write_text(json.dumps(m, indent=2))
    g = lambda k: spec.get(k, "?")
    num = lambda k: f"{spec[k]:g}" if isinstance(spec.get(k), (int, float)) else "?"
    print(f"{g('style')} icons, stroke {num('stroke_px')}px ({num('stroke_24')} on 24 grid), {g('linecap')} caps, "
          f"{g('linejoin')} joins, radius {num('corner_radius_24')}")
    print(f"traits: {g('traits')}")
    for s in spec.get("states") or []:
        if isinstance(s, dict):
            print(f"  state {s.get('state')}: {s.get('treatment')} {s.get('color')}")
    for i in spec["icons"]:
        print(f"  {i['name']:22s} {str(i.get('state')):18s} {str(i.get('treatment')):9s} "
              f"material={i['material_symbol'] or '-':16s} sf={i['sf_symbol'] or '-'}")
    missing = sorted({x["name"] for x in icons} - names.keys())
    if missing:
        print(f"  ! the model didn't describe: {', '.join(missing)} (their symbol names in assets.json are unchanged)")
    extra = sorted(names.keys() - {x["name"] for x in m["assets"]})
    if extra:
        print(f"  ! the model described icons that are not in assets.json: {', '.join(extra)} (kept in icon_spec.json only)")
    print(f"-> {d / 'icon_spec.json'} (symbol names merged into assets.json)")


def _symbol_names(material, sf):
    material = material.strip() if re.fullmatch(r"[a-z0-9_]+", material.strip()) else ""
    sf = sf.strip() if re.fullmatch(r"[a-z0-9]+(\.[a-z0-9]+)*", sf.strip()) else ""
    return material, sf


def _snap_stroke(svg, sw):
    """Every stroke width in the SVG (attribute, style, and the inherited default on the root) set to sw."""
    root = ET.fromstring(svg)
    for el in root.iter():
        st = _style_dict(el)
        if "stroke-width" in st:
            st["stroke-width"] = _fmt(sw)
            el.set("style", ";".join(f"{k}:{v}" for k, v in st.items()))
        elif el.get("stroke-width") is not None or el is root:
            el.set("stroke-width", _fmt(sw))
    return ET.tostring(root, encoding="unicode")


def cmd_icons(a):
    d, m, mock = _load(a.dir)
    concepts = [c.strip() for c in a.concepts.split(",") if c.strip()]
    if not concepts:
        sys.exit('icons: no concepts given (e.g. "notifications, sleep timer, download")')
    slugs = {_norm_name(c): c for c in concepts}
    spec = _load_spec(d)
    out = Path(a.out) if a.out else d / "icons"
    out.mkdir(parents=True, exist_ok=True)
    have = [x for x in m["assets"] if x["format"] == "svg" and x.get("kind") == "icon" and (d / "svg" / f"{x['name']}.svg").exists()]
    if not have:
        sys.exit("icons: no existing svg icons in DIR/svg; run `svg` first so the new icons can match them")
    existing = []
    for x in have:
        try:
            src = _ensure_xmlns((d / "svg" / f"{x['name']}.svg").read_text())
            svg_style(src)
            existing.append({"asset": x, "svg": src})
        except (ET.ParseError, ValueError, OSError) as e:
            print(f"  ! {x['name']}.svg: unreadable ({e}), left out of the reference set")
    if not existing:
        sys.exit("icons: none of the existing icons in DIR/svg is readable")
    sizes = [max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) for e in existing]
    for e, n in zip(existing, _grid_normalize([e["svg"] for e in existing], sizes=sizes)):
        e["svg"] = n or e["svg"]
        e["style"] = svg_style(e["svg"])
        e["sw"] = _stroke24(e["style"])
    size = float(np.median(sizes))
    group = [e for e in existing if max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) <= size * 1.25 and
             max(e["asset"]["bbox"]["w"], e["asset"]["bbox"]["h"]) >= size / 1.25] or existing
    outlined = [e for e in group if e["style"]["stroked"] and not e["style"]["filled"]]
    treatment = a.treatment if a.treatment != "auto" else (
        spec["style"] if spec and spec["style"] in ("outlined", "filled") else ("outlined" if outlined else "filled"))
    pool = outlined if treatment == "outlined" and outlined else group
    sw = a.stroke or (float(np.median([e["sw"] for e in pool if e["sw"]])) if any(e["sw"] for e in pool)
                      else spec["stroke_24"] if spec else 2)
    caps = [c for e in pool for c in e["style"]["caps"]] or [spec["linecap"] if spec else "round"]
    joins = [j for e in pool for j in e["style"]["joins"]] or [spec["linejoin"] if spec else "round"]
    colors = [c for e in pool for c in e["style"]["colors"] if c.startswith("#")]
    color = a.color or (max(set(colors), key=colors.count).upper() if colors else "#000000")
    cap, join = max(set(caps), key=caps.count), max(set(joins), key=joins.count)
    ref = sheet([[(e["asset"]["name"], ref_crop(mock, e["asset"]["bbox"], 0.25, 160)) for e in existing[i:i + 6]]
                 for i in range(0, len(existing), 6)], out / "_existing.png", T=160)
    tr = ("outlined: fill=\"none\" on stroked parts, filled only where the set fills small details" if treatment == "outlined"
          else "filled shapes like the set's filled icons")
    named = ", ".join(f'{c} (file name "{n}")' for n, c in slugs.items())
    prompt = ICONS_PROMPT.format(concepts=named, treatment=tr, sw=f"{sw:g}", cap=cap, join=join, color=color)
    exemplars = "\n".join(f"--- {e['asset']['name']}.svg\n{e['svg']}" for e in pool[:5])
    parts = [prompt + "\n" + SVG_RULES + _spec_text(spec), mock.convert("RGB"), _rgba(ref).convert("RGB"),
             f"Existing icon SVG sources, on the same 24 grid:\n{exemplars}"]
    files, bad, broken = {}, {}, {}
    for attempt in range(2):
        try:
            returned = ask(parts, _svg_schema(list(slugs)), "svg_files")["files"]
        except ApiError as e:
            print(f"  ! icons: request failed: {e}")
            break
        for f in returned:
            n = _norm_name(f["name"])
            if n not in slugs:
                print(f"  ! model returned an unrequested icon {n!r}: ignored")
                continue
            svg = _ensure_xmlns(f["svg"])
            try:
                ET.fromstring(svg)
            except ET.ParseError as e:
                (out / "raw").mkdir(exist_ok=True)
                (out / "raw" / f"{n}_try{attempt + 1}.svg").write_text(svg)
                broken[n] = f"invalid XML: {e}"
                continue
            files[n] = svg
            broken.pop(n, None)
            try:
                svg_to_vd(svg, 24)
                bad.pop(n, None)
            except ValueError as e:
                bad[n] = str(e)
        missing = [n for n in slugs if n not in files]
        redo = {**bad, **{n: broken.get(n, "not returned") for n in missing}}
        if not redo:
            break
        print("  retrying:", "; ".join(f"{k}: {v}" for k, v in redo.items()))
        parts = parts + [f"These icons were invalid or missing, return them again (only these, same names): {redo}"]
    if not files:
        sys.exit(f"icons: no usable icons returned; missing: {', '.join(slugs)}" +
                 (f" ({'; '.join(f'{k}: {v}' for k, v in broken.items())})" if broken else ""))
    for n, s in files.items():
        vbx = _vb(ET.fromstring(s))
        if vbx[2:] != [24.0, 24.0]:
            norm = _grid_normalize([s])[0]
            if norm:
                s = norm
            else:
                print(f"  ~ {n}: kept as drawn ({vbx[2]:g}x{vbx[3]:g} viewBox; transform/percent units), not on the 24 grid")
        if a.snap:
            s = _snap_stroke(s, sw * max(_vb(ET.fromstring(s))[2:]) / 24)
        files[n] = s
        (out / f"{n}.svg").write_text(s)
    renders = render_svgs([e["svg"] for e in existing] + list(files.values()), 192)
    hexbg = lambda x: "#" + "".join(f"{int(v):02X}" for v in _element_mask(mock, x["bbox"])[1])
    bg = hexbg(pool[0]["asset"])
    old_r, new_r = renders[:len(existing)], renders[len(existing):]
    tiles = [(f"{e['asset']['name']} (existing)", r, hexbg(e["asset"])) for e, r in zip(existing, old_r)]
    names = list(files)
    rows = [tiles[i:i + 6] for i in range(0, len(tiles), 6)]
    rows += [[(f"{n} (new)", r, bg) for n, r in zip(names[i:i + 6], new_r[i:i + 6])] for i in range(0, len(names), 6)]
    T = 150
    im = Image.new("RGB", (6 * T, len(rows) * T), "white")
    for ri, row in enumerate(rows):
        for ci, (label, r, b) in enumerate(row):
            im.paste(_tile(r, T, label, bg=b), (ci * T, ri * T))
    im.save(out / "preview.png")
    (out / "_existing.png").unlink(missing_ok=True)
    _, notes = icon_set_report([{"name": e["asset"]["name"], "svg": e["svg"], "size": None} for e in pool] +
                               [{"name": n, "svg": s, "size": None} for n, s in files.items()])
    for n in names:
        try:
            st = svg_style(files[n])
            print(f"  {out / (n + '.svg')}  stroke={_stroke24(st)} caps={'/'.join(sorted(st['caps'])) or '-'} "
                  f"{'filled' if st['filled'] else ''}{' outlined' if st['stroked'] else ''}")
        except ValueError as e:
            print(f"  {out / (n + '.svg')}  (unreadable path data: {e})")
    print(f"target: {treatment}, stroke {sw:g}, {cap} caps, {join} joins, {color}; preview -> {out / 'preview.png'}")
    print("set consistency:", "ok" if not notes else "")
    for n in notes:
        print(f"  ~ {n}")
    for n, err in bad.items():
        print(f"  ! {n}: {err} (kept out of --vd)")
    still = [n for n in slugs if n not in files]
    if still:
        print(f"  ! missing: {', '.join(still)}" + "".join(f" ({n}: {broken[n]})" for n in still if n in broken))
    good = [n for n in names if n not in bad]
    failed_vd = convert_vd([out / f"{n}.svg" for n in good], a.vd, a.dp) if a.vd and good else 0
    if failed_vd:
        print(f"  ! {failed_vd} VectorDrawable(s) failed validation (see above); the SVGs are written, fix them and run `vd`")
    if still or failed_vd:
        sys.exit(1)
