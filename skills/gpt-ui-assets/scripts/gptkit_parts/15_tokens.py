# gptkit part 15_tokens: tokens: Compose / SwiftUI / CSS / Tailwind.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---- tokens: design_spec.json -> Compose / SwiftUI / CSS / Tailwind

_TOKEN_STOP = {"the", "and", "with", "for", "from", "into", "text", "color", "colour", "subtle", "approximately", "main",
               "dominant", "variation", "local", "fill", "shade", "tone", "light", "dark", "very", "slightly", "around",
               "behind", "inside", "outside", "used", "uses", "use", "its", "their", "this", "that", "includes", "plus",
               "icon", "icons", "label", "labels", "background", "a", "an", "of", "on", "to", "in", "at", "is", "or", "by",
               "as", "it", "up", "px", "dp", "sp", "pt"}


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _TOKEN_STOP}


def _word_rel(w, words):
    return any(w == y or (min(len(w), len(y)) >= 5 and (w.startswith(y) or y.startswith(w))) for y in words)


def _link_regions(query, regions, scored=False):
    """Regions whose name (and, weaker, notes) share informative words with `query` (IDF-weighted; the threshold scales
    with the region count so a unique word still links among 2-3 regions). An exact, case-insensitive name match wins
    outright (score 10). `scored` returns every candidate as (score, region), best first, instead of the top group."""
    norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
    exact = [r for r in regions if norm(r["name"]) == norm(query)]
    if exact:
        return [(10.0, r) for r in exact] if scored else exact
    docs = [(_words(r["name"]), _words(r.get("notes", ""))) for r in regions]
    vocab = [nw | tw for nw, tw in docs]
    idf = lambda w: math.log((len(regions) + 1) / (sum(1 for v in vocab if _word_rel(w, v)) + 0.5))
    scores = []
    for (nw, tw), r in zip(docs, regions):
        sc = sum(max(idf(w) if _word_rel(w, nw) else 0, 0.5 * idf(w) if _word_rel(w, tw) else 0) for w in _words(query))
        scores.append((sc, r))
    floor = min(1.0, 0.9 * math.log((len(regions) + 1) / 1.5))
    best = max((sc for sc, _ in scores), default=0)
    if scored:
        return sorted(((sc, r) for sc, r in scores if sc >= floor and sc >= 0.8 * best), key=lambda t: -t[0])
    return [r for sc, r in scores if best >= floor and sc >= 0.8 * best]


def _mode_near(lab_px, rgb_px, start, need=None):
    """Mean-shift from a guessed colour to the nearest dense colour cluster; returns (rgb, share) or (None, 0)."""
    need = need or max(25, int(0.002 * len(lab_px)))
    for r in (6.0, 10.0, 16.0):
        sel = np.linalg.norm(lab_px - start, axis=1) < r
        if sel.sum() < need:
            continue
        cur = lab_px[sel].mean(0)
        for _ in range(20):
            nsel = np.linalg.norm(lab_px - cur, axis=1) < r
            new = lab_px[nsel].mean(0)
            if nsel.sum() < need or np.linalg.norm(new - start) > r + 2:
                break
            done = np.linalg.norm(new - cur) < 0.05
            cur, sel = new, nsel
            if done:
                break
        for rr_ in (2.0, 1.0):  # the flat colour itself: most frequent exact RGB (else the median) within 2, then 1 Lab units
            near = np.linalg.norm(lab_px - cur, axis=1) < rr_
            if near.sum() >= min(need, 25):
                vals, counts = np.unique(rgb_px[near].astype(np.uint8), axis=0, return_counts=True)
                top = int(counts.argmax())
                pick = vals[top].astype(float) if counts[top] >= 0.2 * near.sum() else np.median(rgb_px[near], 0)
                return pick, float(sel.mean())
        return rgb_px[sel].mean(0), float(sel.mean())
    return None, 0.0


def _edge_color(rgb, b):
    """Colour of a thin border line along a box (straight sides, or the inscribed circle for round elements)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1
    t = 7
    clip = lambda v, hi: int(min(max(v, 0), hi - 1))

    def line(xa, ya, xb, yb):
        n = 2 * t + 1
        return rgb[[clip(ya + (yb - ya) * i / (n - 1), H) for i in range(n)],
                   [clip(xa + (xb - xa) * i / (n - 1), W) for i in range(n)]].astype(np.float64)

    def pick(profiles):
        found = []
        for pr in profiles:
            lab = _lab_arr(pr)
            dev = np.minimum(np.linalg.norm(lab - lab[:2].mean(0), axis=1), np.linalg.norm(lab - lab[-2:].mean(0), axis=1))
            i = int(dev.argmax())
            if dev[i] > 4:
                found.append(pr[i])
        return found, len(profiles)
    rect = []
    for f in np.linspace(0.3, 0.7, 24):
        xs, ys = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        rect += [line(xs, y0 - t, xs, y0 + t), line(xs, y1 + t, xs, y1 - t), line(x0 - t, ys, x0 + t, ys),
                 line(x1 + t, ys, x1 - t, ys)]
    sets = [pick(rect)]
    if 0.8 < b["w"] / max(b["h"], 1) < 1.25:
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, min(b["w"], b["h"]) / 2
        sets.append(pick([line(cx + (r + t) * math.cos(g), cy + (r + t) * math.sin(g), cx + (r - t) * math.cos(g),
                               cy + (r - t) * math.sin(g)) for g in np.linspace(0, 2 * math.pi, 72, endpoint=False)]))
    found, n = max(sets, key=lambda s_: len(s_[0]) / s_[1])
    if len(found) < 8:
        return None, 0.0
    return np.median(np.array(found), 0), len(found) / n


_AREA_NAME = re.compile(r"\b(fill|button|surface|background|disc|card|panel|overlay|base|bar|sheet|chip|tile|border|"
                        r"outline|stroke)s?\b", re.I)


def _region_pixels(rgb, boxes):
    H, W = rgb.shape[:2]
    m = np.zeros((H, W), bool)
    for b in boxes:
        m[max(0, b["y"]):min(H, b["y"] + b["h"]), max(0, b["x"]):min(W, b["x"] + b["w"])] = True
    return rgb[m]


def _snap_palette(spec, mock, texts, assets=()):
    rgb = np.asarray(mock.convert("RGB"))
    regions = [c for c in spec.get("components", []) if c["bbox"]["w"] * c["bbox"]["h"] < 0.9 * mock.width * mock.height]
    regions += [{"name": x["name"].replace("_", " "), "notes": x.get("description", ""), "bbox": x["bbox"]} for x in assets]
    by_hex = {}
    for st in spec.get("typography", []):
        t = texts.get(st["role"], {})
        if t.get("found") and st.get("color"):
            by_hex.setdefault(st["color"], []).append((st["role"], t))
    rng = np.random.default_rng(0)
    out = []
    for p in spec.get("palette", []):
        guess = p["hex"].upper()
        text = p["name"] + " " + p.get("usage", "")
        entry = {"name": p["name"], "usage": p.get("usage", ""), "hex_guess": guess, "alpha": float(p.get("alpha", 1.0))}
        linked = by_hex.get(guess, [])
        snapped = None
        if linked and not _AREA_NAME.search(p["name"]):
            inks = np.array([_hexi(t["ink"]) for _, t in linked], dtype=float)
            snapped = np.median(inks, 0)
            entry.update(source="text ink: " + ", ".join(r for r, _ in linked),
                         spread_de=round(max(_de_hex(_to_hex(i), _to_hex(snapped)) for i in inks), 1))
        else:
            screen_bg = re.search(r"\b(screen|canvas|page|app|window)\b", p["name"] + " " + p.get("usage", "")[:60], re.I) and \
                re.search(r"background|canvas|\bbase\b", p["name"] + " " + p.get("usage", "")[:60], re.I)
            picks = [] if screen_bg else _link_regions(text, regions)
            if picks and re.search(r"border|outline|stroke|divider", p["name"], re.I):
                found = [(sh, c, r) for r in picks for c, sh in [_edge_color(rgb, r["bbox"])] if c is not None]
                if found:
                    sh, snapped, r = max(found, key=lambda f: f[0])
                    entry.update(source=f"border of {r['name']}", support=round(sh, 3))
            if snapped is None:
                if picks:
                    px, need = _region_pixels(rgb, [r["bbox"] for r in picks]), None
                else:
                    px, need = rgb.reshape(-1, 3), 25
                if len(px) > 250000:
                    px = px[rng.choice(len(px), 250000, replace=False)]
                snapped, share = _mode_near(_lab_arr(px), px.astype(float), _lab_arr(_hexi(guess)), need)
                entry.update(source=", ".join(r["name"] for r in picks) if picks else "whole screen", support=round(share, 4))
        if snapped is None:
            entry.update(hex=guess, de00=0.0, snapped=False)
        else:
            entry.update(hex=_to_hex(snapped), de00=_de_hex(guess, _to_hex(snapped)), snapped=True)
        out.append(entry)
    return out


def _ident(s, case="pascal", max_words=6):
    words = re.findall(r"[A-Za-z0-9]+", s.replace("'", ""))[:max_words] or ["token"]
    if case == "kebab":
        out = "-".join(w.lower() for w in words)
    else:
        out = "".join(w[:1].upper() + w[1:].lower() for w in words)
        if case == "camel":
            out = out[:1].lower() + out[1:]
    return ("n" if case != "pascal" else "N") + out if out[:1].isdigit() else out


def _uniq(names):
    seen, out = {}, []
    for n in names:
        k = n
        i = 2
        while k in seen:
            k = f"{n}{i}"
            i += 1
        seen[k] = 1
        out.append(k)
    return out


_SWIFT_KEYWORDS = frozenset("as associatedtype async await break case catch class continue default defer deinit do else enum "
                            "extension fallthrough false fileprivate for func guard if import in init inout internal is let "
                            "nil open operator precedencegroup private protocol public repeat rethrows return self static "
                            "struct subscript super switch throw throws true try typealias var where while some any".split())
_SWIFTUI_COLORS = frozenset("primary secondary accentColor clear black white gray red green blue orange yellow pink purple "
                            "brown cyan indigo mint teal".split())


def _names(labels, colors=()):
    """One identifier set per label, unique across the whole token namespace (colours, styles, families, radii, spacing):
    kt (PascalCase), sw (camelCase, Swift keywords backticked, SwiftUI colour names suffixed Token), css (kebab-case).
    `colors` holds the indices of the colour labels."""
    words = [re.findall(r"[A-Za-z0-9]+", str(s).replace("'", ""))[:6] or ["token"] for s in labels]
    pascal = ["".join(w[:1].upper() + w[1:].lower() for w in ws) for ws in words]
    pascal = [("N" + p) if p[:1].isdigit() else p for p in pascal]
    kt = _uniq(pascal)
    sw, css = [], []
    for i, (p, k, ws) in enumerate(zip(pascal, kt, words)):
        suffix = k[len(p):]
        camel = k[:1].lower() + k[1:]
        if i in colors and camel in _SWIFTUI_COLORS:
            camel += "Token"
        sw.append(camel)
        css.append("-".join(w.lower() for w in ws) + (f"-{suffix}" if suffix else ""))
    sw = [f"`{s}`" if s in _SWIFT_KEYWORDS else s for s in _uniq(sw)]
    return [{"kt": k, "sw": s, "css": c} for k, s, c in zip(kt, sw, css)]


def _r05(v):
    return round(v * 2) / 2


def _num_s(v):
    return f"{v:g}"


def _num_ok(v):
    """A finite int/float (not a bool): what a hand-edited JSON number must be before arithmetic touches it."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _json_lists(data, keys, path, rerun):
    """data (from _read_json) with each of `keys` checked to be a list of objects; else a one-line exit naming the file."""
    for k in keys:
        v = data.get(k) if isinstance(data, dict) else None
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            sys.exit(f"gptkit: {path} '{k}' must be a list of objects (hand-edited or older file? re-run `{rerun}`)")
    return data


def _d2c_manifest(d, size=None):
    """DIR/assets.json for tokens/fonts/compare/redline, or None when absent: a non-object exits naming the file; non-numeric
    screen_dp/width/height are dropped; with size=(W, H) assets without a name or a bbox inside the mockup are dropped."""
    p = Path(d) / "assets.json"
    if not p.exists():
        return None
    m = _read_json(p, "assets.json")
    if not isinstance(m, dict):
        sys.exit(f"gptkit: assets.json {p} is not a JSON object (hand-edited? re-run manifest)")
    for k in ("screen_dp", "width", "height"):
        v = m.get(k)
        if v is not None and not (_num_ok(v) and v > 0):
            print(f"  ! {p}: {k} {v!r} is not a positive number; ignored")
            m.pop(k)
    if m.get("mockup") is not None and not (isinstance(m["mockup"], str) and m["mockup"]):
        print(f"  ! {p}: mockup {m.pop('mockup')!r} is not a file name; ignored")
    keep = []
    for x in m.get("assets") if isinstance(m.get("assets"), list) else []:
        if not (isinstance(x, dict) and isinstance(x.get("name"), str) and x["name"]):
            continue
        b = _clip_bbox(x.get("bbox") or {}, *size) if size and isinstance(x.get("bbox"), dict) else x.get("bbox")
        if size and not b:
            continue
        keep.append({**x, "bbox": b, "kind": str(x.get("kind") or ""), "description": str(x.get("description") or "")})
    m["assets"] = keep
    return m


def _d2c_load_spec(dir_, mockup=None, required=True):
    """(dir, design_spec, mockup RGBA) with every palette/typography colour normalised to #RRGGBB (+ alpha) and one-line
    exits for missing or corrupt files. `required=False` (compare) tolerates a missing design_spec.json."""
    d = Path(dir_)
    if not (d / "design_spec.json").exists():
        if required:
            sys.exit(f"gptkit: no {d / 'design_spec.json'}; run `spec {d}` first")
        spec = {}
    else:
        spec = _read_json(d / "design_spec.json", "design_spec.json")
        if not isinstance(spec, dict):
            sys.exit(f"gptkit: design_spec.json {d / 'design_spec.json'} is not a JSON object")
    for key in ("palette", "typography", "radii", "spacing", "components"):
        if not isinstance(spec.get(key), list):
            spec[key] = []
        bad = next((i for i, x in enumerate(spec[key]) if not isinstance(x, dict)), None)
        if bad is not None:
            sys.exit(f"gptkit: design_spec.json {d / 'design_spec.json'}: {key} entry {bad + 1} is not an object "
                     f"({spec[key][bad]!r:.40}); fix it or re-run `spec {d}`")
    for key in ("radii", "spacing"):
        for x in spec[key]:
            if x.get("px") is not None and not _num_ok(x["px"]):
                sys.exit(f"gptkit: design_spec.json {key} '{x.get('element') or x.get('between') or '?'}': px {x['px']!r} "
                         "is not a number")
    if spec.get("px_to_dp") is not None and not (_num_ok(spec["px_to_dp"]) and spec["px_to_dp"] > 0):
        print(f"  ! design_spec.json px_to_dp {spec.pop('px_to_dp')!r} is not a positive number; ignored")
    for p in spec["palette"]:
        try:
            p["hex"], p["alpha"] = _norm_hex(p.get("hex", ""))
        except (ValueError, TypeError) as e:
            sys.exit(f"gptkit: design_spec.json palette '{p.get('name', '?')}': {e}")
    for i, st in enumerate(spec["typography"]):
        for key in ("role", "size_px"):
            if not st.get(key):
                sys.exit(f"gptkit: design_spec.json typography entry {i + 1}: missing '{key}'")
        try:
            st["size_px"] = float(st["size_px"])
            st["weight"] = int(st.get("weight") or 400)
        except (TypeError, ValueError):
            sys.exit(f"gptkit: design_spec.json typography '{st['role']}': size_px {st['size_px']!r} / weight "
                     f"{st.get('weight')!r} must be numbers")
        if st.get("color"):
            try:
                st["color"] = _norm_hex(st["color"])[0]
            except (ValueError, TypeError) as e:
                sys.exit(f"gptkit: design_spec.json typography '{st['role']}': {e}")
        else:
            st["color"] = None
    m = _d2c_manifest(d)
    mpath = Path(mockup) if mockup else d / (m or {}).get("mockup", "mockup.png")
    mock = _open_image(mpath, "mockup", "pass --mockup or fix assets.json 'mockup'")
    if m and m.get("screen_dp") and m.get("width"):
        spec.setdefault("px_to_dp", round(m["screen_dp"] / m["width"], 5))
    if not spec.get("px_to_dp"):
        print("  ! no px_to_dp in design_spec.json and no screen_dp/width in assets.json: using 1 px = 1 dp; "
              "run `spec DIR --recalibrate`")
        spec["px_to_dp"] = 1.0
    good = []
    for c in spec["components"]:
        b = _clip_bbox(c.get("bbox") or {}, mock.width, mock.height)
        if not c.get("name") or not b:
            print(f"  ! design_spec.json: component {c.get('name') or '?'} has no usable bbox inside the mockup; ignored")
            continue
        c["bbox"] = b
        good.append(c)
    spec["components"] = good
    return d, spec, mock


_M3_SCALE = [("displayLarge", 57, 64), ("displayMedium", 45, 52), ("displaySmall", 36, 44), ("headlineLarge", 32, 40),
             ("headlineMedium", 28, 36), ("headlineSmall", 24, 32), ("titleLarge", 22, 28), ("titleMedium", 16, 24),
             ("titleSmall", 14, 20), ("bodyLarge", 16, 24), ("bodyMedium", 14, 20), ("bodySmall", 12, 16), ("labelLarge", 14, 20),
             ("labelMedium", 12, 16), ("labelSmall", 11, 16)]  # M3 baseline type scale (sp size, sp line height); r5 (g)
_HIG_SCALE = [("largeTitle", 34), ("title", 28), ("title2", 22), ("title3", 20), ("headline", 17), ("body", 17), ("callout", 16),
              ("subheadline", 15), ("footnote", 13), ("caption", 12), ("caption2", 11)]  # iOS Dynamic Type defaults (pt); r2/r5


def _m3_slot(role, size_sp, platform_style=None):
    """(slot, source, off) for a text style: the spec's platform_style when it is an M3 slot, else the nearest slot by size
    (ties broken by a role keyword, then body > title > label); off = True when the size is > 15% from the slot's."""
    by_name = {n: (sz, lh) for n, sz, lh in _M3_SCALE}
    if platform_style and platform_style in by_name:
        sz = by_name[platform_style][0]
        return platform_style, "spec", abs(size_sp - sz) / sz > 0.15
    low = role.lower()
    kw = next((f for f in ("display", "headline", "title", "body", "label") if f in low), None)
    order = {"body": 0, "title": 1, "label": 2, "headline": 3, "display": 4}
    best = min(_M3_SCALE, key=lambda t: (abs(size_sp - t[1]), 0 if kw and t[0].startswith(kw) else 1,
                                         order[re.match(r"[a-z]+", t[0]).group(0)]))
    return best[0], "nearest by size", abs(size_sp - best[1]) / best[1] > 0.15


def _hig_style(size_pt, platform_style=None):
    """The spec's platform_style when it is a HIG text style, else the nearest by size."""
    if platform_style in {n for n, _ in _HIG_SCALE}:
        return platform_style
    return min(_HIG_SCALE, key=lambda t: abs(size_pt - t[1]))[0]


def _kt_weight(w):
    """Compose FontWeight literal: the W100..W900 constants exist only for hundreds; anything else is FontWeight(n)."""
    w = int(min(1000, max(1, round(float(w)))))
    return f"FontWeight.W{w}" if w % 100 == 0 and w <= 900 else f"FontWeight({w})"


def _token_model(spec, palette, texts, fonts, redline=None):
    k = spec.get("px_to_dp", 1.0)
    families = {f["family"]: f for f in (fonts or {}).get("families", [])}
    types_in = spec.get("typography", [])
    radii_in = spec.get("radii", [])
    spacing_in = [s for s in spec.get("spacing", []) if s.get("between") and isinstance(s.get("px"), (int, float))
                  and not re.search(r"pixels? per|\bpx ?per\b|per (compose )?dp|px_to_dp|ratio|density|scale factor",
                                    s["between"], re.I)]
    labels = ([p["name"] for p in palette] + [s["role"] for s in types_in] + [r.get("element", "") for r in radii_in]
              + [s["between"] for s in spacing_in] + list(families))
    names = _names(labels, colors=range(len(palette)))
    nc, nt, nr, ns = len(palette), len(types_in), len(radii_in), len(spacing_in)
    colors = [dict(p, alpha=float(p.get("alpha", 1.0)), **n) for p, n in zip(palette, names[:nc])]
    by_guess = {c["hex_guess"]: c for c in colors}
    fstyles = {s["role"]: s for s in (fonts or {}).get("styles", [])}
    fam_names = dict(zip(families, names[nc + nt + nr + ns:]))
    for fam, n in fam_names.items():
        families[fam] = dict(families[fam], kt=n["kt"], css=n["css"])
    types = []
    for st, n in zip(types_in, names[nc:nc + nt]):
        fs = fstyles.get(st["role"], {})
        size_px = fs.get("size_px") or st["size_px"]
        track = fs.get("tracking_em", st.get("letter_spacing_em") or 0)
        t = texts.get(st["role"], {})
        lh_px = t.get("line_height_px")
        slot, slot_src, slot_off = _m3_slot(st["role"], size_px * k, st.get("platform_style"))
        if lh_px:
            lh_src = "measured line pitch"
        elif slot_src == "spec":
            ratio = next(lh / sz for n, sz, lh in _M3_SCALE if n == slot)
            lh_px, lh_src = round(size_px * ratio), f"M3 {slot} ratio ({ratio:.2f} x size)"
        else:
            lh_px, lh_src = round(size_px * 1.25), "estimate: 1.25 x size"
        guess = st.get("family_guess", "")
        generic = "serif" if re.search(r"(?<!sans[- ])serif|baskerville|garamond|times|georgia|playfair|bodoni", guess, re.I) \
            and not re.search(r"sans", guess, re.I) else "sans"
        fam = families.get(fs.get("family"))
        col = by_guess.get(st["color"]) if st.get("color") else None
        ink = t.get("ink") if t.get("found") else None
        weight = int(min(1000, max(1, round(float(fs.get("weight") or st.get("weight") or 400)))))
        types.append({"role": st["role"], **n, "family": fam["family"] if fam else None,
                      "family_kt": fam and fam["kt"], "family_css": fam and fam["css"], "generic": generic,
                      "weight": weight, "italic": bool(fs.get("italic", False)),
                      "size": _r05(size_px * k), "line_height": _r05(lh_px * k), "line_height_source": lh_src,
                      "tracking": round(float(track), 3), "color": col,
                      "color_hex": col["hex"] if col else (ink or st.get("color") or "#000000"),
                      "platform_style": st.get("platform_style"), "m3_slot": slot, "m3_slot_source": slot_src, "m3_slot_off": slot_off,
                      "hig_style": _hig_style(_r05(size_px * k), st.get("platform_style")),
                      "postscript": (fam or {}).get("postscript", {}).get(str(fs.get("weight")) + ("i" if fs.get("italic") else "")),
                      "natural_lh": fs.get("natural_line_height_em"), "size_source": "fonts" if fs.get("size_px") else "spec"})
    measured = {r["element"]: r for r in (redline or {}).get("radii", []) if r.get("measured_px") is not None}
    radii = []
    for r, n in zip(radii_in, names[nc + nt:nc + nt + nr]):
        px = float(r.get("px") or 0)
        if px < 0:  # CSS drops a negative border-radius, Compose/SwiftUI shapes take none: no platform has one
            print(f"  ! radius {r.get('element', '')!r}: {px:g}px is negative; using 0")
            px = 0.0
        hit = measured.get(r.get("element"))
        radii.append({"name": r.get("element", ""), **n, "dp": _r05((hit["measured_px"] if hit else px) * k),
                      "source": f"redline: {hit['component']}" if hit else "spec (GPT estimate)",
                      "native": hit.get("native") if hit else None, "shape": hit.get("shape") if hit else None})
    spacing = [{"name": s["between"], **n, "dp": _r05(float(s["px"]) * k), "source": "spec (GPT estimate)"}
               for s, n in zip(spacing_in, names[nc + nt + nr:nc + nt + nr + ns])]
    for sp in spacing:
        got = _measured_spacing(sp["name"], (redline or {}).get("items", []), k)
        if got:
            sp["dp"], sp["source"] = _r05(got[0]), got[1]
    return {"colors": colors, "types": types, "radii": radii, "spacing": spacing, "families": list(families.values()),
            "px_to_dp": k}


def _measured_spacing(name, items, k):
    """(dp, source) for a spacing name from redline's measured items, or None: 'gap/between/below' -> the gap below the
    linked item, 'edge/screen/margin' -> the item's distance from the screen edge, 'padding/inset' -> the smallest inset
    of the item's children (its content padding) or, for a leaf, its own inset."""
    measured = [i for i in items if i.get("kind") in ("surface", "object", "text")]
    if not measured:
        return None
    hits = _link_regions(name, [{"name": i["name"], "notes": i.get("role", "")} for i in measured], scored=True)
    if not hits:
        return None
    it = next(i for i in measured if i["name"] == hits[0][1]["name"])
    low = name.lower()
    side = ("left" if re.search(r"left|start|horizontal", low) else "right" if re.search(r"right|end", low) else
            "bottom" if re.search(r"bottom|below", low) else "top" if re.search(r"top|above|vertical", low) else None)
    if re.search(r"gap|between|below|above|spacing|distance", low) and it.get("gap_below_dp"):
        return it["gap_below_dp"][1], f"redline: gap below {it['name']} (to {it['gap_below_dp'][0]})"
    if re.search(r"edge|screen|margin", low):
        sd = side or "left"
        v = {"left": it["px"][0], "top": it["px"][1], "right": it["px"][0], "bottom": it["px"][1]}[sd] * k
        return round(v, 1), f"redline: {it['name']} {sd} of screen edge"
    # padding: layout-box insets for text (ink insets overstate the top), the smaller of left/top when no side is named
    inset = lambda i, sd: (i.get("layout_top_inset_dp") if sd == "top" and i.get("layout_top_inset_dp") is not None else i["inset_dp"][sd])
    pick = lambda i: inset(i, side) if side else min(inset(i, "left"), inset(i, "top"))
    kids = [i for i in measured if i.get("parent") == it["name"] and i.get("inset_dp")]
    if kids:
        return min(pick(i) for i in kids), f"redline: {it['name']} {side or 'left/top'} inset of its content ({len(kids)} items)"
    if it.get("inset_dp"):
        return pick(it), f"redline: {it['name']} {side or 'left/top'} inset in {it.get('parent')}"
    return None


_SCHEME_RULES = [("background", r"screen|background|canvas"), ("surface", r"surface|panel|sheet"),
                 ("primary", r"primary.*(fill|button)|accent fill|button fill|\baccent\b"),
                 ("onPrimary", r"on[- ]accent|on[- ]primary|button label"), ("onBackground", r"primary text"),
                 ("onSurface", r"primary text"), ("onSurfaceVariant", r"secondary text"), ("outline", r"border|outline")]


def _kt_header(a, pkg):
    head = f"/**\n * @author {a.author}\n * @since {time.strftime('%d/%m/%y')}\n */\n" if a.author else ""
    return f"{head}package {pkg}\n\n"


def _kt_color(c):
    return f"Color(0x{round(c.get('alpha', 1.0) * 255):02X}{c['hex'][1:]})"


def _emit_compose(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    pkg, P = a.package, a.prefix
    lum = lambda h: sum(v * w for v, w in zip(_hexi(h), (0.2126, 0.7152, 0.0722))) / 255
    scheme = {}
    for role, pat in _SCHEME_RULES:
        c = next((c for c in tm["colors"] if re.search(pat, f"{c['name']} {c.get('usage', '')}", re.I)), None)
        if c and role not in scheme:
            scheme[role] = c["kt"]
    bg = next((c for c in tm["colors"] if c["kt"] == scheme.get("background")), None)
    dark = bg is not None and lum(bg["hex"]) < 0.5
    fn = "darkColorScheme" if dark else "lightColorScheme"
    lines = [f"    val {c['kt']} = {_kt_color(c)}" for c in tm["colors"]]
    args = "".join(f"    {r} = {P}Colors.{v},\n" for r, v in scheme.items())
    other = "lightColorScheme" if dark else "darkColorScheme"
    measured_name, other_name = (f"{P}DarkColorScheme", f"{P}LightColorScheme") if dark else (f"{P}LightColorScheme", f"{P}DarkColorScheme")
    tm["m3_roles_baseline"] = [r for r in _M3_COLOR_ROLES if r not in scheme]
    (out / "Color.kt").write_text(_kt_header(a, pkg) + f"import androidx.compose.material3.ColorScheme\n"
                                  f"import androidx.compose.material3.darkColorScheme\nimport androidx.compose.material3.lightColorScheme\n"
                                  "import androidx.compose.ui.graphics.Color\n\n"
                                  f"object {P}Colors {{\n" + "\n".join(lines) + "\n}\n\n"
                                  f"// Measured on the {'dark' if dark else 'light'} mockup. Roles not listed stay at the M3 baseline: "
                                  + (", ".join(tm["m3_roles_baseline"][:8]) + (" ..." if len(tm["m3_roles_baseline"]) > 8 else "")) + "\n"
                                  f"val {measured_name}: ColorScheme = {fn}(\n{args})\n\n"
                                  f"// TODO: verify against a {'light' if dark else 'dark'} mockup; scaffolded from the same values\n"
                                  f"val {other_name}: ColorScheme = {other}(\n{args})\n\n"
                                  f"val {P}ColorScheme: ColorScheme = {measured_name}\n")
    rpkg = a.r_package or re.sub(r"(\.ui)?(\.theme)?$", "", pkg)
    fam_lines = []
    for f in tm["families"]:
        fonts_ = ",\n".join(f"    Font(R.font.{r['android']}, {_kt_weight(r['weight'])}"
                            + (", FontStyle.Italic" if r.get("italic") else "") + ")" for r in f["files"])
        fam_lines.append(f"val {f['kt']} = FontFamily(\n{fonts_},\n)")
    uses_italic = any(t["italic"] for t in tm["types"]) or any(r.get("italic") for f in tm["families"] for r in f["files"])
    body = []
    for t in tm["types"]:
        fam = t["family_kt"] or ("FontFamily.Serif" if t["generic"] == "serif" else "FontFamily.SansSerif")
        col = f"{P}Colors.{t['color']['kt']}" if t["color"] else f"Color(0xFF{t['color_hex'][1:]})"
        ls = f"({t['tracking']:g}).em" if t["tracking"] < 0 else f"{t['tracking']:g}.em"
        body.append(f"    val {t['kt']} = TextStyle(\n        fontFamily = {fam},\n        fontWeight = {_kt_weight(t['weight'])},\n"
                    + ("        fontStyle = FontStyle.Italic,\n" if t["italic"] else "")
                    + f"        fontSize = {_num_s(t['size'])}.sp,\n        lineHeight = {_num_s(t['line_height'])}.sp,\n"
                    f"        letterSpacing = {ls},\n        color = {col},\n    )")
    imports = ["androidx.compose.material3.Typography", "androidx.compose.ui.graphics.Color", "androidx.compose.ui.text.TextStyle",
               "androidx.compose.ui.text.font.FontFamily", "androidx.compose.ui.text.font.FontWeight",
               "androidx.compose.ui.unit.em", "androidx.compose.ui.unit.sp"]
    if tm["families"]:
        imports += ["androidx.compose.ui.text.font.Font", f"{rpkg}.R"]
    if uses_italic:
        imports.append("androidx.compose.ui.text.font.FontStyle")
    slots, slot_notes = {}, []
    for t in tm["types"]:
        if t["m3_slot"] in slots:
            slot_notes.append(f"{t['role']}: {t['m3_slot']} already taken by {slots[t['m3_slot']]['role']}")
            continue
        slots[t["m3_slot"]] = t
        if t["m3_slot_off"]:
            slot_notes.append(f"{t['role']}: {_num_s(t['size'])}sp is >15% from the {t['m3_slot']} baseline size")
    tm["m3_slot_notes"] = slot_notes
    tm["m3_slots_baseline"] = [s_ for s_ in _M3_TYPE_SLOTS if s_ not in slots]
    m3 = "".join(f"    {s_} = {P}Typography.{slots[s_]['kt']},  // {slots[s_]['m3_slot_source']}\n" for s_ in _M3_TYPE_SLOTS if s_ in slots)
    (out / "Type.kt").write_text(_kt_header(a, pkg) + "".join(f"import {i}\n" for i in sorted(imports)) + "\n"
                                 + "".join(f"{f}\n\n" for f in fam_lines)
                                 + f"object {P}Typography {{\n" + "\n\n".join(body) + "\n}\n\n"
                                 f"// M3 slots not listed keep the Typography() defaults (nearest slot by size when the spec gave none).\n"
                                 f"val {P}M3Typography = Typography(\n{m3})\n")
    shapes = [f"val {r['kt']}Shape = RoundedCornerShape({_num_s(r['dp'])}.dp)  // {r['source']}" for r in tm["radii"]]
    m3_shapes = {}
    for slot, pat in (("small", r"chip"), ("medium", r"card|tile|list item"), ("large", r"fab|drawer|floating"),
                      ("extraLarge", r"sheet|dialog|modal")):
        hit = next((r for r in tm["radii"] if re.search(pat, f"{r['name']} {r.get('native') or ''}", re.I)), None)
        if hit:
            m3_shapes[slot] = hit
    tm["m3_shapes"] = {k_: v["kt"] for k_, v in m3_shapes.items()}
    (out / "Shape.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.foundation.shape.RoundedCornerShape\n"
                                  "import androidx.compose.material3.Shapes\nimport androidx.compose.ui.unit.dp\n\n" + "\n".join(shapes)
                                  + "\n\n// M3 shape slots from the measured radii (small: chips, medium: cards, large: FABs, extraLarge: sheets/dialogs)\n"
                                  f"val {P}Shapes = Shapes(\n" + "".join(f"    {k_} = {v['kt']}Shape,\n" for k_, v in m3_shapes.items()) + ")\n")
    dims = [f"    val {s['kt']} = {_num_s(s['dp'])}.dp" + (f"  // {s['source']}" if s["source"].startswith("redline") else "  // GPT estimate")
            for s in tm["spacing"]]
    (out / "Dimens.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.ui.unit.dp\n\n"
                                   f"object {P}Dimens {{\n" + "\n".join(dims) + "\n}\n")
    wdp = tm.get("width_dp")
    if wdp:
        (out / "Preview.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.runtime.Composable\n"
                                        "import androidx.compose.ui.tooling.preview.Preview\n\n"
                                        f"// Renders at the design's effective width: Roborazzi (ComposablePreviewScanner) or the Compose Preview\n"
                                        f"// Screenshot Testing plugin pick this up; then `gptkit compare DIR shot.png --shot-width-dp {wdp}`.\n"
                                        f"@Preview(name = \"{P} 1:1\", widthDp = {wdp}, showBackground = true)\n@Composable\n"
                                        f"fun {P}ThemePreview() {{\n    {P}Theme {{\n        // TODO: the screen composable built from this theme\n    }}\n}}\n")
    (out / "Theme.kt").write_text(_kt_header(a, pkg) + "import androidx.compose.foundation.isSystemInDarkTheme\n"
                                  "import androidx.compose.material3.MaterialTheme\nimport androidx.compose.runtime.Composable\n\n"
                                  f"@Composable\nfun {P}Theme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {{\n"
                                  f"    MaterialTheme(\n        colorScheme = if (darkTheme) {P}DarkColorScheme else {P}LightColorScheme,\n"
                                  f"        typography = {P}M3Typography,\n        shapes = {P}Shapes,\n        content = content,\n    )\n}}\n")
    return [out / f for f in ("Color.kt", "Type.kt", "Shape.kt", "Dimens.kt", "Theme.kt") + (("Preview.kt",) if wdp else ())]


def _sw_color(hex_, alpha=1.0):
    r, g, b = _hexi(hex_)
    return f"Color(red: {r} / 255, green: {g} / 255, blue: {b} / 255" + (f", opacity: {alpha:.3g})" if alpha < 1 else ")")


def _emit_swift(tm, a, out):
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"    static let {c['sw']} = {_sw_color(c['hex'], c.get('alpha', 1.0))}" for c in tm["colors"]]
    wmap = {100: "ultraLight", 200: "thin", 300: "light", 400: "regular", 500: "medium", 600: "semibold", 700: "bold",
            800: "heavy", 900: "black"}
    styles = []
    for t in tm["types"]:
        if t["postscript"]:
            font = f'.custom("{t["postscript"]}", size: {_num_s(t["size"])}, relativeTo: .{t["hig_style"]})'
            natural = (t["natural_lh"] or 1.2) * t["size"]
        else:
            design = ".serif" if t["generic"] == "serif" else ".default"
            font = f".system(size: {_num_s(t['size'])}, weight: .{wmap.get(min(900, max(100, round(t['weight'] / 100) * 100)), 'regular')}, design: {design})"
            natural = 1.19 * t["size"]
            if t["italic"]:
                font += ".italic()"
        col = f".{t['color']['sw']}" if t["color"] else _sw_color(t["color_hex"])
        styles.append(f"    static let {t['sw']} = TextToken(font: {font}, tracking: {round(t['tracking'] * t['size'], 2):g}, "
                      f"lineSpacing: {max(0.0, round(t['line_height'] - natural, 1)):g}, color: {col})")
    radii = [f"    static let {r['sw']}: CGFloat = {_num_s(r['dp'])}  // {r['source']}" for r in tm["radii"]]
    space = [f"    static let {s['sw']}: CGFloat = {_num_s(s['dp'])}" + ("" if s["source"].startswith("redline") else "  // GPT estimate")
             for s in tm["spacing"]]
    p = a.prefix
    src = ("import SwiftUI\n\nextension Color {\n" + "\n".join(cols) + "\n}\n\n"
           f"struct {p}TextToken {{\n    let font: Font\n    let tracking: CGFloat\n    let lineSpacing: CGFloat\n    let color: Color\n}}\n\n"
           f"extension {p}TextToken {{\n" + "\n".join(styles).replace("TextToken(", f"{p}TextToken(") + "\n}\n\n"
           "@available(iOS 16.0, macOS 13.0, *)\nextension View {\n"
           f"    func textToken(_ t: {p}TextToken) -> some View {{\n"
           "        font(t.font).tracking(t.tracking).lineSpacing(t.lineSpacing).foregroundStyle(t.color)\n    }\n}\n\n"
           f"enum {p}Radius {{\n" + "\n".join(radii) + "\n}\n\n" f"enum {p}Spacing {{\n" + "\n".join(space) + "\n}\n")
    (out / f"{p}Tokens.swift").write_text(src)
    xc = out / "Colors.xcassets"
    xc.mkdir(exist_ok=True)
    (xc / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2))
    for c in tm["colors"]:
        r, g_, b = _hexi(c["hex"])
        cs = xc / f"{c['sw'].strip('`')}.colorset"
        cs.mkdir(exist_ok=True)
        (cs / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}, "colors": [
            {"idiom": "universal", "color": {"color-space": "srgb", "components": {"red": f"0x{r:02X}", "green": f"0x{g_:02X}",
                                                                                    "blue": f"0x{b:02X}", "alpha": f"{c.get('alpha', 1.0):.3f}"}}}]},
                                          indent=2))
    return [out / f"{p}Tokens.swift", xc]


def _css_family(t):
    fallback = "serif" if t["generic"] == "serif" else "sans-serif"
    return f'"{t["family"]}", {fallback}' if t["family"] else ("Georgia, serif" if t["generic"] == "serif" else
                                                              "system-ui, -apple-system, sans-serif")


def _css_hex(c):
    a = c.get("alpha", 1.0)
    return c["hex"] + (f"{round(a * 255):02X}" if a < 1 else "")


def _emit_web(tm, a, out, fonts_css=None):
    out.mkdir(parents=True, exist_ok=True)
    fams = {}
    for t in tm["types"]:
        fams.setdefault(t["family_css"] or t["generic"], _css_family(t))
    v = [f"  --color-{c['css']}: {_css_hex(c)};" for c in tm["colors"]]
    v += [f"  --font-{k}: {f};" for k, f in fams.items()]
    v += [f"  --radius-{r['css']}: {_num_s(r['dp'])}px;" for r in tm["radii"]]
    v += [f"  --space-{s['css']}: {_num_s(s['dp'])}px;" for s in tm["spacing"]]
    cls = []
    for t in tm["types"]:
        col = f"var(--color-{t['color']['css']})" if t["color"] else t["color_hex"]
        cls.append(f".text-{t['css']} {{\n  font-family: var(--font-{t['family_css'] or t['generic']});\n"
                   f"  font-weight: {t['weight']};\n" + ("  font-style: italic;\n" if t["italic"] else "")
                   + f"  font-size: {_num_s(t['size'])}px;\n  line-height: {_num_s(t['line_height'])}px;\n"
                   f"  letter-spacing: {t['tracking']:g}em;\n  color: {col};\n}}")
    head = f'@import "{fonts_css}";\n\n' if fonts_css else ""
    (out / "tokens.css").write_text(head + ":root {\n" + "\n".join(v) + "\n}\n\n" + "\n\n".join(cls) + "\n")
    js = lambda o: json.dumps(o, indent=6).replace("\n}", "\n    }")
    cfg = {"colors": {c["css"]: _css_hex(c) for c in tm["colors"]},
           "fontFamily": {k: [x.strip().strip('"') for x in f.split(",")] for k, f in fams.items()},
           "fontSize": {t["css"]: [f"{_num_s(t['size'])}px", {"lineHeight": f"{_num_s(t['line_height'])}px",
                                                               "letterSpacing": f"{t['tracking']:g}em",
                                                               "fontWeight": str(t["weight"])}] for t in tm["types"]},
           "borderRadius": {r["css"]: f"{_num_s(r['dp'])}px" for r in tm["radii"]},
           "spacing": {s["css"]: f"{_num_s(s['dp'])}px" for s in tm["spacing"]}}
    body = ",\n".join(f"    {k}: {js(val)}" for k, val in cfg.items())
    (out / "tailwind.tokens.cjs").write_text("// Tailwind v3 preset: in tailwind.config.js add  presets: [require('./tailwind.tokens.cjs')]\n"
                                             "/** @type {import('tailwindcss').Config} */\nmodule.exports = {\n"
                                             f"  theme: {{\n    extend: {{\n{body}\n    }},\n  }},\n}};\n")
    th = [f"  --color-{c['css']}: {_css_hex(c)};" for c in tm["colors"]]
    th += [f"  --font-{k}: {f};" for k, f in fams.items()]
    for t in tm["types"]:
        th += [f"  --text-{t['css']}: {_num_s(t['size'])}px;", f"  --text-{t['css']}--line-height: {_num_s(t['line_height'])}px;",
               f"  --text-{t['css']}--letter-spacing: {t['tracking']:g}em;", f"  --text-{t['css']}--font-weight: {t['weight']};"]
    th += [f"  --radius-{r['css']}: {_num_s(r['dp'])}px;" for r in tm["radii"]]
    th += [f"  --spacing-{s['css']}: {_num_s(s['dp'])}px;" for s in tm["spacing"]]
    (out / "tailwind.theme.css").write_text('/* Tailwind v4: in your CSS put  @import "tailwindcss"; @import "./tailwind.theme.css";  */\n\n'
                                            "@theme {\n" + "\n".join(th) + "\n}\n")
    return [out / f for f in ("tokens.css", "tailwind.tokens.cjs", "tailwind.theme.css")]


def _emit_dtcg(tm, out):
    """tokens.tokens.json: W3C Design Tokens (the Style Dictionary v4 subset: $type/$value/$description, dimensions as
    "16px" strings, typography composites) with the measured/spec provenance in $description."""
    tok = lambda typ, val, desc: {"$type": typ, "$value": val, "$description": desc}
    doc = {"$description": "Generated by gptkit tokens; DTCG (Style Dictionary v4 subset). Sizes are dp/pt, written as px.",
           "color": {c["css"]: tok("color", _css_hex(c), f"{c['name']}: " + ("snapped to the mockup (dE00 %s from the spec guess %s)"
                                                                            % (c.get("de00"), c.get("hex_guess")) if c.get("snapped")
                                                                            else f"spec guess {c.get('hex_guess', c['hex'])} (not found in the mockup)"))
                     for c in tm["colors"]},
           "typography": {t["css"]: tok("typography", {"fontFamily": t["family"] or ("serif" if t["generic"] == "serif" else "sans-serif"),
                                                       "fontWeight": t["weight"], "fontSize": f"{_num_s(t['size'])}px",
                                                       "lineHeight": f"{_num_s(t['line_height'])}px", "letterSpacing": f"{t['tracking']:g}em"},
                                        f"{t['role']}: size from {t['size_source']}, line height {t['line_height_source']}, "
                                        f"M3 {t['m3_slot']} ({t['m3_slot_source']}), HIG {t['hig_style']}") for t in tm["types"]},
           "dimension": {"radius": {r["css"]: tok("dimension", f"{_num_s(r['dp'])}px", f"{r['name']}: {r['source']}") for r in tm["radii"]},
                         "spacing": {s_["css"]: tok("dimension", f"{_num_s(s_['dp'])}px", f"{s_['name']}: {s_['source']}") for s_ in tm["spacing"]}}}
    (out / "tokens.tokens.json").write_text(json.dumps(doc, indent=2))
    return out / "tokens.tokens.json"


def _lint_tokens(files):
    """Pure-Python checks of the emitted theme code (no compiler needed): unique identifiers per scope, no Compose style
    whose colour names a sibling of the same object (recursive type-check), FontWeight.W only for hundreds, 8-digit
    Color(0x...) literals, Swift keywords backticked, no duplicate CSS custom properties. Returns a list of findings."""
    out = []
    for f in files:
        f = Path(f)
        text = f.read_text()
        if f.suffix == ".kt":
            scope, vals = "<top>", {}
            for n, line in enumerate(text.splitlines(), 1):
                m_ = re.match(r"^(object|class)\s+(\w+)", line)
                if m_:
                    scope = m_.group(2)
                elif line.startswith("}"):
                    scope = "<top>"
                m_ = re.match(r"^\s*(?:val|var|fun)\s+(\w+)", line)
                if m_:
                    key = (scope if line.startswith(" ") else "<top>", m_.group(1))
                    if key in vals:
                        out.append(f"{f.name}:{n}: duplicate identifier {m_.group(1)} in {key[0]} (first at line {vals[key]})")
                    vals[key] = n
                for w in re.findall(r"FontWeight\.W(\d+)", line):
                    if int(w) not in range(100, 1000, 100):
                        out.append(f"{f.name}:{n}: FontWeight.W{w} does not exist (use FontWeight({w}))")
                for w in re.findall(r"FontWeight\((\d+)\)", line):
                    if not 1 <= int(w) <= 1000:
                        out.append(f"{f.name}:{n}: FontWeight({w}) outside 1..1000")
                for lit in re.findall(r"Color\((0x[0-9A-Fa-f]+)\)", line):
                    if not re.fullmatch(r"0x[0-9A-F]{8}", lit):
                        out.append(f"{f.name}:{n}: {lit} is not an 8-digit ARGB literal")
            for obj, body in re.findall(r"^object (\w+) \{\n(.*?)^\}", text, re.S | re.M):
                members = set(re.findall(r"^\s+val (\w+) =", body, re.M))
                for ref in re.findall(r"^\s+color = (\w+),", body, re.M):
                    if ref in members:
                        out.append(f"{f.name}: object {obj}: `color = {ref}` names a sibling of the same object "
                                   f"(recursive type check); qualify it")
        elif f.suffix == ".swift":
            scope, seen = "<top>", {}
            for n, line in enumerate(text.splitlines(), 1):
                m_ = re.match(r"^(extension|enum|struct|class)\s+([\w.]+)", line)
                if m_:
                    scope = m_.group(2)
                m_ = re.match(r"^\s*(?:static\s+)?(?:let|var|func)\s+(`?)(\w+)(`?)", line)
                if m_:
                    name = m_.group(2)
                    if not m_.group(1) and name in _SWIFT_KEYWORDS:
                        out.append(f"{f.name}:{n}: `{name}` is a Swift keyword; backtick it")
                    if (scope, name) in seen:
                        out.append(f"{f.name}:{n}: duplicate declaration {name} in {scope} (first at line {seen[(scope, name)]})")
                    seen[(scope, name)] = n
        elif f.suffix == ".css":
            for block in re.findall(r"\{([^{}]*)\}", text):
                props = re.findall(r"^\s*(--[\w-]+)\s*:", block, re.M)
                for p in sorted({p for p in props if props.count(p) > 1}):
                    out.append(f"{f.name}: duplicate custom property {p}")
    return out


_KT_STUBS = {
    "AnimationCore.kt": "package androidx.compose.animation.core\ninterface Easing\nobject LinearEasing : Easing\n"
                      "class CubicBezierEasing(val a: Float, val b: Float, val c: Float, val d: Float) : Easing\n"
                      "class KeyframesSpec<T>\nclass KeyframeEntity<T>\nclass KeyframesSpecConfig<T> { var durationMillis: Int = 0; var delayMillis: Int = 0; "
                      "infix fun T.at(timeMillis: Int): KeyframeEntity<T> = KeyframeEntity(); infix fun KeyframeEntity<T>.using(easing: Easing): KeyframeEntity<T> = this }\n"
                      "fun <T> keyframes(init: KeyframesSpecConfig<T>.() -> Unit): KeyframesSpec<T> { KeyframesSpecConfig<T>().init(); return KeyframesSpec() }\n",
    "Color.kt": "package androidx.compose.ui.graphics\nclass Color(val value: Long) { companion object { val White = Color(0xFFFFFFFF); "
                "val Black = Color(0xFF000000); val Transparent = Color(0) } }\ninterface Shape\n",
    "Unit.kt": "package androidx.compose.ui.unit\nclass TextUnit(val v: Float)\nclass Dp(val v: Float)\n"
               "val Int.sp: TextUnit get() = TextUnit(toFloat()); val Double.sp: TextUnit get() = TextUnit(toFloat())\n"
               "val Int.em: TextUnit get() = TextUnit(toFloat()); val Double.em: TextUnit get() = TextUnit(toFloat())\n"
               "val Int.dp: Dp get() = Dp(toFloat()); val Double.dp: Dp get() = Dp(toFloat())\n",
    "Font.kt": "package androidx.compose.ui.text.font\nclass FontWeight(val weight: Int) { init { require(weight in 1..1000) }\n"
               "  companion object { val W100 = FontWeight(100); val W200 = FontWeight(200); val W300 = FontWeight(300); val W400 = FontWeight(400)\n"
               "    val W500 = FontWeight(500); val W600 = FontWeight(600); val W700 = FontWeight(700); val W800 = FontWeight(800); val W900 = FontWeight(900) } }\n"
               "class FontStyle(val v: Int) { companion object { val Normal = FontStyle(0); val Italic = FontStyle(1) } }\n"
               "interface Font\nclass ResFont(val res: Int, val weight: FontWeight, val style: FontStyle) : Font\n"
               "fun Font(resId: Int, weight: FontWeight = FontWeight.W400, style: FontStyle = FontStyle.Normal): Font = ResFont(resId, weight, style)\n"
               "open class FontFamily { companion object { val Serif = FontFamily(); val SansSerif = FontFamily(); val Default = FontFamily(); "
               "val Monospace = FontFamily(); val Cursive = FontFamily() } }\n"
               "class FontListFontFamily(val fonts: List<Font>) : FontFamily()\nfun FontFamily(vararg fonts: Font): FontFamily = FontListFontFamily(fonts.toList())\n",
    "Text.kt": "package androidx.compose.ui.text\nimport androidx.compose.ui.graphics.Color\nimport androidx.compose.ui.text.font.*\n"
               "import androidx.compose.ui.unit.TextUnit\ndata class TextStyle(val color: Color? = null, val fontSize: TextUnit? = null, "
               "val fontWeight: FontWeight? = null, val fontStyle: FontStyle? = null, val fontFamily: FontFamily? = null, "
               "val letterSpacing: TextUnit? = null, val lineHeight: TextUnit? = null)\n",
    "Shape.kt": "package androidx.compose.foundation.shape\nimport androidx.compose.ui.unit.Dp\nimport androidx.compose.ui.graphics.Shape\n"
                "class RoundedCornerShape(val r: Dp) : Shape\n",
    "Runtime.kt": "package androidx.compose.runtime\n@Target(AnnotationTarget.FUNCTION, AnnotationTarget.TYPE, AnnotationTarget.PROPERTY_GETTER)\n"
                  "annotation class Composable\n",
    "ComposeTest.kt": "package androidx.compose.ui.test\nimport androidx.compose.runtime.Composable\n"
                       "class MainTestClock { var autoAdvance: Boolean = true; fun advanceTimeBy(milliseconds: Long) {} }\n"
                       "class SemanticsNodeInteraction\nclass ComposeContentTestRule { val mainClock = MainTestClock(); "
                       "fun setContent(content: @Composable () -> Unit) { content() }; fun onRoot() = SemanticsNodeInteraction() }\n",
    "ComposeTestRule.kt": "package androidx.compose.ui.test.junit4\nimport androidx.compose.ui.test.ComposeContentTestRule\n"
                           "fun createComposeRule() = ComposeContentTestRule()\n",
    "Roborazzi.kt": "package com.github.takahirom.roborazzi\nimport androidx.compose.ui.test.SemanticsNodeInteraction\n"
                    "fun SemanticsNodeInteraction.captureRoboImage(filePath: String) {}\n",
    "JUnit.kt": "package org.junit\n@Target(AnnotationTarget.PROPERTY_GETTER, AnnotationTarget.FIELD)\nannotation class Rule\n"
                "@Target(AnnotationTarget.FUNCTION) annotation class Test\n",
    "Foundation.kt": "package androidx.compose.foundation\nimport androidx.compose.runtime.Composable\n@Composable fun isSystemInDarkTheme(): Boolean = false\n",
    "Preview.kt": "package androidx.compose.ui.tooling.preview\nannotation class Preview(val name: String = \"\", val widthDp: Int = -1, "
                  "val heightDp: Int = -1, val showBackground: Boolean = false, val backgroundColor: Long = 0)\n",
    "Material3.kt": "package androidx.compose.material3\nimport androidx.compose.runtime.Composable\nimport androidx.compose.ui.graphics.Color\n"
                    "import androidx.compose.ui.graphics.Shape\nimport androidx.compose.ui.text.TextStyle\nclass ColorScheme\n"
                    "fun darkColorScheme(ROLES): ColorScheme = ColorScheme()\nfun lightColorScheme(ROLES): ColorScheme = ColorScheme()\n"
                    "class Typography(SLOTS)\nclass Shapes(val extraSmall: Shape? = null, val small: Shape? = null, val medium: Shape? = null, "
                    "val large: Shape? = null, val extraLarge: Shape? = null)\n"
                    "object MaterialTheme { val colorScheme: ColorScheme = ColorScheme(); val typography: Typography = Typography(); val shapes: Shapes = Shapes() }\n"
                    "@Composable fun MaterialTheme(colorScheme: ColorScheme = ColorScheme(), shapes: Shapes = Shapes(), typography: Typography = Typography(), "
                    "content: @Composable () -> Unit) { content() }\n",
}
_M3_COLOR_ROLES = ("primary onPrimary primaryContainer onPrimaryContainer inversePrimary secondary onSecondary secondaryContainer "
                   "onSecondaryContainer tertiary onTertiary tertiaryContainer onTertiaryContainer background onBackground surface "
                   "onSurface surfaceVariant onSurfaceVariant surfaceTint inverseSurface inverseOnSurface error onError errorContainer "
                   "onErrorContainer outline outlineVariant scrim surfaceBright surfaceDim surfaceContainer surfaceContainerHigh "
                   "surfaceContainerHighest surfaceContainerLow surfaceContainerLowest").split()
_M3_TYPE_SLOTS = ("displayLarge displayMedium displaySmall headlineLarge headlineMedium headlineSmall titleLarge titleMedium "
                  "titleSmall bodyLarge bodyMedium bodySmall labelLarge labelMedium labelSmall").split()


def _kotlinc_check(kt_files, rpkg, res_fonts):
    """Type-check the generated Kotlin with kotlinc (PATH or GPTKIT_KOTLINC) against signature stubs of the Compose APIs
    the files use (no Compose jars needed). Returns a one-line verdict."""
    exe = os.environ.get("GPTKIT_KOTLINC") or shutil.which("kotlinc")
    if not exe:
        return "skipped (no kotlinc on PATH; the pure-Python lint ran)"
    with tempfile.TemporaryDirectory() as td:
        stubs = Path(td) / "stubs"
        stubs.mkdir()
        roles = ", ".join(f"{r}: Color? = null" for r in _M3_COLOR_ROLES)
        slots = ", ".join(f"val {s}: TextStyle? = null" for s in _M3_TYPE_SLOTS)
        for name, src in _KT_STUBS.items():
            (stubs / name).write_text(src.replace("ROLES", roles).replace("SLOTS", slots))
        consts = "; ".join(f"const val {n} = {i + 1}" for i, n in enumerate(res_fonts)) or "const val placeholder = 0"
        (stubs / "R.kt").write_text(f"package {rpkg}\nobject R {{ object font {{ {consts} }} }}\n")
        try:
            r = subprocess.run([exe, "-nowarn", "-d", str(Path(td) / "out"), *map(str, kt_files), *map(str, stubs.glob("*.kt"))],
                               capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"FAILED: kotlinc did not run ({type(e).__name__})"
    errs = [l for l in (r.stderr + r.stdout).splitlines() if "error:" in l]
    return "ok (kotlinc against Compose API stubs)" if r.returncode == 0 and not errs else "FAILED: " + "; ".join(errs[:4])[:600]


_CHROME_PROBE_WARNED = False


def _verify_tokens(tm, out, a, platforms=("compose", "swiftui", "web")):
    global _CHROME_PROBE_WARNED
    res = {}
    files = []
    if "compose" in platforms:
        files += sorted((out / "compose").glob("*.kt"))
    if "swiftui" in platforms:
        files += sorted((out / "swiftui").glob("*.swift"))
    if "web" in platforms:
        files += [out / "web" / "tokens.css"]
    lint = _lint_tokens([f for f in files if f.exists()])
    res["lint"] = (f"ok ({len(files)} files: unique identifiers, no self-references, valid weights/colour literals)" if not lint
                   else "FAILED: " + "; ".join(lint[:6]))
    if "compose" in platforms:
        rpkg = a.r_package or re.sub(r"(\.ui)?(\.theme)?$", "", a.package)
        res["compose"] = _kotlinc_check(sorted((out / "compose").glob("*.kt")), rpkg,
                                        [r["android"] for f in tm["families"] for r in f["files"]])
    if "swiftui" in platforms:
        sw = out / "swiftui" / f"{a.prefix}Tokens.swift"
        if sys.platform == "darwin" and shutil.which("xcrun"):
            r = subprocess.run(["xcrun", "--sdk", "iphonesimulator", "swiftc", "-typecheck", "-target",
                                "arm64-apple-ios16.0-simulator", str(sw)], capture_output=True, text=True)
            res["swiftui"] = "ok (swiftc -typecheck, iOS simulator SDK)" if r.returncode == 0 else "FAILED: " + r.stderr.strip()[:600]
        else:
            res["swiftui"] = "skipped (swiftc type-check needs macOS with Xcode; the lint ran)"
    if "web" in platforms:
        if shutil.which("node"):
            r = subprocess.run(["node", "-e", f"const c=require({json.dumps(str((out / 'web' / 'tailwind.tokens.cjs').resolve()))});"
                                              "if(!c.theme.extend.colors)process.exit(3)"], capture_output=True, text=True)
            res["tailwind"] = "ok (node require)" if r.returncode == 0 else "FAILED: " + r.stderr.strip()[:400]
        else:
            res["tailwind"] = "skipped (no node)"
        chrome = _chrome()
        if chrome:
            names = [f"--color-{c['css']}" for c in tm["colors"]] + [f"--radius-{r['css']}" for r in tm["radii"]]
            probe = out / "web" / ".css_probe.html"
            probe.write_text("<!doctype html><meta charset=utf-8><link rel=stylesheet href='tokens.css'><body>"
                             + "".join(f"<p class='text-{t['css']}'>x</p>" for t in tm["types"]) +
                             "<pre id=r></pre><script>const s=getComputedStyle(document.documentElement);const o={vars:{},text:{}};"
                             f"for(const n of {json.dumps(names)})o.vars[n]=s.getPropertyValue(n).trim();"
                             "for(const p of document.querySelectorAll('p')){const c=getComputedStyle(p);"
                             "o.text[p.className]=[c.fontSize,c.lineHeight,c.letterSpacing,c.fontWeight,c.color];}"
                             "document.getElementById('r').textContent=JSON.stringify(o);</script>")
            dom, why = "", ""
            with tempfile.TemporaryDirectory() as td:
                try:
                    r = subprocess.run([chrome, *_chrome_flags(800, 600), "--allow-file-access-from-files", f"--user-data-dir={td}/prof",
                                        "--virtual-time-budget=2000", "--dump-dom", probe.resolve().as_uri()],
                                       capture_output=True, text=True, timeout=60)
                    dom, why = r.stdout, _chrome_err(r)
                except (subprocess.TimeoutExpired, OSError) as e:
                    why = type(e).__name__
            m_ = re.search(r'<pre id="r">(.*?)</pre>', dom, re.S)
            if m_:
                got = json.loads(m_.group(1).replace("&quot;", '"'))
                empty = [n for n, v in got["vars"].items() if not v]
                bad = [t["css"] for t in tm["types"] if got["text"].get(f"text-{t['css']}", [""])[0] != f"{_num_s(t['size'])}px"]
                res["css"] = ("ok (Chrome parsed %d vars, %d text classes)" % (len(got["vars"]), len(got["text"]))
                              if not empty and not bad else f"FAILED: empty vars {empty[:5]}, wrong text sizes {bad[:5]}")
            else:
                res["css"] = f"FAILED: headless Chrome produced no probe output ({why})"
                if not _CHROME_PROBE_WARNED:
                    _CHROME_PROBE_WARNED = True
                    print(f"  ! headless Chrome produced no screenshot: {why}")
            probe.unlink(missing_ok=True)
        else:
            res["css"] = "skipped (no Chrome; set GPTKIT_CHROME)"
    return res


def _palette_sheet(palette, mock, texts, spec, path):
    rows = len(palette)
    im = Image.new("RGB", (560, 26 * rows + 24), "white")
    dr = ImageDraw.Draw(im)
    dr.text((6, 6), "name", fill="black")
    dr.text((250, 6), "spec guess | snapped   dE00  source", fill="black")
    for i, p in enumerate(palette):
        y = 24 + 26 * i
        dr.text((6, y + 6), p["name"][:38], fill="black")
        dr.rectangle([250, y, 300, y + 22], fill=p["hex_guess"])
        dr.rectangle([300, y, 350, y + 22], fill=p["hex"])
        dr.text((356, y + 6), f"{p['de00']:>5}  {p.get('source', '')[:28]}", fill="black" if p["snapped"] else "red")
    im.save(path)


def _platform_set(s):
    alias = {"android": "compose", "kotlin": "compose", "ios": "swiftui", "swift": "swiftui", "css": "web", "tailwind": "web"}
    got = []
    for p in re.split(r"[,\s]+", (s or "").strip()):
        if not p:
            continue
        q = alias.get(p.lower(), p.lower())
        if q not in ("compose", "swiftui", "web"):
            sys.exit(f"gptkit: --platforms: unknown '{p}' (use compose, swiftui, web; aliases android, ios)")
        if q not in got:
            got.append(q)
    if not got:
        sys.exit("gptkit: --platforms: nothing to emit (use compose, swiftui, web)")
    return [p for p in ("compose", "swiftui", "web") if p in got]


def cmd_tokens(a):
    platforms = _platform_set(a.platforms)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", a.prefix or ""):
        sys.exit(f"gptkit: --prefix {a.prefix!r} is not an identifier (letters, digits and _, starting with a letter)")
    pkg_re = r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
    kt_keywords = _JAVA_KEYWORDS | {"as", "fun", "in", "is", "object", "typealias", "typeof", "val", "var", "when"}
    for flag, val in (("--package", a.package), ("--r-package", a.r_package)):
        if val is not None and not re.fullmatch(pkg_re, val):
            sys.exit(f"gptkit: {flag} {val!r} is not a valid package name (dotted lowercase identifiers)")
        bad = [seg for seg in (val or "").split(".") if seg in kt_keywords]
        if bad:
            sys.exit(f"gptkit: {flag} {val!r}: {bad[0]!r} is a Kotlin/Java keyword, which no package segment may be")
    if "compose" in platforms and a.package == "com.example.app.ui.theme" and not a.r_package:
        print("  ! --package not given: Kotlin files use com.example.app.ui.theme and import com.example.app.R; "
              "pass --package (and --r-package if the module namespace differs)")
    d, spec, mock = _d2c_load_spec(a.dir)
    out = Path(a.out) if a.out else d / "tokens"
    out.mkdir(parents=True, exist_ok=True)
    texts = _text_targets(d, spec, mock, a.vision)
    _save_text_boxes(d, mock, texts, out / "text_boxes.png")
    assets = (_d2c_manifest(d, mock.size) or {}).get("assets", [])
    assets = [x for x in assets if isinstance(x, dict) and x.get("name") and _clip_bbox(x.get("bbox") or {}, mock.width, mock.height)]
    for x in assets:
        x["bbox"] = _clip_bbox(x["bbox"], mock.width, mock.height)
    palette = _snap_palette(spec, mock, texts, assets) if not a.no_snap else \
        [{"name": p["name"], "usage": p.get("usage", ""), "hex_guess": p["hex"], "hex": p["hex"], "alpha": p.get("alpha", 1.0),
          "de00": 0.0, "snapped": False} for p in spec.get("palette", [])]
    fonts_path = Path(a.fonts) if a.fonts else d / "fonts" / "fonts.json"
    if a.fonts:
        _need(fonts_path, "fonts.json (--fonts)", "run `fonts DIR` first")
    fonts = _json_lists(_read_json(fonts_path, "fonts.json", ("families", "styles")), ("families", "styles"), fonts_path,
                        f"fonts {d}") if fonts_path.exists() else None
    rl_path = d / "redline" / "redline.json"
    redline = _json_lists(_read_json(rl_path, "redline.json", ("items", "radii")), ("items", "radii"), rl_path,
                          f"redline {d}") if rl_path.exists() else None
    tm = _token_model(spec, palette, texts, fonts, redline)
    tm["width_dp"] = int(spec.get("effective_width_dp") or round(mock.width * tm["px_to_dp"]))
    files = []
    if "compose" in platforms:
        files += _emit_compose(tm, a, out / "compose")
    if "swiftui" in platforms:
        files += _emit_swift(tm, a, out / "swiftui")
    if "web" in platforms:
        fcss = fonts_path.parent / "web" / "fonts.css"
        files += _emit_web(tm, a, out / "web", os.path.relpath(fcss, out / "web") if fonts and fcss.exists() else None)
    if a.dtcg:
        files.append(_emit_dtcg(tm, out))
    _palette_sheet(palette, mock, texts, spec, out / "palette_check.png")
    report = {"palette": palette, "text": {r: {k: v for k, v in t.items() if k not in ("alpha", "instances")}
                                           for r, t in texts.items()},
              "typography": [{k: t[k] for k in ("role", "kt", "sw", "css", "family", "weight", "size", "line_height", "line_height_source",
                                                "tracking", "color_hex", "size_source", "platform_style", "m3_slot", "m3_slot_source",
                                                "hig_style")} for t in tm["types"]],
              "m3": {k_: tm.get(k_) for k_ in ("m3_roles_baseline", "m3_slots_baseline", "m3_slot_notes", "m3_shapes")},
              "prefix": a.prefix,
              "radii": [{k: r[k] for k in ("name", "dp", "source", "shape")} for r in tm["radii"]],
              "spacing": [{k: s[k] for k in ("name", "dp", "source")} for s in tm["spacing"]],
              "platforms": platforms, "px_to_dp": tm["px_to_dp"], "fonts": str(fonts_path) if fonts else None}
    print(f"palette ({sum(p['snapped'] for p in palette)}/{len(palette)} snapped to mockup pixels):")
    for p in palette:
        mark = f"{p['hex_guess']} -> {p['hex']}  dE00 {p['de00']:>4}" if p["snapped"] else f"{p['hex_guess']}  (kept: not found)"
        print(f"  {p['name'][:30]:30s} {mark}   [{p.get('source', '')[:60]}]")
    miss = [r for r, t in texts.items() if not t.get("found")]
    print(f"text samples located: {len(texts) - len(miss)}/{len(texts)}"
          + (f"  missing: {', '.join(miss)} " + ("(vision couldn't find it: check the sample text in design_spec.json)"
                                                 if a.vision else "(add --vision)") if miss else ""))
    if not fonts:
        print("  (no fonts.json: generic serif/sans families; run `fonts` first for matched Google Fonts)")
    if "compose" in platforms:
        print(f"M3 colour roles left at baseline: {', '.join(tm['m3_roles_baseline']) or 'none'}")
        print(f"M3 type slots left at baseline: {', '.join(tm['m3_slots_baseline']) or 'none'}"
              + (f"; shapes: {tm['m3_shapes']}" if tm["m3_shapes"] else "; shapes: none matched (chip/card/fab/sheet names)"))
        for n_ in tm["m3_slot_notes"]:
            print(f"  ! {n_}")
    for s_ in tm["spacing"]:
        if not s_["source"].startswith("redline"):
            print(f"  ~ spacing {s_['name']}: GPT estimate {s_['dp']:g}dp (no redline item matched; run `redline` or name the component)")
    if a.verify:
        report["verify"] = _verify_tokens(tm, out, a, platforms)
        for k, v in report["verify"].items():
            print(f"verify {k:8s}: {v}")
    (out / "tokens.json").write_text(json.dumps(report, indent=2))
    print(*[f"  {f}" for f in files], sep="\n")
    print(f"-> {out}/tokens.json, palette_check.png, text_boxes.png")
