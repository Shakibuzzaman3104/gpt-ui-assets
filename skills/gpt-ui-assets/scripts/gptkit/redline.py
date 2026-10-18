"""Redline geometry."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from .measure import _dominant, _lab_arr, _text_targets, _to_hex
from .tokens import _d2c_load_spec, _d2c_manifest, _link_regions

# ---- redline: pixel-true geometry (edges, corner radii, gaps) for the components of an approved design

def _rise(dev, tau, lab=None, span=4, inside_ok=False):
    """Sub-pixel edge along a profile: the smoothed deviation `dev` first reaches tau at i; the edge is the largest
    colour step (Lab distance between neighbouring pixels of the `lab` profile) within [i-3, i+span], placed where the
    colour is halfway across that step. A drop shadow ramps up slowly before the real edge and anti-aliasing spreads it
    over two pixels; both are handled by the step search. None when the profile never rises, or starts inside (unless
    `inside_ok`, used behind drop shadows, and a clear step is in reach)."""
    idx = np.flatnonzero(dev >= tau)
    if not len(idx) or (idx[0] == 0 and not inside_ok):
        return None
    i = int(idx[0])
    if lab is None:
        return None if i == 0 else i - 1 + (tau - dev[i - 1]) / max(dev[i] - dev[i - 1], 1e-9) + 0.5
    lo, hi = max(0, i - 3), min(len(lab), i + span + 1)
    seg = np.asarray(lab[lo:hi], dtype=np.float64)
    if len(seg) < 3:
        return None
    steps = np.linalg.norm(np.diff(seg, axis=0), axis=-1)
    j = int(np.argmax(steps)) + 1
    if i == 0 and steps[j - 1] < max(2.0, 0.5 * tau):
        return None
    a0, a1 = max(0, j - 2), min(len(seg), j + 2)
    d = seg[a1 - 1] - seg[a0]
    prog = (seg[a0:a1] - seg[a0]) @ d / max(float(d @ d), 1e-9)
    n = next((n for n in range(1, len(prog)) if prog[n - 1] < 0.5 <= prog[n]), None)
    if n is None:
        return lo + j - 0.5 + 0.5
    return lo + a0 + n - 1 + (0.5 - prog[n - 1]) / max(prog[n] - prog[n - 1], 1e-9) + 0.5


def _surface_geometry(rgb, b):
    """Refined edges, per-corner radii and fill of a surface (fill, border or image) around bbox b, or None if its
    outline doesn't stand out from the surroundings near the bbox. Sides within 2 px of the image border are the border
    (full-bleed sheets, bars); corners are measured where the background is visible; a drop shadow is tolerated and
    reported as shadow=True."""
    from .compare import _box_mean  # deferred: compare imports this module
    H, W = rgb.shape[:2]
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    if w < 12 or h < 12:
        return None
    m = max(8, int(0.08 * min(w, h)))
    at_border = {"left": x <= 2, "top": y <= 2, "right": x + w >= W - 2, "bottom": y + h >= H - 2}
    band = lambda y0, y1, x0, x1: rgb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].reshape(-1, 3).astype(np.float64)
    hm = max(3, m // 2)
    outer = [band(y - m, y - hm, x, x + w), band(y + h + hm, y + h + m, x, x + w), band(y, y + h, x - m, x - hm),
             band(y, y + h, x + w + hm, x + w + m)]
    close = [band(y - hm, y - 2, x, x + w), band(y + h + 2, y + h + hm, x, x + w), band(y, y + h, x - hm, x - 2),
             band(y, y + h, x + w + 2, x + w + hm)]
    ring = np.concatenate(outer)
    inner = rgb[y + h // 3:y + h - h // 3, x + w // 3:x + w - w // 3]
    if inner.size == 0 or ring.size == 0:
        return None
    c_out = _dominant(ring)
    fill = _dominant(inner.astype(np.float64))
    c_lab = _lab_arr(c_out)
    contrast = float(np.linalg.norm(_lab_arr(fill) - c_lab))
    ring_dev = np.linalg.norm(_lab_arr(ring) - c_lab, axis=-1)
    tau = max(3.0, min(2 * float(np.percentile(ring_dev, 90)), 0.6 * contrast))
    lum = lambda c: float(np.asarray(c, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722]))
    shadow = False
    for o, c in zip(outer, close):
        if len(o) and len(c):
            od, cd = np.linalg.norm(_lab_arr(o) - c_lab, axis=-1), np.linalg.norm(_lab_arr(c) - c_lab, axis=-1)
            if np.percentile(cd, 50) > max(4.0, 1.5 * np.percentile(od, 50)) and np.percentile(od, 50) > 1.5 \
                    and lum(np.median(c, 0)) < lum(c_out) - 3:
                shadow = True
    oy, ox = max(0, y - m), max(0, x - m)
    lab = _lab_arr(rgb[oy:min(H, y + h + m), ox:min(W, x + w + m)].astype(np.float64))
    raw = np.linalg.norm(lab - c_lab, axis=-1)
    dev = _box_mean(raw, 3)
    edges = {}
    for side in ("top", "bottom", "left", "right"):
        if at_border[side]:
            edges[side] = {"top": 0.0, "left": 0.0, "bottom": float(H), "right": float(W)}[side]
            continue
        vals = []
        for f in np.linspace(0.4, 0.6, 9):
            if side in ("top", "bottom"):
                col = int(x + f * w) - ox
                prof, lprof = (dev[:, col], lab[:, col]) if side == "top" else (dev[::-1, col], lab[::-1, col])
            else:
                row = int(y + f * h) - oy
                prof, lprof = (dev[row, :], lab[row, :]) if side == "left" else (dev[row, ::-1], lab[row, ::-1])
            c = _rise(prof[:2 * m + 2], tau, lprof[:2 * m + 2], m + 4, shadow)
            if c is not None:
                vals.append(c)
        if len(vals) < 5:
            return None
        v = float(np.median(vals))
        edges[side] = {"top": oy + v, "left": ox + v, "bottom": oy + dev.shape[0] - v, "right": ox + dev.shape[1] - v}[side]
    tol_x, tol_y = max(6, 0.05 * w), max(6, 0.05 * h)
    if ((not at_border["left"] and abs(edges["left"] - x) > tol_x) or (not at_border["right"] and abs(edges["right"] - x - w) > tol_x)
            or (not at_border["top"] and abs(edges["top"] - y) > tol_y)
            or (not at_border["bottom"] and abs(edges["bottom"] - y - h) > tol_y)):
        return None
    ew, eh = edges["right"] - edges["left"], edges["bottom"] - edges["top"]
    radii = [None, None, None, None]  # top-left, top-right, bottom-left, bottom-right
    for ci, (flip_x, flip_y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        dd, ll = (dev[:, ::-1], lab[:, ::-1]) if flip_x else (dev, lab)
        dd, ll = (dd[::-1], ll[::-1]) if flip_y else (dd, ll)
        top = (oy + dev.shape[0] - edges["bottom"]) if flip_y else edges["top"] - oy
        left = (ox + dev.shape[1] - edges["right"]) if flip_x else edges["left"] - ox
        pts = []
        lim = int(left + min(ew, eh) / 2)
        for r_ in range(int(math.ceil(top)), min(dd.shape[0], int(top + min(ew, eh) / 2))):
            c = _rise(dd[r_, :lim], tau, ll[r_, :lim], m + 4, shadow)
            if c is not None:
                pts.append((r_ + 0.5 - top, c - left))
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        best = None
        for r in np.arange(0, min(ew, eh) / 2 + 0.25, 0.25):
            yy = np.clip(pts[:, 0], 0, None)
            pred = np.where(yy < r, r - np.sqrt(np.maximum(0, r * r - (r - yy) ** 2)), 0)
            err = float(np.mean((pred - pts[:, 1]) ** 2))
            if best is None or err < best[0]:
                best = (err, r)
        radii[ci] = float(best[1])
    got = [v for v in radii if v is not None]
    if not got:
        return None
    r = float(np.median(got))
    round_ = r > 0.45 * min(ew, eh)
    r = min(ew, eh) / 2 if round_ else r
    return {"box": [round(edges["left"], 1), round(edges["top"], 1), round(ew, 1), round(eh, 1)], "radius": round(r, 1),
            "corner_radii": [None if v is None else round(v, 1) for v in radii], "fill": _to_hex(fill), "outside": _to_hex(c_out),
            "shape": "circle" if round_ and abs(ew - eh) < 0.03 * max(ew, eh) else ("pill" if round_ else "rect"),
            "full_bleed": [s for s, v in at_border.items() if v], "shadow": shadow}


def cmd_redline(a):
    from .compare import _object_box, _regions  # deferred: compare imports this module
    d, spec, mock = _d2c_load_spec(a.dir)
    rgb = np.asarray(mock.convert("RGB"))
    k = spec.get("px_to_dp", 1.0)
    m = _d2c_manifest(d, mock.size)
    texts = _text_targets(d, spec, mock, a.vision) if spec.get("typography") else {}
    out = Path(a.out) if a.out else d / "redline"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for r in _regions(spec, m, mock.width, mock.height):
        b = r["bbox"]
        text_comp = sum(inst["box"][2] * inst["box"][3] for t in texts.values() for inst in t.get("instances", [])
                        if inst.get("component") == r["name"]) > 0.5 * b["w"] * b["h"]
        group = any(cb["w"] * cb["h"] >= 0.5 * b["w"] * b["h"] for cb in r["children"])
        geo = _surface_geometry(rgb, b) if r.get("asset_kind") not in ("icon", "illustration") and not text_comp and not group else None
        item = {"name": r["name"], "asset": r.get("asset"), "spec_px": [b["x"], b["y"], b["w"], b["h"]]}
        if geo:
            item.update(kind="surface", px=geo["box"], radius_px=geo["radius"], corner_radii_px=geo["corner_radii"], fill=geo["fill"],
                        shape=geo["shape"], shadow=geo["shadow"], full_bleed=geo["full_bleed"])
        else:
            pad = 4
            x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
            ob = _object_box(rgb[y0:b["y"] + b["h"] + pad, x0:b["x"] + b["w"] + pad].astype(np.float64))
            if not ob:
                item.update(kind="unmeasured", px=[float(v) for v in item["spec_px"]],
                            reason=("text component" if text_comp else "container of other components" if group else
                                    "no edge or object stands out from its surroundings near the bbox"))
                items.append(item)
                continue
            item.update(kind="object", px=[round(x0 + ob[0], 1), round(y0 + ob[1], 1), round(ob[2], 1), round(ob[3], 1)])
        item["dp"] = [round(v * k, 1) for v in item["px"]]
        if "radius_px" in item:
            item["radius_dp"] = round(item["radius_px"] * k, 1)
        items.append(item)
    fpath = d / "fonts" / "fonts.json"
    fstyles = {}
    if fpath.exists():
        try:
            fstyles = {x["role"]: x for x in json.loads(fpath.read_text()).get("styles", []) if x.get("ascent_em") and x.get("size_px")}
        except (OSError, ValueError, TypeError, KeyError):
            fstyles = {}
    for role, t in texts.items():
        for inst in t.get("instances", []):
            bx = inst["box"]
            items.append({"name": f"text: {inst['sample'][:30]}", "role": role, "kind": "text", "px": bx,
                          "dp": [round(v * k, 1) for v in bx], "baseline_px": inst["baseline"], "baseline_dp": round(inst["baseline"] * k, 1),
                          "ink": inst["ink"], "note": "px/dp are the ink box; pad from baseline_inset_dp or layout_top_inset_dp"})
    measured = [it for it in items if it["kind"] != "unmeasured"]
    for it in measured:
        x, y, w, h = it["px"]
        par = [o for o in items if o is not it and o["kind"] == "surface" and o["px"][0] <= x + 0.5 and o["px"][1] <= y + 0.5
               and o["px"][0] + o["px"][2] >= x + w - 0.5 and o["px"][1] + o["px"][3] >= y + h - 0.5 and o["px"][2] * o["px"][3] > w * h]
        if par:
            p = min(par, key=lambda o: o["px"][2] * o["px"][3])
            it["parent"] = p["name"]
            it["inset_dp"] = {"left": round((x - p["px"][0]) * k, 1), "top": round((y - p["px"][1]) * k, 1),
                              "right": round((p["px"][0] + p["px"][2] - x - w) * k, 1),
                              "bottom": round((p["px"][1] + p["px"][3] - y - h) * k, 1)}
            if it["kind"] == "text":  # ink insets are not padding: give the baseline (paddingFromBaseline) and the layout-box top
                it["baseline_inset_dp"] = round((it["baseline_px"] - p["px"][1]) * k, 1)
                fs = fstyles.get(it.get("role"))
                if fs:
                    it["layout_top_inset_dp"] = round((it["baseline_px"] - fs["ascent_em"] * fs["size_px"] - p["px"][1]) * k, 1)
    col = sorted(measured, key=lambda o: o["px"][1])
    for it in col:
        x, y, w, h = it["px"]
        below = [o for o in col if o["px"][1] >= y + h - 0.5 and min(x + w, o["px"][0] + o["px"][2]) - max(x, o["px"][0]) > 0.3 * min(w, o["px"][2])
                 and o.get("parent") == it.get("parent")]
        if below:
            nb = min(below, key=lambda o: o["px"][1])
            it["gap_below_dp"] = [nb["name"], round((nb["px"][1] - y - h) * k, 1)]
    notes = {c["name"]: c.get("notes", "") for c in spec.get("components", [])}
    native = {c["name"]: c.get("native") for c in spec.get("components", [])}
    surfaces = {i["name"]: i for i in items if i["kind"] == "surface"}
    docs = [{"name": n, "notes": notes.get(n, "")} for n in surfaces]
    pairs = []  # (score, radius index, surface name): each surface is assigned to at most one radius element, best first
    for ri, rr in enumerate(spec.get("radii", [])):
        if not rr.get("element") or not isinstance(rr.get("px"), (int, float)):
            continue
        for sc, reg in _link_regions(rr["element"], docs, scored=True):
            if 0.5 <= (surfaces[reg["name"]]["radius_px"] + 1) / (rr["px"] + 1) <= 2:
                pairs.append((sc, ri, reg["name"]))
    assigned, used = {}, set()
    for sc, ri, name in sorted(pairs, key=lambda t: -t[0]):
        if ri not in assigned and name not in used:
            assigned[ri] = (name, sc)
            used.add(name)
    radii_cmp = []
    for ri, rr in enumerate(spec.get("radii", [])):
        name, sc = assigned.get(ri, (None, None))
        hit = surfaces.get(name)
        radii_cmp.append({"element": rr.get("element"), "spec_px": rr.get("px"), "measured_px": hit["radius_px"] if hit else None,
                          "corner_radii_px": hit["corner_radii_px"] if hit else None, "component": name,
                          "shape": hit["shape"] if hit else None, "native": native.get(name) if hit else None,
                          "match_score": None if sc is None else ("exact" if sc >= 10 else round(sc, 2))})
    ov = mock.convert("RGB")
    dr = ImageDraw.Draw(ov)
    for it in measured:
        x, y, w, h = it["px"]
        c = {"surface": "#00e5ff", "object": "#ffd60a", "text": "#ff2d55"}[it["kind"]]
        dr.rectangle([x, y, x + w, y + h], outline=c, width=1)
        lab = f"{it['dp'][2]:g}x{it['dp'][3]:g}" + (f" r{it['radius_dp']:g}" if "radius_dp" in it else "")
        if it["kind"] != "text":
            dr.text((x + 2, y + 2), lab, fill=c)
        if "gap_below_dp" in it and it["gap_below_dp"][1] > 0:
            dr.line([x + w / 2, y + h, x + w / 2, y + h + it["gap_below_dp"][1] / k], fill="#7CFF6B", width=1)
            dr.text((x + w / 2 + 3, y + h + 1), f"{it['gap_below_dp'][1]:g}", fill="#7CFF6B")
    ov.save(out / "redline.png")
    (out / "redline.json").write_text(json.dumps({"px_to_dp": k, "items": items, "radii": radii_cmp}, indent=2))
    n_s = sum(i["kind"] == "surface" for i in items)
    unm = [i for i in items if i["kind"] == "unmeasured"]
    print(f"{len(items)} items ({n_s} surfaces with measured edges + corner radius, "
          f"{sum(i['kind'] == 'object' for i in items)} objects, {sum(i['kind'] == 'text' for i in items)} text lines"
          + (f", {len(unm)} unmeasured: " + ", ".join(f"{i['name']} ({i['reason']})" for i in unm) if unm else "") + ")")
    for i in items:
        if i["kind"] == "surface" and (i.get("shadow") or i.get("full_bleed")):
            print(f"  {i['name']}: " + ", ".join(filter(None, ["drop shadow detected" if i.get("shadow") else "",
                  ("full-bleed at " + "/".join(i["full_bleed"]) + f"; corner radii {i['corner_radii_px']}") if i.get("full_bleed") else ""])))
    for rc in radii_cmp:
        got = (f"measured {rc['measured_px']:>5}px ({rc['shape']}, {rc['component']}, match {rc['match_score']})" if rc["component"]
               else "no matching surface measured")
        print(f"  radius {str(rc['element'])[:30]:30s} spec {rc['spec_px']!s:>5}px  {got}")
    print(f"-> {out}/redline.json, redline.png (cyan surfaces, yellow objects, red text; green = gap below in dp)")
