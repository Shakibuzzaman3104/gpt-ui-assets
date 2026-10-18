"""Compare a built screen with the design."""
import concurrent.futures as cf
import json
import math
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

from .core import _need, _open_image, _read_json, ask
from .imaging import sheet
from .measure import _blobs, _de00, _de_hex, _dominant, _hexi, _lab_arr, _locate_text, _text_targets, _to_hex
from .tokens import _d2c_load_spec, _d2c_manifest, _json_lists
from .fonts import _fit_font, _gf_meta, _gf_ttf, _shape_similarity

# ---- compare: implemented screen vs approved mockup

def _gray(rgb):
    return np.asarray(rgb, dtype=np.float64)[..., :3] @ np.array([0.299, 0.587, 0.114])


def _box_mean(a, k):
    """Mean over a k x k window (same size, edge-clamped) via an integral image."""
    p = k // 2
    ap_ = np.pad(a, ((p + 1, p), (p + 1, p)), mode="edge")
    c = ap_.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _ssim_map(a, b, k=7):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _box_mean(a, k), _box_mean(b, k)
    va, vb = _box_mean(a * a, k) - ma ** 2, _box_mean(b * b, k) - mb ** 2
    cov = _box_mean(a * b, k) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def _ncc_map(S, T):
    """Normalised cross-correlation of template T at every valid position of S (FFT + integral images)."""
    h, w = T.shape
    H, W = S.shape
    if h > H or w > W:
        return None
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64) - T.mean()
    tn = np.sqrt((t * t).sum())
    if tn < 1e-6:
        return None
    num = np.fft.irfft2(np.fft.rfft2(S) * np.conj(np.fft.rfft2(t, s=S.shape)), s=S.shape)[:H - h + 1, :W - w + 1]
    c1 = np.pad(S, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    c2 = np.pad(S * S, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    win = lambda c: c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    var = win(c2) - win(c1) ** 2 / (h * w)
    ok = var > h * w
    return np.where(ok, np.clip(num / (tn * np.sqrt(np.where(ok, var, 1.0))), -1, 1), 0.0)


def _find_region(Mg, Sg, b, dy0, R, scales=(0.9, 0.95, 1.0, 1.05, 1.1), ctx=10):
    """Best (score, x, y, scale, score_at_expected) of mockup box b in the screenshot near the global offset.

    Shifts and scales are only accepted when they beat the expected placement by a clear margin."""
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    H, W = Mg.shape
    px0, py0 = max(0, x - ctx), max(0, y - ctx)
    px1, py1 = min(W, x + w + ctx), min(H, y + h + ctx)
    T = Mg[py0:py1, px0:px1]
    if T.size == 0 or T.std() < 2:
        return None
    tw0, th0 = px1 - px0, py1 - py0
    f = 2 if tw0 * th0 > 20000 else 1
    cands, expected = [], None
    for s in scales:
        tw, th = max(4, round(tw0 * s / f)), max(4, round(th0 * s / f))
        Ts = np.asarray(Image.fromarray(T).resize((tw, th), Image.BILINEAR if f == 1 else Image.BOX))
        ex, ey = px0 + (x + w / 2 - px0) * (1 - s), py0 + dy0 + (y + h / 2 - py0) * (1 - s)
        X0, Y0 = int(max(0, ex - R)), int(max(0, ey - R))
        X1, Y1 = int(min(Sg.shape[1], ex + tw0 * s + R)), int(min(Sg.shape[0], ey + th0 * s + R))
        win = Sg[Y0:Y1, X0:X1]
        if f > 1:
            win = np.asarray(Image.fromarray(win).resize((max(1, win.shape[1] // f), max(1, win.shape[0] // f)), Image.BOX))
        m = _ncc_map(win, Ts)
        if m is None or not m.size:
            continue
        iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
        cands.append((float(m[iy, ix]), X0 + ix * f, Y0 + iy * f, s))
        if s == 1.0:
            qy, qx = int(round((ey - Y0) / f)), int(round((ex - X0) / f))
            if 0 <= qy < m.shape[0] and 0 <= qx < m.shape[1]:
                expected = float(m[qy, qx])
    if not cands:
        return None
    best = max(cands, key=lambda c: c[0] - 0.03 * (c[3] != 1.0))
    if expected is not None and expected >= best[0] - 0.05:
        best = (expected, px0, py0 + dy0, 1.0)
    sc, bx, by, s = best
    if f > 1:
        tw, th = max(4, round(tw0 * s)), max(4, round(th0 * s))
        Ts = np.asarray(Image.fromarray(T).resize((tw, th), Image.BILINEAR))
        X0, Y0 = max(0, bx - 3), max(0, by - 3)
        m = _ncc_map(Sg[Y0:min(Sg.shape[0], by + th + 3), X0:min(Sg.shape[1], bx + tw + 3)], Ts)
        if m is not None and m.size:
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            sc, bx, by = float(m[iy, ix]), X0 + ix, Y0 + iy
    return sc, bx + round((x - px0) * s), by + round((y - py0) * s), s, expected


def _global_offset(Mg, Sg):
    """Vertical offset of the screenshot content vs the mockup (row-profile matching) and its cost."""
    k = 64
    ms = np.asarray(Image.fromarray(Mg).resize((k, Mg.shape[0]), Image.BOX))
    ss = np.asarray(Image.fromarray(Sg).resize((k, Sg.shape[0]), Image.BOX))
    H, Hs = len(ms), len(ss)
    best = (1e9, 0)
    for dy in range(-int(0.15 * H), Hs - int(0.6 * H) + 1):
        m0, m1 = max(0, -dy), min(H, Hs - dy)
        if m1 - m0 < 0.6 * H:
            continue
        cost = float(np.abs(ms[m0:m1] - ss[m0 + dy:m1 + dy]).mean())
        if cost < best[0]:
            best = (cost, dy)
    return best[1], best[0]


def _object_box(rgb):
    """Bounding box (x, y, w, h) of what stands out from the background of a region (the dominant colour of the crop's
    2-4 px border ring, so an object filling most of its bbox is not taken for the background), or None."""
    H, W = rgb.shape[:2]
    t = max(2, min(4, min(H, W) // 8))
    ring = np.concatenate([rgb[:t].reshape(-1, 3), rgb[-t:].reshape(-1, 3), rgb[t:-t, :t].reshape(-1, 3), rgb[t:-t, -t:].reshape(-1, 3)])
    d = np.linalg.norm(_lab_arr(rgb) - _lab_arr(_dominant(ring if len(ring) else rgb)), axis=-1)
    m = d > max(12.0, 0.25 * np.percentile(d, 99.5))
    if m.sum() < 12:
        return None
    rows, cols = m.sum(1), m.sum(0)
    ys = np.flatnonzero(rows >= max(2, 0.02 * rows.max()))  # rows/columns with more than isolated specks
    xs = np.flatnonzero(cols >= max(2, 0.02 * cols.max()))
    if not len(ys) or not len(xs):
        return None
    return float(xs[0]), float(ys[0]), float(xs[-1] - xs[0] + 1), float(ys[-1] - ys[0] + 1)


def _tolerant_de(ref, cur):
    """Per-pixel CIEDE2000 after a light blur, minimised over +-1px shifts (ignores anti-aliasing jitter)."""
    blur = lambda x: _lab_arr(np.asarray(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                                         dtype=np.float64))
    lm, lc = blur(ref), blur(cur)
    de = _de00(lm, lc)
    sel = de > 0  # identical pixels stay 0 under every shift: evaluate the shifts only where the images differ
    part = sel.mean() < 0.5
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if (oy, ox) == (0, 0):
                continue
            rolled = np.roll(lc, (oy, ox), (0, 1))
            if part:
                de[sel] = np.minimum(de[sel], _de00(lm[sel], rolled[sel]))
            else:
                de = np.minimum(de, _de00(lm, rolled))
    return de


def _impl_font_check(fonts, role, sample, alpha, size_px, platform=None):
    """Which font explains implemented text best: the one fonts.json chose, a neighbouring weight, another family, or
    (Android) Roboto, the default when fontFamily is not set? SF Pro (the iOS default) is not on Google Fonts.

    Returns (spec (family, weight, sim), best (family, weight, sim)); relative ranking, so rendering noise cancels out."""
    st = next((x for x in fonts.get("styles", []) if x["role"] == role), None)
    if not st:
        return None, None
    it = st["italic"]
    weights = lambda fam: sorted({int(k.rstrip("i")) for f in _gf_meta() if f["family"] == fam for k in f["fonts"] if k.endswith("i") == it})
    cands = [(st["family"], w) for w in weights(st["family"]) if abs(w - st["weight"]) <= 200] or [(st["family"], st["weight"])]
    for f in fonts.get("families", []):
        ws = weights(f["family"])
        if f["family"] != st["family"] and ws:
            cands.append((f["family"], min(ws, key=lambda w: abs(w - st["weight"]))))
    cands += [(fa, w) for fa, w, _ in st.get("alternatives", [])[:3] if (fa, w) not in cands]
    if platform == "android" and st["family"] != "Roboto":
        ws = weights("Roboto")
        if ws:
            cands.append(("Roboto", min(ws, key=lambda w: abs(w - st["weight"]))))

    def one(c):
        try:
            path = _gf_ttf(c[0], c[1], it)
            fit = _fit_font(path, sample, alpha, size_px) if path else None
        except (requests.RequestException, OSError, ValueError):
            return None
        return (c[0], c[1], fit["similarity"]) if fit else None
    with cf.ThreadPoolExecutor(4) as ex:
        res = [r for r in ex.map(one, cands) if r]
    spec = next((r for r in res if r[:2] == (st["family"], st["weight"])), None)
    return spec, (max(res, key=lambda r: r[2]) if res else None)


def _impl_box(box, r, dy0, k):
    """The design crop box moved by the global offset and the region's measured shift (same size, so tiles scale alike)."""
    sx, sy = (round(v / k) for v in r["shift_dp"])
    return box[0] + sx, box[1] + dy0 + sy, box[2] + sx, box[3] + dy0 + sy


def _heat_rgb(v):
    stops = np.array([[0, 0, 0], [80, 0, 110], [200, 30, 40], [255, 150, 0], [255, 255, 160]], dtype=np.float64)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    t = (x - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)


def _regions(spec, m, W, H):
    regs = [{"name": c["name"], "bbox": c["bbox"], "notes": c.get("notes", ""), "kind": "component"}
            for c in spec.get("components", []) if c["bbox"]["w"] * c["bbox"]["h"] < 0.8 * W * H]
    iou = lambda a, b: max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])) * \
        max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])) / max(1, a["w"] * a["h"] + b["w"] * b["h"] -
        max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])) * max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])))
    for x in (m or {}).get("assets", []):
        hit = next((r for r in regs if iou(r["bbox"], x["bbox"]) > 0.5), None)
        if hit:
            hit["asset"], hit["asset_kind"] = x["name"], x.get("kind")
        else:
            regs.append({"name": x["name"], "bbox": x["bbox"], "notes": x.get("description", ""), "kind": "asset", "asset": x["name"],
                         "asset_kind": x.get("kind")})
    for r in regs:
        b = r["bbox"]
        r["children"] = [o["bbox"] for o in regs if o is not r and o["bbox"]["x"] >= b["x"] and o["bbox"]["y"] >= b["y"]
                         and o["bbox"]["x"] + o["bbox"]["w"] <= b["x"] + b["w"] and o["bbox"]["y"] + o["bbox"]["h"] <= b["y"] + b["h"]
                         and o["bbox"]["w"] * o["bbox"]["h"] < 0.9 * b["w"] * b["h"]]
    return regs


COMPARE_PROMPT = ("Each numbered row shows LEFT: a region of the APPROVED design, RIGHT: the same region of the implemented "
                  "app screen. For every row list the concrete visual differences a developer must fix (size, position, "
                  "colour, font family/weight, missing or extra elements, image content). Be specific and brief; say "
                  "'matches' if there is no real difference. Ignore sub-pixel anti-aliasing noise.")
COMPARE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rows"], "properties": {"rows": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["index", "differences"],
                               "properties": {"index": {"type": "integer"},
                                              "differences": {"type": "array", "items": {"type": "string"}}}}}}}


def _compare_load_inputs(a):
    """Validate comparison options and load, crop, and size the design and screenshot."""
    if a.crop_top < 0 or a.crop_bottom < 0:
        sys.exit("compare: --crop-top/--crop-bottom must be >= 0")
    if a.fail_below is not None and not 0 <= a.fail_below <= 100:
        sys.exit("compare: --fail-below is a score between 0 and 100")
    d, spec, mock_rgba = _d2c_load_spec(a.dir, mockup=a.mockup, required=False)
    m = _d2c_manifest(d, mock_rgba.size)
    mock = mock_rgba.convert("RGB")
    W, H = mock.size
    k = spec.get("px_to_dp") or 1.0
    design_dp = spec.get("effective_width_dp") or (m or {}).get("screen_dp") or round(W * k)
    print(f"  (native scale: for a 1:1 check the screenshot should show the screen at ~{design_dp}dp wide; "
          "a narrower device reflows the layout)")
    shot = _open_image(a.screenshot, "screenshot").convert("RGB")
    if a.crop_top + a.crop_bottom >= shot.height:
        sys.exit(f"compare: --crop-top {a.crop_top} + --crop-bottom {a.crop_bottom} leaves nothing of the {shot.height}px screenshot")
    if a.crop_top or a.crop_bottom:
        shot = shot.crop((0, a.crop_top, shot.width, shot.height - a.crop_bottom))
    shot_dp = a.shot_width_dp or (shot.width / a.density if a.density else None)
    if shot_dp and abs(shot_dp - design_dp) / design_dp > 0.05:
        print(f"  ! the screenshot shows a ~{shot_dp:.0f}dp-wide screen, the design is {design_dp}dp: the layout reflows and every "
              f"size reads {shot_dp / design_dp:.2f}x; re-shoot at {design_dp}dp (Preview widthDp={design_dp})")
    shot = shot.resize((W, round(shot.height * W / shot.width)), Image.LANCZOS)
    out = Path(a.out) if a.out else d / "compare"
    out.mkdir(parents=True, exist_ok=True)
    return {"d": d, "spec": spec, "manifest": m, "mock_rgba": mock_rgba, "mock": mock, "shot": shot,
            "W": W, "H": H, "k": k, "design_dp": design_dp, "shot_dp": shot_dp, "out": out}


def _compare_align_screens(data):
    """Align the screenshot vertically and compute whole-screen difference metrics."""
    mock, shot, W, H = data["mock"], data["shot"], data["W"], data["H"]
    M, S = np.asarray(mock, dtype=np.float64), np.asarray(shot, dtype=np.float64)
    Mg, Sg = _gray(M), _gray(S)
    dy0, cost = _global_offset(Mg.astype(np.float32), Sg.astype(np.float32))
    top_extra = max(0, dy0)
    bottom_extra = max(0, S.shape[0] - (H + dy0))
    A = np.zeros_like(M)
    y0, y1 = max(0, -dy0), min(H, S.shape[0] - dy0)
    A[y0:y1] = S[y0 + dy0:y1 + dy0]
    valid = np.zeros((H, W), bool)
    valid[y0:y1] = True
    de = _tolerant_de(M, A)
    de[~valid] = 0
    ssim = _ssim_map(Mg, _gray(A))
    overall = {"ssim": round(float(ssim[valid].mean()), 4), "mean_de00": round(float(de[valid].mean()), 2),
               "pct_pixels_de_gt_10": round(float((de[valid] > 10).mean() * 100), 1),
               "vertical_offset_px": int(dy0), "status_bar_px": int(top_extra), "extra_bottom_px": int(bottom_extra),
               "coverage": round(float(valid.mean()), 3), "compared_rows": [int(y0), int(y1)]}
    overall["score"] = round(100 * overall["ssim"] * max(0.0, 1 - overall["mean_de00"] / 50), 1)
    overall["ssim_method"] = "SSIM (7px box window, luma)"
    data.update({"M": M, "S": S, "A": A, "de": de, "valid": valid, "Mg": Mg, "Sg": Sg,
                 "dy0": dy0, "top_extra": top_extra, "bottom_extra": bottom_extra, "y0": y0, "y1": y1,
                 "overall": overall})
    return data


def _compare_typography(data, a):
    """Load typography roles, token names, and font context for region diagnostics."""
    d, spec, mock_rgba, m = data["d"], data["spec"], data["mock_rgba"], data["manifest"]
    H, W, Sg, Mg = data["H"], data["W"], data["Sg"], data["Mg"]
    texts = _text_targets(d, spec, mock_rgba) if spec.get("typography") else {}
    role_of = {}
    for r_, t_ in texts.items():
        for inst in t_.get("instances", []):
            if inst.get("component"):
                role_of.setdefault(inst["component"], []).append((r_, inst))
    fpath = Path(a.fonts) if a.fonts else d / "fonts" / "fonts.json"
    if a.fonts:
        _need(fpath, "fonts.json (--fonts)", "run `fonts DIR` first")
    fonts_json = _json_lists(_read_json(fpath, "fonts.json", ("families", "styles")), ("families", "styles"), fpath,
                             f"fonts {d}") if fpath.exists() else None
    tk_path = d / "tokens" / "tokens.json"
    token_names = {}
    if tk_path.exists():
        try:
            tk = json.loads(tk_path.read_text())
            token_names = {t["role"]: f"{tk.get('prefix', 'App')}Typography.{t['kt']}" for t in tk.get("typography", []) if t.get("kt")}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    platform = spec.get("platform") or (m or {}).get("platform")
    styles = {st["role"]: st for st in spec.get("typography", [])}
    Sg32, Mg32 = Sg.astype(np.float32), Mg.astype(np.float32)
    R = max(24, int(0.04 * H))
    data.update({"texts": texts, "role_of": role_of, "fonts_json": fonts_json, "fpath": fpath, "token_names": token_names,
                 "platform": platform, "styles": styles, "Sg32": Sg32, "Mg32": Mg32, "R": R,
                 "font_check_off": False, "size_hints": []})
    return data


def _compare_measure_region(r, ctx):
    """Align a region and measure its pixels, fill colour, and child coverage."""
    b = r["bbox"]
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    valid = ctx["valid"]
    inside = float(valid[y:y + h, x:x + w].mean()) if valid[y:y + h, x:x + w].size else 0.0
    rep = {"name": r["name"], "asset": r.get("asset"), "bbox": b}
    if inside < 0.5:
        rep.update(status="outside screenshot", inside=round(inside, 2), hints=[], badness=0.0, match=None, shift_dp=[0.0, 0.0],
                   scale=1.0, de00_mean=None, ssim=None)
        return rep, None
    hit = _find_region(ctx["Mg32"], ctx["Sg32"], b, ctx["dy0"], ctx["R"])
    if hit and hit[0] < 0.5 and ctx["bottom_extra"] + ctx["top_extra"] > ctx["R"]:
        far = _find_region(ctx["Mg32"], ctx["Sg32"], b, ctx["dy0"] + (ctx["bottom_extra"] + ctx["top_extra"]) // 2,
                           (ctx["bottom_extra"] + ctx["top_extra"]) // 2 + ctx["R"])
        hit = far if far and far[0] > hit[0] + 0.1 else hit
    if hit:
        sc, sx, sy, s, _ = hit
        sw, sh = max(1, round(w * s)), max(1, round(h * s))
        cur = ctx["S"][sy:sy + sh, sx:sx + sw]
        cur = np.asarray(Image.fromarray(cur.astype(np.uint8)).resize((w, h), Image.LANCZOS), dtype=np.float64)
        rep.update(match=round(sc, 3), shift_dp=[round((sx + (sw - w) / 2 - x) * ctx["k"], 1),
                                                round((sy + (sh - h) / 2 - y - ctx["dy0"]) * ctx["k"], 1)],
                   scale=round(s, 3), impl_bbox=[int(sx), int(sy - ctx["dy0"]), int(sw), int(sh)])
    else:
        cur = ctx["A"][y:y + h, x:x + w]
        rep.update(match=None, shift_dp=[0.0, 0.0], scale=1.0)
    own = np.ones((h, w), bool)
    for cb in r["children"]:
        own[max(0, cb["y"] - y):cb["y"] - y + cb["h"], max(0, cb["x"] - x):cb["x"] - x + cb["w"]] = False
    if own.mean() < 0.15:
        own[:] = True
    ref = ctx["M"][y:y + h, x:x + w]
    dmap = _tolerant_de(ref, cur)
    rep["de00_mean"] = round(float(dmap[own].mean()), 2)
    rep["ssim"] = round(float(_ssim_map(_gray(ref), _gray(cur))[own].mean()), 3)
    fill_m, fill_i = _dominant(ref), _dominant(cur)
    share = lambda img, c: float((np.linalg.norm(_lab_arr(img) - _lab_arr(c), axis=-1) < 6).mean())
    rep["fill"] = [_to_hex(fill_m), _to_hex(fill_i), round(float(_de00(_lab_arr(fill_m), _lab_arr(fill_i))), 1),
                   round(min(share(ref, fill_m), share(cur, fill_i)), 2)]
    return rep, {"x": x, "y": y, "w": w, "h": h, "cur": cur, "ref": ref, "dmap": dmap,
                 "own": own, "image": r.get("asset_kind") in ("background", "photo"), "hints": [], "text_boxes": []}


def _compare_region_text(r, rep, part, ctx):
    """Locate typography in a region and add size, colour, position, and font hints."""
    x, y, w, h = part["x"], part["y"], part["w"], part["h"]
    hints, text_boxes = part["hints"], part["text_boxes"]
    for role, t in ctx["role_of"].get(r["name"], []):
        st = ctx["styles"][role]
        ib = rep.get("impl_bbox") or [x, y, w, h]
        tx, ty, tw_, th_ = t["box"]
        px_, py_ = int(0.6 * st["size_px"]) + 6, int(0.35 * st["size_px"]) + 4
        found_ = []
        for ox, oy in {(0, ctx["dy0"]), (0, ib[1] + ctx["dy0"] - y), (ib[0] - x, ib[1] + ctx["dy0"] - y)}:
            win = [tx + ox - px_, ty + oy - py_, tw_ + 2 * px_, th_ + 2 * py_]
            g_ = _locate_text(ctx["S"], win, t["sample"], st["size_px"], st.get("letter_spacing_em", 0), t["ink"])
            if g_ and _de_hex(g_["ink"], t["ink"]) < 20 and 0.6 < g_["box"][3] / th_ < 1.6:
                bx_, sh_ = g_["box"], ctx["S"].shape
                cut = (bx_[1] <= win[1] + 1 and win[1] > 0) or (bx_[1] + bx_[3] >= win[1] + win[3] - 1 and win[1] + win[3] < sh_[0])
                found_.append((abs(math.log(g_["box"][3] / th_)) + abs(math.log(g_["box"][2] / tw_)) + (10 if cut else 0), g_))
        got = min(found_, key=lambda f_: f_[0])[1] if found_ else None
        text_boxes.append((tx - x, ty - y, tw_, th_))
        if not got:
            hints.append(f"text '{t['sample'][:24]}' not found in the implementation")
            continue
        gx, gy, gw, gh = got["box"]
        text_boxes.append((gx - (ib[0] - x) - x, gy - ctx["dy0"] - (ib[1] - y) - y, gw, gh))
        hr, wr = gh / th_, gw / tw_
        dei = round(float(_de00(_lab_arr(_hexi(t["ink"])), _lab_arr(_hexi(got["ink"])))), 1)
        shape = round(_shape_similarity(t["alpha"], got["alpha"], t["sample"])[0], 3)
        fspec, fbest = None, None
        if ctx["fonts_json"] and th_ >= 16 and not ctx["font_check_off"]:
            try:
                fspec, fbest = _impl_font_check(ctx["fonts_json"], role, t["sample"], got["alpha"], st["size_px"], ctx["platform"])
            except (requests.RequestException, RuntimeError, ValueError, KeyError, OSError) as e:
                ctx["font_check_off"] = True
                print(f"  (font verdict skipped: Google Fonts metadata unavailable: {str(e)[:160]})")
        sh = [round((gx - tx) * ctx["k"], 1), round((got["baseline"] - ctx["dy0"] - t["baseline"]) * ctx["k"], 1)]
        rep.setdefault("text", []).append({"role": role, "sample": t["sample"], "height_ratio": round(hr, 3),
                                           "width_ratio": round(wr, 3), "ink": [t["ink"], got["ink"]], "ink_de00": dei,
                                           "glyph_similarity_vs_design": shape, "spec_font_fit": fspec,
                                           "best_font_fit": fbest, "shift_dp": sh})
        lab = f"'{t['sample'][:20]}'"
        if abs(hr - 1) <= 0.1 and len(t["sample"].strip()) >= 3:  # wrong-size or 1-2 character text can't place its neighbours
            ctx.setdefault("text_nodes", []).append({"rep": rep, "label": lab, "role": role, "comp": r["name"], "d": list(t["box"]),
                                                     "i": [gx, gy - ctx["dy0"], gw, gh], "own": r["bbox"],
                                                     "tol": 0.1 * st["size_px"] * ctx["k"]})
        token = f", token {ctx['token_names'][role]}" if role in ctx["token_names"] else ""
        if abs(hr - 1) >= 0.04 and abs(gh - th_) >= 2:
            design_sp = st["size_px"] * ctx["k"]
            hint = (f"{lab} {abs(hr - 1) * 100:.0f}% {'larger' if hr > 1 else 'smaller'}: renders like ~{design_sp * hr:.1f}sp "
                    f"(design {design_sp:g}sp{token}; ink height {th_} -> {gh}px)")
            if fspec and fbest and fbest[:2] == fspec[:2]:
                hint += f"; if it already uses {design_sp:g}sp the font runs {abs(hr - 1) * 100:.0f}% {'large' if hr > 1 else 'small'}: use {design_sp / hr:.1f}sp"
            hints.append(hint)
            ctx["size_hints"].append((rep, hint))
        elif abs(wr - 1) >= 0.05 and abs(gw - tw_) >= 3:
            hints.append(f"{lab} {abs(wr - 1) * 100:.0f}% {'wider' if wr > 1 else 'narrower'} at the same height "
                         "(letter-spacing or font family/weight differs)")
        if fspec and fbest and fbest[2] > fspec[2] + (0.08 if fbest[0] != fspec[0] else 0.05):
            what = f"{fbest[0]} {fbest[1]}" if fbest[0] != fspec[0] else f"weight {fbest[1]}"
            hints.append(f"{lab} looks like {what} (glyph fit {fbest[2]:.2f}) rather than the specified {fspec[0]} {fspec[1]} "
                         f"({fspec[2]:.2f})" + (" (fontFamily not set?)" if fbest[0] == "Roboto" and ctx["platform"] == "android" else ""))
        if dei >= 3:
            hints.append(f"{lab} colour off by dE {dei} ({t['ink']} -> {got['ink']})")
        if abs(sh[0]) >= 2 or abs(sh[1]) >= 2:
            hints.append(f"{lab} shifted {sh[0]:+.0f}dp x, {sh[1]:+.0f}dp y")


def _compare_finish_region(r, rep, part):
    """Classify difference blobs, add structural hints, and calculate region badness."""
    x, y, cur, ref, dmap, own = (part[key] for key in ("x", "y", "cur", "ref", "dmap", "own"))
    image, hints, text_boxes = part["image"], part["hints"], part["text_boxes"]
    blob_mask = (dmap > 12) & own
    for bx_, by_, bw_, bh_ in text_boxes:
        blob_mask[max(0, int(by_) - 3):int(by_ + bh_) + 3, max(0, int(bx_) - 3):int(bx_ + bw_) + 3] = False
    if blob_mask.any():
        bl = max(_blobs(blob_mask), key=lambda t: len(t[0])) if blob_mask.sum() < 60000 else None
        if bl is not None and len(bl[0]) >= max(30, 0.01 * own.sum()):
            ys_, xs_ = bl
            bb = (int(xs_.min()), int(ys_.min()), int(xs_.max() - xs_.min() + 1), int(ys_.max() - ys_.min() + 1))
            sub = lambda img: img[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            inkish = lambda img: float((np.linalg.norm(_lab_arr(sub(img)) - _lab_arr(_dominant(img)), axis=-1) > 15).mean())
            dm, di = inkish(ref), inkish(cur)
            what = ("differs inside the artwork (redrawn image)" if image else
                    "missing in the implementation" if dm > 0.15 and di < 0.5 * dm else
                    "extra in the implementation" if di > 0.15 and dm < 0.5 * di else "drawn differently")
            rep["diff_blob"] = {"bbox": [x + bb[0], y + bb[1], bb[2], bb[3]], "px": int(len(ys_)), "what": what}
            hints.append(f"element at x={x + bb[0]},y={y + bb[1]} ({bb[2]}x{bb[3]}px) {what}")
    strong = (rep["match"] is not None and rep["match"] < 0.6) or \
        rep.get("diff_blob", {}).get("what", "").startswith(("missing", "extra"))
    if strong:
        rep["shift_dp"], rep["scale"] = [0.0, 0.0], 1.0
        rep.pop("impl_bbox", None)
    sx_, sy_ = rep["shift_dp"]
    if not strong and (abs(sx_) >= 2 or abs(sy_) >= 2):
        hints.append(f"shifted {'%+.0fdp horizontally' % sx_ if abs(sx_) >= 2 else ''}"
                     f"{' and ' if abs(sx_) >= 2 and abs(sy_) >= 2 else ''}{'%+.0fdp vertically' % sy_ if abs(sy_) >= 2 else ''}")
    if not strong and "text" not in rep and not r["children"] and r.get("asset_kind") not in ("background", "photo"):
        ob_m, ob_i = _object_box(ref), _object_box(cur)
        if ob_m and ob_i:
            ratio = rep["scale"] * math.sqrt((ob_i[2] * ob_i[3]) / (ob_m[2] * ob_m[3]))
            rep["object"] = {"design_px": [round(v) for v in ob_m[2:]], "impl_px": [round(v * rep["scale"]) for v in ob_i[2:]],
                             "size_ratio": round(ratio, 3)}
            rep["scale"] = round(ratio, 3)
        if abs(rep["scale"] - 1) >= 0.06:
            o = rep.get("object")
            hints.append(f"element ~{abs(rep['scale'] - 1) * 100:.0f}% {'larger' if rep['scale'] > 1 else 'smaller'}"
                         + (f" (visible {o['design_px'][0]}x{o['design_px'][1]} -> {o['impl_px'][0]}x{o['impl_px'][1]}px)" if o else ""))
    if 3 <= rep["fill"][2] < 25 and rep["fill"][3] >= 0.3:
        hints.append(f"fill colour off by dE {rep['fill'][2]} ({rep['fill'][0]} -> {rep['fill'][1]})")
    if rep["match"] is not None and rep["match"] < 0.6 and not rep.get("diff_blob", {}).get("what", "").startswith(("missing", "extra")):
        hints.append(f"content differs strongly (match {rep['match']}): missing, replaced or restyled element")
    elif rep["ssim"] < 0.75 and not hints:
        hints.append(f"structure differs (SSIM {rep['ssim']}): check shapes, icon/image content, font")
    tx_ = rep.get("text", [])
    badness = ((0.4 if image else 1) * (min(1, (1 - rep["ssim"]) / 0.5) + min(1, rep["de00_mean"] / 12))
               + min(1, math.hypot(sx_, sy_) / 8) + min(1, abs(math.log(rep["scale"])) / 0.1)
               + (min(1.5, rep["fill"][2] / 5) if 3 <= rep["fill"][2] < 25 and rep["fill"][3] >= 0.3 else 0)
               + sum(min(1.5, abs(math.log(t_["height_ratio"])) / 0.06) + min(1, t_["ink_de00"] / 10)
                     + (min(1.5, max(0, t_["best_font_fit"][2] - t_["spec_font_fit"][2] - 0.05) / 0.05)
                        * (1 if t_["best_font_fit"][0] != t_["spec_font_fit"][0] else 0.4)
                        if t_["spec_font_fit"] and t_["best_font_fit"] else 0) for t_ in tx_)
               + ((0.3 if image else 0.5 if r.get("asset") else 1) * min(1, rep["diff_blob"]["px"] / 300)
                  + (1.0 if rep["diff_blob"]["what"].startswith(("missing", "extra")) else 0) if "diff_blob" in rep else 0)
               + (1.0 if strong else 0))
    rep["badness"] = round(badness, 3)
    rep["hints"] = hints
    return rep


def _text_scale(found):
    """The common height ratio of the implemented text when nearly all of it is off by the same factor, or None."""
    ratios = [t_["height_ratio"] for r in found for t_ in r.get("text", [])]
    if len(ratios) >= 3:
        med = float(np.median(ratios))
        if abs(med - 1) > 0.05 and sum(abs(v - med) <= 0.05 for v in ratios) >= max(3, 0.75 * len(ratios)):
            return med
    return None


LAYOUT_TOL_DP = {"radius": 2.0, "radius_rel": 0.25, "inset": 3.0, "gap": 3.0, "size": 4.0, "size_rel": 0.03}


def _layout_tokens(d, platform):
    """(kind, dp, name, source) for the radius and spacing tokens in DIR/tokens/tokens.json, named for the platform."""
    try:
        tk = json.loads((d / "tokens" / "tokens.json").read_text())
    except (OSError, ValueError):
        return []
    p = tk.get("prefix") or "App"
    out = []
    for kind, key in (("radius", "radii"), ("spacing", "spacing")):
        for t in tk.get(key) or []:
            if not isinstance(t, dict) or not isinstance(t.get("dp"), (int, float)):
                continue
            if platform in ("ios", "macos") and t.get("sw"):
                name = f"{p}{'Radius' if kind == 'radius' else 'Spacing'}.{t['sw']}"
            elif platform == "web" and t.get("css"):
                name = f"var(--{'radius' if kind == 'radius' else 'space'}-{t['css']})"
            elif t.get("kt"):
                name = f"{t['kt']}Shape" if kind == "radius" else f"{p}Dimens.{t['kt']}"
            else:
                continue
            out.append((kind, float(t["dp"]), name, str(t.get("source", ""))))
    return out


def _token_for(tokens, kind, dp, source):
    """', token X' for the token holding dp whose redline source is this measurement, or ''."""
    pick = next((t for t in tokens if t[0] == kind and abs(t[1] - dp) <= 0.5 and t[3].startswith(source)), None)
    return f", token {pick[2]}" if pick else ""


def _compare_layout(regs, found, data):
    """Corner radii, surface sizes, padding and gaps of the built screen against the design, measured with redline's
    surface geometry on both. A wrong gap is reported once, and the shifts it causes below are dropped."""
    from .redline import _not_surface, _surface_geometry  # deferred: redline imports this module
    k, dy0 = data["k"], data["dy0"]
    shot_dp, design_dp = data["shot_dp"], data["design_dp"]
    if shot_dp and abs(shot_dp - design_dp) / design_dp > 0.05:
        data["overall"]["layout"] = {"skipped": "screenshot width differs from the design"}
        return
    Mu, Su = data["M"].astype(np.uint8), data["S"].astype(np.uint8)
    tokens = _layout_tokens(data["d"], data["platform"])
    T = LAYOUT_TOL_DP
    text_boxes = [inst["box"] for t in data["texts"].values() for inst in t.get("instances", [])]
    nodes = []
    reach = max(8, round(0.02 * data["W"]))
    system = {c.get("name"): str(c.get("native", "")).lower() == "system" for c in data["spec"].get("components", []) if isinstance(c, dict)}
    for r, rep in zip(regs, found):
        if rep.get("status") or _not_surface(r, data["texts"]) == "text component" or system.get(r["name"]):
            continue  # the OS draws system UI (status bar, home indicator), not the app
        b = r["bbox"]
        why = _not_surface(r, data["texts"])
        if why == "container of other components" or min(b["w"], b["h"]) * k < 20:
            continue  # tiny elements (icons in rows) match ambiguously; their rows and texts carry the layout
        tx, ty = round(rep["shift_dp"][0] / k), round(rep["shift_dp"][1] / k)  # 0, 0 when the region didn't match
        gd0 = _surface_geometry(Mu, b) if not why else None
        gd = gi = None
        if gd0:  # re-measured from the found edges with a short reach on both screens, so nearby content can't pull them
            box = lambda v, dx=0, dy=0: {"x": round(v[0] + dx), "y": round(v[1] + dy), "w": round(v[2]), "h": round(v[3])}
            gd = _surface_geometry(Mu, box(gd0["box"]), reach)
            for c in ([box(gd["box"], sx_, sy_ + dy0) for sx_, sy_ in ((0, 0), (0, ty), (tx, ty), (tx, 0))] if gd else ()):
                if c["x"] >= 0 and c["y"] >= 0 and c["x"] + c["w"] <= Su.shape[1] and c["y"] + c["h"] <= Su.shape[0]:
                    gi = _surface_geometry(Su, c, reach)
                    if gi and _de_hex(gi["fill"], gd["fill"]) < 10 and all(abs(gi["box"][j] / gd["box"][j] - 1) <= 0.15 for j in (2, 3)):
                        break
                    gi = None
        solid = lambda g_: g_["shadow"] or _de_hex(g_["fill"], g_["outside"]) >= 1.5  # an outline or a phantom has no fill of its own
        if gd and gi and solid(gd) and solid(gi):
            di, ii = list(gd["box"]), list(gi["box"])
            ii[1] -= dy0
            x_, y_, w_, h_ = (round(v) for v in di)
            inner = Mu[y_ + h_ // 4:y_ + 3 * h_ // 4, x_ + w_ // 4:x_ + 3 * w_ // 4].reshape(-1, 3).astype(np.float64)
            photo = len(inner) > 0 and float(np.linalg.norm(inner - np.median(inner, 0), axis=1).mean()) > 12
            nodes.append({"rep": rep, "label": r["name"], "comp": r["name"], "d": di, "i": ii, "gd": gd, "gi": gi, "photo": photo})
            continue
        if "impl_bbox" not in rep or not r.get("asset") or any(min(b["x"] + b["w"], ib_[0] + ib_[2]) - max(b["x"], ib_[0]) > 0.5 * ib_[2]
               and min(b["y"] + b["h"], ib_[1] + ib_[3]) - max(b["y"], ib_[1]) > 0.5 * ib_[3] for ib_ in text_boxes):
            continue  # only artwork has a box of its own; a text's box follows the font, a row's its faint edges
        pad = 4
        x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
        ob_d = _object_box(data["M"][y0:b["y"] + b["h"] + pad, x0:b["x"] + b["w"] + pad])
        ix0, iy0 = max(0, b["x"] + tx - pad), max(0, b["y"] + ty + dy0 - pad)
        ob_i = _object_box(data["S"][iy0:b["y"] + ty + dy0 + b["h"] + pad, ix0:b["x"] + tx + b["w"] + pad])
        if ob_d and ob_i and 0.7 <= ob_i[2] / ob_d[2] <= 1.4 and 0.7 <= ob_i[3] / ob_d[3] <= 1.4:
            nodes.append({"rep": rep, "label": r["name"], "comp": r["name"], "d": [x0 + ob_d[0], y0 + ob_d[1], ob_d[2], ob_d[3]],
                          "i": [ix0 + ob_i[0], iy0 + ob_i[1] - dy0, ob_i[2], ob_i[3]]})
    surf = [n for n in nodes if "gd" in n]
    ratios = [n["i"][2] / n["d"][2] for n in surf if not set(n["gd"]["full_bleed"]) & {"left", "right"}]
    scales = [rep["scale"] for rep in found if "impl_bbox" in rep]
    if len(ratios) < 2 and len(scales) >= 3:
        ratios = scales
    if len(ratios) >= 2 and abs(float(np.median(ratios)) - 1) > 0.04:
        data["overall"]["layout"] = {"skipped": f"the surfaces are ~{abs(float(np.median(ratios)) - 1) * 100:.0f}% "
                                                f"{'wider' if np.median(ratios) > 1 else 'narrower'}: the screenshot width differs from the design"}
        return
    if not _text_scale(found):  # all text off by one factor (font scale): the text can't place anything
        nodes += [dict(t, text=True) for t in data.get("text_nodes", [])]
    issues = []

    def add(n, kind, delta, hint, weight):
        if hint in n["rep"]["hints"]:
            return
        n["rep"]["hints"].append(hint)
        n["rep"].setdefault("layout", []).append({"kind": kind, "delta_dp": round(delta, 1), "hint": hint})
        n["rep"]["badness"] = round(n["rep"]["badness"] + min(1.5, abs(delta) / weight), 3)
        issues.append(kind)
    dp = lambda v: round(v * k, 1)
    tol = lambda n: max(T["inset"], n.get("tol", 0))  # ink edges of large text depend on the font: side bearings, descenders
    for n in surf:
        gd, gi = n["gd"], n["gi"]
        if not (gd["shape"] in ("pill", "circle") and gi["shape"] in ("pill", "circle")):
            pairs = [(c, a_, b_) for c, a_, b_ in zip(("top-left", "top-right", "bottom-left", "bottom-right"), gd["corner_radii"], gi["corner_radii"])
                     if a_ is not None and b_ is not None and not ({"top", "left"} if c == "top-left" else {"top", "right"} if c == "top-right"
                                                                    else {"bottom", "left"} if c == "bottom-left" else {"bottom", "right"})
                     & set(gd["full_bleed"] + gi["full_bleed"])]
            rt, rr = (4.0, 0.3) if n["photo"] else (T["radius"], T["radius_rel"])  # corners over a photo measure less cleanly
            bad = [(c, dp(a_), dp(b_)) for c, a_, b_ in pairs if abs(b_ - a_) * k >= max(rt, rr * a_ * k)]
            if bad:
                full = "fully rounded" if gd["shape"] in ("pill", "circle") else None
                if len(bad) == len(pairs):
                    rd, ri = float(np.median([b_[1] for b_ in bad])), float(np.median([b_[2] for b_ in bad]))
                    hint = f"corner radius {ri:g}dp, design {full or f'{rd:g}dp'}" + ("" if full else _token_for(tokens, "radius", rd, f"redline: {n['comp']}"))
                else:
                    hint = "corners " + ", ".join(f"{c} {ri:g}dp (design {rd:g}dp)" for c, rd, ri in bad)
                add(n, "radius", max((ri - rd for _, rd, ri in bad), key=abs), hint, 4)
        if not (set(gd["full_bleed"]) & {"left", "right"}):
            wd, wi = dp(n["d"][2]), dp(n["i"][2])
            if abs(wi - wd) >= max(T["size"], T["size_rel"] * wd):
                add(n, "size", wi - wd, f"{wi - wd:+g}dp {'wider' if wi > wd else 'narrower'} ({wi:g}dp, design {wd:g}dp)", 8)
    contains = lambda p, c: (p["d"][0] <= c["d"][0] + 2 and p["d"][1] <= c["d"][1] + 2 and p["d"][0] + p["d"][2] >= c["d"][0] + c["d"][2] - 2
                             and p["d"][1] + p["d"][3] >= c["d"][1] + c["d"][3] - 2 and p["d"][2] * p["d"][3] > c["d"][2] * c["d"][3])
    for n in nodes:
        par = [p for p in surf if p is not n and contains(p, n)]
        n["parent"] = min(par, key=lambda p: p["d"][2] * p["d"][3]) if par else None
    W = data["W"]
    every = [[r["bbox"][c] for c in "xywh"] for r in regs] + list(text_boxes)
    for n in nodes:
        P = n["parent"]
        sib = [o for o in nodes if o is not n and o["parent"] is P]
        x, y, w, h = n["d"]
        above = [o for o in sib if o["d"][1] + o["d"][3] <= y + 2
                 and min(x + w, o["d"][0] + o["d"][2]) - max(x, o["d"][0]) > 0.3 * min(w, o["d"][2])]
        A = max(above, key=lambda o: o["d"][1] + o["d"][3]) if above else None
        if A:  # a gap is only checked between neighbours: nothing unmeasured may sit between them
            y0_, y1_ = A["d"][1] + A["d"][3], y
            x0_, x1_ = max(x, A["d"][0]), min(x + w, A["d"][0] + A["d"][2])
            if any(bb[1] >= y0_ - 2 and bb[1] + bb[3] <= y1_ + 2 and bb[3] > 0 and min(x1_, bb[0] + bb[2]) - max(x0_, bb[0]) > 0
                   for bb in every):
                A = None
        n["pred"] = A
        pd, pi = (P["d"], P["i"]) if P else ([0, 0, W, data["H"]], [0, 0, W, data["H"]])
        where = f"in {P['label']}" if P else "from the screen edge"
        li, ld = n["i"][0] - pi[0], x - pd[0]
        ri_, rd_ = pi[0] + pi[2] - n["i"][0] - n["i"][2], pd[0] + pd[2] - x - w
        centred = abs(ld - rd_) <= max(4, 0.02 * W)
        o_ = n.get("own")  # text centred in an outlined chip or button: the font's width moves its edges, not the layout
        chip = bool(o_) and not any(p["comp"] == n["comp"] for p in surf) and o_["w"] > 1.3 * w \
            and abs((x - o_["x"]) - (o_["x"] + o_["w"] - x - w)) <= max(4, 0.15 * w) \
            or n.get("text") and centred and w >= 0.6 * pd[2]  # a full-width line: centred or left-aligned can't be told apart
        A = n["pred"]
        cx = lambda o, s_: o[s_][0] + o[s_][2] / 2
        if chip:
            pass
        elif A and abs(cx(n, "d") - cx(A, "d")) <= max(3, 0.01 * W) and not centred and w < 0.9 * A["d"][2] \
                and abs(x - A["d"][0]) > 0.05 * A["d"][2] and y - A["d"][1] - A["d"][3] <= max(h, A["d"][3]):
            delta = ((cx(n, "i") - cx(A, "i")) - (cx(n, "d") - cx(A, "d"))) * k
            if abs(delta) >= max(tol(n), tol(A)):
                add(n, "padding", delta, f"{n['label']} off-centre under {A['label']} by {delta:+.0f}dp", 6)
        elif centred and (n.get("text") or w < 0.6 * pd[2]):
            delta = ((li - ri_) - (ld - rd_)) / 2 * k
            if abs(delta) >= tol(n):
                add(n, "padding", delta, f"{n['label']} off-centre {where} by {delta:+.0f}dp", 6)
        else:
            sides = [("left", li, ld), ("right", ri_, rd_)] if centred else [("left", li, ld) if ld <= rd_ else ("right", ri_, rd_)]
            for side, vi, vd in sides:
                if abs(vi - vd) * k >= tol(n) and not (P and side in P["gd"]["full_bleed"] and vd < 1):
                    src = f"redline: {P['comp']} " if P else f"redline: {n['comp']} {side} of screen edge"
                    add(n, "padding", (vi - vd) * k, f"{n['label']} {side} {'padding' if P else 'margin'} {where} {dp(vi):g}dp, "
                        f"design {dp(vd):g}dp" + (" (ink edge)" if n.get("text") else _token_for(tokens, "spacing", dp(vd), src)), 6)
        if A:
            gi_, gd_ = n["i"][1] - A["i"][1] - A["i"][3], y - A["d"][1] - A["d"][3]
            if abs(gi_ - gd_) * k >= max(tol(n), tol(A), T["gap"]):
                line = n.get("text") and A.get("text") and A["comp"] == n["comp"] and A["role"] == n["role"]
                add(n, "gap", (gi_ - gd_) * k, (f"line gap above {n['label']} {dp(gi_):g}dp, design {dp(gd_):g}dp (line height)" if line else
                    f"gap above {n['label']} (below {A['label']}) {dp(gi_):g}dp, design {dp(gd_):g}dp"
                    + _token_for(tokens, "spacing", dp(gd_), f"redline: gap below {A['comp']} (to {n['comp']})")), 6)
        elif P and "top" not in P["gd"]["full_bleed"]:
            ti, td = n["i"][1] - pi[1], y - pd[1]
            if abs(ti - td) * k >= tol(n):
                add(n, "padding", (ti - td) * k, f"{n['label']} top padding {where} {dp(ti):g}dp, design {dp(td):g}dp"
                    + (" (ink edge)" if n.get("text") else _token_for(tokens, "spacing", dp(td), f"redline: {P['comp']} ")), 6)
    first = [n for n in nodes if n["parent"] is None and n["pred"] is None]
    offs = [(n["i"][1] - n["d"][1]) * k for n in first]
    common = bool(offs) and max(offs) - min(offs) <= 2
    boxed = {id(n["rep"]) for n in nodes if not n.get("text")}
    for n in nodes:
        if n in first and not common:
            continue
        drop = (f"{n['label']} shifted",) + (() if id(n["rep"]) in boxed else ("shifted ",)) if n.get("text") else ("shifted ",)
        n["rep"]["hints"] = [h_ for h_ in n["rep"]["hints"] if not h_.startswith(drop)]
    data["overall"]["layout"] = {"surfaces": len(surf), "elements": len(nodes),
                                 **{kind: issues.count(kind) for kind in ("radius", "size", "padding", "gap")}}
    if common and abs(float(np.median(offs))) >= 2:
        data["overall"]["layout"]["top_offset_dp"] = round(float(np.median(offs)), 1)


def _compare_rank_regions(found, data):
    """Suppress uniform text-scale hints, rank regions, and locate unlabelled differences."""
    global_scale = _text_scale(found)
    if global_scale:
        for rep, hint in data["size_hints"]:
            rep["hints"] = [h_ for h_ in rep["hints"] if h_ != hint]
        data["overall"]["text_scale"] = round(global_scale, 3)
    found.sort(key=lambda r: -r["badness"])
    H, W = data["H"], data["W"]
    covered = np.zeros((H, W), bool)
    for r in found:
        b = r["bbox"]
        covered[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = True
    g, unl = 32, []
    for yy in range(0, H - g + 1, g):
        for xx in range(0, W - g + 1, g):
            cell = data["de"][yy:yy + g, xx:xx + g]
            if data["valid"][yy:yy + g, xx:xx + g].all() and not covered[yy:yy + g, xx:xx + g].any() and cell.mean() > 8:
                unl.append({"bbox": [xx, yy, g, g], "de00_mean": round(float(cell.mean()), 1)})
    ranked = [r for r in found if r.get("status") != "outside screenshot"]
    return found, global_scale, unl, ranked


def _compare_vision_rows(a, data, ranked):
    """Build and submit the optional vision comparison for the highest-ranked regions."""
    if not a.vision or not ranked:
        return
    mock, shot, W, H = data["mock"], data["shot"], data["W"], data["H"]
    top, rows = ranked[:min(6, len(ranked))], []
    for i, r in enumerate(top):
        b = r["bbox"]
        pad = 12
        box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(W, b["x"] + b["w"] + pad), min(H, b["y"] + b["h"] + pad))
        rows.append([(f"{i + 1}. design", mock.crop(box)),
                     (f"{i + 1}. implemented", shot.crop(_impl_box(box, r, data["dy0"], data["k"])))])
    vis = data["out"] / "vision_rows.png"
    sheet(rows, vis, T=300)
    try:
        res = ask([COMPARE_PROMPT, Image.open(vis)], COMPARE_SCHEMA, "compare_rows")
        for row in res["rows"]:
            if isinstance(row, dict) and isinstance(row.get("index"), int) and 1 <= row["index"] <= len(top) \
                    and isinstance(row.get("differences"), list):
                top[row["index"] - 1]["vision"] = [str(v) for v in row["differences"]]
    except Exception as e:
        print(f"  ! vision pass failed: {e}")


def _compare_write_report(a, data, found, unl, ranked):
    """Render heatmaps and region sheets, then write the JSON comparison report."""
    W, H, out = data["W"], data["H"], data["out"]
    de, M, A = data["de"], data["M"], data["A"]
    mock, shot = data["mock"], data["shot"]
    hm = Image.fromarray((_heat_rgb(de / 20) * 0.75 + M * 0.25).astype(np.uint8))
    dr = ImageDraw.Draw(hm)
    for i, r in enumerate(ranked[:a.top]):
        b = r["bbox"]
        dr.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline="#00e5ff", width=3)
        dr.text((b["x"] + 3, b["y"] + 2), str(i + 1), fill="#00e5ff")
    for u in unl:
        x_, y_, w_, h_ = u["bbox"]
        dr.rectangle([x_, y_, x_ + w_, y_ + h_], outline="#ffffff", width=1)
    panel = Image.new("RGB", (W * 3 + 40, H), "white")
    panel.paste(mock, (0, 0))
    panel.paste(Image.fromarray(A.astype(np.uint8)), (W + 20, 0))
    panel.paste(hm, (2 * W + 40, 0))
    panel.save(out / "heatmap.png")
    hm.save(out / "heatmap_only.png")
    rows = []
    for i, r in enumerate(ranked[:a.top]):
        b = r["bbox"]
        pad = 10
        box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(W, b["x"] + b["w"] + pad), min(H, b["y"] + b["h"] + pad))
        rows.append([(f"#{i + 1} {r['name'][:30]} design", mock.crop(box)),
                     ("implemented", shot.crop(_impl_box(box, r, data["dy0"], data["k"]))), ("dE heat", hm.crop(box))])
    if rows:
        sheet(rows, out / "regions.png", T=260)
    report = {"overall": data["overall"], "regions": found, "unlabelled": unl,
              "mockup": str(a.mockup or data["d"] / (data["manifest"] or {}).get("mockup", "mockup.png")),
              "screenshot": str(a.screenshot), "px_to_dp": data["k"], "design_width_dp": data["design_dp"],
              "screenshot_width_dp": data["shot_dp"], "fonts": str(data["fpath"]) if data["fonts_json"] else None,
              "fail_below": a.fail_below}
    (out / "report.json").write_text(json.dumps(report, indent=2))


def _compare_print_report(a, data, found, unl, ranked, global_scale):
    """Print comparison scores, alignment context, ranked fixes, and the fail-below gate."""
    overall, H = data["overall"], data["H"]
    top_extra, bottom_extra, k = data["top_extra"], data["bottom_extra"], data["k"]
    y0, y1 = data["y0"], data["y1"]
    design_dp, platform, fonts_json = data["design_dp"], data["platform"], data["fonts_json"]
    out = data["out"]
    print(f"overall score {overall['score']}/100  (SSIM (7px box) {overall['ssim']}, mean dE00 {overall['mean_de00']}, "
          f"{overall['pct_pixels_de_gt_10']}% px dE>10)")
    lay = overall.get("layout") or {}
    if lay.get("skipped"):
        print(f"  (layout checks skipped: {lay['skipped']})")
    elif lay:
        print(f"  layout: {lay['surfaces']} surfaces measured on both screens, {lay['elements']} elements placed; "
              + ", ".join(f"{lay[x]} {x}" for x in ("radius", "size", "padding", "gap")) + " differences")
        if lay.get("top_offset_dp"):
            t_ = lay["top_offset_dp"]
            print(f"  ~ the top elements sit {abs(t_):g}dp {'lower' if t_ > 0 else 'higher'} than the alignment of the rest: the top inset "
                  "or a gap below is off (gaps are checked between neighbours, so the gap hints name it)")
    if top_extra or bottom_extra:
        print(f"  aligned with a {top_extra}px ({top_extra * k:.0f}dp) top offset (status bar / extra top inset); "
              f"{bottom_extra}px extra at the bottom (taller screen or nav bar) ignored")
    if overall["coverage"] < 0.98:
        print(f"  ! compared {overall['coverage'] * 100:.0f}% of the design (screenshot covers rows {y0}-{y1} of {H}); "
              f"scroll and capture the rest ({sum(r.get('status') == 'outside screenshot' for r in found)} regions outside the screenshot)")
    if global_scale:
        est = design_dp * global_scale
        print(f"  ! all text is ~{abs(global_scale - 1) * 100:.0f}% {'larger' if global_scale > 1 else 'smaller'}: the screenshot shows a "
              f"~{est:.0f}dp-wide screen (or a {global_scale:.2f}x font scale), the design is {design_dp}dp; re-shoot at {design_dp}dp "
              f"(Preview widthDp={design_dp}, fontScale=1) -- per-text size hints suppressed")
    if platform == "ios" and fonts_json:
        print("  (SF Pro, the iOS default, is not on Google Fonts: an unset font can't be named; check the 'looks like' verdicts)")
    print(f"worst regions (of {len(ranked)}" + (f"; {len(found) - len(ranked)} outside the screenshot" if len(found) > len(ranked) else "") + "):")
    for i, r in enumerate(ranked[:a.top]):
        hint = "; ".join(r["hints"]) or f"minor (dE {r['de00_mean']}, SSIM {r['ssim']})"
        print(f"  {i + 1:2d}. {r['name'][:34]:34s} bad {r['badness']:.2f}  {hint}")
        for v in r.get("vision", []):
            print(f"        vision: {v}")
    if unl:
        print(f"  + {len(unl)} unlabelled 32px cells with dE>8 (white boxes on the heatmap)")
    print(f"-> {out}/report.json, heatmap.png (design | implementation aligned | dE heatmap), regions.png")
    if a.fail_below is not None:
        ok = overall["score"] >= a.fail_below
        print(f"gate: score {overall['score']} {'>=' if ok else '<'} --fail-below {a.fail_below:g}: {'pass' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


def cmd_compare(a):
    data = _compare_typography(_compare_align_screens(_compare_load_inputs(a)), a)
    d, spec, m, mock_rgba, mock, shot = (data[k] for k in ("d", "spec", "manifest", "mock_rgba", "mock", "shot"))
    W, H, k, design_dp, shot_dp, out = (data[x] for x in ("W", "H", "k", "design_dp", "shot_dp", "out"))
    M, S, A, de, valid, Mg, Sg = (data[x] for x in ("M", "S", "A", "de", "valid", "Mg", "Sg"))
    dy0, top_extra, bottom_extra, y0, y1, overall = (data[x] for x in
        ("dy0", "top_extra", "bottom_extra", "y0", "y1", "overall"))
    texts, role_of, fonts_json = data["texts"], data["role_of"], data["fonts_json"]
    token_names, platform, styles = data["token_names"], data["platform"], data["styles"]
    Sg32, Mg32, R = data["Sg32"], data["Mg32"], data["R"]
    found = []
    font_check_off, size_hints = data["font_check_off"], data["size_hints"]
    regs = _regions(spec, m, W, H)
    for r in regs:
        rep, part = _compare_measure_region(r, data)
        if part is None:
            found.append(rep)
            continue
        _compare_region_text(r, rep, part, data)
        x, y, w, h = part["x"], part["y"], part["w"], part["h"]
        cur, ref, dmap, own = part["cur"], part["ref"], part["dmap"], part["own"]
        image, hints, text_boxes = part["image"], part["hints"], part["text_boxes"]
        found.append(_compare_finish_region(r, rep, part))
    _compare_layout(regs, found, data)
    found, global_scale, unl, ranked = _compare_rank_regions(found, data)
    _compare_vision_rows(a, data, ranked)
    _compare_write_report(a, data, found, unl, ranked)
    _compare_print_report(a, data, found, unl, ranked, global_scale)
