"""Trace, rig, animate."""
import argparse
import base64
import concurrent.futures as cf
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .core import (
    ApiError, _chrome_err, _chrome_flags, _open_image, _png_bytes, _pos_float, _pos_int, _read_json, _res_name,
    _unit_float, ask, edit,
)
from .imaging import _rgba, sheet
from .vectordrawable import _aapt2_line, _vd_lint, aapt2_check, aapt2_level, svg_to_vd, vd_to_svg
from .frames import _square
from .vectorize import _TOK, _d, _hex_rgb, _subpaths, _validate_vec, build_lottie
from .previews import _check_vec, _chrome, _lottie_js, _vec_svg

# ---- raster -> vector tracing, part rigs, procedural keyframes

_CHROME_FAILED_REASON = None  # why the last headless capture produced nothing (printed once per process)


def _tr_chrome_reason():
    """One line saying why fidelity was not measured (no Chrome, or Chrome started and produced no screenshot)."""
    return _CHROME_FAILED_REASON or ("no Chrome/Chromium found (set GPTKIT_CHROME)" if not _chrome()
                                     else "headless Chrome produced no screenshot")


def chrome_shot(html, w, h, budget=3000):
    """Render an HTML string in headless Chrome at w x h CSS px on a transparent background -> RGBA image or None."""
    global _CHROME_FAILED_REASON
    import tempfile
    exe = _chrome()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        page, png = Path(td) / "page.html", Path(td) / "shot.png"
        page.write_text(html)
        try:
            r = subprocess.run([exe, *_chrome_flags(w, h), f"--user-data-dir={td}/prof", f"--virtual-time-budget={budget}",
                                f"--screenshot={png}", page.as_uri()], capture_output=True, text=True, timeout=180)
            reason = None if png.exists() else _chrome_err(r)
        except (subprocess.TimeoutExpired, OSError) as e:
            reason = f"{exe}: {type(e).__name__}"
        if reason is None:
            return Image.open(png).convert("RGBA").crop((0, 0, w, h))  # the window is padded (C1); the page is not
    if _CHROME_FAILED_REASON is None:
        print(f"  ! headless Chrome produced no screenshot: {reason}")
    _CHROME_FAILED_REASON = reason
    return None


def _tr_render_svgs(svgs, sizes):
    """Rasterise SVG strings at the given (w, h) with real alpha, batched into few Chrome calls. None if no Chrome."""
    out, batch, bw = [], [], 0
    for s, (w, h) in list(zip(svgs, sizes)) + [(None, (10 ** 9, 0))]:
        if batch and (s is None or bw + w > 8000):
            H = max(h for _, (_, h) in batch)
            x, tags = 0, []
            for bs, (w_, h_) in batch:
                tags.append(f'<img src="data:image/svg+xml;base64,{base64.b64encode(bs.encode()).decode()}" '
                            f'style="position:absolute;left:{x}px;top:0;width:{w_}px;height:{h_}px">')
                x += w_
            strip = chrome_shot(f'<!doctype html><body style="margin:0;background:transparent">{"".join(tags)}</body>', x, H)
            if strip is None:
                return None
            x = 0
            for _, (w_, h_) in batch:
                out.append(strip.crop((x, 0, x + w_, h_)))
                x += w_
            batch, bw = [], 0
        if s is not None:
            batch.append((s, (w, h)))
            bw += w
    return out


def trace_metrics(src, ren, display=None):
    """Fidelity of a rendered vector vs its raster source: grey-composited mean diff, alpha MAE, silhouette IoU."""
    def m(a, b):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        comp = lambda x: x[..., :3] * (x[..., 3:] / 255) + 128 * (1 - x[..., 3:] / 255)
        ma, mb = a[..., 3] > 127, b[..., 3] > 127
        ga, gb = (np.abs(np.diff(comp(x).mean(2), axis=0))[:, :-1] + np.abs(np.diff(comp(x).mean(2), axis=1))[:-1]
                  for x in (a, b))
        return {"diff": round(float(np.abs(comp(a) - comp(b)).mean()), 2),
                "edge_err": round(float(np.abs(ga - gb).mean()), 2),
                "alpha_mae": round(float(np.abs(a[..., 3] - b[..., 3]).mean()), 2),
                "iou": round(float((ma & mb).sum() / max(1, (ma | mb).sum())), 4)}
    res = m(src, ren.resize(src.size, Image.LANCZOS))
    if display:
        s = display / max(src.size)
        ds = (max(1, round(src.width * s)), max(1, round(src.height * s)))
        res["display"] = {"px": ds, **m(src.resize(ds, Image.LANCZOS), ren.resize(ds, Image.LANCZOS))}
    return res


def _label(mask):
    """4-connected components of a boolean mask -> (int32 labels, 0 = background, count)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab.astype(np.int32), int(n)
    except ImportError:
        pass
    H, W = mask.shape
    parent = [0]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    rows, prev = [], []
    for y in range(H):
        dd = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        cur, j = [], 0
        for s, e in zip(np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]):
            lid = None
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                r = find(prev[k][2])
                if lid is None:
                    lid = r
                elif r != lid:
                    parent[max(r, lid)] = min(r, lid)
                    lid = min(r, lid)
                k += 1
            if lid is None:
                lid = len(parent)
                parent.append(lid)
            cur.append((s, e, lid))
        rows.append(cur)
        prev = cur
    roots = {}
    lab = np.zeros((H, W), np.int32)
    for y, cur in enumerate(rows):
        for s, e, lid in cur:
            lab[y, s:e] = roots.setdefault(find(lid), len(roots) + 1)
    return lab, len(roots)


def _fill_holes(mask):
    lab, n = _label(~mask)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return mask | ((lab > 0) & ~np.isin(lab, list(edge)))


def _dilate(mask, r=1):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * r + 1))) > 127


def _blur(arr, sigma):
    out = np.asarray(arr, np.float32)
    if sigma <= 0:
        return out
    r = max(1, int(3 * sigma + 0.5))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for ax in (0, 1):
        p = np.pad(out, [(r, r) if i == ax else (0, 0) for i in range(out.ndim)], mode="edge")
        n = out.shape[ax]
        out = sum(k[i] * np.take(p, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1))
    return out.astype(np.float32)


def _to_lab(rgb):
    c = np.asarray(rgb, np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _kmeans(X, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(len(X), 20000), replace=False)] if len(X) > 20000 else X
    k = max(1, min(k, len(np.unique(S.round(1), axis=0))))
    C = [S[rng.integers(len(S))]]
    d2 = ((S - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        C.append(S[rng.choice(len(S), p=d2 / d2.sum())] if d2.sum() > 0 else S[rng.integers(len(S))])
        d2 = np.minimum(d2, ((S - C[-1]) ** 2).sum(1))
    C = np.array(C, np.float32)
    for _ in range(iters):
        lab = ((S[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = S[lab == j].mean(0)
    out = np.empty(len(X), np.int32)
    for i in range(0, len(X), 50000):
        out[i:i + 50000] = ((X[i:i + 50000, None] - C[None]) ** 2).sum(2).argmin(1)
    return C, out


# marching-squares edge pairs per case (corners tl=8 tr=4 br=2 bl=1; edges T R B L); saddles resolved by the centre
_MS = {1: ["LB"], 2: ["BR"], 3: ["LR"], 4: ["TR"], 6: ["TB"], 7: ["TL"], 8: ["TL"], 9: ["TB"], 11: ["TR"],
       12: ["LR"], 13: ["BR"], 14: ["LB"]}


def _contours(field, level=0.5):
    """Closed iso-contours of a 2-D field as (n, 2) float arrays in pixel-centre coordinates."""
    f = np.pad(np.asarray(field, np.float32), 1, constant_values=min(0.0, level - 1))
    b = f >= level
    case = b[:-1, :-1] * 8 + b[:-1, 1:] * 4 + b[1:, 1:] * 2 + b[1:, :-1] * 1
    ys, xs = np.nonzero((case > 0) & (case < 15))
    pts, adj = {}, {}

    def key(e, y, x):
        k = {"T": ("h", y, x), "B": ("h", y + 1, x), "L": ("v", y, x), "R": ("v", y, x + 1)}[e]
        if k not in pts:
            _, yy, xx = k
            a0 = f[yy, xx]
            a1 = f[yy, xx + 1] if k[0] == "h" else f[yy + 1, xx]
            t = 0.5 if a1 == a0 else min(1.0, max(0.0, (level - a0) / (a1 - a0)))
            pts[k] = (xx + t - 0.5, yy - 0.5) if k[0] == "h" else (xx - 0.5, yy + t - 0.5)
        return k
    for y, x, c in zip(ys.tolist(), xs.tolist(), case[ys, xs].tolist()):
        if c in (5, 10):
            centre = f[y:y + 2, x:x + 2].mean() >= level
            segs = ["TL", "BR"] if (c == 5) == centre else ["TR", "LB"]
        else:
            segs = _MS[c]
        for s in segs:
            k1, k2 = key(s[0], y, x), key(s[1], y, x)
            adj.setdefault(k1, []).append(k2)
            adj.setdefault(k2, []).append(k1)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = next((n for n in adj[cur] if n != prev and n not in seen), None)
            if nxt is None:
                break
            seen.add(nxt)
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.array([pts[k] for k in loop], np.float64))
    return loops


def _area(P):
    return 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


def _inside(p, P):
    x, y = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = (y > p[1]) != (y2 > p[1])
    xi = x + (p[1] - y) * (x2 - x) / np.where(y2 == y, 1e-12, y2 - y)
    return bool(np.count_nonzero(cross & (p[0] < xi)) % 2)


def _orient(loops):
    """Wind outer loops one way and holes the other so the nonzero fill rule (VD/Lottie default) is correct."""
    out = []
    for i, P in enumerate(loops):
        depth = sum(_inside(P[0], Q) for j, Q in enumerate(loops) if j != i and abs(_area(Q)) > abs(_area(P)))
        want_pos = depth % 2 == 0
        out.append(P if (_area(P) > 0) == want_pos else P[::-1].copy())
    return out


def _bez(b, t):
    mt = 1 - t
    return ((mt ** 3)[:, None] * b[0] + (3 * mt * mt * t)[:, None] * b[1] + (3 * mt * t * t)[:, None] * b[2]
            + (t ** 3)[:, None] * b[3])


def _gen_bezier(P, u, t1, t2):
    p0, p3 = P[0], P[-1]
    mt = 1 - u
    A1 = (3 * mt * mt * u)[:, None] * t1
    A2 = (3 * mt * u * u)[:, None] * t2
    c00, c01, c11 = (A1 * A1).sum(), (A1 * A2).sum(), (A2 * A2).sum()
    tmp = P - _bez(np.array([p0, p0, p3, p3]), u)
    x0, x1 = (A1 * tmp).sum(), (A2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = np.linalg.norm(p3 - p0)
    al, ar = ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det) if abs(det) > 1e-12 else (0, 0)
    if al < 1e-6 * seg or ar < 1e-6 * seg or al > 3 * seg or ar > 3 * seg:
        al = ar = seg / 3
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _fit_cubic(P, t1, t2, tol, depth=0):
    """Schneider's curve fitting: cubic Beziers through points P within `tol` px; t2 points back from the end."""
    n = len(P)
    seg = np.linalg.norm(P[-1] - P[0])
    if n <= 2 or depth > 14:
        return [np.array([P[0], P[0] + t1 * seg / 3, P[-1] + t2 * seg / 3, P[-1]])]
    u = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, n)
    b = _gen_bezier(P, u, t1, t2)
    for it in range(5):
        d = np.linalg.norm(_bez(b, u) - P, axis=1)
        split = int(d.argmax())
        if d[split] <= tol:
            return [b]
        if d[split] > 4 * tol or it == 4:
            break
        # Newton-Raphson reparameterisation
        q = _bez(b, u) - P
        d1 = 3 * ((1 - u) ** 2)[:, None] * (b[1] - b[0]) + (6 * (1 - u) * u)[:, None] * (b[2] - b[1]) + \
            (3 * u * u)[:, None] * (b[3] - b[2])
        d2 = (6 * (1 - u))[:, None] * (b[2] - 2 * b[1] + b[0]) + (6 * u)[:, None] * (b[3] - 2 * b[2] + b[1])
        den = (d1 * d1).sum(1) + (q * d2).sum(1)
        u = np.clip(u - np.where(np.abs(den) > 1e-12, (q * d1).sum(1) / np.where(den == 0, 1, den), 0), 0, 1)
        u[0], u[-1] = 0, 1
        u = np.maximum.accumulate(u)
        b = _gen_bezier(P, u, t1, t2)
    split = min(max(split, 1), n - 2)
    tc = P[split - 1] - P[split + 1]
    tc = tc / (np.linalg.norm(tc) or 1)
    return _fit_cubic(P[:split + 1], t1, tc, tol, depth + 1) + _fit_cubic(P[split:], -tc, t2, tol, depth + 1)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _fit_loop(P, tol=0.8, corner_deg=70, k=3):
    """Closed polyline -> list of cubic Beziers (4x2), keeping corners sharp and everything else smooth."""
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
    P = P[keep]
    n = len(P)
    if n < 4:
        return []
    k = min(k, max(1, n // 4))
    v1 = P - np.roll(P, k, axis=0)
    v2 = np.roll(P, -k, axis=0) - P
    cos = (v1 * v2).sum(1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    corners = [i for i in range(n) if ang[i] >= corner_deg and ang[i] == max(ang[(i + j) % n] for j in range(-k, k + 1))]
    if not corners:
        Q = np.vstack([P, P[:1]])
        t = _unit(P[k % n] - P[-k])
        return _fit_cubic(Q, t, -t, tol)
    out = []
    for ci, a in enumerate(corners):
        b_ = corners[(ci + 1) % len(corners)]
        idx = list(range(a, (b_ if b_ > a else b_ + n) + 1))
        seg = P[[i % n for i in idx]]
        if len(seg) < 2:
            continue
        kk = min(k, len(seg) - 1)
        out += _fit_cubic(seg, _unit(seg[kk] - seg[0]), _unit(seg[-1 - kk] - seg[-1]), tol)
    return out


def _tr_fmt(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else ("-0" if s == "-0.0" else s)


def _beziers_d(loops_bez, scale=1.0, dx=0.0, dy=0.0):
    """Beziers -> absolute M/L/C/Z path data (the command set build_lottie and svg_to_vd accept)."""
    out = []
    for bz in loops_bez:
        if not bz:
            continue
        P = lambda p: f"{_tr_fmt(p[0] * scale + dx)} {_tr_fmt(p[1] * scale + dy)}"
        s = [f"M{P(bz[0][0])}"]
        for b in bz:
            chord = b[3] - b[0]
            L = np.linalg.norm(chord)
            straight = L > 0 and all(abs(chord[0] * (c - b[0])[1] - chord[1] * (c - b[0])[0]) / L < 0.15
                                     for c in (b[1], b[2]))
            s.append(f"L{P(b[3])}" if straight else f"C{P(b[1])} {P(b[2])} {P(b[3])}")
        out.append(" ".join(s) + "Z")
    return " ".join(out)


def mask_to_path(mask, tol=0.8, corner_deg=70, sigma=0.7, min_area=4.0, field=None, dx=0.0, dy=0.0):
    """Binary mask (or a soft `field` in 0..1) -> smooth, correctly wound absolute path data, offset by (dx, dy)."""
    f = field if field is not None else _blur(mask.astype(np.float32), sigma)
    loops = [L for L in _contours(f, 0.5) if abs(_area(L)) >= min_area]
    return _beziers_d([_fit_loop(L, tol, corner_deg) for L in _orient(loops)], 1.0, dx, dy)


def _lmask(L, shape):
    """A traced layer's boolean mask on the full work canvas (stored cropped to its bbox: memory O(area), not O(canvas))."""
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = L["bbox"]
    m[y0:y1, x0:x1] = L["mask"]
    return m


def _lab_range(stops):
    """Largest colour distance (CIE76) between gradient stops: a gradient that spans < 6 is flat to the eye."""
    lab = _to_lab(np.array([c for _, c in stops], np.float32))
    return float(max(np.linalg.norm(lab[i] - lab[j]) for i in range(len(lab)) for j in range(i + 1, len(lab))))


def _profile_stops(s, C, s0, s1, nb=24, tol=3.0, max_stops=6):
    """1-D colour profile along s -> gradient stops [(offset, rgb)] simplified with RDP, plus per-pixel prediction."""
    span = max(s1 - s0, 1e-6)
    o = np.clip((s - s0) / span, 0, 1)
    bi = np.minimum((o * nb).astype(int), nb - 1)
    cnt = np.bincount(bi, minlength=nb)
    prof = np.stack([np.bincount(bi, C[:, c], nb) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    ok = cnt > 0
    xs = (np.arange(nb) + 0.5) / nb
    prof = np.stack([np.interp(xs, xs[ok], prof[ok, c]) for c in range(3)], 1)
    xs = np.r_[0, xs, 1]
    prof = np.vstack([prof[:1], prof, prof[-1:]])

    def rdp(i, j, t):
        if j <= i + 1:
            return []
        f = (xs[i + 1:j] - xs[i]) / (xs[j] - xs[i])
        interp = prof[i] + f[:, None] * (prof[j] - prof[i])
        e = np.linalg.norm(prof[i + 1:j] - interp, axis=1)
        m = int(e.argmax())
        return [] if e[m] <= t else rdp(i, i + 1 + m, t) + [i + 1 + m] + rdp(i + 1 + m, j, t)
    t = tol
    while True:
        keep = [0] + rdp(0, len(xs) - 1, t) + [len(xs) - 1]
        if len(keep) <= max_stops:
            break
        t *= 1.5
    stops = [(float(xs[i]), prof[i]) for i in keep]
    pred = np.stack([np.interp(o, [x for x, _ in stops], [c[ch] for _, c in stops]) for ch in range(3)], 1)
    return stops, pred


def fit_fill(P, C, kinds=("flat", "linear", "radial"), max_stops=6, tol=3.0):
    """Best SVG/VD-expressible fill for pixels at P (n,2) with colours C (n,3): flat, linear or circular radial."""
    rng = np.random.default_rng(0)
    if len(P) > 6000:
        idx = rng.choice(len(P), 6000, replace=False)
        P, C = P[idx], C[idx]
    C = C.astype(np.float64)
    mean = C.mean(0)
    rmse = lambda pred: float(np.sqrt(((pred - C) ** 2).sum(1).mean()))
    best = {"type": "flat", "color": mean, "rmse": rmse(mean[None])}
    if len(P) < 12 or best["rmse"] < 2.5:
        return best
    if "linear" in kinds:
        A = np.c_[np.ones(len(P)), P]
        coef = np.linalg.lstsq(A, C, rcond=None)[0]
        _, _, vt = np.linalg.svd(coef[1:].T)
        u = vt[0]
        s = P @ u
        s0, s1 = np.percentile(s, 1), np.percentile(s, 99)
        stops, pred = _profile_stops(s, C, s0, s1, tol=tol, max_stops=max_stops)
        r = rmse(pred)
        # a gradient must earn its API-24 dependency: a real colour span and a clear, not just relative, improvement
        if r < 0.8 * best["rmse"] and best["rmse"] - r > 2.0 and _lab_range(stops) > 6.0:
            best = {"type": "linear", "p0": u * s0, "p1": u * s1, "stops": stops, "rmse": r}
    if "radial" in kinds:
        lo, hi = P.min(0), P.max(0)
        ext = (hi - lo) * 0.5
        spanx, spany = (hi - lo + 2 * ext)
        c0 = lo - ext

        def ev(c):
            s = np.linalg.norm(P - c, axis=1)
            st, pred = _profile_stops(s, C, 0, np.percentile(s, 99), nb=16, tol=0.0, max_stops=18)
            return rmse(pred)
        grid = [c0 + np.array([spanx * i / 8, spany * j / 8]) for i in range(9) for j in range(9)]
        scores = [(ev(c), tuple(c)) for c in grid]
        r_best, c_best = min(scores)
        step = np.array([spanx, spany]) / 8
        for _ in range(3):
            step = step / 2
            cs = [np.array(c_best) + step * np.array([i, j]) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            r_best, c_best = min([(ev(c), tuple(c)) for c in cs] + [(r_best, c_best)])
        c = np.array(c_best)
        s = np.linalg.norm(P - c, axis=1)
        R = float(np.percentile(s, 99))
        stops, pred = _profile_stops(s, C, 0, R, tol=tol, max_stops=max_stops)
        r = rmse(pred)
        if r < 0.9 * best["rmse"] and best["rmse"] - r > 2.0 and R > 1 and _lab_range(stops) > 6.0:
            best = {"type": "radial", "c": c, "r": R, "stops": stops, "rmse": r}
    return best


def _tr_hex(c, alpha=None):
    c = np.clip(np.round(c), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c[:3])


def _fill_svg(fill, gid, scale=1.0):
    """fill spec -> (fill attribute, <defs> entry)."""
    if fill["type"] == "flat":
        return _tr_hex(fill["color"]), ""
    stops = "".join(f'<stop offset="{o:.3f}" stop-color="{_tr_hex(c)}"/>' for o, c in fill["stops"])
    if fill["type"] == "linear":
        (x1, y1), (x2, y2) = fill["p0"] * scale, fill["p1"] * scale
        return f"url(#{gid})", (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{_tr_fmt(x1)}" '
                                f'y1="{_tr_fmt(y1)}" x2="{_tr_fmt(x2)}" y2="{_tr_fmt(y2)}">{stops}</linearGradient>')
    cx, cy = fill["c"] * scale
    return f"url(#{gid})", (f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{_tr_fmt(cx)}" cy="{_tr_fmt(cy)}" '
                            f'r="{_tr_fmt(fill["r"] * scale)}">{stops}</radialGradient>')


def _quad_residual(xy, lab):
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[np.ones(len(x)), x, y, x * x, x * y, y * y]
    coef = np.linalg.lstsq(A, lab, rcond=None)[0]
    return ((A @ coef - lab) ** 2).sum(1)


def segment_regions(rgb, core, k=16, min_frac=0.0005, edge_tol=7.0, fit_tol=4.0, soft_tol=2.0):
    """Opaque core -> region map (-1 outside): k-means, components, speckle folding, smooth-gradient merging."""
    H, W = core.shape
    blur = np.stack([_blur(rgb[..., c], 0.8) for c in range(3)], -1)
    lab = _to_lab(blur)
    _, q = _kmeans(lab[core], k)
    Q = np.full((H, W), -1, np.int32)
    Q[core] = q
    R = np.full((H, W), -1, np.int32)
    n = 0
    for c in range(int(q.max()) + 1):
        lb, m = _label(Q == c)
        R[lb > 0] = lb[lb > 0] - 1 + n
        n += m
    if n <= 1:
        return R
    pairs = []
    for a_, b_, la, lb_ in ((R[:, :-1], R[:, 1:], lab[:, :-1], lab[:, 1:]), (R[:-1], R[1:], lab[:-1], lab[1:])):
        v = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
        lo, hi = np.minimum(a_[v], b_[v]), np.maximum(a_[v], b_[v])
        pairs.append((lo.astype(np.int64) * n + hi, np.linalg.norm(la[v] - lb_[v], axis=1)))
    keys = np.concatenate([p[0] for p in pairs])
    diffs = np.concatenate([p[1] for p in pairs])
    uk, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv)
    contrast = np.bincount(inv, diffs) / cnt
    edges = sorted(zip(contrast.tolist(), (uk // n).tolist(), (uk % n).tolist(), cnt.tolist()))
    flat = R.ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(n + 1))
    members = {i: order[starts[i]:starts[i + 1]] for i in range(n)}
    area = {i: len(v) for i, v in members.items()}
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if area[ra] < area[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        area[ra] += area[rb]
        members[ra] = np.concatenate([members[ra], members[rb]])
        return ra
    min_area = max(4, int(min_frac * core.sum()))
    shared = {}
    for c_, a, b, ct in edges:
        shared.setdefault(a, []).append((ct, b))
        shared.setdefault(b, []).append((ct, a))
    # boundary length per region (4-neighbour edges to any other label, the outside or the image border)
    perim = np.zeros(n)
    for a_, b_ in ((R[:, :-1], R[:, 1:]), (R[:-1], R[1:])):
        v = a_ != b_
        perim += np.bincount(a_[v & (a_ >= 0)], minlength=n) + np.bincount(b_[v & (b_ >= 0)], minlength=n)
    border = np.concatenate([R[0], R[-1], R[:, 0], R[:, -1]])
    perim += np.bincount(border[border >= 0], minlength=n)
    for i in sorted(range(n), key=lambda i: area[i]):
        # speckle (too small) and slivers (a 1-3 px resampling seam: area / boundary < 1.5 px) join their best neighbour
        if (area[find(i)] < min_area or len(members[i]) / max(1.0, perim[i]) < 1.5) and shared.get(i):
            best = max(shared[i], key=lambda t: t[0] if find(t[1]) != find(i) else -1)
            if find(best[1]) != find(i):
                union(i, best[1])
    rng = np.random.default_rng(0)
    yy, xx = np.divmod(np.arange(H * W), W)
    xy_all = np.c_[xx / W, yy / H]
    lab_flat = lab.reshape(-1, 3)
    for c_, a, b, ct in edges:
        if c_ > edge_tol:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        small, big = sorted((members[ra], members[rb]), key=len)
        ia = rng.choice(big, min(len(big), 3000), replace=False)
        ib = rng.choice(small, min(len(small), 2000), replace=False)
        res = _quad_residual(xy_all[np.r_[ia, ib]], lab_flat[np.r_[ia, ib]])
        # an invisible source boundary is worse as a seam than as a looser fit, unless the smaller side is a
        # distinct feature (a highlight) that the joint fit would erase
        if (np.sqrt(res.mean()) <= fit_tol * (2.0 if c_ < soft_tol else 1.0)
                and np.sqrt(res[len(ia):].mean()) <= 2 * fit_tol):
            union(ra, rb)
    roots = {}
    out = np.full(H * W, -1, np.int32)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
            out[members[r]] = roots[r]
    return out.reshape(H, W)


def _alpha_bands(alpha, rgb, core, n, tol=1.2, fill_kinds=("flat",), gamma=1.0):
    """Soft alpha (glow, shadow, feathering) as nested iso-alpha bands whose stacked opacities rebuild the falloff."""
    soft = (alpha > 0.02) & ~core
    if n <= 0 or soft.sum() < 0.03 * max(1, core.sum()):
        return []
    # the soft ring must be wider than resampling ringing (mean width over the core's boundary > 2 px)
    edge = core & ~(np.asarray(Image.fromarray(core.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 127)
    if core.any() and soft.sum() / max(1, int(edge.sum())) <= 2.0:
        return []
    # smooth contours, but never spill past a crisp edge (a halo there reads as a dark outline)
    a_s = np.minimum(_blur(alpha, 1.2), np.asarray(Image.fromarray(alpha.astype(np.float32), "F").filter(ImageFilter.MaxFilter(3))))
    thr = [round(0.5 * ((i + 1) / (n + 1)) ** gamma, 3) for i in range(n)]
    layers, prev_a, prev_c = [], 0.0, np.zeros(3)
    for i, t in enumerate(thr):
        inner = thr[i + 1] if i + 1 < len(thr) else 0.5
        region = a_s >= t
        ring = region & (a_s < inner) & ~core
        if ring.sum() < 12:
            continue
        a_i = float(alpha[ring].mean())
        w = alpha[ring][:, None]
        c_i = (rgb[ring] * w).sum(0) / max(float(w.sum()), 1e-6)
        o = float(np.clip(1 - (1 - a_i) / (1 - prev_a) if a_i > prev_a else 0.01, 0.01, 1))
        if o < 0.03:
            continue  # invisible band: only adds a VD path
        c_draw = np.clip((a_i * c_i - (1 - o) * prev_a * prev_c) / o, 0, 255)
        fill = {"type": "flat", "color": c_draw, "rmse": 0.0}
        if "linear" in fill_kinds or "radial" in fill_kinds:
            ys, xs = np.nonzero(ring)
            fill = fit_fill(np.c_[xs + 0.5, ys + 0.5], rgb[ring], fill_kinds, 3, 6.0)
            if fill["type"] != "flat":
                shift = c_draw - c_i
                fill["stops"] = [(o_, np.clip(c + shift, 0, 255)) for o_, c in fill["stops"]]
            else:
                fill["color"] = c_draw
        d = mask_to_path(region, tol=max(2.0, tol), sigma=1.5, min_area=12)
        if d:
            ys, xs = np.nonzero(region)
            layers.append({"name": f"glow_{len(layers) + 1}", "role": "glow", "d": d, "fill": fill, "opacity": round(o, 3),
                           "area": int(region.sum()), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                           "color": _tr_hex(c_i), "mask": region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]})
        prev_a, prev_c = a_i, c_i
    return layers


TRACE_PRESETS = {
    "flat": dict(work=512, k=8, edge_tol=8.0, fit_tol=3.0, min_frac=0.001, kinds=("flat", "linear"), stops=3, tol=0.8,
                 corner=60, bands=4, band_gamma=1.6, stop_tol=4.0),
    "illustration": dict(work=512, k=16, edge_tol=7.0, fit_tol=4.0, min_frac=0.0006, kinds=("flat", "linear", "radial"),
                         stops=6, tol=0.8, corner=75, bands=8, band_gamma=1.6, stop_tol=2.5),
    "gradient": dict(work=512, k=16, edge_tol=7.0, fit_tol=4.0, soft_tol=3.0, min_frac=0.0006,
                     kinds=("flat", "linear", "radial"), stops=8, tol=0.8, corner=75, bands=12, band_gamma=1.6, stop_tol=1.5),
}


def _vtracer():
    try:
        import vtracer
        return vtracer
    except ImportError:
        return None


def _bake_translate(d, dx, dy):
    """Fold a translate() into absolute M/L/C path data (VectorDrawable has no SVG transforms)."""
    out, k = [], 0
    for t in _TOK.findall(d):
        if t.isalpha():
            if t not in "MLCZ":
                raise ValueError(f"unsupported path command {t}")
            out.append(t)
            continue
        out.append(_tr_fmt(float(t) + (dx if k % 2 == 0 else dy)))
        k += 1
    return " ".join(out).replace("Z ", "Z")


VT_PATH = re.compile(r'<path d="([^"]*)" fill="(#[0-9A-Fa-f]{6})"(?: transform="translate\(([-\d.]+),([-\d.]+)\)")?')


def trace_vtracer(img, work=512, bands=8, band_gamma=1.6, **kw):
    """vtracer colour mode on the opaque core (soft alpha stripped first, then rebuilt as glow bands)."""
    s = min(1.0, work / max(img.size))
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    arr = np.asarray(img.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255
    core = alpha >= 0.5
    flat = np.zeros((h, w, 4), np.uint8)
    flat[core, :3] = rgb[core].astype(np.uint8)
    flat[core, 3] = 255
    opts = dict(colormode="color", hierarchical="stacked", mode="spline", filter_speckle=4, color_precision=6,
                layer_difference=16, corner_threshold=60, length_threshold=4.0, splice_threshold=45, path_precision=2)
    opts.update(kw)
    svg = _vtracer().convert_raw_image_to_svg(_png_bytes(Image.fromarray(flat)), img_format="png", **opts)
    layers = _alpha_bands(alpha, rgb, core, bands, 1.2, ("flat",), band_gamma)
    for m_ in VT_PATH.finditer(svg):
        d = _bake_translate(m_.group(1), float(m_.group(3) or 0), float(m_.group(4) or 0))
        layers.append({"name": f"region_{len(layers) + 1}", "role": "region", "d": d, "opacity": 1.0,
                       "fill": {"type": "flat", "color": np.array(_hex_rgb(m_.group(2))) * 255, "rmse": 0.0},
                       "color": m_.group(2).lower(), "area": 0, "bbox": [0, 0, w, h]})
    if core.any() and not any(L["role"] == "region" for L in layers):
        print("  ! vtracer returned no <path d fill> the kit recognises (a vtracer version with another attribute order?): "
              "the vtracer candidate has no regions")
    return {"preset": "vtracer", "params": {"work": work, "bands": bands, **opts}, "size": (w, h), "src_size": img.size,
            "layers": layers}


def trace_image(img, preset="illustration", max_regions=400, **over):
    """RGBA image -> layered vector: glow bands (bottom) + gradient-filled regions. Returns layers + svg text.
    Every per-region step runs on the region's bbox (+ 4 px), so cost is O(total area), not O(regions x canvas);
    more than `max_regions` regions is photographic and returns {"skipped": ...} with no layers."""
    P = dict(TRACE_PRESETS[preset])
    P.update({k: v for k, v in over.items() if v is not None})
    s = min(1.0, P["work"] / max(img.size))  # never upscale: that traces the source's own pixel staircase
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    arr = np.asarray(img.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] / 255
    core = alpha >= 0.5
    out = {"preset": preset, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
           "size": (w, h), "src_size": img.size, "layers": []}
    layers = _alpha_bands(alpha, rgb, core, P["bands"], P["tol"] * 1.5, P.get("band_kinds", ("flat",)),
                          P.get("band_gamma", 1.0))
    R = segment_regions(rgb, core, P["k"], P["min_frac"], P["edge_tol"], P["fit_tol"], P.get("soft_tol", 2.0)) \
        if core.any() else np.full(core.shape, -1, np.int32)
    n = int(R.max()) + 1
    if n > max_regions:
        out["skipped"] = f"photographic: {n} colour regions at {w}x{h} (> {max_regions}); keep the raster (export)"
        return out
    flat = R.ravel()
    order_ = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order_], np.arange(n + 1))
    win, crops = {}, {}
    for r in range(n):
        idx = order_[starts[r]:starts[r + 1]]
        if len(idx) < 4:
            continue
        ys, xs = np.divmod(idx, w)
        x0, y0, x1, y1 = max(0, xs.min() - 4), max(0, ys.min() - 4), min(w, xs.max() + 5), min(h, ys.max() + 5)
        win[r] = (x0, y0, x1, y1)
        m = R[y0:y1, x0:x1] == r
        crops[r] = (m, (_fill_holes(m) & core[y0:y1, x0:x1]) | m, len(idx))
    order = sorted(win, key=lambda r: -int(crops[r][1].sum()))
    later = np.zeros_like(core)
    draw = {}
    for r in reversed(order):
        x0, y0, x1, y1 = win[r]
        m, filled, _ = crops[r]
        draw[r] = filled | (_dilate(m, 2) & later[y0:y1, x0:x1])
        later[y0:y1, x0:x1] |= m
    for r in order:
        x0, y0, x1, y1 = win[r]
        m = crops[r][0]
        ys, xs = np.nonzero(m)
        px = rgb[y0:y1, x0:x1][ys, xs]
        fill = fit_fill(np.c_[xs + x0 + 0.5, ys + y0 + 0.5], px, P["kinds"], P["stops"], P["stop_tol"])
        d = mask_to_path(draw[r], P["tol"], P["corner"], 0.7, 4.0, dx=x0, dy=y0)
        if d:
            bx0, by0, bx1, by1 = int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0 + 1, int(ys.max()) + y0 + 1
            layers.append({"name": f"region_{len(layers) + 1}", "role": "region", "d": d, "fill": fill, "opacity": 1.0,
                           "area": int(len(xs)), "bbox": [bx0, by0, bx1, by1], "color": _tr_hex(px.mean(0)),
                           "mask": m[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]})
    out["layers"] = layers
    return out


def _tr_gid(name):
    """Gradient id prefix per asset: two traced SVGs inlined in one HTML page must not share ids."""
    return (re.sub(r"[^A-Za-z0-9_-]", "_", str(name)) + "_") if name else ""


def layers_svg(layers, vb, size=None, bg=None, name=None):
    """Traced layers -> SVG text (absolute M/L/C/Z paths, userSpaceOnUse gradients, no transforms)."""
    w, h = vb
    W, H = size or vb
    defs, body = [], []
    for i, L in enumerate(layers):
        fill, gd = _fill_svg(L["fill"], f"{_tr_gid(name)}g{i}")
        defs.append(gd)
        op = f' opacity="{L["opacity"]:g}"' if L.get("opacity", 1) < 1 else ""
        body.append(f'<path d="{L["d"]}" fill="{fill}"{op}/>')
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {w} {h}">'
            f'<defs>{"".join(defs)}</defs>{rect}' + "\n".join(body) + "</svg>")


def layers_vec(res, asset=None):
    """A trace as a build_lottie/animate-ready layers.json: static layers 'glow' (bands) and 'art' (regions)."""
    defs, static = [], []
    for role, name in (("glow", "glow"), ("region", "art")):
        paths = []
        for L in (x for x in res["layers"] if x["role"] == role):
            f, gd = _fill_svg(L["fill"], f"{_tr_gid(asset)}{name}_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": L.get("opacity", 1)})
        if paths:
            static.append({"name": name, "paths": paths})
    w, h = res["size"]
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static]}


def _tr_path_error(layers):
    """One line naming the first layer whose pathData Android could not inflate (nan/inf or unparsable), else None."""
    for L in layers:
        d = L.get("d") or ""
        if re.search(r"nan|inf", d, re.I):
            return f"{L.get('name', '?')}: pathData contains nan/inf"
        try:
            _subpaths(d)
        except (ValueError, IndexError, TypeError) as e:
            return f"{L.get('name', '?')}: {e}"
    return None


def _tr_aapt2(vd_text, res_name):
    """aapt2 verdict for one VectorDrawable, written under its final resource name (contract C4 wording):
    {"aapt2": message, "aapt2_ok": True | False | None, "aapt2_level": "link" | "compile" | None}."""
    import tempfile
    level = aapt2_level()
    if level is None:
        return {"aapt2": _aapt2_line(), "aapt2_ok": None, "aapt2_level": None}
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / f"{res_name}.xml"
        f.write_text(vd_text)
        errs = aapt2_check([str(f)])
    if errs:
        return {"aapt2": "aapt2 error: " + "; ".join(f"{k}: {v}" if k != "*" else v for k, v in errs.items())[:300],
                "aapt2_ok": False, "aapt2_level": level}
    return {"aapt2": _aapt2_line(), "aapt2_ok": True, "aapt2_level": level}


def vector_advice(m, layers, dp=None):
    """Vector or raster? Fidelity bar plus Android guidance (VDs <= 200 dp, lint VectorPath > 800 chars, kit size heuristic)."""
    lens = [len(re.sub(r"\s+", " ", L["d"])) for L in layers]
    grads = sum(L["fill"]["type"] != "flat" for L in layers)
    heavy = (f"{len(layers)} paths / {sum(lens)} pathData chars (kit heuristic: > 64 paths or > 40000 chars is a heavy "
             f"first draw; prefer WebP for that slot)") if len(layers) > 64 or sum(lens) > 40000 else None
    if not m:
        # no render, no fidelity: say so instead of guessing (only the size hint can be given from the paths alone)
        return {"fidelity": "unknown", "vector_ok": None, "vd_ok": None, "prefer": "unknown", "paths": len(layers),
                "pathdata_chars": sum(lens), "max_path_chars": max(lens, default=0), "gradients": grads,
                "reasons": [f"fidelity not measured: {_tr_chrome_reason()}"] + ([heavy] if heavy else [])}
    disp = m.get("display", m)
    fid = ("faithful" if disp["diff"] <= 2 else "close" if disp["diff"] <= 3.5 else
           "stylised" if disp["diff"] <= 5 else "lossy")
    if disp["iou"] < 0.95:
        fid = "lossy"
    reasons = [f"display-size diff {disp['diff']} / silhouette IoU {disp['iou']} -> {fid}"]
    vec_ok = fid in ("faithful", "close")
    vd_ok, why_not_vd = vec_ok, (None if vec_ok else reasons[0])
    if dp and max(dp) > 200:
        vd_ok = False
        reasons.append(f"{max(dp):g}dp > 200dp: Android recommends raster above 200x200dp (draw time)")
        why_not_vd = why_not_vd or reasons[-1]
    if heavy:
        vd_ok = False
        reasons.append(heavy)
        why_not_vd = why_not_vd or heavy
    long_ = sum(n > 800 for n in lens)
    if long_:
        reasons.append(f"{long_} path(s) over 800 chars: lint VectorPath warns (MAX_PATH_DATA_LENGTH 800; it is documented as "
                       f"a draw-time hint, not an error)")
    if grads:
        reasons.append(f"{grads} gradient fill(s): VectorDrawable gradients need API 24+ or AndroidX VectorDrawableCompat")
    if fid == "stylised":
        reasons.append("visibly simplified vs the raster; acceptable only if a flatter look is fine")
    prefer = "vd" if vd_ok else ("svg/lottie" if vec_ok else "png/webp")
    return {"fidelity": fid, "vector_ok": vec_ok, "vd_ok": vd_ok, "prefer": prefer, "paths": len(layers),
            "pathdata_chars": sum(lens), "max_path_chars": max(lens, default=0), "gradients": grads, "reasons": reasons,
            "why_not_vd": why_not_vd}


def _asset_dp(png):
    """dp size of a forged asset, looked up from the build dir's assets.json when the PNG lives in <dir>/assets/."""
    p = Path(png).resolve()
    mf = p.parent.parent / "assets.json"
    if p.parent.name != "assets" or not mf.exists():
        return None
    try:
        a = next((x for x in json.loads(mf.read_text())["assets"] if x.get("name") == p.stem), None)
        dp = a.get("dp") if a else None
        if not dp:
            return None
        aspect = Image.open(p).height / max(1, Image.open(p).width)
        if isinstance(dp, dict):  # {"w": 48, "h": 48}; h optional
            w = float(dp["w"])
            return w, float(dp["h"]) if dp.get("h") else w * aspect
        if isinstance(dp, (list, tuple)):  # [w, h] or [w]
            w = float(dp[0])
            return w, float(dp[1]) if len(dp) > 1 else w * aspect
        return float(dp), float(dp) * aspect  # a bare number is the width
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
        print(f"  ! {mf}: dp of {p.stem!r} unreadable ({e}); pass --dp")
        return None


def _diff_heat(a, b):
    comp = lambda x: np.asarray(x, np.float32)[..., :3] * (np.asarray(x, np.float32)[..., 3:] / 255)
    d = np.abs(comp(a) - comp(b.resize(a.size))).mean(2)
    v = np.clip(d * 4, 0, 255).astype(np.uint8)
    return Image.merge("RGB", (Image.fromarray(v), Image.fromarray(v // 3), Image.fromarray(255 - v)))


def _on(img, bg):
    base = Image.new("RGBA", img.size, bg)
    base.alpha_composite(img)
    return base


def trace_asset(src, presets, dp=None, display=None, name=None, **over):
    """Trace with each preset, render all with Chrome, score, and pick the smallest near-best candidate."""
    cands = []
    for pr in presets:
        t = time.time()
        res = trace_vtracer(src, **{k: v for k, v in over.items() if k in ("work", "bands") and v is not None}) \
            if pr == "vtracer" else trace_image(src, pr, **over)
        svg = layers_svg(res["layers"], res["size"], src.size, name=name)
        cands.append({"preset": pr, "res": res, "svg": svg, "secs": round(time.time() - t, 2), "bytes": len(svg)})
        if res.get("skipped"):
            cands[-1]["render"] = None  # photographic under one preset is photographic under all: stop here
            break
        if len(cands) == 1 and len(presets) > 1:
            # without a renderer the presets cannot be compared: check on the first one and do not trace the rest
            first = _tr_render_svgs([svg], [src.size])
            if first is None:
                break
            cands[0]["render"] = first[0]
    # display-size fidelity: 3x the dp width, else at least 192 px on the long side and 96 px on the short side
    display = display or (round(max(dp) * 3) if dp else max(192, round(96 * max(src.size) / max(1, min(src.size)))))
    todo = [c for c in cands if "render" not in c]
    rens = _tr_render_svgs([c["svg"] for c in todo], [src.size] * len(todo)) if todo else []
    for c, r in zip(todo, rens or [None] * len(todo)):
        c["render"] = r
    for c in cands:
        c["metrics"] = trace_metrics(src, c["render"], display) if c["render"] is not None else None
    scored = [c for c in cands if c["metrics"]]
    if scored:
        best = min(c["metrics"]["display"]["diff"] for c in scored)
        near = [c for c in scored if c["metrics"]["display"]["diff"] <= best * 1.1 + 0.15]
        chosen = min(near, key=lambda c: c["bytes"])
    else:
        live = [c for c in cands if not c["res"].get("skipped")] or cands
        chosen = next((c for c in live if c["preset"] == "illustration"), live[0])
    return chosen, cands


def cmd_trace(a):
    out = Path(a.out)
    if a.name and len(a.files) > 1:
        sys.exit("trace: --name needs a single input file (the second file would overwrite the first)")
    if a.preset == "auto":
        presets = ["flat", "illustration", "gradient"] + (["vtracer"] if _vtracer() else [])
    else:
        presets = [a.preset]
        if a.preset == "vtracer" and not _vtracer():
            sys.exit("trace: vtracer not installed: pip install vtracer (or use --preset flat|illustration|gradient)")
    if a.preset == "vtracer" and (a.colors is not None or a.tol is not None):
        print("  ! --preset vtracer ignores --colors and --tol (vtracer has its own colour/curve settings)")
    if not _chrome():
        print("  (no Chrome/Chromium found: fidelity is not measured; set GPTKIT_CHROME to enable it)")
    (out / "review").mkdir(parents=True, exist_ok=True)
    failed = 0
    for f in a.files:
        try:
            if not Path(f).is_file():
                raise OSError("no such file")
            src = _rgba(f)
        except (OSError, ValueError) as e:  # UnidentifiedImageError is an OSError
            print(f"  ! {f}: cannot read image: {e}")
            failed += 1
            continue
        if not (np.asarray(src.getchannel("A")) >= 128).any():
            print(f"  ! {f}: nothing opaque to trace (fully transparent); skipped")
            failed += 1
            continue
        name = a.name or Path(f).stem
        vd_name = _res_name(a.prefix + name)
        if vd_name != a.prefix + name:
            print(f"  ({a.prefix + name!r} -> {vd_name}: Android resource names are [a-z0-9_] starting with a letter)")
        dp = (a.dp, a.dp * src.height / src.width) if a.dp else _asset_dp(f)
        chosen, cands = trace_asset(src, presets, dp, a.display, name, work=a.work, k=a.colors, bands=a.bands, tol=a.tol)
        res = chosen["res"]
        report = {"source": str(f), "src_size": src.size, "dp": dp, "chosen": chosen["preset"],
                  "candidates": [{k: c[k] for k in ("preset", "bytes", "secs", "metrics")} | {"paths": len(c["res"]["layers"])}
                                 for c in cands]}
        for c in cands:
            if c["res"].get("skipped"):
                report["candidates"][cands.index(c)]["skipped"] = c["res"]["skipped"]
        if res.get("skipped"):
            report["advice"] = {"fidelity": "not traced", "vector_ok": False, "vd_ok": False, "prefer": "png/webp",
                                "reasons": [res["skipped"]]}
            (out / f"{name}.trace.json").write_text(json.dumps(report, indent=2, default=str))
            print(f"{name}: not traced ({res['skipped']}) -> {out / f'{name}.trace.json'}")
            print("  advice: prefer png/webp (photographic input; export writes the density set)")
            continue
        svg_path = out / f"{name}.svg"
        svg_path.write_text(chosen["svg"])
        m = chosen["metrics"]
        report["advice"] = vector_advice(m, res["layers"], dp)
        adv = report["advice"]
        bad = _tr_path_error(res["layers"])
        if bad:
            report["vd"] = {"error": f"not written: {bad}"}
        else:
            try:
                vd_w = round(dp[0], 1) if dp else round(48 * src.width / max(src.size), 1)
                vd = svg_to_vd(chosen["svg"], vd_w)
                (out / f"{vd_name}.xml").write_text(vd)
                report["vd"] = {"file": str(out / f"{vd_name}.xml"), "res_name": vd_name, "bytes": len(vd), "width_dp": vd_w,
                                "width_from": "dp" if dp else "default 48dp long side (pass --dp)", **_tr_aapt2(vd, vd_name),
                                "lint": _vd_lint(vd)}
                for e in report["vd"]["lint"]:
                    print(f"  ! {vd_name}.xml lint: {e}")
                vr = _tr_render_svgs([vd_to_svg(vd)], [src.size])
                if vr and chosen.get("render") is not None:
                    report["vd"]["vs_svg_diff"] = trace_metrics(chosen["render"], vr[0])["diff"]
            except ValueError as e:
                report["vd"] = {"error": str(e)}
        (out / f"{name}.layers.json").write_text(json.dumps(layers_vec(res, name), indent=1))
        (out / f"{name}.trace.json").write_text(
            json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        rows = []
        if chosen.get("render") is not None:
            r = chosen["render"]
            rows.append([("source", _on(src, a.bg)), (f"{chosen['preset']} vector", _on(r, a.bg)), ("source", src),
                         ("vector", r), ("|diff| x4", _diff_heat(src, r))])
            rows.append([(f"{c['preset']} d={c['metrics']['display']['diff']} {c['bytes'] // 1024}KB p={len(c['res']['layers'])}",
                          _on(c["render"], a.bg)) for c in cands if c.get("render") is not None])
            sheet(rows, out / "review" / f"{name}_trace.png", T=300)
        print(f"{name}: {chosen['preset']} -> {svg_path} ({chosen['bytes'] // 1024} KB, {len(res['layers'])} paths)"
              + (f"  display diff={m['display']['diff']} iou={m['display']['iou']}" if m else ""))
        for c in cands:
            if c["res"].get("skipped"):
                print(f"    {c['preset']:13s} skipped: {c['res']['skipped']}")
            elif c["metrics"] and len(cands) > 1:
                print(f"    {c['preset']:13s} d={c['metrics']['display']['diff']:5.2f} iou={c['metrics']['display']['iou']:.3f} "
                      f"{c['bytes'] // 1024:4d} KB {len(c['res']['layers']):3d} paths")
        if m:
            print(f"  advice: prefer {adv['prefer']} ({adv['fidelity']})" + "".join(f"; {r}" for r in adv["reasons"][1:]))
        else:
            print(f"  advice: prefer unknown ({'; '.join(adv['reasons'])})")
        v = report["vd"]
        if "file" in v:
            note = f"  (not recommended: {adv['why_not_vd']})" if adv.get("why_not_vd") else ""
            print(f"  vd: {v['file']}  {v['aapt2']}" + (f"  VD->SVG round-trip diff={v['vs_svg_diff']}" if "vs_svg_diff" in v else "")
                  + note)
        else:
            print(f"  ! vd: {v['error']}")
    if failed:
        sys.exit(f"trace: {failed} of {len(a.files)} input(s) not traced (see the lines above)")


def map_path(d, fn):
    """Apply fn(x, y) -> (x', y') to every point of absolute M/L/C/Z path data; the command sequence never changes."""
    toks = _TOK.findall(d)
    out, nums = [], []
    for t in toks + ["Z"]:
        if t.isalpha():
            for i in range(0, len(nums) - 1, 2):
                x, y = fn(nums[i], nums[i + 1])
                out.append(f"{_tr_fmt(x)} {_tr_fmt(y)}")
            nums = []
            out.append(t)
        else:
            nums.append(float(t))
    return " ".join(out[:-1]).replace(" Z", "Z")


def _path_points(d):
    v = [float(t) for t in _TOK.findall(d) if not t.isalpha()]
    return np.array(v[:len(v) // 2 * 2]).reshape(-1, 2)


TR_MOTIONS = {  # default amplitude per motion (fraction of the part's width/height)
    "sway": 0.06, "flicker": 0.05, "breathe": 0.04, "squash": 0.08, "bob": 0.04, "wave": 0.04, "ripple": 0.06,
}


def motion_field(kind, box, t, amp=1.0, phase=0.0, seed=0):
    """Deterministic loop-periodic deformation at time t in [0, 1); t=0 is the rest pose (the approved design)."""
    ft, f0 = _field(kind, box, t, amp, phase, seed), _field(kind, box, 0.0, amp, phase, seed)

    def f(x, y):
        (xt, yt), (x0, y0) = ft(x, y), f0(x, y)
        return x + xt - x0, y + yt - y0
    return f


def _field(kind, box, t, amp, phase, seed):
    if kind not in TR_MOTIONS:
        raise ValueError(f"unknown motion {kind!r} (choose {', '.join(TR_MOTIONS)})")
    x0, y0, x1, y1 = box
    # a flat box (a horizontal stroke, a thin bar) still gets a usable height/width: 10 % of the other side, >= 1 unit
    W, H = max(x1 - x0, (y1 - y0) * 0.1, 1.0), max(y1 - y0, (x1 - x0) * 0.1, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    A = TR_MOTIONS[kind] * amp
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 6)
    tau = 2 * np.pi * (t + phase)

    def h(y):  # 0 at the base (bottom), 1 at the top
        return min(1.0, max(0.0, (y1 - y) / H))
    if kind == "sway":
        return lambda x, y: (x + A * W * np.sin(tau) * h(y) ** 1.5, y)
    if kind == "flicker":
        def f(x, y):
            k = h(y)
            lat = 0.6 * np.sin(tau + ph[0]) + 0.4 * np.sin(2 * tau + 3.0 * k + ph[1]) + 0.25 * np.sin(3 * tau + 5.0 * k + ph[2])
            st = 0.5 * np.sin(2 * tau + ph[3]) + 0.35 * np.sin(3 * tau + ph[4])
            tip = k ** 1.4
            return (x + A * W * lat * tip + 0.35 * A * (x - cx) * np.sin(2 * tau + ph[5]) * k,
                    y1 - (y1 - y) * (1 + 1.2 * A * st * tip))
        return f
    if kind == "breathe":
        s = 1 + A * np.sin(tau)
        return lambda x, y: (cx + (x - cx) * s, cy + (y - cy) * s)
    if kind == "squash":
        sy = 1 + A * np.sin(tau)
        sx = 1 / np.sqrt(sy)
        return lambda x, y: (cx + (x - cx) * sx, y1 - (y1 - y) * sy)
    if kind == "bob":
        return lambda x, y: (x, y - A * H * (0.5 - 0.5 * np.cos(tau)))
    if kind == "wave":
        return lambda x, y: (x + A * W * np.sin(tau - 2 * np.pi * (y - y0) / H), y)
    if kind == "ripple":
        s = 1 + A * (0.5 - 0.5 * np.cos(tau))
        return lambda x, y: (cx + (x - cx) * s, cy + (y - cy) * (1 + (s - 1) * 0.6))
    raise ValueError(f"unknown motion {kind!r} (choose {', '.join(TR_MOTIONS)})")


def procedural_keyframes(paths, kind, n=8, amp=1.0, phase=0.0, seed=0, box=None):
    """Paths -> n morph-compatible keyframes each, all deformed by one shared field so a part stays registered."""
    paths = [_d(_subpaths(d)) for d in paths]  # relative / H V S Q A commands -> absolute M/L/C/Z first
    if box is None:
        pts = np.vstack([_path_points(d) for d in paths])
        box = (*pts.min(0), *pts.max(0))
    return [[map_path(d, motion_field(kind, box, j / n, amp, phase, seed)) for j in range(n)] for d in paths]


def _grad_geom(defs, gid):
    """(kind, [(x, y), (x, y)]) of gradient #gid in the defs text: linear -> start/end, radial -> centre and a point at
    the radius. None for gradients the mover does not handle (gradientTransform, objectBoundingBox units)."""
    m = re.search(rf'<(linear|radial)Gradient\b[^>]*\bid="{re.escape(gid)}"[^>]*>', defs or "")
    if not m or "gradientTransform" in m.group(0) or 'gradientUnits="userSpaceOnUse"' not in m.group(0):
        return None
    at = dict(re.findall(r'([\w:-]+)="([^"]*)"', m.group(0)))
    try:
        if m.group(1) == "linear":
            return "linear", [(float(at["x1"]), float(at["y1"])), (float(at["x2"]), float(at["y2"]))]
        cx, cy, r = float(at["cx"]), float(at["cy"]), float(at["r"])
        return "radial", [(cx, cy), (cx + r, cy)]
    except (KeyError, ValueError):
        return None


def _grad_patch(defs, gid, kind, pts):
    """defs text with gradient #gid moved to pts (the shapes _grad_geom returns)."""
    (x1, y1), (x2, y2) = pts
    new = {"x1": x1, "y1": y1, "x2": x2, "y2": y2} if kind == "linear" else {"cx": x1, "cy": y1, "r": math.hypot(x2 - x1, y2 - y1)}

    def sub(m):
        tag = m.group(0)
        for k, v in new.items():
            tag = re.sub(rf'\b{k}="[^"]*"', f'{k}="{_tr_fmt(v)}"', tag)
        return tag
    return re.sub(rf'<(linear|radial)Gradient\b[^>]*\bid="{re.escape(gid)}"[^>]*>', sub, defs, count=1)


def animate_layers(vec, specs, n=8):
    """Static layers -> animated ones. specs: {layer: (motion, amp, phase[, anchor layer sharing its field])}."""
    vec = json.loads(json.dumps(vec))
    specs = dict(specs)
    names_ = {l["name"] for l in vec["static_layers"]}
    for name, spec in list(specs.items()):
        if f"glow_{name}" in names_ and f"glow_{name}" not in specs:
            specs[f"glow_{name}"] = (*spec[:3], name)
    order = vec.get("layer_order") or [l["name"] for l in vec["static_layers"]]
    boxes = dict((vec.get("rig") or {}).get("boxes", {}))
    for l in vec["static_layers"]:
        if l["name"] not in boxes:
            pts = np.vstack([_path_points(_d(_subpaths(p["d"]))) for p in l["paths"]])
            boxes[l["name"]] = [float(v) for v in (*pts.min(0), *pts.max(0))]
    for name, spec in specs.items():
        kind, amp, phase = spec[:3]
        anchor = (spec[3] if len(spec) > 3 else None) or name
        sl = next((l for l in vec["static_layers"] if l["name"] == name), None)
        if sl is None:
            raise ValueError(f"no static layer {name!r} (have: {', '.join(l['name'] for l in vec['static_layers'])})")
        if anchor not in boxes:
            raise ValueError(f"unknown anchor layer {anchor!r}")
        keys = procedural_keyframes([p["d"] for p in sl["paths"]], kind, n, amp, phase,
                                    seed=sum(map(ord, anchor)), box=boxes[anchor])
        names = [name if len(sl["paths"]) == 1 else f"{name}__{i + 1}" for i in range(len(sl["paths"]))]
        paint = ("fill", "opacity", "stroke", "stroke_width", "cap", "join")  # every paint key survives the morph
        for nm, p, k in zip(names, sl["paths"], keys):
            al = {"name": nm, "fill": p.get("fill", "none"), "opacity": p.get("opacity", 1)}
            al.update({key: p[key] for key in paint if key in p})
            if sl.get("motion"):
                al["motion"] = sl["motion"]  # the layer's rigid motion (vectorize schema) rides on top of the morph
            al["keyframes"] = k
            ref = re.match(r"url\(#([^)]+)\)", al["fill"] or "")
            geom = _grad_geom(vec.get("defs", ""), ref.group(1)) if ref else None
            if geom:
                # the gradient's control points ride the same field, so the shading stays on the part as it moves
                fields = [motion_field(kind, boxes[anchor], j / n, amp, phase, sum(map(ord, anchor))) for j in range(n)]
                al["grad_kind"] = geom[0]
                al["grad_keys"] = [[[round(float(c), 2) for c in f(*pt)] for pt in geom[1]] for f in fields]
            vec["animated_layers"].append(al)
        vec["static_layers"] = [l for l in vec["static_layers"] if l["name"] != name]
        i = order.index(name) if name in order else len(order)
        order = order[:i] + names + order[i + 1:]
    vec["layer_order"] = order
    return vec


def _tr_animate_gradients(lot, vec, linear=True):
    """Animate gf/gs start+end of morphing layers whose gradient rides the motion field (animate_layers' grad_keys)."""
    vb = [float(x) for x in vec["viewBox"].split()]
    k = lot["w"] / max(1e-9, vb[2])
    grads = {al["name"]: al for al in vec["animated_layers"] if al.get("grad_keys")}
    ease = ({"i": {"x": [1], "y": [1]}, "o": {"x": [0], "y": [0]}} if linear else
            {"i": {"x": [0.58], "y": [1]}, "o": {"x": [0.42], "y": [0]}})
    for layer in lot["layers"]:
        al = grads.get(layer["nm"])
        if not al:
            continue
        gk, n, op = al["grad_keys"], len(al["grad_keys"]), lot["op"]
        pt = lambda p: [round((p[0] - vb[0]) * k, 2), round((p[1] - vb[1]) * k, 2)]
        for gr in layer["shapes"]:
            for it in gr.get("it", []):
                if it.get("ty") in ("gf", "gs"):
                    for prop, idx in (("s", 0), ("e", 1)):
                        it[prop] = {"a": 1, "k": [{"t": round(op * j / n, 3), "s": pt(gk[j][idx]), **ease} for j in range(n)]
                                    + [{"t": op, "s": pt(gk[0][idx])}]}
    return lot


def _tidy_lottie(lot, linear=True):
    """Round shape vertices to 2 decimals and optionally interpolate keys linearly (no ease-to-stop at every key)."""
    def rnd(sh):
        for key in ("i", "o", "v"):
            sh[key] = [[round(c, 2) for c in pt] for pt in sh[key]]
    for layer in lot["layers"]:
        for gr in layer["shapes"]:
            for it in gr.get("it", []):
                ks = it.get("ks", {})
                if it.get("ty") != "sh":
                    continue
                if ks.get("a") == 1:
                    for k in ks["k"]:
                        for sh in k.get("s", []):
                            rnd(sh)
                        if linear and "i" in k:
                            k["i"], k["o"] = {"x": [1], "y": [1]}, {"x": [0], "y": [0]}
                else:
                    rnd(ks["k"])
    return lot


TR_LOTTIE_JS = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"


def lottie_filmstrip(lot, frames, size=256, bg="#1b1d33"):
    """Render Lottie frames with lottie-web in headless Chrome -> one strip image (None without Chrome)."""
    w = round(size * lot["w"] / max(lot["w"], lot["h"]))
    h = round(size * lot["h"] / max(lot["w"], lot["h"]))
    divs = "".join(f'<div id="f{i}" style="position:absolute;left:{i * w}px;top:0;width:{w}px;height:{h}px;background:{bg}"></div>'
                   for i in range(len(frames)))
    js = (f"var d={json.dumps(lot, separators=(',', ':'))};var fr={json.dumps(list(frames))};"
          "fr.forEach(function(f,i){var an=lottie.loadAnimation({container:document.getElementById('f'+i),renderer:'svg',"
          "loop:false,autoplay:false,animationData:JSON.parse(JSON.stringify(d))});an.addEventListener('DOMLoaded',"
          "function(){an.goToAndStop(f,true)});});")
    lib = _lottie_js()
    html = (f"<!doctype html><body style='margin:0;background:{bg}'>{divs}"
            + (f"<script>{lib}</script>" if lib else f"<script src='{TR_LOTTIE_JS}'></script>") + f"<script>{js}</script></body>")
    return chrome_shot(html, w * len(frames), h, budget=6000)


def region_components(res):
    """Group traced regions into connected pieces of the silhouette -> {region number: component number}."""
    regs = [dict(L, mask=_lmask(L, res["size"][::-1])) for L in res["layers"] if L["role"] == "region"]
    if not regs:
        return {}
    lab, _ = _label(np.logical_or.reduce([L["mask"] for L in regs]))
    comp = {}
    for i, L in enumerate(regs):
        ids = lab[L["mask"]]
        comp[i + 1] = int(np.bincount(ids[ids > 0]).argmax()) if (ids > 0).any() else 0
    order = sorted(set(comp.values()), key=lambda c: -sum(regs[i - 1]["area"] for i, v in comp.items() if v == c))
    rank = {c: k + 1 for k, c in enumerate(order)}
    return {i: rank[c] for i, c in comp.items()}


def region_overlay(src, res, bg="#1b1d33"):
    """Source | numbered regions | connected components c1..cN: what Claude or the vision model reads to name parts."""
    w, h = res["size"]
    regs = [dict(L, mask=_lmask(L, (h, w))) for L in res["layers"] if L["role"] == "region"]
    comps = region_components(res)
    rng = np.random.default_rng(7)
    S = max(1, 768 // max(w, h))
    panels = []
    for mode in ("regions", "components"):
        vis = np.zeros((h, w, 3), np.uint8)
        cols = {}
        for i, L in enumerate(regs):
            key = i + 1 if mode == "regions" else comps[i + 1]
            vis[L["mask"]] = cols.setdefault(key, rng.integers(60, 256, 3))
        im = Image.fromarray(vis).resize((w * S, h * S), Image.NEAREST)
        d = ImageDraw.Draw(im)
        seen = set()
        for i, L in enumerate(regs):
            label = str(i + 1) if mode == "regions" else f"c{comps[i + 1]}"
            if label in seen:
                continue
            seen.add(label)
            m_ = L["mask"] if mode == "regions" else np.logical_or.reduce(
                [R["mask"] for j, R in enumerate(regs) if comps[j + 1] == comps[i + 1]])
            ys, xs = np.nonzero(m_)
            o = np.lexsort((xs, ys))
            x, y = xs[o[len(o) // 2]] * S, ys[o[len(o) // 2]] * S
            d.rectangle([x - 2, y - 2, x + 8 * len(label) + 2, y + 12], fill="black")
            d.text((x, y), label, fill="white")
        panels.append(im)
    left = _on(src.resize((w * S, h * S), Image.LANCZOS), bg).convert("RGB")
    out = Image.new("RGB", (left.width * 3 + 32, left.height), "white")
    for k, im in enumerate([left] + panels):
        out.paste(im, (k * (left.width + 16), 0))
    return out


def name_regions(src, res, parts, model=None):
    """One vision call: assign every numbered region to a named part (or 'other')."""
    n = sum(L["role"] == "region" for L in res["layers"])
    names = list(parts) + ["other"]
    schema = {"type": "object", "additionalProperties": False, "required": ["assignments"], "properties": {
        "assignments": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                   "required": ["region", "part"],
                                                   "properties": {"region": {"type": "integer"},
                                                                  "part": {"type": "string", "enum": names}}}}}}
    desc = "; ".join(f"{k}: {v}" for k, v in parts.items())
    r = ask([f"LEFT is an illustration. MIDDLE is the same image split into {n} numbered colour regions (RIGHT only "
             "shows which regions touch, as c1..cN; ignore those labels). Assign EVERY "
             f"region number 1..{n} to exactly one part: {desc}; use 'other' only if it belongs to none. Judge by position "
             "and by what the region depicts in the LEFT image.", region_overlay(src, res)], schema, "region_parts", model)
    return {int(x["region"]): x["part"] for x in r["assignments"] if 1 <= int(x["region"]) <= n}


def _reg_search(S, part, grid, scales, shifts, cap=60.0, centre=(0.0, 0.0)):
    """Best (cost, scale, dx, dy) placing `part` on the square reference S (grid px) under a truncated colour cost."""
    Sa = S[..., 3] > 128
    best = None
    for s in scales:
        n = max(2, round(grid * s))
        Pm = np.asarray(part.resize((n, n), Image.LANCZOS), np.float32)
        ys, xs = np.nonzero(Pm[..., 3] > 128)
        if len(xs) < 8:
            continue
        cols = Pm[ys, xs, :3]
        c0 = (grid - n) / 2
        for dy in shifts:
            for dx in shifts:
                yy = np.round(ys + c0 + dy + centre[1]).astype(int)
                xx = np.round(xs + c0 + dx + centre[0]).astype(int)
                ok = (yy >= 0) & (yy < grid) & (xx >= 0) & (xx < grid)
                d = np.full(len(xs), cap, np.float32)
                v = np.nonzero(ok)[0]
                v = v[Sa[yy[v], xx[v]]]
                d[v] = np.minimum(np.abs(S[yy[v], xx[v], :3] - cols[v]).mean(1), cap)
                # summed, not averaged: a mean rewards shrinking the part into its best-matching patch
                c = float((d - cap / 2).sum()) / (grid * grid)
                if best is None or c < best[0]:
                    best = (c, s, dx + centre[0], dy + centre[1])
    return best


def register_part(part_sq, src):
    """Register a GPT-isolated part (square canvas) onto the source; GPT drifts up to ~25% in scale and position."""
    sq, off = _square(src)
    part_sq = part_sq.resize(sq.size, Image.LANCZOS)
    S96 = np.asarray(sq.resize((96, 96), Image.LANCZOS), np.float32)
    _, s, dx, dy = _reg_search(S96, part_sq, 96, np.arange(0.7, 1.31, 0.03), range(-20, 21))
    G = 256
    S2 = np.asarray(sq.resize((G, G), Image.LANCZOS), np.float32)
    f = G / 96
    _, s, dx, dy = _reg_search(S2, part_sq, G, np.arange(s - 0.03, s + 0.031, 0.01), range(-4, 5), centre=(dx * f, dy * f))
    n = round(sq.width * s)
    k = sq.width / G
    pad = sq.width
    canvas = Image.new("RGBA", (sq.width + 2 * pad, sq.height + 2 * pad), (0, 0, 0, 0))
    canvas.alpha_composite(part_sq.resize((n, n), Image.LANCZOS),
                           (pad + round((sq.width - n) / 2 + dx * k), pad + round((sq.width - n) / 2 + dy * k)))
    out = canvas.crop((pad + off[0], pad + off[1], pad + off[0] + src.width, pad + off[1] + src.height))
    return out, {"scale": round(float(s), 3), "dx": round(dx * k), "dy": round(dy * k)}


def isolate_parts(src, parts, model=None, jobs=2, cache=None):
    """Route (a): GPT redraws each named part alone on the asset's canvas (occluded areas completed).
    Returns (done {part: png bytes}, failed {part: (error, prompt)}); each result is written to cache[part] on arrival,
    so a later failure never loses a paid call."""
    sq, off = _square(src)
    names = list(parts)

    def one(name):
        others = ", ".join(parts[o] for o in names if o != name) or "everything else"
        prompt = (f"The attached image is an APPROVED illustration. Keep ONLY {parts[name]}, exactly as drawn: same "
                  f"size, position, angle, colours and gradients on the same canvas. Remove {others} completely. Where "
                  f"the removed parts covered {parts[name]}, complete its hidden shape naturally. Transparent background, "
                  "no glow, no shadow, nothing else.")
        raw, _ = edit(_png_bytes(sq), prompt, "1024x1024", "transparent", model)
        if cache and name in cache:
            Path(cache[name]).write_bytes(raw)
        return name, raw, prompt
    done, failed = {}, {}
    with cf.ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(one, n): n for n in names}
        for fut in cf.as_completed(futs):
            try:
                name, raw, _ = fut.result()
                done[name] = raw
            except ApiError as e:  # refusal, exhausted retries, no/undecodable image: all ApiError (C2)
                name = futs[fut]
                failed[name] = (str(e)[:300], f"Keep ONLY {parts[name]}, ...")
    return done, failed


def _hull_mask(mask, min_frac=0.02):
    """Filled convex hull of each sizeable connected component (a cheap amodal-completion prior)."""
    lab, n = _label(mask)
    out = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    d = ImageDraw.Draw(out)
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < max(8, min_frac * mask.sum()):
            continue
        pts = sorted(set(zip(xs.tolist(), ys.tolist())))
        cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, hi = [], []
        for p in pts:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
                lo.pop()
            lo.append(p)
        for p in reversed(pts):
            while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
                hi.pop()
            hi.append(p)
        d.polygon(lo[:-1] + hi[:-1], fill=255)
    return (np.asarray(out) > 0) | mask


def build_rig(src, res, assign, order, amodal=None, ext_frac=0.06, tol=0.8, corner=75, min_detail=0.0, glow_mode="part"):
    """Traced regions -> named, gap-free parts (extended under the parts above them) as a layers.json dict."""
    w, h = res["size"]
    sc = w / src.width
    regs = [dict(L, mask=_lmask(L, (h, w))) for L in res["layers"] if L["role"] == "region"]
    glow = [dict(L, mask=_lmask(L, (h, w))) for L in res["layers"] if L["role"] == "glow"]
    parts, merged = {}, {}
    for i, L in enumerate(regs):
        parts.setdefault(assign.get(i + 1, "other"), []).append(L)
    if "other" in parts and "other" not in order and len(parts) > 1:
        # unnamed slivers join the named part they share the most boundary with
        for L in parts.pop("other"):
            ring = _dilate(L["mask"], 2) & ~L["mask"]
            near = {p: sum(int((ring & R["mask"]).sum()) for R in rs) for p, rs in parts.items()}
            parts[max(near, key=near.get)].append(L)
            merged[regs.index(L) + 1] = max(near, key=near.get)
    order = [p for p in order if p in parts or (p == "glow" and glow)]
    order = order + [p for p in parts if p not in order]
    pmask = {p: np.logical_or.reduce([L["mask"] for L in parts[p]]) for p in parts}
    glows = {}
    if glow and glow_mode == "part":
        # each part owns the glow nearest to it (Voronoi cells), so the glow travels with whatever emits it
        cell = np.zeros((h, w), np.uint8)
        named = [p for p in order if p in pmask]
        for k_, p in enumerate(named):
            cell[pmask[p]] = k_ + 1
        arr = np.asarray(src.convert("RGBA").resize((w, h), Image.LANCZOS), np.float32)
        alpha = arr[..., 3] / 255
        for _ in range(4 * max(w, h)):
            todo = (cell == 0) & (alpha > 0.01)
            if not todo.any():
                break
            grown = np.asarray(Image.fromarray(cell).filter(ImageFilter.MaxFilter(3)))
            cell = np.where(cell == 0, grown, cell)
        P_ = res["params"]
        for k_, p in enumerate(named):
            a_p = np.where(cell == k_ + 1, alpha, 0)
            gl = _alpha_bands(a_p, arr[..., :3], a_p >= 0.5, P_.get("bands", 8), P_.get("tol", 0.8) * 1.5,
                              tuple(P_.get("band_kinds", ("flat",))), P_.get("band_gamma", 1.6))
            if gl:
                glows[f"glow_{p}"] = gl
        order = list(glows) + order
    elif glow:
        glows = {"glow": [L for L in res["layers"] if L["role"] == "glow"]}  # bbox-cropped masks, as _alpha_bands stores them
        order = ["glow"] + [p for p in order if p != "glow"]
    am = {}
    if amodal:
        for p, img in amodal.items():
            am[p] = np.asarray(img.resize((w, h), Image.LANCZOS).getchannel("A")) > 128
    defs, static, boxes, info = [], [], {}, {}
    for zi, p in enumerate(order):
        if p in glows:
            paths = []
            for L in glows[p]:
                f, gd = _fill_svg(L["fill"], f"{p}_{len(defs)}")
                defs.append(gd)
                paths.append({"d": L["d"], "fill": f, "opacity": L["opacity"]})
            m = np.logical_or.reduce([_lmask(L, (h, w)) for L in glows[p]])
        else:
            above = np.zeros((h, w), bool)
            for q in order[zi + 1:]:
                if q in pmask:
                    above |= pmask[q]
            pm = pmask[p]
            r = max(2, int(ext_frac * max(np.ptp(np.nonzero(pm)[0]) + 1, np.ptp(np.nonzero(pm)[1]) + 1)))
            if p in am:
                m = pm | (am[p] & above)
            else:
                m = pm | ((_fill_holes(pm) | _hull_mask(pm) | _dilate(pm, r)) & above)
            rs = sorted(parts[p], key=lambda L: -int(_fill_holes(L["mask"]).sum()))
            later = np.zeros((h, w), bool)
            draws = {}
            for j in range(len(rs) - 1, -1, -1):
                L = rs[j]
                draws[j] = m if j == 0 else (_fill_holes(L["mask"]) & pm) | (_dilate(L["mask"], 2) & later)
                later |= L["mask"]
            paths = []
            for j, L in enumerate(rs):
                if j and L["area"] < min_detail * pm.sum():
                    continue
                d = mask_to_path(draws[j], tol, corner, 0.7, 4.0)
                if not d:
                    continue
                f, gd = _fill_svg(L["fill"], f"{p}_{len(defs)}")
                defs.append(gd)
                paths.append({"d": d, "fill": f, "opacity": 1})
            info[p] = {"regions": [regs.index(L) + 1 for L in rs], "amodal_px": int((m & ~pm).sum())}
        if not paths:
            continue
        ys, xs = np.nonzero(m)
        boxes[p] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        static.append({"name": p, "paths": paths})
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static],
            "rig": {"parts": info, "boxes": boxes, "src_size": list(src.size), "work_scale": round(sc, 5),
                    "merged": merged}}


def _vec_render(vec, size, k=0, only=None):
    """layers.json -> SVG text at keyframe k, optionally only some layers (exploded previews)."""
    v = json.loads(json.dumps(vec))
    if only is not None:
        v["static_layers"] = [l for l in v["static_layers"] if l["name"] in only]
        v["animated_layers"] = [l for l in v["animated_layers"] if l["name"].split("__")[0] in only]
    for al in v["animated_layers"]:
        al["keyframes"] = [al["keyframes"][k % len(al["keyframes"])]]
        if al.get("grad_keys"):
            gid = re.match(r"url\(#([^)]+)\)", al["fill"]).group(1)
            v["defs"] = _grad_patch(v["defs"], gid, al["grad_kind"], al["grad_keys"][k % len(al["grad_keys"])])
    return _vec_svg(v, 0, size=1).replace('width="1" height="1"', f'width="{size[0]}" height="{size[1]}"', 1)


def _parse_kv(s, sep=";"):
    out = {}
    for item in (s or "").split(sep):
        k, eq, v = item.partition("=")
        if eq and k.strip():
            out[k.strip()] = v.strip()
    return out


def _parse_motions(s):
    """'outer_flame=flicker:1.2:0.3; glow=flicker:0.8@outer_flame' -> {layer: (motion, amp, phase, anchor)}"""
    out = {}
    for k, v in _parse_kv(s).items():
        v, _, anchor = v.partition("@")
        bits = v.split(":")
        if bits[0].strip() not in TR_MOTIONS:
            raise ValueError(f"unknown motion {bits[0].strip()!r} in '{k}={v}' (choose {', '.join(TR_MOTIONS)})")
        try:
            out[k] = (bits[0].strip(), float(bits[1]) if len(bits) > 1 and bits[1] else 1.0,
                      float(bits[2]) if len(bits) > 2 and bits[2] else 0.0, anchor.strip() or None)
        except ValueError:
            raise ValueError(f"bad amp/phase in '{k}={v}' (use layer=motion[:amp[:phase]][@anchor], numbers only)")
        if not (abs(out[k][1]) <= 10 and abs(out[k][2]) <= 100):  # also nan/inf: they became 'nan' path data
            raise ValueError(f"amp/phase out of range in '{k}={v}' (amp is a multiplier around 1, at most 10)")
    return out


def _motion_stats(vec):
    """Mean / max point travel (viewBox units) of each animated layer across its keyframes."""
    st = {}
    for al in vec["animated_layers"]:
        P = [_path_points(k) for k in al["keyframes"]]
        dev = np.array([np.linalg.norm(p - P[0], axis=1) for p in P])
        st[al["name"]] = {"mean": round(float(dev.mean()), 2), "max": round(float(dev.max()), 2)}
    return st


def write_animation(vec, out, name, seconds=1.6, fps=30, size=512, ease="linear", bg="#1b1d33", strip=8):
    """layers.json with animated layers -> Lottie + preview.html + keyframe sheet + lottie-web filmstrip."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    errs = _validate_vec(vec)
    if errs:
        raise ValueError("; ".join(errs))
    (out / "layers.json").write_text(json.dumps(vec, indent=1))
    lot = build_lottie(vec, size, seconds, fps)
    lot = _tidy_lottie(lot, ease == "linear")
    _tr_animate_gradients(lot, vec, ease == "linear")
    (out / f"{name}.json").write_text(json.dumps(lot, separators=(",", ":")))
    lib = _lottie_js()  # inline the cached lottie-web so the preview also opens offline; CDN only as a fallback
    (out / "preview.html").write_text(
        f"<!doctype html><meta charset=utf-8><body style='margin:0;background:{bg}'>"
        f"<div id=a style='width:{lot['w']}px;height:{lot['h']}px'></div>"
        + (f"<script>{lib}</script>" if lib else f"<script src='{TR_LOTTIE_JS}'></script>")
        + f"<script>lottie.loadAnimation({{container:a,renderer:'svg',loop:true,autoplay:true,animationData:{json.dumps(lot)}}})</script>")
    vb = [float(x) for x in vec["viewBox"].split()]
    sz = (round(300 * vb[2] / max(vb[2:])), round(300 * vb[3] / max(vb[2:])))
    nk = max((len(al["keyframes"]) for al in vec["animated_layers"]), default=1)
    stats = _motion_stats(vec)
    report = {"lottie": str(out / f"{name}.json"), "bytes": len(json.dumps(lot, separators=(",", ":"))),
              "frames": lot["op"], "fps": fps, "motion": stats}
    for nm, v in stats.items():
        if v["max"] < 0.2:
            print(f"  ! {nm}: max travel {v['max']} viewBox units (< 0.2): the motion is invisible; raise the amplitude or "
                  f"check the layer's box")
    # the art without its glow bands: soft halos would hide a gap in the tear metric
    art = [n.split("__")[0] for n in vec["layer_order"] if not n.split("__")[0].startswith("glow")]
    art = list(dict.fromkeys(art)) or None
    svgs = [_vec_render(vec, sz, k) for k in range(nk)] + [_vec_render(vec, sz, k, only=art) for k in range(nk)]
    rens = _tr_render_svgs(svgs, [sz] * len(svgs))
    if rens:
        keys, bare = rens[:nk], rens[nk:]
        sheet([[(f"key {k + 1}", _on(im, bg)) for k, im in enumerate(keys)]], out / "keys_review.png", T=220)
        report["tear_px"], report["tear_frac_max"], report["tear_radius_px"] = _tear_px(bare, vec, stats, sz)
    else:
        report["keys_review"] = f"not rendered: {_tr_chrome_reason()}"
        print(f"  ! keys_review.png and tear_px NOT measured (Chrome failed: {_tr_chrome_reason()})")
    frames = [round(lot["op"] * i / strip) for i in range(strip)]
    film = lottie_filmstrip(lot, frames, 220, bg)
    if film is not None:
        film.save(out / "lottie_frames.png")
        tiles = [np.asarray(film.crop((i * film.width // strip, 0, (i + 1) * film.width // strip, film.height)), np.float32)
                 for i in range(strip)]
        report["lottie_frame_delta"] = [round(float(np.abs(tiles[i] - tiles[i - 1]).mean()), 2) for i in range(1, strip)]
        report["lottie_rendered"] = bool(np.asarray(film)[..., :3].reshape(-1, 3).std(0).max() > 3)
    else:
        report["lottie_rendered"] = f"not measured: {_tr_chrome_reason()}"
        print(f"  ! lottie_frames.png NOT rendered (Chrome failed: {_tr_chrome_reason()})")
    (out / "animation_report.json").write_text(json.dumps(report, indent=2))
    return report


def _tear_px(keys, vec, stats, sz):
    """Tears per key: rest-pose interior (eroded by the largest travel + 1 px) left uncovered by that key's art.
    Open and enclosed gaps both count; key 0 is the rest pose, so its value is 0 by construction."""
    vb = [float(x) for x in vec["viewBox"].split()]
    travel = max((v["max"] for v in stats.values()), default=0.0) * max(sz) / max(vb[2:])
    r = int(min(max(2, math.ceil(travel) + 1), max(sz) // 4))
    ops = [np.asarray(im.getchannel("A")) > 127 for im in keys]
    core = np.asarray(Image.fromarray(ops[0].astype(np.uint8) * 255).filter(ImageFilter.MinFilter(2 * r + 1))) > 127
    tears = [int((core & ~op).sum()) for op in ops]
    return tears, round(max(tears) / max(1, int(ops[0].sum())), 5), r


def cmd_animate(a):
    src = Path(a.layers)
    vec = _read_json(src, "layers.json", required=("viewBox", "static_layers"))
    errs = _check_vec(vec)
    if errs:
        sys.exit(f"animate: {src} invalid: {errs[0]}" + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else "") +
                 "; fix it or re-export it from rig, vectorize or trace")
    vec.setdefault("animated_layers", [])
    try:
        specs = _parse_motions(a.part)
        if not specs:
            raise ValueError("give --part 'layer=motion[:amp[:phase]]; ...' (motions: " + ", ".join(TR_MOTIONS) + ")")
        vec = animate_layers(vec, specs, a.keyframes)
    except ValueError as e:
        sys.exit(f"animate: {e}")
    stem = src.name[:-len(".layers.json")] if src.name.endswith(".layers.json") else ""
    out = Path(a.out) if a.out else src.parent / "anim" / stem
    try:
        rep = write_animation(vec, out, a.name or stem or src.parent.name, a.seconds, a.fps, a.size, a.ease, a.preview_bg)
    except ValueError as e:
        sys.exit(f"animate: {e}")
    made = [f for f in ("keys_review.png", "lottie_frames.png", "preview.html") if (out / f).exists()]
    print(f"-> {out}/layers.json, {rep['lottie']} ({rep['bytes'] // 1024} KB, {rep['frames']} frames), {', '.join(made)}")
    for k, v in rep["motion"].items():
        print(f"  {k:24s} travel mean={v['mean']} max={v['max']} (viewBox units)")
    if "lottie_frame_delta" in rep:
        print(f"  lottie-web frame-to-frame delta: {rep['lottie_frame_delta']}  rendered={rep['lottie_rendered']}")


def cmd_rig(a):
    d = Path(a.dir)
    src = _open_image(d / "assets" / f"{a.name}.png", "asset", "forge it first or copy the PNG to DIR/assets/")
    out = d / "rig" / a.name
    parts = _parse_kv(a.parts)
    if not parts:
        sys.exit("rig: give --parts 'name=description; ...' listed bottom to top")
    order = [p.strip() for p in (a.order.split(",") if a.order else parts) if p.strip()]
    unknown = [p for p in order if p not in parts]
    if unknown:
        sys.exit(f"rig: --order names {', '.join(unknown)} which are not in --parts ({', '.join(parts)})")
    if a.route == "gpt" and (a.assign or a.min_detail):
        print("  ! --route gpt traces GPT's redraws directly: --assign, --namer and --min-detail are not used")
    res = trace_image(src, a.preset)
    if not any(L["role"] == "region" for L in res["layers"]):
        sys.exit(f"rig: nothing opaque to rig in {d / 'assets' / f'{a.name}.png'} (no traced region)")
    out.mkdir(parents=True, exist_ok=True)
    region_overlay(src, res).save(out / "regions.png")
    amodal, gpt_imgs = None, {}
    if a.route in ("gpt", "hybrid"):
        (out / "parts_gpt").mkdir(exist_ok=True)
        cached = {p: out / "parts_gpt" / f"{p}_raw.png" for p in parts}
        todo = {p: v for p, v in parts.items() if a.force or not cached[p].exists()}
        if todo:
            print(f"isolating {len(todo)} part(s) with the image model ({len(todo)} image calls)...")
            done, failed = isolate_parts(src, todo, a.model, cache=cached)
            if failed:
                for p, (err, prompt) in failed.items():
                    print(f"  ! part {p!r}: {err} — prompt \"{prompt}\" — edit the part description or use --route local --assign")
                kept = len(parts) - len(todo)
                sys.exit(f"rig: {len(failed)} of {len(todo)} part isolation(s) failed; {len(done)} succeeded and are cached in "
                         f"{out / 'parts_gpt'}" + (f" next to {kept} from an earlier run" if kept else "") + " (re-run to resume)")
        reg_info = {}
        for p in parts:
            gpt_imgs[p], reg_info[p] = register_part(Image.open(cached[p]).convert("RGBA"), src)
            gpt_imgs[p].save(out / "parts_gpt" / f"{p}.png")
        print("  registered GPT parts (scale, shift px):", reg_info)
        amodal = gpt_imgs
    hint = f"regions -> {out / 'regions.png'}: Read it, then re-run with --assign 'part=1,2,3; other_part=4,5'"
    if a.route == "gpt":
        vec = _rig_from_parts(src, gpt_imgs, order, res, a.preset)
    else:
        if a.assign:
            assign, issues = parse_assign(a.assign, res)
            for line in issues:
                print(f"  ! {line}")
        elif a.route == "hybrid":
            assign = _assign_by_masks(res, gpt_imgs, order)
        elif a.namer == "gpt":
            print("naming regions with the vision model (1 call)...")
            try:
                assign = name_regions(src, res, parts, a.text_model)
            except (ApiError, ValueError, LookupError, TypeError) as e:  # ApiError, or an answer the int()/dict parse rejects
                print(f"  ! region naming failed: {str(e)[:300]}")
                sys.exit(hint)
        else:
            print(hint)
            return
        stray = sorted({p for p in assign.values() if p not in parts and p != "other"})
        for p in stray:
            print(f"  ! part {p!r} is not in --parts (it becomes an extra layer; a typo?)")
        empty = [p for p in parts if p not in assign.values()]
        if empty:
            print(f"  ! parts with no regions: {', '.join(empty)}")
            if not a.allow_empty:
                sys.exit(f"rig: {', '.join(empty)} got no region; fix --assign (regions.png) or pass --allow-empty")
        vec = build_rig(src, res, assign, order, amodal if a.route == "hybrid" else None, min_detail=a.min_detail,
                        glow_mode=a.glow)
        merged = vec["rig"].get("merged", {})
        if merged:
            regs = [L for L in res["layers"] if L["role"] == "region"]
            frac = sum(regs[i - 1]["area"] for i in merged) / max(1, sum(L["area"] for L in regs))
            print(f"  unassigned regions {','.join(str(i) for i in sorted(merged))} ({100 * frac:.1f}% of area) merged into "
                  f"their neighbours: {', '.join(f'{i}->{p}' for i, p in sorted(merged.items()))}")
        final = {str(i): {"part": assign.get(i, merged.get(i, "other")), **({"merged": True} if i in merged else {})}
                 for i in range(1, sum(L["role"] == "region" for L in res["layers"]) + 1)}
        (out / "assign.json").write_text(json.dumps(final, indent=1))
    vec["rig"]["route"] = a.route
    (out / "layers.json").write_text(json.dumps(vec, indent=1))
    names = vec["layer_order"]
    svgs = [_vec_render(vec, src.size)] + [_vec_render(vec, src.size, only=[p]) for p in names]
    rens = _tr_render_svgs(svgs, [src.size] * len(svgs))
    rep = {"route": a.route, "layers": names, "parts": vec["rig"]["parts"]}
    if rens:
        rep["recomposite_vs_source"] = trace_metrics(src, rens[0], 192)
        rows = [[("source", _on(src, a.preview_bg)), (f"rig ({a.route})", _on(rens[0], a.preview_bg))] +
                [(p, _on(r, a.preview_bg)) for p, r in zip(names, rens[1:])]]
        if gpt_imgs:
            rows.append([(f"gpt {p}", _on(im, a.preview_bg)) for p, im in gpt_imgs.items()])
        sheet(rows, out / "rig_preview.png", T=260)
    print(f"rig ({a.route}) -> {out / 'layers.json'}  layers bottom->top: {', '.join(names)}")
    if rens:
        r_ = rep["recomposite_vs_source"]["display"]
        print(f"  recomposite vs source (display): diff={r_['diff']} iou={r_['iou']}   preview -> {out / 'rig_preview.png'}")
    else:
        rep["recomposite_vs_source"] = f"not measured: {_tr_chrome_reason()}"
        print(f"  ! recomposite fidelity NOT measured, no rig_preview.png (Chrome failed: {_tr_chrome_reason()})")
    if a.animate:
        try:
            vec = animate_layers(vec, _parse_motions(a.animate), a.keyframes)
            rep["animation"] = write_animation(vec, out / "anim", a.name, a.seconds, a.fps, 512, "linear", a.preview_bg)
        except ValueError as e:
            (out / "rig_report.json").write_text(json.dumps(rep, indent=2, default=str))
            sys.exit(f"rig --animate: {e} (the rig itself was written)")
        made = [f for f in ("keys_review.png", "lottie_frames.png", "preview.html") if (out / "anim" / f).exists()]
        print(f"  animated -> {out / 'anim'} (Lottie {a.name}.json, {', '.join(made)})")
        for k, v in rep["animation"]["motion"].items():
            print(f"    {k:24s} travel mean={v['mean']} max={v['max']}")
    (out / "rig_report.json").write_text(json.dumps(rep, indent=2, default=str))


def parse_assign(spec, res):
    """'drop=c1; inner_ring=c2; outer_ring=rest; logs=3,4,7' -> {region number: part} (cN = a whole component)."""
    comps = region_components(res)
    n, nc = len(comps), max(comps.values(), default=0)
    assign, rest, bad = {}, None, []
    for part, ids in _parse_kv(spec).items():
        for tok in re.split(r"[,\s]+", ids.strip()):
            if tok == "rest":
                rest = part
            elif tok.startswith("c") and tok[1:].isdigit():
                hit = {i: part for i, c in comps.items() if c == int(tok[1:])}
                assign.update(hit) if hit else bad.append(tok)
            elif tok.isdigit() and 1 <= int(tok) <= n:
                assign[int(tok)] = part
            elif tok:
                bad.append(tok)
    if rest:
        assign.update({i: rest for i in comps if i not in assign})
    issues = [f"--assign: {', '.join(bad)} do not exist (regions 1..{n}, components c1..c{nc})"] if bad else []
    return assign, issues


def _assign_by_masks(res, masks, order):
    """Hybrid route: each traced region goes to the top-most GPT part that covers most of it."""
    w, h = res["size"]
    M = {p: np.asarray(im.resize((w, h), Image.LANCZOS).getchannel("A")) > 128 for p, im in masks.items()}
    assign = {}
    for i, L in enumerate(r for r in res["layers"] if r["role"] == "region"):
        x0, y0, x1, y1 = L["bbox"]
        cover = {p: (M[p][y0:y1, x0:x1] & L["mask"]).sum() / max(1, L["mask"].sum()) for p in M}
        top = [p for p in reversed(order) if p in cover and cover[p] >= 0.5]
        assign[i + 1] = top[0] if top else max(cover, key=cover.get)
    return assign


def _rig_from_parts(src, imgs, order, res, preset):
    """Route (a): trace every GPT-isolated part on its own and stack them (glow bands from the source trace)."""
    w, h = res["size"]
    defs, static, boxes, info = [], [], {}, {}
    glow = [L for L in res["layers"] if L["role"] == "glow"]
    if glow:
        paths = []
        for L in glow:
            f, gd = _fill_svg(L["fill"], f"glow_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": L["opacity"]})
        static.append({"name": "glow", "paths": paths})
    for p in order:
        r = trace_image(imgs[p], preset, bands=0, work=max(w, h))
        paths = []
        for L in r["layers"]:
            f, gd = _fill_svg(L["fill"], f"{p}_{len(defs)}")
            defs.append(gd)
            paths.append({"d": L["d"], "fill": f, "opacity": 1})
        if paths:
            pts = np.vstack([_path_points(x["d"]) for x in paths])
            boxes[p] = [float(v) for v in (*pts.min(0), *pts.max(0))]
            static.append({"name": p, "paths": paths})
            info[p] = {"paths": len(paths)}
    return {"viewBox": f"0 0 {w} {h}", "defs": "".join(defs), "static_layers": static, "animated_layers": [],
            "layer_order": [l["name"] for l in static], "rig": {"parts": info, "boxes": boxes, "src_size": list(src.size)}}


def _tr_int_min(lo, hi=None):
    """argparse type: integer >= lo (e.g. --work 16, --keyframes 2), and <= hi when given."""
    def t(s):
        v = _pos_int(s)
        if v < lo or (hi is not None and v > hi):
            raise argparse.ArgumentTypeError(f"expected an integer >= {lo}" + (f" and <= {hi}" if hi else "") + f", got {v}")
        return v
    return t


def _tr_nonneg_int(s):
    """argparse type: integer >= 0 (--bands 0 = no bands)."""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {s!r}")
    if v < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {v}")
    return v


def _add_trace_rig_parsers(sub):
    from .cli import _color_arg  # deferred: cli imports this module
    p = sub.add_parser("trace", help="raster asset -> multi-colour SVG + VectorDrawable with gradient fits, glow "
                                     "bands, measured fidelity and a vector-or-raster recommendation")
    p.add_argument("files", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=["auto", "flat", "illustration", "gradient", "vtracer"], default="auto")
    p.add_argument("--dp", type=_pos_float, help="display width in dp (default: from assets.json when tracing DIR/assets/*.png, "
                                                 "else the VD is 48dp on its long side)")
    p.add_argument("--display", type=_tr_int_min(16), help="px size (long side) the display-size fidelity is measured at "
                                                          "(default: 3x the dp width, else max(192, 96 x aspect))")
    p.add_argument("--work", type=_tr_int_min(16), help="tracing resolution, long side (default 512, never above the source size)")
    p.add_argument("--colors", type=_pos_int, help="k-means colours before region merging")
    p.add_argument("--bands", type=_tr_nonneg_int, help="alpha bands used to rebuild glow/soft edges (0 = none)")
    p.add_argument("--tol", type=_pos_float, help="curve-fit tolerance in work px")
    p.add_argument("--name", help="output stem (single input only; the VD is --prefix + name, sanitised to [a-z0-9_])")
    p.add_argument("--prefix", default="img_", help="VectorDrawable file prefix")
    p.add_argument("--bg", type=_color_arg, default="#1b1d33", help="preview backdrop"); p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("rig", help="split a forged illustration into named, registered, gap-free vector parts "
                                   "(layers.json for Lottie), optionally animated procedurally")
    p.add_argument("dir"); p.add_argument("name")
    p.add_argument("--parts", required=True, help="'name=description; ...' listed bottom to top")
    p.add_argument("--route", choices=["local", "hybrid", "gpt"], default="local",
                   help="local: traced regions named by --assign or vision; hybrid: GPT part masks assign the traced "
                        "regions and complete occluded shapes; gpt: trace GPT's isolated part redraws")
    p.add_argument("--assign", help="'part=1,2,3; other=4,5' region numbers from regions.png (skips the vision call)")
    p.add_argument("--namer", choices=["gpt", "none"], default="gpt", help="none: only write regions.png for Claude")
    p.add_argument("--order", help="comma list bottom->top (default: --parts order)")
    p.add_argument("--preset", choices=list(TRACE_PRESETS), default="illustration")
    p.add_argument("--min-detail", type=_unit_float, default=0.0, help="drop detail regions smaller than this fraction of "
                                                                       "their part (smaller Lottie; the part base covers them)")
    p.add_argument("--allow-empty", action="store_true", help="continue when a --parts part gets no region (default: exit)")
    p.add_argument("--animate", help="'part=motion[:amp[:phase]][@anchor_part]; ...' motions: " + ", ".join(TR_MOTIONS))
    p.add_argument("--keyframes", type=_tr_int_min(2, 120), default=8, help="procedural morph keys per loop (2-120)")
    p.add_argument("--seconds", type=_pos_float, default=1.6, help="loop length")
    p.add_argument("--fps", type=_pos_int, default=30)
    p.add_argument("--force", action="store_true", help="re-run the GPT part isolation even when parts_gpt/ is cached")
    p.add_argument("--model", default=None, help="image model for --route hybrid|gpt")
    p.add_argument("--text-model", default=None, help="vision model for --namer gpt")
    p.add_argument("--glow", choices=["part", "shared"], default="part",
                   help="part: each part owns (and moves with) the glow nearest to it; shared: one glow layer")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="backdrop of rig_preview.png / keys_review.png"); p.set_defaults(fn=cmd_rig)
    p = sub.add_parser("animate", help="procedural morph keyframes (sway/flicker/breathe/squash/bob/wave/ripple) "
                                       "for named layers of a layers.json -> Lottie; no API calls")
    p.add_argument("layers", help="layers.json from rig, vectorize or trace")
    p.add_argument("--part", required=True, help="'layer=motion[:amp[:phase]][@anchor_layer]; ...' (anchor: move "
                                                 "with that layer's field, e.g. glow=flicker@outer_flame)")
    p.add_argument("--keyframes", type=_tr_int_min(2, 120), default=8, help="procedural morph keys per loop (2-120)")
    p.add_argument("--seconds", type=_pos_float, default=1.6, help="loop length")
    p.add_argument("--fps", type=_pos_int, default=30); p.add_argument("--size", type=_tr_int_min(16), default=512,
                                                                     help="Lottie canvas long side in px")
    p.add_argument("--ease", choices=["linear", "keys"], default="linear",
                   help="linear between procedural keys (smooth); keys = build_lottie's ease at every key")
    p.add_argument("--out", help="output dir (default: <layers dir>/anim/<stem>)")
    p.add_argument("--name", help="Lottie file stem (default: the layers.json stem)")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="backdrop of keys_review.png / lottie_frames.png / preview.html")
    p.set_defaults(fn=cmd_animate)
