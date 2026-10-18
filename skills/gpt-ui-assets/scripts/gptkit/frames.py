"""Frames."""
import base64
import concurrent.futures as cf
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from .core import ApiError, _config, _open_image, _plan, _png_bytes, _res_name, _size_for, edit, pick_image_model
from .imaging import _lab, _rgba, content_box, sheet
from .vectordrawable import _aapt2_tree

# ---- animation frames

def _runs(v):
    """(start, end) pairs of the True runs of a 1-D array."""
    d = np.diff(np.concatenate([[0], np.asarray(v, np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _remove_dividers(img):
    """Clear drawn grid lines: thin lines that span nearly the whole sheet."""
    arr = np.array(img.convert("RGBA"))
    solid = arr[..., 3] > 100
    for axis in (0, 1):
        cov = solid.mean(axis=axis)
        n = cov.size
        for s, e in _runs(cov >= 0.9):
            side = cov[[i for i in (s - 3, e + 2) if 0 <= i < n]]
            if e - s <= max(4, n // 150) and (side < 0.6).all():
                sl = slice(max(0, s - 1), min(n, e + 1))
                arr[(slice(None), sl) if axis == 0 else sl] = 0
    return Image.fromarray(arr)


def _bands(occ, gap):
    """Occupied runs of a 1-D profile; gaps under `gap` are bridged and slivers join their nearest neighbour."""
    runs = []
    for s, e in _runs(occ):
        if runs and s - runs[-1][1] < gap:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    while len(runs) > 1:
        sizes = [e - s for s, e in runs]
        i = int(np.argmin(sizes))
        if sizes[i] >= 0.3 * max(sizes):
            break
        left = runs[i][0] - runs[i - 1][1] if i else math.inf
        right = runs[i + 1][0] - runs[i][1] if i + 1 < len(runs) else math.inf
        j = i - 1 if left <= right else i
        runs[j:j + 2] = [(runs[j][0], runs[j + 1][1])]
    return runs


def _gutter_cells(A):
    """XY cut along empty gutters: row bands, then column bands inside each row, in reading order."""
    H, W = A.shape
    m = A > 64
    rows = _bands(m.sum(1) > max(2, W * 0.005), max(4, H // 60))
    cells = []
    for r, (y0, y1) in enumerate(rows):
        top = 0 if r == 0 else (rows[r - 1][1] + y0) // 2
        bot = H if r == len(rows) - 1 else (y1 + rows[r + 1][0]) // 2
        cols = _bands(m[top:bot].sum(0) > max(2, (bot - top) * 0.005), max(4, W // 60))
        for c, (x0, x1) in enumerate(cols):
            left = 0 if c == 0 else (cols[c - 1][1] + x0) // 2
            right = W if c == len(cols) - 1 else (x1 + cols[c + 1][0]) // 2
            cells.append((left, top, right, bot))
    return cells


def _reading_order(boxes):
    boxes = sorted(boxes, key=lambda b: b[1] + b[3])
    tol = 0.5 * float(np.median([b[3] - b[1] for b in boxes]))
    rows = []
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        if rows and abs(cy - rows[-1][0]) < tol:
            rows[-1][1].append(b)
        else:
            rows.append((cy, [b]))
    return [b for _, r in rows for b in sorted(r, key=lambda b: b[0])]


def _blob_cells(A, count):
    """(box, mask) cells from connected blobs, for sheets whose frames are not separated by straight gutters."""
    f = max(1, max(A.shape) // 400)
    m = A[::f, ::f] > 96
    h, w = m.shape
    ml, lab, boxes = m.tolist(), [[0] * w for _ in range(h)], []
    for sy, sx in zip(*np.nonzero(m)):
        if lab[sy][sx]:
            continue
        n = len(boxes) + 1
        lab[sy][sx] = n
        stack, x0, y0, x1, y1, area = [(sy, sx)], sx, sy, sx, sy, 0
        while stack:
            y, x = stack.pop()
            area += 1
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < h and 0 <= nx < w and ml[ny][nx] and not lab[ny][nx]:
                        lab[ny][nx] = n
                        stack.append((ny, nx))
        boxes.append([x0, y0, x1 + 1, y1 + 1, area, {n}])
    boxes = [b for b in boxes if b[4] >= h * w * 0.0005]

    def union(p, q):
        return [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]), p[4] + q[4], p[5] | q[5]]

    def nested(p, q):
        ix, iy = min(p[2], q[2]) - max(p[0], q[0]), min(p[3], q[3]) - max(p[1], q[1])
        return ix > 0 and iy > 0 and ix * iy >= 0.5 * min((b[2] - b[0]) * (b[3] - b[1]) for b in (p, q))
    merged = True
    while merged:
        merged = False
        for i, j in ((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            if nested(boxes[i], boxes[j]):
                boxes[i] = union(boxes[i], boxes.pop(j))
                merged = True
                break
    if not boxes:
        return []
    floor = 0.15 * float(np.median(sorted(b[4] for b in boxes)[-count:]))
    main = [b for b in boxes if b[4] >= floor]
    for b in boxes:
        if b[4] < floor:
            k = min(range(len(main)), key=lambda k: abs(main[k][0] + main[k][2] - b[0] - b[2]) +
                    abs(main[k][1] + main[k][3] - b[1] - b[3]))
            main[k] = union(main[k], b)
    H, W = A.shape
    g = max(1, int(max(h, w) * 0.02))
    labels = np.array(lab)
    cells = []
    for b in _reading_order(main):
        own = Image.fromarray(np.isin(labels, list(b[5])).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * g + 1))
        cells.append(((max(0, (b[0] - g) * f), max(0, (b[1] - g) * f), min(W, (b[2] + g) * f), min(H, (b[3] + g) * f)),
                      own.resize((w * f, h * f), Image.NEAREST)))
    return cells


def _robust_box(alpha, thr=24, frac=0.004):
    """Content bbox that ignores stray specks (rows/columns with only a few pixels over thr)."""
    m = alpha > thr
    ys = np.nonzero(m.sum(1) > max(1, frac * m.shape[1]))[0]
    xs = np.nonzero(m.sum(0) > max(1, frac * m.shape[0]))[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(xs) and len(ys) else None


def _slice_sheet(img, count, cols):
    """Frames of a sprite sheet in reading order: gutter cuts, else a clean equal grid, else connected blobs."""
    A = np.asarray(img.getchannel("A"))
    H, W = A.shape
    rows_n = math.ceil(count / cols)
    grid = [((round(i % cols * W / cols), round(i // cols * H / rows_n), round((i % cols + 1) * W / cols),
              round((i // cols + 1) * H / rows_n)), None) for i in range(count)]
    m = A > 64
    xs, ys = [round(c * W / cols) for c in range(1, cols)], [round(r * H / rows_n) for r in range(1, rows_n)]
    cuts = [m[:, max(0, x - 1):x + 2].mean() for x in xs] + [m[max(0, y - 1):y + 2].mean() for y in ys]
    how, cells = "gutters", [(c, None) for c in _gutter_cells(A)]
    if len(cells) != count and max(cuts, default=0) < 0.01:
        how, cells = "grid", grid
    elif len(cells) != count:
        blobs = _blob_cells(A, count)
        if abs(len(blobs) - count) < abs(len(cells) - count):
            how, cells = "blobs", blobs
        if abs(len(cells) - count) > count // 4:
            how, cells = "grid", grid
    crops = []
    for i, (box, mask) in enumerate(cells):
        cell = img.crop(box)
        if mask is not None:
            cell.putalpha(ImageChops.multiply(cell.getchannel("A"), mask.crop(box)))
        bb = _robust_box(np.asarray(cell.getchannel("A")))
        if not bb:
            print(f"  ! frame {i + 1} empty")
            continue
        if bb[0] <= 1 or bb[1] <= 1 or bb[2] >= cell.width - 1 or bb[3] >= cell.height - 1:
            print(f"  ! frame {i + 1} touches its cell edge (may be cut off)")
        crops.append(cell.crop(bb))
    print(f"  sliced {len(crops)} frames by {how}" + ("" if len(crops) == count else f" (asked for {count})"))
    return crops


def _anchor_point(img, anchor):
    """Base centre (bottom anchor) or centroid (center anchor) of the solid part."""
    s = np.asarray(img.getchannel("A")) > 128
    if not s.any():
        return img.width / 2, img.height if anchor == "bottom" else img.height / 2
    ys, xs = np.nonzero(s)
    if anchor == "bottom":
        band = ys >= ys.max() - 0.15 * (ys.max() - ys.min() + 1)
        return float(xs[band].mean()), float(ys.max() + 1)
    return float(xs.mean()), float(ys.mean())


def _align_frames(crops, anchor):
    """Place crops on one shared canvas with their anchor points coincident."""
    pts = [_anchor_point(c, anchor) for c in crops]
    left, up = max(p[0] for p in pts), max(p[1] for p in pts)
    right = max(c.width - p[0] for c, p in zip(crops, pts))
    down = max(c.height - p[1] for c, p in zip(crops, pts))
    pad = int(max(left + right, up + down) * 0.06)
    size = (math.ceil(left + right) + 2 * pad, math.ceil(up + down) + 2 * pad)
    out = []
    for c, (px, py) in zip(crops, pts):
        f = Image.new("RGBA", size, (0, 0, 0, 0))
        f.alpha_composite(c, (round(pad + left - px), round(pad + up - py)))
        out.append(f)
    return out


def _square(img):
    side = max(img.size)
    c = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    off = ((side - img.width) // 2, (side - img.height) // 2)
    c.alpha_composite(img, off)
    return c, off


def _register(layer, target, grid=160):
    """Best scale/offset placing `layer` (a part of `target`) onto target's canvas; returns the placed layer.
    Score = overlap - 2 x layer pixels outside the target, over 11 scales and a +-10% grid of offsets."""
    tm = np.asarray(target.getchannel("A").resize((grid, grid))) > 100
    lm_full = layer.getchannel("A")
    r = grid // 10
    best = (-1, 1.0, 0, 0)
    for s10 in range(85, 116, 3):
        s = s10 / 100
        size = max(1, round(grid * s))
        lm = np.asarray(lm_full.resize((size, size))) > 100
        o = (grid - size) // 2
        for dx in range(-r, r + 1, 2):
            for dy in range(-r, r + 1, 2):
                x, y = o + dx, o + dy                       # the layer's top-left on the canvas; clipped like a paste
                lx0, ly0, lx1, ly1 = max(0, -x), max(0, -y), min(size, grid - x), min(size, grid - y)
                if lx1 <= lx0 or ly1 <= ly0:
                    continue
                sub = lm[ly0:ly1, lx0:lx1]
                inter = int((sub & tm[y + ly0:y + ly1, x + lx0:x + lx1]).sum())
                sc = inter - 2 * (int(sub.sum()) - inter)
                if sc > best[0]:
                    best = (sc, s, dx, dy)
    _, s, dx, dy = best
    W, H = target.size
    placed = layer.resize((round(W * s), round(H * s)), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(placed, ((W - placed.width) // 2 + round(dx * W / grid), (H - placed.height) // 2 + round(dy * H / grid)))
    return out


def _xcorr(a, k):
    """Integer cross-correlation of boolean a with kernel k over every offset (FFT): the value at index
    (y + kh - 1, x + kw - 1) is the count of kernel pixels that land on true pixels of a when k sits at (x, y)."""
    H, W = a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1
    c = np.fft.irfft2(np.fft.rfft2(a.astype(np.float64), (H, W)) * np.fft.rfft2(k[::-1, ::-1].astype(np.float64), (H, W)),
                      (H, W))
    return np.rint(c).astype(np.int64)


def _fit_moving(src, static, frame, grid=128):
    """Scale + bottom-centre anchor that best overlays `frame` (moving part) on `src`, given its static layer.
    Score per placement: pixels on the visible moving part - 2 x pixels outside the asset (rows above the canvas
    count) - 0.7 x visible pixels missed; every offset of every scale is scored at once by cross-correlation."""
    k = grid / max(src.size)
    tw, th = max(1, round(src.width * k)), max(1, round(src.height * k))
    sa = np.asarray(src.getchannel("A").resize((tw, th))) > 100
    st = np.asarray(static.getchannel("A").resize((tw, th))) > 30
    visible = sa & ~st
    allowed = (np.asarray(src.getchannel("A").resize((tw, th))) > 40) | st
    fb = content_box(frame, 100)
    ys, xs = np.nonzero(visible)
    if not fb or not len(xs):
        return None
    fcrop = frame.crop(fb)
    s0 = (xs.max() - xs.min() + 1) / (fcrop.width * k)
    vis_total = int(visible.sum())
    best = None
    for f in np.linspace(0.55, 1.25, 15):
        sc = s0 * f
        fw, fh = max(1, round(fcrop.width * sc * k)), max(1, round(fcrop.height * sc * k))
        if fw > tw or fh > th + th // 2:
            continue
        fm = np.asarray(fcrop.getchannel("A").resize((fw, fh))) > 100
        y0, y1 = -(fh // 2), th - fh // 2
        outside = np.pad(~allowed, ((fh, 0), (0, 0)), constant_values=True)
        hit = _xcorr(visible, fm)[y0 + fh - 1:y1 + fh - 1, fw - 1:tw]
        spill = _xcorr(outside, fm)[y0 + 2 * fh - 1:y1 + 2 * fh - 1, fw - 1:tw]
        score = hit - 2 * spill - 0.7 * (vis_total - hit)
        idx = int(np.argmax(score))
        yy, xx = divmod(idx, score.shape[1])
        if best is None or score.flat[idx] > best[0]:
            best = (float(score.flat[idx]), sc, xx, y0 + yy, fw, fh)
    if best is None:
        return None
    _, sc, x, y, fw, fh = best
    return sc, (x + fw / 2) / k, (y + fh) / k, fb


def _placed_alpha(size, frame, s, org):
    m = Image.new("L", size, 0)
    m.paste(frame.getchannel("A").resize((max(1, round(frame.width * s)), max(1, round(frame.height * s)))),
            (round(org[0]), round(org[1])))
    return np.asarray(m)


def _box_gap(a, b):
    return max(0, a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])


def _target_clusters(target, size):
    """Spatial clusters of the target's components (e.g. coin + rays vs slivers the static redraw left behind)."""
    from .trace_rig import _label  # deferred: trace_rig imports this module
    lab, n = _label(target)
    if n < 2:
        return []
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = [i + 1 for i in np.argsort(-areas)[:40] if areas[i] >= 0.01 * target.sum()]
    boxes = {}
    for i in keep:
        ys, xs = np.nonzero(lab == i)
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    clusters = [[i] for i in keep]
    merged = True
    while merged:
        merged = False
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                if min(_box_gap(boxes[i], boxes[j]) for i in clusters[x] for j in clusters[y]) < 0.06 * max(size):
                    clusters[x] += clusters.pop(y)
                    merged = True
                    break
            if merged:
                break
    if len(clusters) < 2:
        return []
    return [m for m in (np.isin(lab, c) for c in clusters) if m.sum() >= 0.03 * target.sum()]


def _placement(src, static, frames):
    """One scale + origin for every frame: silhouette fit of the middle frame or extent fit of any frame, by IoU."""
    raw = (np.asarray(src.getchannel("A")) > 60) & _moving_mask(static)
    # an opening drops slivers the static redraw leaves along its edges; thin moving parts keep the raw mask
    k = max(3, round(max(src.size) * 0.015)) | 1
    opened = np.asarray(Image.fromarray((raw * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(k))
                        .filter(ImageFilter.MaxFilter(k))) > 127
    target = opened if opened.sum() >= 0.3 * raw.sum() else raw
    if not target.any():
        return None
    mid = len(frames) // 2
    fit = _fit_moving(src, static, frames[mid])

    def best_for(t, silhouette):
        ys, xs = np.nonzero(t)
        tb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        cands = []
        if silhouette and fit:
            s, cx, by, fb = fit
            cands.append(("silhouette", mid, s, (cx - (fb[0] + fb[2]) / 2 * s, by - fb[3] * s)))
        for i, f in enumerate(frames):
            b = content_box(f, 60)
            if b:
                s = math.sqrt((tb[2] - tb[0]) / (b[2] - b[0]) * (tb[3] - tb[1]) / (b[3] - b[1]))
                cands.append(("extent", i, s, ((tb[0] + tb[2] - (b[0] + b[2]) * s) / 2,
                                               (tb[1] + tb[3] - (b[1] + b[3]) * s) / 2)))

        def iou(c):
            m = _placed_alpha(src.size, frames[c[1]], c[2], c[3]) > 60
            return (m & t).sum() / max(1, (m | t).sum())
        return max(((iou(c), c) for c in cands), key=lambda x: x[0], default=None)
    best = best_for(target, True)
    if best is None:
        return None
    for cl in _target_clusters(target, src.size):
        alt = best_for(cl, False)
        if alt and alt[0] > best[0] + 0.15:
            best = (alt[0], (alt[1][0] + " (cluster)",) + alt[1][1:])
    v, (how, i, s, org) = best
    print(f"  registered by {how} fit of f{i + 1} (IoU {v:.2f} with the asset's moving part)")
    return s, org, v


def _composite(src, static, frames, order):
    """Place every aligned moving frame with one shared transform onto the asset and stack the static layer."""
    fit = _placement(src, static, frames)
    if not fit:
        print("  ! could not locate the moving part in the asset; composites skipped")
        return [], 0.0
    s, org, v = fit
    P = max(src.size)
    size = (src.width + 2 * P, src.height + 2 * P)
    pos = (P + round(org[0]), P + round(org[1]))
    st = Image.new("RGBA", size, (0, 0, 0, 0))
    st.alpha_composite(static, (P, P))
    comps = []
    for f in frames:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        layer.paste(f.resize((max(1, round(f.width * s)), max(1, round(f.height * s))), Image.LANCZOS), pos)
        comp = Image.new("RGBA", size, (0, 0, 0, 0))
        for lyr in ((layer, st) if order == "front" else (st, layer)):
            comp.alpha_composite(lyr)
        comps.append(comp)
    # one shared crop box keeps every frame registered
    boxes = [content_box(c, 8) for c in comps]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    return [c.crop(box) for c in comps], v


# -- frame QA

_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])




def _lab_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    c = (np.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * _D65) @ np.linalg.inv(_XYZ).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def _solid_lab(img, mask=None, thr=160):
    arr = np.asarray(img.convert("RGBA"))
    sel = arr[..., 3] > thr
    if mask is not None:
        sel &= mask
    return _lab(arr[..., :3][sel]) if sel.any() else np.zeros((1, 3))


def _moving_mask(static):
    """Pixels of the approved asset that belong to the moving part (the asset minus a dilated static layer)."""
    return np.asarray(static.getchannel("A").filter(ImageFilter.MaxFilter(9))) <= 30


def _premul_thumb(img, side=128):
    t = img.copy()
    t.thumbnail((side, side), Image.LANCZOS)
    a = np.asarray(t, np.float64) / 255
    return np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1) * 255


def _alpha_stats(img, mask=None):
    """Alpha-weighted mean Lab colour and alpha mass (robust for glows and fading frames)."""
    arr = np.asarray(img.convert("RGBA"))
    w = arr[..., 3] / 255 * (1 if mask is None else mask)
    sel = w > 0.1
    if not sel.any():
        return np.zeros(3), 1.0
    return (_lab(arr[..., :3][sel]) * w[sel, None]).sum(0) / w[sel].sum(), float(w.sum())


def _outliers(X, thr, spike):
    """Frames off the median by > thr in an otherwise steady series, or sticking out from both neighbours by > spike."""
    X = np.asarray(X, np.float64).reshape(len(X), -1)
    dev = np.linalg.norm(X - np.median(X, 0), axis=1)
    out = set(np.nonzero((dev > thr) & (dev > 6 * np.median(dev)))[0].tolist())
    for i in range(len(X) if len(X) >= 4 else 0):
        p, q = X[i] - X[i - 1], X[i] - X[(i + 1) % len(X)]
        if min(np.linalg.norm(p), np.linalg.norm(q)) > spike and np.dot(p, q) > 0:
            out.add(i)
    return out


def _frame_qa(frames, ref, ref_mask=None):
    """Colour (Lab) and size outliers, centre drift and loop continuity of an aligned frame sequence."""
    n = len(frames)
    ref_lab = _alpha_stats(ref, ref_mask)[0]
    stats = [_alpha_stats(f) for f in frames]
    labs, mass = np.array([x[0] for x in stats]), np.array([max(1.0, x[1]) for x in stats])
    side = max(frames[0].size)
    pts = np.array([_anchor_point(f, "center") for f in frames]) / side
    med_lab, med_mass, med_pt = np.median(labs, 0), float(np.median(mass)), np.median(pts, 0)
    flags = {"colour": _outliers(labs, 8, 10), "size": _outliers(np.log(mass) / 2, 0.14, 0.26),
             "drift": _outliers(pts, 0.06, 0.08)}
    thumbs = [_premul_thumb(f) for f in frames]
    adj = [float(np.abs(thumbs[i] - thumbs[(i + 1) % n]).mean()) for i in range(n)]
    typical = max(1e-6, float(np.median(adj[:-1])) if n > 2 else adj[0])

    def near(i):  # steps just outside frame i's in/out steps, floored at the median step
        return max(typical, (adj[(i - 2) % n] + adj[(i + 1) % n]) / 2) if n > 4 else typical
    rows = []
    for i in range(n):
        r = {"frame": i + 1, "dE_seq": round(float(np.linalg.norm(labs[i] - med_lab)), 1),
             "dE_ref": round(float(np.linalg.norm(labs[i] - ref_lab)), 1),
             "size": round(math.sqrt(mass[i] / med_mass), 3),
             "drift": round(float(np.linalg.norm(pts[i] - med_pt)), 3),
             "step_in": round(adj[i - 1] / near(i), 2), "step_out": round(adj[i] / near(i), 2)}
        r["flags"] = [k for k, v in flags.items() if i in v]
        if n > 3 and min(r["step_in"], r["step_out"]) > 2:
            r["flags"].append("pop")
        rows.append(r)
    seq = {"frames": n, "dE_ref": round(float(np.linalg.norm(med_lab - ref_lab)), 1),
           "loop_ratio": round(adj[-1] / near(n - 1), 2), "step_median": round(typical, 2), "flags": []}
    if seq["dE_ref"] > 10:
        seq["flags"].append("style-drift")
    if seq["loop_ratio"] > 2:
        seq["flags"].append("loop-seam")
    return {"sequence": seq, "frames": rows}


def _qa_report(qa, label=""):
    s = qa["sequence"]
    bad = [r for r in qa["frames"] if r["flags"]]
    print(f"  QA{label}: dE vs asset {s['dE_ref']}, loop seam {s['loop_ratio']}x a typical step"
          f"{' ' + ','.join(s['flags']) if s['flags'] else ''}; flagged frames: "
          + (", ".join(f"f{r['frame']}({'+'.join(r['flags'])})" for r in bad) or "none")
          + (" (try --match-colors)" if "style-drift" in s["flags"] else ""))
    return [r["frame"] - 1 for r in bad]


def _badness(r):
    return max(r["dE_seq"] / 8, abs(math.log(max(r["size"], 1e-3))) / 0.14, r["drift"] / 0.06,
               min(r["step_in"], r["step_out"]) / 2)


def _match_colors(frames, ref, ref_mask=None):
    """One LCh transform for all frames: lightness mean/std, chroma ratio and hue rotation onto the approved asset."""
    r, p = _solid_lab(ref, ref_mask), np.concatenate([_solid_lab(f) for f in frames])
    gain = np.clip(r[:, 0].std() / (p[:, 0].std() + 1e-6), 0.6, 1.6)
    chroma = np.hypot(r[:, 1], r[:, 2]).mean() / max(1e-6, np.hypot(p[:, 1], p[:, 2]).mean())
    turn = np.arctan2(r[:, 2].mean(), r[:, 1].mean()) - np.arctan2(p[:, 2].mean(), p[:, 1].mean())
    out = []
    for f in frames:
        arr = np.array(f)
        vis = arr[..., 3] > 0
        lab = _lab(arr[..., :3][vis])
        c, h = np.hypot(lab[:, 1], lab[:, 2]) * chroma, np.arctan2(lab[:, 2], lab[:, 1]) + turn
        arr[..., :3][vis] = _lab_rgb(np.stack([(lab[:, 0] - p[:, 0].mean()) * gain + r[:, 0].mean(),
                                               c * np.cos(h), c * np.sin(h)], -1))
        out.append(Image.fromarray(arr))
    return out


# -- procedural motion

MOTIONS = {"bounce": 0.9, "pulse": 1.0, "float": 2.0, "wiggle": 1.0, "spin": 1.2, "breathe": 4.0}


def _motion(kind, t, amt):
    """Transform at loop phase t: (scale x, scale y, degrees clockwise, dy in content heights, opacity, pivot)."""
    wave = 0.5 - 0.5 * math.cos(2 * math.pi * t)
    if kind == "bounce":
        land = 0.14
        if t < land:
            sy = 1 - 0.16 * amt * math.sin(math.pi * t / land)
            return 1 / sy, sy, 0, 0, 1, "bottom"
        u = (t - land) / (1 - land)
        sy = 1 + 0.08 * amt * abs(1 - 2 * u)
        return 1 / sy, sy, 0, -0.22 * amt * 4 * u * (1 - u), 1, "bottom"
    if kind == "pulse":
        return 1 + 0.08 * amt * wave, 1 + 0.08 * amt * wave, 0, 0, 1, "center"
    if kind == "breathe":
        e = wave * wave * (3 - 2 * wave)
        return 1 + 0.06 * amt * e, 1 + 0.06 * amt * e, 0, 0, 0.8 + 0.2 * e, "center"
    if kind == "float":
        return 1, 1, 2 * amt * math.cos(2 * math.pi * t), -0.035 * amt * math.sin(2 * math.pi * t), 1, "center"
    if kind == "wiggle":
        return 1, 1, 8 * amt * math.sin(2 * math.pi * t), 0, 1, "bottom"
    return 1, 1, 360 * t, 0, 1, "center"


def _affine(p, w, h):
    sx, sy, deg, dy, _, pivot = p
    px, py = w / 2, h if pivot == "bottom" else h / 2
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]])
    return A, np.array([px, py + dy * h]) - A @ [px, py]


def _procedural(src, kind, count, amt):
    """Loop frames of `src` under an eased affine motion; returns frames, per-frame params, canvas origin, art."""
    art = src.crop(content_box(src, 8) or (0, 0, src.width, src.height))
    w, h = art.size
    params = [_motion(kind, i / count, amt) for i in range(count)]
    mats = [_affine(p, w, h) for p in params]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64)
    pts = np.concatenate([corners @ A.T + t for A, t in mats])
    lo, hi = np.floor(pts.min(0)) - 2, np.ceil(pts.max(0)) + 2
    size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    pm = art.convert("RGBa")
    frames = []
    for (A, t), p in zip(mats, params):
        inv = np.linalg.inv(A)
        b = -inv @ (t - lo)
        f = pm.transform(size, Image.AFFINE, (inv[0, 0], inv[0, 1], b[0], inv[1, 0], inv[1, 1], b[1]),
                         Image.BICUBIC).convert("RGBA")
        if p[4] < 1:
            f.putalpha(f.getchannel("A").point(lambda v, o=p[4]: round(v * o)))
        frames.append(f)
    return frames, params, lo, art


# -- outputs

def _lottie_ks_image(w, h):
    return {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [w / 2, h / 2, 0]},
            "a": {"a": 0, "k": [w / 2, h / 2, 0]}, "s": {"a": 0, "k": [100, 100, 100]}}


def _lottie_image(img, i):
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return {"id": f"img_{i}", "w": img.width, "h": img.height, "u": "", "e": 1,
            "p": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def lottie_image_sequence(frames, fps, name="gptkit"):
    """Lottie image sequence: each frame is an embedded PNG shown for exactly one Lottie frame."""
    w, h = frames[0].size
    layers = [{"ddd": 0, "ind": i + 1, "ty": 2, "nm": f"frame {i + 1}", "refId": f"img_{i}", "sr": 1, "ip": i,
               "op": i + 1, "st": 0, "bm": 0, "ks": _lottie_ks_image(w, h)} for i in range(len(frames))]
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": len(frames), "w": w, "h": h, "nm": name, "ddd": 0,
            "assets": [_lottie_image(f, i) for i, f in enumerate(frames)], "layers": layers}


def lottie_motion(art, kind, count, amt, lo, size, fps, px, name="gptkit"):
    """Lottie of one embedded image driven by transform keyframes (smooth at any frame rate)."""
    k = min(1.0, px / max(size))
    w, h = art.size
    img = art.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    params = [_motion(kind, i / count, amt) for i in range(count + 1)]
    pivot = (w / 2, h if params[0][5] == "bottom" else h / 2)

    def track(vals):
        if all(v == vals[0] for v in vals):
            return {"a": 0, "k": vals[0] if len(vals[0]) > 1 else vals[0][0]}
        return {"a": 1, "k": [{"t": i, "s": v, "o": {"x": [0], "y": [0]}, "i": {"x": [1], "y": [1]}}
                              for i, v in enumerate(vals[:-1])] + [{"t": count, "s": vals[-1]}]}
    ks = {"a": {"a": 0, "k": [pivot[0] * k, pivot[1] * k, 0]},
          "p": {"s": True, "x": {"a": 0, "k": (pivot[0] - lo[0]) * k},
                "y": track([[(pivot[1] + p[3] * h - lo[1]) * k] for p in params])},
          "s": track([[p[0] * 100, p[1] * 100, 100] for p in params]),
          "r": track([[p[2]] for p in params]), "o": track([[p[4] * 100] for p in params])}
    layer = {"ddd": 0, "ind": 1, "ty": 2, "nm": name, "refId": "img_0", "sr": 1, "ip": 0, "op": count, "st": 0, "bm": 0,
             "ks": ks}
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": count, "w": round(size[0] * k), "h": round(size[1] * k), "nm": name,
            "ddd": 0, "assets": [_lottie_image(img, 0)], "layers": [layer]}


def _lottie_html(lot, bg, size=512):
    return ("<!doctype html><meta charset=utf-8><body style='margin:0;background:" + bg + "'>"
            f"<div id=a style='width:{size}px;height:{size}px'></div>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js'></script>"
            "<script>const f=parseFloat(location.hash.slice(1));const an=lottie.loadAnimation({container:a,renderer:'svg',"
            f"loop:true,autoplay:isNaN(f),animationData:{json.dumps(lot, separators=(',', ':'))}}});"
            "if(!isNaN(f))an.addEventListener('DOMLoaded',()=>an.goToAndStop(f,true))</script>")


def _fit_px(img, px):
    if max(img.size) <= px:
        return img
    k = px / max(img.size)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def _durations(n, fps, unit=1):
    """Per-frame ms (multiples of `unit`) whose sum is the exact loop length: one rounded value per frame drifts
    0.8% at 24 fps. GIF stores centiseconds, so its list uses unit=10."""
    return [(round((i + 1) * 1000 / fps / unit) - round(i * 1000 / fps / unit)) * unit for i in range(n)]


def _write_anim(out, anim, seq, fps, a, labels=None, lottie=None):
    """preview.gif, animated WebP/APNG, Lottie, the Android animation-list (frames + XML under android/) and the
    review sheet for a frame sequence."""
    formats = {f.strip() for f in a.formats.split(",") if f.strip()}
    durs = _durations(len(seq), fps)
    small = [_fit_px(f, a.px) for f in seq]
    wrote = []
    if "gif" in formats:
        bg = Image.new("RGBA", small[0].size, a.preview_bg)
        flat = [Image.alpha_composite(bg, f).convert("RGB") for f in small]
        # one palette for the whole loop (per-frame adaptive palettes flicker between frames)
        contact = Image.new("RGB", (flat[0].width * len(flat), flat[0].height))
        for i, f in enumerate(flat):
            contact.paste(f, (i * f.width, 0))
        pal = contact.quantize(256)
        gif = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in flat]
        gif[0].save(out / "preview.gif", save_all=True, append_images=gif[1:], duration=_durations(len(seq), fps, 10),
                    loop=0, disposal=2)
        wrote.append("preview.gif")
    if "webp" in formats:
        # Pillow's animated writer defaults to method=0 (fastest, largest); exact keeps RGB under transparent pixels
        small[0].save(out / f"{anim}.webp", save_all=True, append_images=small[1:], duration=durs, loop=0, quality=90,
                      method=6, exact=True)
        wrote.append(f"{anim}.webp")
    if "apng" in formats:
        small[0].save(out / f"{anim}.apng", format="PNG", save_all=True, append_images=small[1:], duration=durs, loop=0)
        wrote.append(f"{anim}.apng")
    if "lottie" in formats:
        lot = lottie or lottie_image_sequence(small, fps, anim)
        (out / f"{anim}.json").write_text(json.dumps(lot, separators=(",", ":")))
        (out / "preview.html").write_text(_lottie_html(lot, a.preview_bg))
        wrote += [f"{anim}.json", "preview.html"]
    # Android: pixel-sized frames belong in drawable-nodpi, and the animation-list must name files that exist
    res = _res_name(anim)
    if res != anim:
        print(f"  (android resource name: {res}, from {anim!r})")
    nodpi, drw = out / "android" / "drawable-nodpi", out / "android" / "drawable"
    for sub, pat in ((nodpi, f"{res}_frame_*.png"), (drw, f"{res}_anim.xml")):  # only this animation's files:
        sub.mkdir(parents=True, exist_ok=True)                                  # `rain_*` would also delete rain_pulse_*
        for old in sub.glob(pat):
            old.unlink()
    for i, f in enumerate(small):
        f.save(nodpi / f"{res}_frame_{i + 1:02d}.png")
    items = "\n".join(f'    <item android:drawable="@drawable/{res}_frame_{i + 1:02d}" android:duration="{durs[i]}" />'
                      for i in range(len(seq)))
    (drw / f"{res}_anim.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<animation-list xmlns:android="http://schemas.android.com/apk/res/android"\n'
        f'    android:oneshot="false">\n{items}\n</animation-list>\n')
    labels = labels or [f"f{i + 1}" for i in range(len(seq))]
    sheet([list(zip(labels, seq))[j:j + 8] for j in range(0, len(seq), 8)], out / "frames_review.png", T=200)
    print(f"{len(seq)} frames @ {fps:g} fps ({sum(durs)} ms loop) -> {out}  "
          f"({', '.join(wrote + ['frames_review.png', f'android/drawable/{res}_anim.xml'])})")
    verdict = _aapt2_tree(out / "android")  # the animation-list must compile next to its PNGs, hence the whole tree
    print(f"  {'! ' if 'FAILED' in verdict else ''}{verdict}")


def _frames_procedural(a, src, out):
    period = MOTIONS[a.procedural]
    if max(src.size) > 2048:  # every frame is a full-size RGBA copy: an 8000 px source ran out of memory
        k = 2048 / max(src.size)
        print(f"  ~ assets/{a.name}.png is {src.width}x{src.height}: animating a {round(src.width * k)}x{round(src.height * k)} copy")
        src = src.resize((max(1, round(src.width * k)), max(1, round(src.height * k))), Image.LANCZOS)
    fps = a.fps or (24 if period * 24 <= 48 else 12)
    count = a.count or max(8, round(period * fps))
    frames, params, lo, art = _procedural(src, a.procedural, count, a.amount)
    for i, f in enumerate(frames):
        f.save(out / f"frame_{i + 1:02d}.png")
    (out / "motion.json").write_text(json.dumps({
        "motion": a.procedural, "amount": a.amount, "seconds": round(count / fps, 3), "fps": round(fps, 3),
        "pivot": params[0][5],
        "frames": [{"scale_x": round(p[0], 4), "scale_y": round(p[1], 4), "rotation_deg": round(p[2], 3),
                    "translate_y_of_height": round(p[3], 4), "opacity": round(p[4], 3)} for p in params]},
        indent=1))
    lot = lottie_motion(art, a.procedural, count, a.amount, lo, frames[0].size, fps, a.px, f"{a.name}_{a.procedural}")
    _write_anim(out, f"{a.name}_{a.procedural}", frames, fps, a, lottie=lot)


def _opaque_share(img):
    """(share of pixels at alpha 255, whether all four corners are opaque): the signature of a render that ignored
    background=transparent (a gateway that rejects the parameter, or a model that painted a backdrop)."""
    a = np.asarray(img.getchannel("A"))
    return float((a == 255).mean()), bool(min(int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])) == 255)


def _decode_result(raw, what):
    """RGBA image of the bytes an image call returned; RuntimeError naming `what` when they are not an image
    (a gateway can answer with HTML or a truncated file)."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im.convert("RGBA")
    except (OSError, ValueError):
        raise RuntimeError(f"{what}: the response is not a decodable image ({len(raw)} bytes starting {raw[:24]!r})")


def _opaque_error(img, what, model=None):
    """One-line reason when an image asked for with a transparent background came back opaque, else None."""
    share, corners = _opaque_share(img)
    if share > 0.95 or corners:
        c = _config()
        return (f"{what}: the image came back with an opaque background ({share:.0%} of it is opaque; "
                f"background=transparent was ignored by {pick_image_model(model, 'precise', 'transparent')} at {c['base']})")
    return None


def _retryable(e):
    """Whether a caller should try an image call again after the API layer gave up: transport failures, 408/429 and
    5xx only. A 4xx (moderation, bad request) or an opaque/undecodable result would come back the same."""
    st, code = getattr(e, "status", None), str(getattr(e, "code", None))
    if code == "insufficient_quota" or code.startswith("billing"):  # _call never retries these either: no credit left
        return False
    return isinstance(e, ApiError) and (st in (0, 408, 429) or (st or 0) >= 500)


def _fail_note(what, e, printed, prompt=None):
    """Print the whole error the first time it is seen (code, server message, the prompt), one line afterwards."""
    msg = str(e).replace("\n", " ")
    code = getattr(e, "code", None)  # ApiError.code (C2); a local RuntimeError (opaque/undecodable result) has none
    if msg[:160] in printed:
        print(f"  ! {what} failed with the same error{f' ({code})' if code else ''}", flush=True)
        return
    printed.add(msg[:160])
    hint = "; rephrase --moving/--locked/--look" if code == "moderation_blocked" or "moderation" in msg else ""
    print(f"  ! {what} failed: {msg}{hint}" + (f"\n    prompt: {prompt[:200]!r}" if prompt and "prompt:" not in msg else ""),
          flush=True)


def _pool(jobs, fn, items):
    """Run fn over items in a thread pool; results as they finish (None dropped). Ctrl-C cancels the queued jobs
    instead of waiting for every submitted call to time out."""
    ex = cf.ThreadPoolExecutor(max(1, jobs))
    futs = [ex.submit(fn, it) for it in items]
    try:
        out = [r for r in (f.result() for f in cf.as_completed(futs)) if r is not None]
    except BaseException:  # Ctrl-C or a job's own error: the queued calls must not start before the exit
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return out


FRAME_REROLL = ("The attached image is frame {j} of {n} of a seamless looping animation of {moving} ({motion}). Draw "
                "frame {i}: the SAME element in exactly the same style, colours, gradients, glow, size and anchor, with "
                "only the small natural change of shape between neighbouring frames. Only this one element, centred, "
                "transparent background, nothing else.")


def _reroll(a, crops, qa, bad, ref, ref_mask, motion, out):
    """Redraw flagged frames from their nearest good neighbour; keep a redraw only if its QA badness drops."""
    good = [i for i in range(len(crops)) if i not in bad]
    if not good:
        print("  ! every frame is flagged; re-roll skipped (re-run the sheet instead)")
        return crops, qa

    def job(i):
        j = min(good, key=lambda g: min(abs(g - i), len(crops) - abs(g - i)))
        prompt = FRAME_REROLL.format(i=i + 1, j=j + 1, n=len(crops), moving=a.moving, motion=motion)
        raw = edit(_png_bytes(_square(crops[j])[0]), prompt, "1024x1024", "transparent", a.model)[0]
        name = f"reroll_f{i + 1:02d}.png"
        img = _decode_result(raw, name)
        err = _opaque_error(img, name, a.model)
        if err:
            raise RuntimeError(err)
        (out / name).write_bytes(raw)                    # each redraw is kept as soon as it arrives
        return i, j, img
    ex = cf.ThreadPoolExecutor(2)
    futs = {ex.submit(job, i): i for i in bad}
    redraws = []
    try:
        for f in cf.as_completed(futs):
            try:
                redraws.append(f.result())
            except (ApiError, RuntimeError) as e:
                print(f"  ! frames: the re-roll f{futs[f] + 1} call failed: {e}")
    except BaseException:  # Ctrl-C or an unexpected error: the queued calls must not start before the exit
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    for i, j, img in sorted(redraws, key=lambda r: r[0]):
        bb = _robust_box(np.asarray(img.getchannel("A")))
        if not bb:
            print(f"  re-roll f{i + 1}: came back empty, discarded")
            continue
        img = img.crop(bb)
        k = math.sqrt(_alpha_stats(crops[j])[1] / _alpha_stats(img)[1])
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
        trial = crops[:i] + [img] + crops[i + 1:]
        tq = _frame_qa(_align_frames(trial, a.anchor), ref, ref_mask)
        old, new = _badness(qa["frames"][i]), _badness(tq["frames"][i])
        keep = new < old
        print(f"  re-roll f{i + 1}: badness {old:.2f} -> {new:.2f} {'KEPT' if keep else 'discarded'}")
        if keep:
            crops, qa = trial, tq
    return crops, qa


FRAME_FORMATS = ("gif", "webp", "apng", "lottie")


def cmd_frames(a):
    d = Path(a.dir)
    src = _open_image(d / "assets" / f"{a.name}.png", "asset", "run forge first, or check the name")
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    if not formats or set(formats) - set(FRAME_FORMATS):
        sys.exit(f"frames: --formats takes a comma list of {','.join(FRAME_FORMATS)}, got {a.formats!r}")
    if a.count is not None and a.count > 240:
        sys.exit(f"frames: --count {a.count} is more than 240 frames (10 s at 24 fps); a loop needs far fewer")
    if a.fps is not None and not 1 <= a.fps <= 120:
        sys.exit(f"frames: --fps {a.fps:g} must be between 1 and 120")
    if a.amount > 10:
        sys.exit(f"frames: --amount {a.amount:g} is a strength multiplier around 1 (at most 10)")
    if not np.asarray(src.getchannel("A")).any():
        sys.exit(f"frames: assets/{a.name}.png is fully transparent: there is nothing to animate")
    share, corners = _opaque_share(src)
    if share > 0.9 and corners:
        print(f"  ! assets/{a.name}.png has no transparent background ({share:.0%} opaque): the whole rectangle will "
              "move; forge it with transparent=true or cut the backdrop out first")
    out = d / "frames" / (f"{a.name}_{a.procedural}" if a.procedural else a.name)
    out.mkdir(parents=True, exist_ok=True)
    stale = ["frame_*.png", "moving_f*.png"] + ([] if a.reuse else ["reroll_f*.png", "static.png", "qa.json"])
    for pat in stale:
        for old in out.glob(pat):
            old.unlink()
    if a.count is not None and a.count < 2:
        sys.exit(f"frames: --count must be at least 2, got {a.count}")
    if a.procedural:
        return _frames_procedural(a, src, out)
    if not a.moving:
        sys.exit("frames: --moving is required (or use --procedural)")
    count, fps = a.count or 8, a.fps or 12
    cols = a.cols
    if cols > count:
        print(f"  (--cols {cols} > --count {count}: using {count} columns)")
        cols = count
    rows_n = math.ceil(count / cols)
    sheet_size = _size_for(cols * src.width, rows_n * src.height)
    canvas = {"1536x1024": "wide landscape", "1024x1536": "tall portrait"}.get(sheet_size, "square")
    motion = a.motion or "a gentle, natural looping motion"
    plural = lambda n, w: f"{n} {w}{'s' if n != 1 else ''}"  # noqa: E731
    moving_prompt = (f"The attached image is an APPROVED asset. Produce ONLY its moving part — {a.moving}"
                     f"{' with NO ' + a.static + ' at all' if a.static else ''} — as {count} animation frames of {motion}, "
                     f"laid out as a sprite sheet of {plural(cols, 'column')} x {plural(rows_n, 'row')} of equal cells on a "
                     f"{canvas} canvas (left-to-right, top-to-bottom). "
                     "Every frame: same style, colours, gradients and glow as the attached asset; same overall size and "
                     f"the SAME anchor point at the {'bottom centre' if a.anchor == 'bottom' else 'centre'} of its cell; "
                     "only the natural motion changes, and the "
                     "last frame flows back into the first. Transparent background everywhere. Absolutely no grid lines, "
                     "dividers, borders, numbers or labels. Leave empty transparent space between cells.")
    jobs = {}
    if a.reuse != "all":
        jobs["sheet"] = (moving_prompt, sheet_size)
    if a.static and not a.reuse:
        jobs["static"] = (f"The attached image is an APPROVED asset. Keep ONLY {a.static}, exactly as drawn: same size, "
                          f"position, angle and colours on the same canvas. Remove {a.moving} completely and repair "
                          "what it covered. Transparent background.", "1024x1024")
    for k in {"sheet", "static"} - set(jobs):
        if (k == "sheet" or a.static) and not (out / f"{k}_raw.png").exists():
            sys.exit(f"frames: --reuse needs {out / (k + '_raw.png')} from a previous run")
    _plan(len(jobs), 0, image_model=a.model, task="precise", background="transparent", edits=True)
    sq_src, sq_off = _square(src)
    ex = cf.ThreadPoolExecutor(2)
    futs = {ex.submit(edit, _png_bytes(sq_src if k == "static" else src), p, sz, "transparent", a.model): k
            for k, (p, sz) in jobs.items()}
    failed = []
    try:
        for f in cf.as_completed(futs):               # each result is written as it arrives, whatever the other does
            k = futs[f]
            try:
                raw = f.result()[0]
                img = _decode_result(raw, f"{k}_raw.png")
                err = _opaque_error(img, f"{k}_raw.png", a.model)
                if err:
                    (out / f"{k}_raw.opaque.png").write_bytes(raw)
                    raise RuntimeError(f"{err}; saved as {k}_raw.opaque.png")
                (out / f"{k}_raw.png").write_bytes(raw)
            except (ApiError, RuntimeError) as e:
                failed.append(k)
                print(f"  ! frames: the {k} call failed: {e}")
    except BaseException:  # Ctrl-C or an unexpected error: the queued call must not start before the exit
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    if failed:
        kept = [k for k in jobs if k not in failed]
        sys.exit(f"frames: the {' and '.join(failed)} call{'s' if len(failed) > 1 else ''} failed (see above)"
                 + (f"; {kept[0]}_raw.png was kept: re-run with --reuse {kept[0]}" if kept == ["static"] else ""))
    crops = _slice_sheet(_remove_dividers(_rgba(out / "sheet_raw.png")), count, cols)
    if not crops:
        sys.exit("frames: no frames found in the sheet")
    static = ref_mask = None
    if a.static:
        sq_static = _register(_rgba(out / "static_raw.png").resize(sq_src.size, Image.LANCZOS), sq_src)
        static = sq_static.crop((sq_off[0], sq_off[1], sq_off[0] + src.width, sq_off[1] + src.height))
        static.save(out / "static.png")
        ref_mask = _moving_mask(static)
    qa = _frame_qa(_align_frames(crops, a.anchor), src, ref_mask)
    if len(crops) != count:
        qa["sequence"]["flags"].append("frame-count")
        print(f"  ! only {len(crops)} of {count} frames: the loop timing changes and the seam may jump; re-run the sheet "
              "(no --reuse) or accept it")
    bad = _qa_report(qa)
    if a.reroll and bad:
        crops, qa = _reroll(a, crops, qa, bad, src, ref_mask, motion, out)
        _qa_report(qa, " after re-roll")
    frames = _align_frames(crops, a.anchor)
    if a.match_colors:
        frames = _match_colors(frames, src, ref_mask)
        flags = qa["sequence"]["flags"]
        qa = _frame_qa(frames, src, ref_mask)
        qa["sequence"]["flags"] += [f for f in flags if f == "frame-count"]
        _qa_report(qa, " after colour match")
    for i, f in enumerate(frames):
        f.save(out / f"moving_f{i + 1:02d}.png")
    comps, reg = _composite(src, static, frames, a.static_order) if static is not None else ([], None)
    if reg is not None:
        qa["sequence"]["registration_iou"] = round(float(reg), 3)
        if reg < 0.3:
            qa["sequence"]["flags"].append("misregistered")
            print(f"  ! MISREGISTERED: the moving frames match the asset's moving part poorly (IoU {reg:.2f}). Check "
                  "static.png (it must contain only the static part), describe --moving/--static more precisely, or "
                  "re-run with --reuse static; moving_fNN.png are still usable on their own")
    (out / "qa.json").write_text(json.dumps(qa, indent=1))
    seq = comps or frames
    for i, f in enumerate(seq):
        f.save(out / f"frame_{i + 1:02d}.png")
    labels = [f"f{r['frame']} {' '.join(r['flags'])}".strip() for r in qa["frames"]]
    _write_anim(out, a.name, seq, fps, a, labels)
