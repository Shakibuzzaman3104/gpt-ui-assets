# gptkit part 01_imaging: image utilities and perceptual scoring.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---------------------------------------------------------------- image utils

def _rgba(p):
    """Open as RGBA: EXIF orientation applied, 16-bit greyscale (I;16*, I) scaled to 8 bits first (plan contract C5)."""
    from PIL import ImageOps
    im = ImageOps.exif_transpose(Image.open(p))
    if im.mode == "I" or im.mode.startswith("I;16"):
        a = np.asarray(im).astype(np.float64)
        if a.size and a.max() > 255:
            a = a / 257.0  # 16-bit range -> 8-bit (65535/257 = 255)
        im = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8), "L")
    return im.convert("RGBA")


def content_box(img, thr=24):
    return img.getchannel("A").point(lambda v: 255 if v > thr else 0).getbbox()


def trim(img, thr=8, pad=0.0):
    bb = content_box(img, thr) or (0, 0, img.width, img.height)
    c = img.crop(bb)
    if pad:
        p = int(max(c.size) * pad)
        out = Image.new("RGBA", (c.width + 2 * p, c.height + 2 * p), (0, 0, 0, 0))
        out.alpha_composite(c, (p, p))
        return out
    return c


def ref_crop(mock, bbox, margin=0.6, long_side=1024, center=False):
    """The element with `margin` context on each side, long side resized to long_side. center=True pads the crop
    with the design's backdrop colour where the mockup edge cuts the margin, so the element is always centred."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    full = (int(x - w * margin), int(y - h * margin), int(x + w * (1 + margin)), int(y + h * (1 + margin)))
    box = (max(0, full[0]), max(0, full[1]), min(mock.width, full[2]), min(mock.height, full[3]))
    c = mock.crop(box).convert("RGB")
    if center and box != full:
        padded = Image.new("RGB", (full[2] - full[0], full[3] - full[1]), _backdrop(mock, bbox))
        padded.paste(c, (box[0] - full[0], box[1] - full[1]))
        c = padded
    s = long_side / max(c.size)
    return c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)


def checker(w, h, c=16):
    im = Image.new("RGB", (w, h), "#bdbdbd")
    d = ImageDraw.Draw(im)
    for y in range(0, h, c):
        for x in range(0, w, c):
            if (x // c + y // c) % 2:
                d.rectangle([x, y, x + c - 1, y + c - 1], fill="#ececec")
    return im


def _tile(img, T, label=None, bg="checker"):
    im = img.convert("RGBA").copy() if img is not None else Image.new("RGBA", (1, 1))
    im.thumbnail((T, T))
    base = checker(T, T) if bg == "checker" else Image.new("RGB", (T, T), bg)
    base.paste(im, ((T - im.width) // 2, (T - im.height) // 2), im)
    if label:
        d = ImageDraw.Draw(base)
        lines, rest = [], str(label)
        while rest and len(lines) < 2:
            cut = len(rest)
            while cut > 1 and d.textlength(rest[:cut]) > T - 6:
                cut = rest.rfind(" ", 0, cut) if " " in rest[:cut - 1] else cut - 1
            lines.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        d.rectangle([0, 0, T, 2 + 12 * len(lines)], fill="white")
        for i, ln in enumerate(lines):
            d.text((3, 1 + 12 * i), ln, fill="black")
    return base


def sheet(rows, out, T=220):
    """rows: list of lists of (label, PIL image)."""
    cols = max(len(r) for r in rows)
    im = Image.new("RGB", (cols * T, len(rows) * T), "white")
    for r, row in enumerate(rows):
        for c, (label, img) in enumerate(row):
            im.paste(_tile(img, T, label), (c * T, r * T))
    im.save(out)
    return out


# ---------------------------------------------------------------- scoring

def _mean_diff(a, b):
    return sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3


def _edge_var(img):
    return ImageStat.Stat(img.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]


_M_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])
WORK = 128  # long side of the working view used by the perceptual metrics


def _lab(rgb):
    c = np.asarray(rgb, np.float64)[..., :3] / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M_XYZ.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    c = np.clip(xyz @ np.linalg.inv(_M_XYZ).T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def _morph(mask, *ops):
    im = Image.fromarray(np.uint8(mask) * 255)
    for op in ops:
        im = im.filter(ImageFilter.MinFilter(3) if op == "min" else ImageFilter.MaxFilter(3))
    return np.asarray(im) > 127


def _clean(mask):
    return _morph(mask, "min", "max", "max", "min")


def _backdrop(mock, bbox, k=4):
    """The design's backdrop colour: median of a k-px ring OUTSIDE the bbox (clamped to the mockup). Only when the
    bbox is the whole mockup does the box's own edge serve, so tight or tiny bboxes never sample the element itself."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    M = mock.convert("RGB")
    box = (max(0, x - k), max(0, y - k), min(M.width, x + w + k), min(M.height, y + h + k))
    a = np.asarray(M.crop(box)).astype(int)
    inner = np.zeros(a.shape[:2], bool)
    inner[y - box[1]:y - box[1] + h, x - box[0]:x - box[0] + w] = True
    ring = a[~inner] if (~inner).any() else np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(ring, 0))


def _backdrop_edge_frac(asset, backdrop, thr=36):
    """Fraction of the asset's silhouette boundary (alpha > 200 minus a 5 px erosion at 128 px) within thr RGB of
    the backdrop. A kept card/circle puts backdrop colour on the outline; a light icon with its own outline does not."""
    g = asset.crop(content_box(asset, 40) or (0, 0, asset.width, asset.height))
    g.thumbnail((WORK, WORK))
    a = np.asarray(g).astype(int)
    solid = a[..., 3] > 200
    eroded = np.asarray(Image.fromarray(np.uint8(solid) * 255).filter(ImageFilter.MinFilter(5))) > 127
    edge = solid & ~eroded
    if edge.sum() < 8:
        return 0.0
    near = np.abs(a[..., :3] - np.array(backdrop)).sum(-1) < thr
    return float((near & edge).sum() / edge.sum())


def _design_view(mock, bbox):
    """The design element at WORK scale in Lab, its backdrop and a foreground mask segmented from it."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    s = WORK / max(w, h)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    arr = np.asarray(mock.crop((x, y, x + w, y + h)).convert("RGB").resize(size, Image.LANCZOS))
    backdrop = _backdrop(mock, bbox)
    L, bl = _lab(arr), _lab(np.array(backdrop))
    dist = np.linalg.norm(L - bl, axis=-1)
    thr = min(22.0, max(8.0, 0.45 * float(np.percentile(dist, 95))))
    return {"L": L, "mask": _clean(dist > thr), "backdrop": backdrop, "bl": bl, "thr": thr, "size": size, "up": s}


def _align_view(asset, view):
    """Scale/offset of the asset (composited on the design backdrop) that best overlaps the design silhouette."""
    w, h = view["size"]
    M = max(w, h) // 5
    g = asset.crop(content_box(asset, 40) or (0, 0, asset.width, asset.height))
    if max(g.size) > 4 * WORK:
        k = 4 * WORK / max(g.size)
        g = g.resize((max(1, round(g.width * k)), max(1, round(g.height * k))), Image.LANCZOS)
    s0 = min(w / g.width, h / g.height)
    dm = view["mask"]
    search = dm.sum() >= 16
    best = None
    for s in (np.arange(0.88, 1.14, 0.025) if search else [1.0]):
        gw, gh = max(1, round(g.width * s0 * s)), max(1, round(g.height * s0 * s))
        big = Image.new("RGBA", (w + 2 * M, h + 2 * M), view["backdrop"] + (255,))
        big.alpha_composite(g.resize((gw, gh), Image.LANCZOS), ((w + 2 * M - gw) // 2, (h + 2 * M - gh) // 2))
        big = big.convert("RGB")
        if view["up"] > 1.25:  # blur like the upscaled low-res design so colours at edges compare fairly
            small = (max(1, round(big.width / view["up"])), max(1, round(big.height / view["up"])))
            big = big.resize(small, Image.LANCZOS).resize(big.size, Image.LANCZOS)
        BL = _lab(np.asarray(big))
        BM = _clean(np.linalg.norm(BL - view["bl"], axis=-1) > view["thr"])
        if search:  # IoU of every offset at once: FFT correlation for |am & dm|, an integral image for |am|
            lo, hi = M + (-M // 2), M + M // 2 + 1
            inter = np.rint(_corr_valid(BM.astype(np.float64), dm.astype(np.float64)))[lo:hi, lo:hi]
            S = np.pad(BM.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            area = (S[h:, w:] - S[:-h, w:] - S[h:, :-w] + S[:-h, :-w])[lo:hi, lo:hi]
            iou = inter / np.maximum(area + dm.sum() - inter, 1)
            k = int(np.argmax(iou))  # first maximum in (dy, dx) row-major order, like the former loop
            dy, dx = divmod(k, iou.shape[1])
            dy, dx, val = dy + (-M // 2), dx + (-M // 2), float(iou.flat[k])
        else:
            dy = dx = 0
            am0 = BM[M:M + h, M:M + w]
            val = (am0 & dm).sum() / max((am0 | dm).sum(), 1)
        if best is None or val > best[0]:
            best = (val, BL[M + dy:M + dy + h, M + dx:M + dx + w], BM[M + dy:M + dy + h, M + dx:M + dx + w])
    return best


def _corr_valid(A, B):
    """sum(A[i:i+h, j:j+w] * B) for every valid (i, j), via a circular FFT whose tail is wrap-free."""
    (H, W), (h, w) = A.shape, B.shape
    full = np.fft.irfft2(np.fft.rfft2(A, (H, W)) * np.fft.rfft2(B[::-1, ::-1], (H, W)), (H, W))
    return full[h - 1:, w - 1:]


def _qemd(a, b, n=32):
    """Mean 1-D earth mover's distance of the L, a, b distributions (quantile form)."""
    q = np.linspace(0.02, 0.98, n)
    return float(np.mean([np.abs(np.quantile(a[:, i], q) - np.quantile(b[:, i], q)).mean() for i in range(3)]))


def _fill_view(asset, view):
    """An opaque asset cropped to the element's aspect and resized onto the design view, in Lab."""
    w, h = view["size"]
    tw, th = asset.size
    if tw / th > w / h:
        nw = round(th * w / h)
        asset = asset.crop(((tw - nw) // 2, 0, (tw - nw) // 2 + nw, th))
    else:
        nh = round(tw * h / w)
        asset = asset.crop((0, (th - nh) // 2, tw, (th - nh) // 2 + nh))
    return _lab(np.asarray(asset.convert("RGB").resize((w, h), Image.LANCZOS)))


def perceptual(mock, bbox, asset, transparent=True):
    """iou: silhouette overlap after alignment; de: mean CIELAB distance; col: Lab distribution distance."""
    view = _design_view(mock, bbox)
    L, dm = view["L"], view["mask"]
    if transparent:
        iou, AL, am = _align_view(asset, view)
        if dm.sum() >= 16:
            return {"iou": float(iou), "de": float(np.linalg.norm(AL - L, axis=-1)[am | dm].mean()),
                    "col": _qemd(AL[am], L[dm]) if am.sum() > 8 else 50.0}
    else:
        AL = _fill_view(asset, view)
    # opaque art, or an element that can't be separated from its backdrop: compare the whole box
    return {"iou": None, "de": float(np.linalg.norm(AL - L, axis=-1).mean()), "col": _qemd(AL.reshape(-1, 3), L.reshape(-1, 3))}


def _fid(diff, p):
    return 0.3 * diff + 0.5 * p["de"] + p["col"] + (40 * (1 - p["iou"]) if p["iou"] is not None else 0)


def color_match(mock, bbox, asset, strength=0.75):
    """Reinhard-style transfer of the design element's Lab mean/std onto a transparent asset; alpha is kept."""
    view = _design_view(mock, bbox)
    _, AL, am = _align_view(asset, view)
    src, dst = AL[_morph(am, "min")], view["L"][_morph(view["mask"], "min")]
    if len(src) < 16 or len(dst) < 16:
        return asset
    k = np.clip(dst.std(0) / np.maximum(src.std(0), 1e-3), 0.75, 1.33).astype(np.float32)
    a = np.asarray(asset.convert("RGBA"))
    # the transfer is a per-colour map: compute it on the unique colours only (a 4096^2 asset no longer needs 3 GB)
    keys = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    rgb = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], -1).astype(np.uint8)
    L = _lab(rgb).astype(np.float32)
    L = L + np.float32(strength) * (dst.mean(0).astype(np.float32) + (L - src.mean(0).astype(np.float32)) * k - L)
    out = np.dstack([_lab_to_rgb(L)[inv.reshape(a.shape[:2])], a[..., 3]])
    out[a[..., 3] == 0, :3] = 0
    return Image.fromarray(out, "RGBA")


def score(mock, bbox, asset, ref=None, transparent=True):
    """Fidelity vs the mockup element plus copy/crop red flags. Lower 'fid' (and 'diff') = closer to the design."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    crop = mock.crop((x, y, x + w, y + h)).convert("RGB")
    backdrop = _backdrop(mock, bbox)
    flags = []
    A = asset.getchannel("A")
    transp = sum(A.histogram()[:8]) / (asset.width * asset.height)
    cb = content_box(asset, 40)
    empty = cb is None or (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.002 * asset.width * asset.height
    bb = cb or (0, 0, asset.width, asset.height)
    g = asset.crop(bb)
    if empty:
        flags.append("EMPTY")
    if transparent:
        if transp < 0.15:
            flags.append("OPAQUE")
        if not empty and (bb[0] <= 2 or bb[1] <= 2 or bb[2] >= asset.width - 2 or bb[3] >= asset.height - 2):
            flags.append("CUT-OFF")
        if not empty and _backdrop_edge_frac(asset, backdrop) > 0.5:
            flags.append("BACKDROP")
    s = min(w / g.width, h / g.height)
    gs = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), backdrop + (255,))
    canvas.alpha_composite(gs, ((w - gs.width) // 2, (h - gs.height) // 2))
    diff = _mean_diff(canvas, crop)
    flat = Image.new("RGBA", asset.size, backdrop + (255,))
    flat.alpha_composite(asset)
    small = flat.convert("RGB").resize((64, 64), Image.LANCZOS)
    if _mean_diff(small, mock.convert("RGB").resize((64, 64), Image.LANCZOS)) < 10:
        flags.append("COPY:returned-mockup")
    if ref is not None and _mean_diff(small, ref.convert("RGB").resize((64, 64), Image.LANCZOS)) < 8:
        flags.append("COPY:returned-reference")
    sharp = _edge_var(g) / max(_edge_var(crop.resize(g.size, Image.BICUBIC)), 1e-6)
    warn = ["soft"] if sharp < 2 else []
    p = perceptual(mock, bbox, asset, transparent)
    if p["iou"] is not None and p["iou"] < 0.5:
        warn.append("silhouette")
    return {"fid": round(_fid(diff, p), 1), "diff": round(diff, 1), "iou": None if p["iou"] is None else round(p["iou"], 3),
            "de": round(p["de"], 1), "col": round(p["col"], 1), "flags": flags, "warn": warn,
            "sharp_ratio": round(sharp, 1), "transparent_frac": round(transp, 3),
            "aspect_err": round(abs((g.width / g.height) / (w / h) - 1), 3)}


FLAG_WEIGHT = {"EMPTY": 1000, "COPY": 100, "OPAQUE": 10, "BACKDROP": 10, "CUT-OFF": 3}


def _flag_weight(flags):
    return sum(FLAG_WEIGHT.get(f.split(":")[0], 1) for f in flags)


def _rank(sc, meta):
    """Sort key: flag severity (EMPTY 1000 > COPY 100 > OPAQUE, BACKDROP 10 > CUT-OFF 3), then fid (+6 for low quality)."""
    return (_flag_weight(sc["flags"]), sc.get("fid", sc["diff"]) + (6 if (meta or {}).get("quality") == "low" else 0))


def _unusable(sc):
    """The flag that makes a candidate unshippable (EMPTY or COPY:*), or None."""
    return next((f for f in sc["flags"] if f == "EMPTY" or f.startswith("COPY")), None)
