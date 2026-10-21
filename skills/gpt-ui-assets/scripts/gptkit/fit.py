"""Small-phone and large-text fit of single-line text in a phone mockup: GPT proposes the lines, the pixels decide."""
import numpy as np

from .core import _clip_bbox, ask
from .measure import _d2c_runs
from .trace_rig import _label

SMALL = {"android": (360, 640), "ios": (375, 667)}  # small phone, logical units: compact Android, iPhone SE
SCALES = {  # default size -> size at a large and the largest text setting; big text grows less
    "android": ([8, 10, 12, 14, 18, 20, 24, 30, 100],  # Android 14 FontScaleConverterFactory at 130% and 200%
                {"large": [10.4, 13, 15.6, 18.8, 21.6, 23.6, 26.4, 30, 100], "max": [16, 20, 24, 26, 30, 34, 36, 38, 100]}),
    "ios": ([11, 12, 13, 15, 16, 17, 20, 22, 28, 34],  # HIG Dynamic Type text styles at xxxLarge and AX2
            {"large": [17, 18, 19, 21, 22, 23, 26, 28, 34, 40], "max": [24, 26, 27, 30, 32, 33, 37, 39, 43, 48]})}
FIXED = {"ios": ("tab", "segment")}  # UIKit tab bars and segmented controls keep their text size
GAP, MARGIN, SLACK = 12, 16, 4  # dp: floors between items and at edges; a shortfall under SLACK is noise
PADS = {"button": 16, "segment": 8, "tab": 2, "chips": MARGIN, "tile": MARGIN, "row": MARGIN, "text": MARGIN}

FIT_PROMPT = """This is a {platform} phone screen ({W}x{H} px, {dp} logical units wide). List the single-line text that
could run out of room on a smaller phone or at the largest text size: button labels, tab and navigation
labels, segmented-control segments, chips, stat values and their labels, prices, times, the value side of
label-value rows, and a hero figure or one-word heading. Skip paragraphs and titles that can wrap to two lines,
lettering drawn as artwork (logos, wordmarks, game titles, letter tiles), and rows holding a slider or progress bar.
One entry per container whose width limits the text: kind (button, segment, tab for a tab or navigation-bar item,
chips for a row of chips, tile, row for a list or card row, text for text placed directly on the screen), the
container's bbox (for chips and text directly on the screen: the full screen width), every single-line text inside it
with a tight bbox, and share: how many equal containers sit side by side across the screen's
width (1 when this container spans the width). Bboxes in this image's pixels."""
FIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["lines"],
              "properties": {"lines": {"type": "array", "items": {
                                 "type": "object", "additionalProperties": False, "required": ["kind", "container", "share", "texts"],
                                 "properties": {
                                     "kind": {"type": "string", "enum": list(PADS)},
                                     "container": {"type": "object", "additionalProperties": False,
                                                   "required": ["x", "y", "w", "h"],
                                                   "properties": {k: {"type": "integer"} for k in "xywh"}},
                                     "share": {"type": "integer"},
                                     "texts": {"type": "array", "items": {
                                         "type": "object", "additionalProperties": False, "required": ["text", "bbox"],
                                         "properties": {"text": {"type": "string"},
                                                        "bbox": {"type": "object", "additionalProperties": False,
                                                                 "required": ["x", "y", "w", "h"],
                                                                 "properties": {k: {"type": "integer"} for k in "xywh"}}}}}}}}}}


def _grow(size, level, platform="android"):
    """Size factor of text at a text-size level; outside the table the end ratio holds."""
    xs, t = SCALES.get(platform, SCALES["android"])[0], SCALES.get(platform, SCALES["android"])[1][level]
    if size <= xs[0] or size >= xs[-1]:
        return t[0] / xs[0] if size <= xs[0] else t[-1] / xs[-1]
    return float(np.interp(size, xs, t)) / size


def _ink_em(text):
    """Ink height of a string in em: caps and digits only, or with descenders and tall marks."""
    return 0.92 if any(ch in "gjpqy$,;()[]|/@" for ch in text) else 0.74 if not any(ch.islower() for ch in text) or \
        any(ch in "bdfhklt" for ch in text) else 0.55


def _rows(boxes):
    """Text boxes grouped into lines: boxes whose vertical centres fall inside each other's span share a line."""
    rows = []
    for b in sorted(boxes, key=lambda b: b["y"] + b["h"] / 2):
        cy = b["y"] + b["h"] / 2
        for r in rows:
            if r[0]["y"] <= cy <= r[0]["y"] + r[0]["h"]:
                r.append(b)
                break
        else:
            rows.append([b])
    return rows


def measure_line(rgb, k, box, texts, pad_dp=MARGIN, strings=None, slot=None):
    """Free width (px) on one line of text inside its container, with each gap kept down to its floor (the edge gaps
    to pad_dp, the gaps between items to GAP), plus each text's ink width and estimated size (dp). None when the line
    can't be measured (busy background, boxes that don't hold text)."""
    H, W = rgb.shape[:2]
    tb = [b for b in (_clip_bbox(t, W, H) for t in texts) if b]
    c = _clip_bbox(box, W, H)
    if not tb or not c:
        return None
    y0, y1 = max(min(b["y"] for b in tb), c["y"]), min(max(b["y"] + b["h"] for b in tb), c["y"] + c["h"])
    if y1 - y0 < 4:
        return None
    px = rgb[y0:y1].astype(np.float64)
    out = np.ones(W, bool)
    for b in tb:
        out[max(0, b["x"] - 2):b["x"] + b["w"] + 2] = False
    cx0, cx1 = c["x"], c["x"] + c["w"]
    around = px[:, cx0:cx1][:, out[cx0:cx1]].reshape(-1, 3)
    if len(around) < 20:
        return None
    bg = np.median(around, 0)
    if np.mean(np.linalg.norm(around - bg, axis=-1) < 20) < 0.6:
        return None
    dist = np.linalg.norm(px - bg, axis=-1)
    lh = y1 - y0
    wa, wb = max(0, y0 - lh // 2), min(H, y1 + lh // 2)
    win = np.linalg.norm(rgb[wa:wb].astype(np.float64) - bg, axis=-1) > 40
    win &= ~((win[:, cx0:cx1].mean(1) > 0.7)[:, None])  # a border or fill crossing the line is not text
    lab, n = _label(win)
    if n:
        ys = np.arange(wa, wb)[:, None].repeat(W, 1)
        lo = np.full(n + 1, H); hi = np.zeros(n + 1, int)
        np.minimum.at(lo, lab.ravel(), ys.ravel()); np.maximum.at(hi, lab.ravel(), ys.ravel())
        ring = (lo < y0 - lh // 4) & (hi > y1 + lh // 4)  # an outline runs above and below the line; glyphs don't
        ring[0] = False
        clean = win & ~ring[lab]
    else:
        clean = win
    ink = win[y0 - wa:y1 - wa].sum(0) >= 2  # outlines stay: they bound the items
    glyphs = clean[y0 - wa:y1 - wa].sum(0) >= 2
    edge = (dist > 15).mean(0) >= 0.9  # an edge (fill boundary, outline) crosses the band and continues above and below it
    for a, b in ((y0 - lh // 2, y0 - lh // 4), (y1 + lh // 4, y1 + lh // 2)):
        a, b = max(0, a), min(H, b)
        if b > a:
            edge &= (np.linalg.norm(rgb[a:b].astype(np.float64) - bg, axis=-1) > 15).mean(0) >= 0.9
    tol = int(0.03 * W)
    left, right = _snap(edge, cx0, tol, -1, W), _snap(edge, cx1 - 1, tol, 1, W) + 1
    if left == cx0 and right == cx1 and slot:  # no visible edge: an equal share of the screen around the text
        mid = sum(b["x"] + b["w"] / 2 for b in tb) / len(tb)
        left, right = max(0, int(mid - slot / 2)), min(W, int(mid + slot / 2))
    ems = [_glyph_h(clean[:, b["x"] + b["w"] // 4:b["x"] + 3 * b["w"] // 4 + 1], y0 - wa, y1 - wa) / (_ink_em(sv) if sv else 0.85)
           for b, sv in zip(tb, strings or [""] * len(tb))]
    merge = max(2, int(0.45 * float(np.median(ems))))  # bridges word spaces, not the gaps between items
    items = [(s + left, e + left) for s, e in _d2c_runs(ink[left:right], merge=merge)]
    words = [(s + left, e + left) for s, e in _d2c_runs(glyphs[left:right], merge=merge)]
    near = lambda s, e: any(min(e, b["x"] + b["w"]) - max(s, b["x"]) > 0 for b in tb)
    border = lambda s, e: e - s <= 3 * k and (s <= left + 2 * k or e >= right - 2 * k) and not near(s, e)
    items = [(s, e) for s, e in items if not border(s, e)]
    if not items:
        return None
    textual = []
    for b, sv in zip(tb, strings or [""] * len(tb)):
        cx = b["x"] + b["w"] / 2
        hit = [it for it in words if it[0] <= cx < it[1]] or \
              sorted(words, key=lambda it: max(0.0, it[0] - cx, cx - it[1]))[:1]
        if not hit:
            continue
        s, e = hit[0]
        if max(0.0, s - cx, cx - e) > 0.5 * b["w"]:
            continue
        em = _ink_em(sv) if sv else 0.85
        textual.append({"w": int(e - s), "size_dp": round(_glyph_h(clean[:, s:e], y0 - wa, y1 - wa) / k / em, 1)})
    if not textual:
        return None
    lg, rg = items[0][0] - left, right - items[-1][1]
    pad = min(pad_dp * k, lg, rg)  # the design's own padding: its smaller edge gap, at most pad_dp
    free = lg + rg - 2 * pad + sum(max(0, b[0] - a[1] - GAP * k) for a, b in zip(items, items[1:]))
    return {"left": int(left), "right": int(right), "free": float(free), "texts": textual, "items": len(items)}


def _glyph_h(win, y0, y1):
    """Height (px) of the ink in a column strip: the unbroken run of inked rows through the line, looked for in a window
    half a line taller on each side, so a box that clips the glyphs doesn't shrink them."""
    rows = win.any(1)
    mid = (y0 + y1) // 2
    if not rows.any():
        return y1 - y0
    if not rows[mid]:
        near = np.flatnonzero(rows)
        mid = int(near[np.argmin(np.abs(near - mid))])
    top = bot = mid
    while top > 0 and rows[top - 1]:
        top -= 1
    while bot < len(rows) - 1 and rows[bot + 1]:
        bot += 1
    return bot - top + 1


def _snap(edge, x, tol, step, W):
    """The container edge near x: the nearest column (within tol) whose outward neighbour is an edge column, else x."""
    best = None
    for i in range(max(0, x - tol), min(W - 1, x + tol) + 1):
        j = i + step
        if 0 <= j < W and not edge[i] and edge[j] and (best is None or abs(i - x) < abs(best - x)):
            best = i
    return best if best is not None else x


def fit_lines(img, dp, platform, lines):
    """Each line's shortfall (dp) on a small phone and at a large and the largest text size; failures only where even generous padding
    and gaps can't absorb it. Containers that hug their content are skipped: they grow instead of clipping."""
    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    k = W / dp
    small_w = SMALL.get(platform, SMALL["android"])[0]
    out = []
    for ln in lines:
        kind = ln.get("kind") if ln.get("kind") in PADS else "text"
        share = max(1, int(ln.get("share") or 1))
        boxes = {id(t["bbox"]): t.get("text", "") for t in ln["texts"]}
        slot = W / share if share > 1 and kind in ("tab", "segment") else None
        for row in _rows([t["bbox"] for t in ln["texts"]]):
            strings = [boxes.get(id(b), "") for b in row]
            res = {"kind": kind, "texts": strings, "container": ln["container"], "share": share}
            m = measure_line(rgb, k, ln["container"], row, PADS[kind], strings, slot)
            if m is None:
                out.append(dict(res, status="unmeasured"))
                continue
            if (m["right"] - m["left"]) * share < 0.8 * (W - 2 * MARGIN * k):
                out.append(dict(res, status="hugs"))
                continue
            need = {"small": max(0.0, dp - small_w) * k / share}
            if kind not in FIXED.get(platform, ()):
                need.update({lv: sum(t["w"] * (_grow(t["size_dp"], lv, platform) - 1) for t in m["texts"])
                             for lv in ("large", "max")})
            short = {s: round(float(n - m["free"]) / k, 1) for s, n in need.items() if n - m["free"] > SLACK * k}
            out.append(dict(res, status="fails" if short else "fits", free_dp=round(m["free"] / k, 1), short_dp=short))
    return out


def fit_check(img, dp, platform, model=None):
    """One vision call for the candidate lines, then the measurement. Phone screens only."""
    res = ask([FIT_PROMPT.format(platform=platform, W=img.width, H=img.height, dp=dp), img.convert("RGB")],
              FIT_SCHEMA, "fit", model=model)
    return {"lines": fit_lines(img, dp, platform, res.get("lines") or [])}


FIXES = {"button": "stack the buttons or let the label wrap to two lines", "segment": "shorten the labels or use a menu",
         "tab": "shorten the labels; navigation labels shouldn't truncate", "chips": "let the chips scroll sideways or wrap",
         "tile": "let the value wrap, or stack the tiles", "row": "let the label wrap to two lines",
         "text": "let it wrap to two lines or shrink the hero size"}
LEVELS = {"android": {"small": "on a 360 dp-wide phone", "large": "at 130% text", "max": "at 200% text",
                     "unit": "dp", "head": "a 360 dp-wide phone; text at 130% and 200%, Android 14 scaling"},
          "ios": {"small": "on a 375 pt-wide iPhone SE", "large": "at text size xxxLarge", "max": "at text size AX2",
                  "unit": "pt", "head": "a 375 pt-wide iPhone SE; Dynamic Type xxxLarge and AX2, tab bars and "
                                        "segmented controls stay fixed"}}


def print_fit(res, platform):
    levels = LEVELS.get(platform, LEVELS["android"])
    lines = res["lines"]
    print(f"fit ({levels['head']}):")
    for ln in lines:
        if ln["status"] != "fails":
            continue
        where = "; ".join(f"{levels[lv]} needs {v:g} {levels['unit']} more" for lv, v in ln["short_dp"].items())
        print(f"  {ln['kind']:8s} {' | '.join(repr(t) for t in ln['texts'])}: {where}\n           fix: {FIXES[ln['kind']]}")
    n = {s: sum(ln["status"] == s for ln in lines) for s in ("fails", "fits", "hugs", "unmeasured")}
    print(f"  {n['fails']} fail, {n['fits']} fit, {n['hugs']} grow with their content, {n['unmeasured']} not measured "
          f"(busy background or unreadable box). Height is not checked.")
