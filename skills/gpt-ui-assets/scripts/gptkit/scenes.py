"""Painted store scenes with perspective-keyed captures and local lettering."""
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .core import _plan, _pos_int, edit, generate
from .frames import _decode_result
from .fonts import _gf_ttf
from . import store

SCENE_TARGETS = {"ios-phone": (1206, 2622), "ios-phone-large": (1320, 2868),
                 "ios-phone-65": (1242, 2688)}
MIN_SCREEN_AREA = 0.16
MIN_SCREEN_HEIGHT = 0.42
MAX_OCCLUDED = 0.06
MAX_OCCLUDED_BOTTOM = 0.10
PERSPECTIVE_RATIO = (0.85, 1.18)
AFFINE_RATIO = (0.88, 1.14)
MIN_FOCAL = 0.6
AFFINE_FOCAL = 1.2
MAX_SKEW = 5
DEFAULT_ASPECT = 1320 / 2868
SCENE_PROMPT = ("Art direction: {style}. Scene: {brief}. The phone is the hero and BIG: its screen alone spans at "
                "least 55% of the image height and about two thirds of the image width; the phone body runs from just "
                "below the headline band to near the bottom edge, close to the camera, whole body inside the frame "
                "with a small margin; the world and props are around and behind it, never shrinking it. Its display "
                "is one flat uniform pure chroma-key green #00FF00 edge to "
                "edge, with no UI, text, icons, glare or reflections; nothing else is that green. Objects may overlap "
                "the phone edges and float in front of it. A prop may overlap at most one corner of the phone; all four edges "
                "of the screen stay visible along most of their length. Keep the top {head}% calm open background with no objects "
                "for a headline added later. No text, captions, logos, watermarks or status bar anywhere unless the "
                "brief asks for lettering on an object. Portrait 9:19.5.")
REF_PROMPT = ("Use the reference image only for its art direction: palette, light, materials and rendering. "
              "Paint a NEW composition from the scene description below: a different camera angle, phone pose, "
              "setting and props. Do not reuse the reference's layout. ")
LOOK_PROMPT = ("{look}. Keep exactly the same composition, camera, phone size and position and every object unchanged; "
               "keep the display one flat uniform pure #00FF00 edge to edge; keep the top {head}% clean; no text.")
SCENE_VIEWS = {
    "front": "The phone faces the camera straight on: upright, no tilt, no rotation, no perspective, its screen parallel "
             "to the image and fully readable, centred horizontally.",
    "angled": "The phone is turned and tilted in gentle perspective.",
}


def _need_cv():
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.exit("store scene/compose need OpenCV: pip install opencv-python-headless")


def green_alpha(rgb):
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    return np.clip((g - np.maximum(r, b) - 40) / 70, 0, 1)


def quad_from_mask(m):
    import cv2
    H, W = m.shape
    k = W / 853
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("no green screen")
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(cnt)
    ap = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(ap) < 4:
        raise ValueError("could not find four screen edges")
    segs = [(ap[i], ap[(i + 1) % len(ap)]) for i in range(len(ap))]
    segs = sorted(segs, key=lambda s: -np.linalg.norm(s[1] - s[0]))[:4]
    c = ap.mean(0)
    segs.sort(key=lambda s: np.arctan2(*((s[0] + s[1]) / 2 - c)[::-1]))
    top = int(np.argmin([s[0][1] + s[1][1] for s in segs]))
    segs = segs[top:] + segs[:top]
    lines = []
    for i, (p0, p1) in enumerate(segs):
        d = p1 - p0
        L = np.linalg.norm(d)
        if not L:
            raise ValueError(f"edge {i}: too few points")
        u = d / L
        rel = cnt - p0
        t = rel @ u
        band = (t > L * 0.2) & (t < L * 0.8)
        sel = cnt[band & (np.abs(rel @ np.array([-u[1], u[0]])) < 25 * k)]
        for tol in (6 * k, 3 * k):
            if len(sel) < 20:
                raise ValueError(f"edge {i}: too few points")
            vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            dd = np.abs((cnt - [x0, y0]) @ np.array([-vy, vx]))
            sel = cnt[band & (dd < tol)]
        if len(sel) < 20:
            raise ValueError(f"edge {i}: too few points")
        border = (sel[:, 0] <= 1) | (sel[:, 0] >= W - 2) | (sel[:, 1] <= 1) | (sel[:, 1] >= H - 2)
        if border.mean() > 0.3:
            raise ValueError("a screen edge is cut by the canvas")
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    try:
        for i in range(4):
            (pa, da), (pb, db) = lines[i - 1], lines[i]
            t = np.linalg.solve(np.array([da, -db]).T, pb - pa)
            corners.append(pa + da * t[0])
    except np.linalg.LinAlgError as e:
        raise ValueError("screen edges do not intersect") from e
    return np.array(corners, dtype=np.float32)


def screen_perspective(quad, width, height, aspect):
    q = np.asarray(quad, dtype=np.float64)
    angles = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        angles.append(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))) if denom else 0)
    skew = float(max(abs(a - 90) for a in angles))
    m1, m2, m3, m4 = [np.array([q[i, 0] - width / 2, q[i, 1] - height / 2, 1])
                      for i in (0, 1, 3, 2)]
    d2 = np.dot(np.cross(m2, m4), m3)
    d3 = np.dot(np.cross(m3, m4), m2)
    if abs(d2) < 1e-9 or abs(d3) < 1e-9 or aspect <= 0:
        return {"focal": None, "aspect": None, "skew": skew, "ok": False}
    n2 = np.dot(np.cross(m1, m4), m3) / d2 * m2 - m1
    n3 = np.dot(np.cross(m1, m4), m2) / d3 * m3 - m1
    z = n2[2] * n3[2]
    f2 = -np.dot(n2[:2], n3[:2]) / z if abs(z) > 1e-9 else float("nan")
    focal = float(np.sqrt(f2) / height) if f2 > 0 else None
    perspective = focal is not None and focal >= MIN_FOCAL
    f2 = f2 if perspective else (AFFINE_FOCAL * height) ** 2
    numerator = np.dot(n2[:2], n2[:2]) + f2 * n2[2] ** 2
    denominator = np.dot(n3[:2], n3[:2]) + f2 * n3[2] ** 2
    ratio = float(np.sqrt(numerator / denominator) / aspect) if denominator > 0 and numerator >= 0 else None
    limits = PERSPECTIVE_RATIO if perspective else AFFINE_RATIO
    ok = ratio is not None and limits[0] <= ratio <= limits[1] and (perspective or skew < MAX_SKEW)
    return {"focal": focal, "aspect": ratio, "skew": skew, "ok": bool(ok)}


def _perspectives(quads, width, height, aspects):
    return [screen_perspective(q, width, height, a) for q, a in zip(quads, aspects)]


def _worst_perspective(items):
    return max(items, key=lambda p: (not p["ok"], abs(np.log(p["aspect"])) if p["aspect"] else float("inf"), p["skew"]))


def _perspective_note(p):
    ratio = f"{p['aspect']:.2f}" if p["aspect"] is not None else "unknown"
    return f"impossible perspective (ratio {ratio}, skew {p['skew']:.0f}°)"


def screen_quads(alpha, count=1):
    import cv2
    n, labels, stats, cents = cv2.connectedComponentsWithStats((alpha > 0.5).astype(np.uint8))
    order = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:count]
    return [quad_from_mask(labels == i) for i in sorted(order, key=lambda i: cents[i][0])]


def quad_fit(alpha, quad):
    import cv2
    poly = np.zeros(alpha.shape, np.uint8)
    cv2.fillConvexPoly(poly, quad.astype(np.int32), 1)
    g = alpha > 0.5
    return float((poly & g).sum() / max(1, (poly | g).sum()))


def _fits(alpha, quads):
    import cv2
    # Measure each screen independently when there are multiple screens.
    n, lab, stats, cents = cv2.connectedComponentsWithStats((alpha > 0.5).astype(np.uint8))
    ids = sorted(1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:len(quads)], key=lambda i: cents[i][0])
    return [quad_fit((lab == i).astype(np.float32), q) for i, q in zip(ids, quads)]


def _occlusion(alpha, quads):
    import cv2
    values = []
    for q in quads:
        region = np.zeros(alpha.shape, np.uint8)
        cv2.fillConvexPoly(region, q.astype(np.int32), 1)
        mask = region.astype(bool)
        bottom = mask & (np.arange(alpha.shape[0])[:, None] >= q[:, 1].min() + 0.75 * np.ptp(q[:, 1]))
        values.append((1 - float((alpha[mask] > 0.3).mean()), 1 - float((alpha[bottom] > 0.3).mean())))
    return [max(v[i] for v in values) for i in range(2)]


def _occluded(values):
    return values[0] > MAX_OCCLUDED or values[1] > MAX_OCCLUDED_BOTTOM


def key_capture(scene, captures, pad=3):
    import cv2
    rgb = np.asarray(scene.convert("RGB"))
    alpha = green_alpha(rgb)
    quads = screen_quads(alpha, len(captures))
    if len(quads) != len(captures):
        raise ValueError(f"found {len(quads)} screens, slot needs {len(captures)}")
    out = rgb.astype(np.float32)
    limit = np.maximum(out[..., 0], out[..., 2])
    out[..., 1] = np.where(out[..., 1] - limit > 25, np.minimum(out[..., 1], limit + 6), out[..., 1])
    for quad, capture in zip(quads, captures):
        c = np.asarray(capture.convert("RGB"))
        ch, cw = c.shape[:2]
        centre = quad.mean(0)
        grown = centre + (quad - centre) * (1 + pad / max(scene.size))
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), grown)
        warped = cv2.warpPerspective(c, M, scene.size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        region = np.zeros(alpha.shape, np.uint8)
        cv2.fillConvexPoly(region, (centre + (quad - centre) * 1.04).astype(np.int32), 1)
        r = region.astype(bool)
        rad = max(2, round(4 * scene.width / 1320))
        keyed = cv2.dilate((alpha > 0.3).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1,) * 2))
        n, lab, st, _ = cv2.connectedComponentsWithStats((r & (keyed == 0)).astype(np.uint8))
        specks = np.isin(lab, 1 + np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < 400 * (scene.width / 1320) ** 2))
        keyed = np.where(specks, 1, keyed)
        ra = cv2.GaussianBlur(np.where(r, keyed, 0).astype(np.float32), (0, 0), 1.2)[..., None]
        out[..., 1] = np.where(r, np.minimum(out[..., 1], np.maximum(out[..., 0], out[..., 2])), out[..., 1])
        out = out * (1 - ra) + warped.astype(np.float32) * ra
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), quads, _fits(alpha, quads)


def clear_status_bar(scene):
    import cv2
    a = np.array(scene.convert("RGB"))
    h = max(1, int(a.shape[0] * 0.045))
    band = a[:h]
    smooth = cv2.medianBlur(band, 21)
    diff = np.abs(band.astype(int) - smooth.astype(int)).sum(2)
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[:h] = (diff > 60).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return Image.fromarray(cv2.inpaint(a, mask, 9, cv2.INPAINT_TELEA))


def _names(slot):
    return slot.get("scene_captures", [slot["capture"]])


def _slots(sb, arg):
    want = set(arg.split(",")) if arg else None
    slots = [s for s in sb["slots"] if not want or s["id"] in want]
    if want and want - {s["id"] for s in slots}:
        raise ValueError(f"unknown slots: {', '.join(sorted(want - {s['id'] for s in slots}))}")
    return slots


def _capture(d, sb, setname, name, folder=None):
    folder = folder or sb.get("scenes", {}).get("sets", {}).get(setname, {}).get("captures")
    if folder:
        p = Path(folder)
        p = (p if p.is_absolute() else d / p) / (name + ".png")
    else:
        caps = store._captures(d)
        c = next((c for c in caps if c["name"] == name and c["platform"] == "ios" and
                  c["device"] == "phone" and c["orientation"] == "portrait"), None)
        p = d / c["file"] if c else None
    if p is None or not p.is_file():
        raise ValueError(f"missing capture {name!r}")
    return Image.open(p).convert("RGB")


def _screen_size(quads, width, height):
    import cv2
    area = sum(cv2.contourArea(q) for q in quads) / (width * height)
    span = max(q[:, 1].max() - q[:, 1].min() for q in quads) / height
    return span, area < MIN_SCREEN_AREA or span < (0.40 if len(quads) >= 2 else MIN_SCREEN_HEIGHT)


def _check(scene, count, head, aspects=None):
    import cv2
    if aspects is not None and len(aspects) != count:
        raise ValueError("one capture aspect is required per screen")
    alpha = green_alpha(np.asarray(scene))
    if not (alpha > 0.5).any():
        return None, "no green screen", [], None
    try:
        quads = screen_quads(alpha, count)
        if len(quads) != count:
            return None, f"found {len(quads)} screens, slot needs {count}", [], None
        fit = min(_fits(alpha, quads))
    except ValueError as e:
        return None, str(e), [], None
    region = np.zeros(alpha.shape, np.uint8)
    for q in quads:
        c = q.mean(0)
        cv2.fillConvexPoly(region, (c + (q - c) * 1.04).astype(np.int32), 1)
    screen, small = _screen_size(quads, scene.width, scene.height)
    flags = ["SMALL-PHONE"] if small else []
    if _occluded(_occlusion(alpha, quads)):
        flags.append("OCCLUDED")
    perspectives = _perspectives(quads, scene.width, scene.height, aspects if aspects is not None else [DEFAULT_ASPECT] * count)
    bad = [p for p in perspectives if not p["ok"]]
    if bad:
        flags.append("BAD-PERSPECTIVE")
    outside = region == 0
    if outside.any() and np.mean((alpha > 0.5)[outside]) > 0.005:
        flags.append("GREEN-SPILL")
    g = np.asarray(scene.convert("L").resize((256, round(256 * scene.height / scene.width))), np.float32)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    top = e[:max(1, int(e.shape[0] * head))].mean()
    if top > 1.25 * e.mean() and top > 6:
        flags.append("BUSY-TOP")
    return round(fit, 3), _perspective_note(_worst_perspective(bad)) if bad else None, flags, round(float(screen), 2)


def _copies_ref(quads, size, ref_quad, ref_size):
    scale = np.array(ref_size) / np.array(size)
    return any(np.all(np.linalg.norm(q * scale - ref_quad, axis=1) <= 0.03 * ref_size[0]) for q in quads)


def _load_manifest(path, setname, look=None):
    return json.loads(path.read_text()) if path.exists() else {"set": setname, "look": look, "slots": {}}


def _review(d, sb, setname, manifest, out):
    rows = []
    for slot in sb["slots"]:
        entry = manifest["slots"].get(slot["id"])
        if not entry or not entry["variants"]:
            continue
        ims, labels = [], []
        for v in entry["variants"]:
            p = out / v["file"]
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            if v["error"] is None:
                try:
                    caps = [_capture(d, sb, setname, n) for n in _names(slot)]
                    im = key_capture(im, caps)[0]
                except ValueError:
                    pass
            ims.append(im)
            persp = v.get("perspective")
            label = f" persp {persp['ratio']:.2f}" if persp and persp["ratio"] is not None else ""
            occ = v.get("occluded")
            label += f" occ {occ[0]:.2f}/{occ[1]:.2f}" if occ else ""
            labels.append(f"{v['file']}{'*' if entry.get('pick') == v['file'] else ''}: "
                          f"{v['error'] or v['fit']} screen {v.get('screen')}{label} {' '.join(v['flags'])}")
        if ims:
            rows.append((slot["id"] + " " + " | ".join(labels), ims))
    if rows:
        store._contact(rows, out / "review.jpg")


def cmd_store_scene(a):
    _need_cv()
    d = Path(a.dir)
    sb = store._storyboard(d)
    settings = sb.setdefault("scenes", {})
    head = settings.get("headroom", 0.22)
    slots = _slots(sb, a.slots)
    out = d / "scenes" / a.set
    manifest = _load_manifest(out / "scenes.json", a.set)
    if a.pick:
        for part in a.pick.split(","):
            sid, num = part.split("=", 1)
            file = f"{sid}_v{num}.png"
            entry = manifest["slots"].get(sid)
            if not entry or file not in [v["file"] for v in entry["variants"]] or not (out / file).exists():
                raise ValueError(f"{sid}: variant {num} does not exist")
            entry["pick"] = file
        out.mkdir(parents=True, exist_ok=True)
        (out / "scenes.json").write_text(json.dumps(manifest, indent=2))
        _review(d, sb, a.set, manifest, out)
        return
    look = None
    if a.set == "main":
        if not settings.get("style"):
            raise ValueError("store scene: scenes.style is required")
        ref = a.style_ref or settings.get("style_ref")
        ref = (Path(ref) if Path(ref).is_absolute() else d / ref) if ref else None
    else:
        sets = settings.setdefault("sets", {})
        look = a.look or sets.get(a.set, {}).get("look")
        if not look:
            raise ValueError(f"store scene: set {a.set} needs a look")
        ref = None
    work = []
    main = _load_manifest(d / "scenes" / "main" / "scenes.json", "main") if a.set != "main" else None
    for slot in slots:
        if not slot.get("scene_brief"):
            continue
        sid = slot["id"]
        entry = manifest["slots"].get(sid, {})
        if entry.get("variants") and not a.force:
            continue
        source = ref
        if main is not None:
            pick = main["slots"].get(sid, {}).get("pick")
            source = d / "scenes" / "main" / pick if pick else None
            if source is None or not source.is_file():
                raise ValueError(f"store scene: main set has no pick for {sid}")
        prompt = (LOOK_PROMPT.format(look=look, head=round(head * 100)) if main is not None else
                  SCENE_PROMPT.format(style=settings["style"], brief=slot["scene_brief"], head=round(head * 100)))
        if main is None and source:
            prompt = REF_PROMPT + prompt
        if main is None and "scene_view" in slot:
            view = slot["scene_view"]
            if view not in SCENE_VIEWS:
                raise ValueError(f"{sid}: scene_view {view!r} is not one of {', '.join(SCENE_VIEWS)}")
            prompt += " " + SCENE_VIEWS[view]
        for n in range(1, a.variants + 1):
            work.append((slot, n, source, prompt))
    _plan(len(work), model=a.model, image_model=a.model, size="1152x2496", edits=bool(ref or main), dry_run=a.dry_run)
    if a.dry_run:
        print(f"store scene: {len(work)} image calls")
        return
    out.mkdir(parents=True, exist_ok=True)
    if a.set != "main" and a.look:
        sets.setdefault(a.set, {})["look"] = a.look
        (d / "storyboard.json").write_text(json.dumps(sb, indent=2))
    manifest["look"] = look
    ref_quad, ref_size, ref_data = None, None, None
    if ref:
        try:
            ref_data = store._ref_png(ref)
            with Image.open(ref) as ref_im:
                ref_im = ref_im.convert("RGB")
                ref_size = ref_im.size
                quads = screen_quads(green_alpha(np.asarray(ref_im)))
                ref_quad = quads[0] if quads else None
        except (OSError, ValueError):
            pass
    ref_slots = {slot["id"] for slot in slots if ref and
                 manifest["slots"].get(slot["id"], {}).get("pick") and
                 (out / manifest["slots"][slot["id"]]["pick"]).resolve() == ref.resolve()}
    for slot in slots:
        if a.force and any(w[0]["id"] == slot["id"] for w in work):
            for p in out.glob(f"{slot['id']}_v*.png"):
                if re.fullmatch(re.escape(slot["id"]) + r"_v\d+\.png", p.name):
                    p.unlink()
            manifest["slots"].pop(slot["id"], None)
    def job(item):
        slot, n, source, prompt = item
        file = f"{slot['id']}_v{n}.png"
        try:
            if source:
                raw, _ = edit(ref_data if ref_data is not None and main is None else store._ref_png(source),
                              prompt, size="1152x2496", model=a.model, fidelity="low",
                              task="concept")
            else:
                raw, _ = generate(prompt, size="1152x2496", model=a.model, task="concept")
            im = _decode_result(raw, f"scene {slot['id']}").convert("RGB")
        except Exception as e:  # one failed call must not lose the other variants
            return slot["id"], {"file": file, "fit": None, "error": f"image call failed: {e}", "flags": [],
                                "screen": None, "perspective": None, "occluded": None}
        im.save(out / file)
        aspects = []
        for name in _names(slot):
            try:
                cap = _capture(d, sb, a.set, name)
                aspects.append(cap.width / cap.height)
            except ValueError:
                aspects.append(DEFAULT_ASPECT)
        fit, error, flags, screen = _check(im, len(aspects), head, aspects)
        perspective = None
        occluded = None
        if fit is not None:
            quads = screen_quads(green_alpha(np.asarray(im)), len(aspects))
            occluded = [round(v, 3) for v in _occlusion(green_alpha(np.asarray(im)), quads)]
            if ref_quad is not None and slot["id"] not in ref_slots and _copies_ref(quads, im.size, ref_quad, ref_size):
                error = "copies the style reference's composition"
                flags.append("COPIES-REF")
            p = _worst_perspective(_perspectives(quads, im.width, im.height, aspects))
            perspective = {"ratio": p["aspect"], "skew": p["skew"], "focal": p["focal"]}
        return slot["id"], {"file": file, "fit": fit, "error": error, "flags": flags, "screen": screen,
                            "perspective": perspective, "occluded": occluded}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for sid, v in ex.map(job, work):
            entry = manifest["slots"].setdefault(sid, {"variants": [], "pick": None})
            entry["variants"].append(v)
            status = f"FAIL {v['error']}" if v["error"] else f"fit {v['fit']:.2f} screen {v['screen']:.2f}"
            p = v.get("perspective")
            label = f" persp {p['ratio']:.2f}" if p and p["ratio"] is not None else ""
            occ = v.get("occluded")
            label += f" occ {occ[0]:.2f}/{occ[1]:.2f}" if occ else ""
            print(f"{a.set}/{v['file']} {status}{label} {' '.join(v['flags'])}".rstrip())
    for entry in manifest["slots"].values():
        if not entry.get("pick") or not (out / entry["pick"]).is_file():
            passing = [v for v in entry["variants"] if v["error"] is None and v["fit"] >= 0.85 and (out / v["file"]).exists()]
            entry["pick"] = max(passing, key=lambda v: (not any(f in v["flags"] for f in ("SMALL-PHONE", "OCCLUDED")), v["fit"]))["file"] if passing else None
    (out / "scenes.json").write_text(json.dumps(manifest, indent=2))
    _review(d, sb, a.set, manifest, out)


def _scene_lines(slot):
    if "scene_lines" in slot:
        lines = slot["scene_lines"]
    else:
        caption = slot["caption"]
        lines = re.split(r"(?<=\.)\s+", caption)
        if len(lines) == 1:
            words = caption.split()
            if len(words) > 2:
                i = min(range(1, len(words)), key=lambda n: abs(len(' '.join(words[:n])) - len(' '.join(words[n:]))))
                lines = [' '.join(words[:i]), ' '.join(words[i:])]
    return [dict(text=x, size=1) if isinstance(x, str) else {"size": 1, **x} for x in lines]


def _font(family, weight, size, notes):
    path = _gf_ttf(family, weight)
    if path:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    if "font fallback" not in notes:
        notes.append("font fallback")
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _headline(canvas, slot, sb, notes):
    W, H = canvas.size
    settings = sb.get("scenes", {}).get("type", {})
    brand = sb.get("brand", {})
    color = store._store_hex(settings.get("color") or brand.get("ink_light"), (255, 255, 255))
    accent = store._store_hex(settings.get("accent"), color)
    shadow = store._store_hex(settings.get("shadow") or brand.get("ink_dark"), (31, 27, 36))
    family = settings.get("font") or brand.get("head_font") or "Nunito"
    sub_family = settings.get("sub_font") or brand.get("body_font") or "DM Sans"
    lines = _scene_lines(slot)
    upper = settings.get("case") == "upper"
    for line in lines:
        if upper:
            line["text"] = line["text"].upper()
    size = max(1, round(settings.get("size", 0.127) * W))
    margin = settings.get("margin", 0.064) * W
    while True:
        fonts = [_font(family, settings.get("weight", 900), max(1, round(size * line["size"])), notes) for line in lines]
        if size <= 1 or all(f.getlength(line["text"]) <= W - 2 * margin for f, line in zip(fonts, lines)):
            break
        size -= 1
    align = settings.get("align", "left")
    y = settings.get("top", 0.042) * H
    top = y
    items = []
    for i, (line, font) in enumerate(zip(lines, fonts)):
        text = line["text"]
        x = margin if align == "left" else (W - font.getlength(text)) / 2
        col = line.get("color")
        ink = accent if col == "accent" else color if col == "color" else store._store_hex(col, color) if col else (accent if i >= settings.get("accent_from", 1) else color)
        items.append((text, font, x, y, ink))
        y += size * line["size"] * settings.get("line_gap", 0.94)
    sub = slot.get("scene_sub", slot.get("sub", ""))
    if sub:
        if upper:
            sub = sub.upper()
        y += 0.018 * H
        font = _font(sub_family, settings.get("sub_weight", 700), max(1, round(settings.get("sub_size", 0.041) * W)), notes)
        x = margin if align == "left" else (W - font.getlength(sub)) / 2
        items.append((sub, font, x, y, color))
        y += font.size
    boxes = [ImageDraw.Draw(canvas).textbbox((x, yy), text, font=font) for text, font, x, yy, _ in items]
    bottom = max([b[3] for b in boxes] + [top])
    if boxes:
        x0 = max(0, int(min(b[0] for b in boxes)))
        x1 = min(W, int(max(b[2] for b in boxes)) + 1)
        y0, y1 = max(0, int(top)), min(H, int(bottom) + 1)
        bg = np.asarray(canvas.convert("RGB").crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))), dtype=np.float32) / 255
        rgb = np.where(bg <= 0.04045, bg / 12.92, ((bg + 0.055) / 1.055) ** 2.4)
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        low = False
        for _, _, _, _, ink in items:
            t = np.array(ink, dtype=np.float32) / 255
            t = np.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4) @ np.array([0.2126, 0.7152, 0.0722])
            candidates = np.percentile(lum, [5, 95])
            b = min(candidates, key=lambda v: abs(v - t))
            low |= (max(t, b) + 0.05) / (min(t, b) + 0.05) < 3
    else:
        low = False
    if settings.get("glow", True):
        glow = Image.new("RGBA", canvas.size)
        dr = ImageDraw.Draw(glow)
        for text, font, x, yy, _ in items:
            dr.text((x, yy), text, font=font, fill=(*shadow, 180))
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(size * 0.2)))
    dr = ImageDraw.Draw(canvas)
    for text, font, x, yy, ink in items:
        dr.text((x + 0.03 * size, yy + 0.045 * size), text, font=font, fill=(*shadow, 255))
        dr.text((x, yy), text, font=font, fill=(*ink, 255))
    return bottom, low


def _flat(d, report, target, sid, size, head):
    mapped = {"ios-phone": "ios-phone", "ios-phone-large": "ios-phone-large",
              "ios-phone-65": "ios-phone-large"}[target]
    def find(key):
        shots = report.get("targets", {}).get(key, {}).get("shots", [])
        entry = next((s for s in shots if s.get("slot") == sid), None)
        path = d / entry["file"] if entry and entry.get("file") else None
        if not path or not path.is_file():
            path = next(iter(sorted((d / "build" / key).glob(f"[0-9][0-9]_{sid}.png"))), None)
        return path, entry
    path, entry = find(mapped)
    notes = []
    if not path:
        return None
    im = Image.open(path).convert("RGB")
    scale = max(size[0] / im.width, size[1] / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    left, top = (im.width - size[0]) // 2, (im.height - size[1]) // 2
    im = im.crop((left, top, left + size[0], top + size[1]))
    box = entry.get("screen_box") if entry else None
    if box:
        phone_top = box["y"] * scale - top
        notes.append("phone top from build screen box")
    else:
        a = np.asarray(im, dtype=np.int16)
        bg = np.median(a[0], axis=0)
        diff = np.max(np.abs(a - bg), axis=2)
        rows = np.flatnonzero((diff > 35).mean(axis=1)[int(head * size[1]):] > 0.05)
        phone_top = int(head * size[1]) + int(rows[0]) if len(rows) else size[1]
        notes.append("phone top detected from background difference" if len(rows) else "phone top not detected")
    return im, phone_top, notes


def cmd_store_compose(a):
    targets = a.targets.split(",")
    for target in targets:
        if target not in SCENE_TARGETS:
            if target.startswith("play"):
                raise ValueError("store compose: painted scenes are for the App Store only; Google Play asks for interface-only screenshots without device frames or added backgrounds. Build Play shots with `store build` (frameless play-phone, 9:16).")
            raise ValueError(f"unknown scene target {target}; available targets: {', '.join(SCENE_TARGETS)}")
    _need_cv()
    d = Path(a.dir)
    sb = store._storyboard(d)
    slots = _slots(sb, a.slots)
    source = d / "scenes" / a.set
    manifest = _load_manifest(source / "scenes.json", a.set)
    flat_dir = Path(a.flat) if a.flat else None
    flat_report = {}
    if flat_dir and (flat_dir / "build" / "report.json").exists():
        flat_report = json.loads((flat_dir / "build" / "report.json").read_text())
        if not flat_report.get("no_caption"):
            raise ValueError(f"store compose: flat build has captions; run `store build {flat_dir} --no-caption` first")
    out = Path(a.out) if a.out else source / "build"
    report = {}
    for target in targets:
        W, H = SCENE_TARGETS[target]
        folder = out / target
        folder.mkdir(parents=True, exist_ok=True)
        report[target] = {}
        files = []
        for slot in slots:
            sid = slot["id"]
            pick = manifest["slots"].get(sid, {}).get("pick")
            if not pick or not (source / pick).exists():
                flat = _flat(flat_dir, flat_report, target, sid, (W, H), sb.get("scenes", {}).get("headroom", 0.22)) if flat_dir else None
                if not flat:
                    print(f"store compose: {target}/{sid}: no picked scene or flat build render; skipped")
                    continue
                im, phone_top, notes = flat
                flags = []
                canvas = im.convert("RGBA")
                bottom, low = _headline(canvas, slot, sb, notes)
                if bottom > phone_top - H * 0.01:
                    flags.append("OVERLAP")
                if low:
                    flags.append("LOW-CONTRAST")
                index = next(i for i, s in enumerate(sb["slots"], 1) if s["id"] == sid)
                path = store._save_store_png(canvas, folder / f"{index:02d}_{sid}.png")
                files.append(path)
                report[target][sid] = {"file": str(path.relative_to(out)), "kind": "flat", "fit": None, "flags": flags, "notes": notes}
                continue
            notes, flags = [], []
            im = Image.open(source / pick).convert("RGB")
            caps = [_capture(d, sb, a.set, name, a.captures) for name in _names(slot)]
            scale = max(W / im.width, H / im.height)
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            try:
                qs = screen_quads(green_alpha(np.asarray(im)), len(_names(slot)))
                if len(qs) != len(_names(slot)):
                    raise ValueError(f"found {len(qs)} screens, slot needs {len(_names(slot))}")
                offset = max(0, int(np.ceil(max(q[:, 1].max() for q in qs) - H * 0.98)))
                if offset > im.height - H:
                    raise ValueError("screen does not fit target crop")
                left = (im.width - W) // 2
                im = im.crop((left, offset, left + W, offset + H))
                if a.clean_top:
                    im = clear_status_bar(im)
                alpha = green_alpha(np.asarray(im))
                occ = _occlusion(alpha, screen_quads(alpha, len(caps)))
                im, qs, fits = key_capture(im, caps)
            except ValueError as e:
                report[target][sid] = {"file": None, "kind": "painted", "fit": None, "flags": [], "notes": [str(e)]}
                print(f"store compose: {target}/{sid}: {e}")
                continue
            fit = round(min(fits), 3)
            perspectives = _perspectives(qs, W, H, [c.width / c.height for c in caps])
            bad = [p for p in perspectives if not p["ok"]]
            if bad:
                note = _perspective_note(_worst_perspective(bad))
                report[target][sid] = {"file": None, "kind": "painted", "fit": fit, "flags": ["BAD-PERSPECTIVE"] + (["OCCLUDED"] if _occluded(occ) else []), "occluded": [round(v, 3) for v in occ], "notes": [note]}
                print(f"store compose: {target}/{sid}: {note} BAD-PERSPECTIVE")
                continue
            if _screen_size(qs, W, H)[1]:
                flags.append("SMALL-PHONE")
            if _occluded(occ):
                flags.append("OCCLUDED")
            canvas = im.convert("RGBA")
            bottom, low = _headline(canvas, slot, sb, notes)
            if bottom > min(q[:, 1].min() for q in qs) - H * 0.01:
                flags.append("OVERLAP")
            if low:
                flags.append("LOW-CONTRAST")
            if fit < 0.85:
                flags.append("LOW-FIT")
            index = next(i for i, s in enumerate(sb["slots"], 1) if s["id"] == sid)
            path = store._save_store_png(canvas, folder / f"{index:02d}_{sid}.png")
            files.append(path)
            report[target][sid] = {"file": str(path.relative_to(out)), "kind": "painted", "fit": fit, "occluded": [round(v, 3) for v in occ], "flags": flags, "notes": notes}
        store._set_sheet(files, folder / "sheet.jpg")
        print(f"store compose: {target}: {len(files)} shots -> {folder}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))


def _scene_parsers(sub):
    q = sub.add_parser("scene", help="paint and review green-screen scenes")
    q.add_argument("dir")
    q.add_argument("--set", default="main")
    q.add_argument("--slots")
    q.add_argument("--variants", type=_pos_int, default=2)
    q.add_argument("--look")
    q.add_argument("--style-ref")
    q.add_argument("--pick")
    q.add_argument("--jobs", type=_pos_int, default=4)
    q.add_argument("--force", action="store_true")
    q.add_argument("--model")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_store_scene)
    q = sub.add_parser("compose", help="key captures and set headlines on painted scenes")
    q.add_argument("dir")
    q.add_argument("--set", default="main")
    q.add_argument("--slots")
    q.add_argument("--targets", default="ios-phone,ios-phone-large")
    q.add_argument("--captures")
    q.add_argument("--flat", help="store project with captionless build renders for unpainted slots")
    q.add_argument("--clean-top", action="store_true")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_store_compose)
