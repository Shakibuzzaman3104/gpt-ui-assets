"""Mockup, critique, manifest, spec."""
import argparse
import concurrent.futures as cf
import io
import json
import math
import re
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import (
    ApiError, _clip_bbox, _need, _open_image, _plan, _png_bytes, _read_json, _size_for, ask, edit, generate,
)
from .imaging import _mean_diff, _rgba, ref_crop
from .prompts import (
    ACCENT_PROMPT, ACCENT_SCHEMA, ADAPT_PROMPT, ASSET_PROMPT, BACKGROUND_PROMPT, CANDIDATE_FOCUS, CRITIQUE_PROMPT,
    CRITIQUE_REF_RULE, CRITIQUE_SCHEMA, DEVICES, MANIFEST_PROMPT, MANIFEST_SCHEMA, MOCKUP_WRAP, PLATFORMS, SEVERITY,
    SPEC_SCHEMA, STYLE_REF_PROMPT, VERIFY_SCHEMA, _aspect_words, _design_meta, _device_record, _house_style, _platform_rule,
    _screen_dp,
)

# ---------------------------------------------------------------- commands

ASSET_DEFAULTS = (("kind", "illustration"), ("format", "png"), ("description", ""), ("colors", []))


def _load(d):
    """(dir, manifest, mockup RGBA) with one-line exits and normalised manifest entries (plan contract C5):
    missing screen_dp is filled from the platform, per-asset defaults are applied and printed, every bbox becomes
    ints clamped inside the mockup (clamps printed; no overlap or no name/bbox -> exit naming the asset)."""
    d = Path(d)
    if not d.is_dir():
        sys.exit(f"gptkit: project dir not found: {d} (run manifest first)")
    mf = d / "assets.json"
    m = _read_json(mf, "manifest", required=("mockup", "width", "height", "assets"))
    if not isinstance(m["assets"], list):
        sys.exit(f"gptkit: manifest {mf}: 'assets' must be a list of assets")
    if not isinstance(m["mockup"], str) or not m["mockup"]:
        sys.exit(f"gptkit: manifest {mf}: 'mockup' must be the mockup file name")
    mock = _open_image(d / m["mockup"], "mockup", f"named by {mf}")
    W, H = mock.size
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0  # noqa: E731
    for k in ("width", "height", "screen_dp"):  # hand-edited values: a wrong type would surface after the API calls
        if k in m and not num(m[k]):
            sys.exit(f"gptkit: {mf}: '{k}' must be a positive number, got {m[k]!r}")
    if (m["width"], m["height"]) != (W, H):  # bboxes are mockup pixels: the mockup's size is the truth
        print(f"  ~ {mf.name}: width/height {m['width']}x{m['height']} differ from the mockup {W}x{H}; using the mockup's")
        m["width"], m["height"] = W, H
    if "screen_dp" not in m:
        plat = m.get("platform") if m.get("platform") in PLATFORMS else None
        m["screen_dp"] = _screen_dp(plat)
        print(f"  ~ {mf}: no screen_dp; assuming {m['screen_dp']} dp ({plat or 'no known platform'}; run spec to fix)")
    defaults, seen = [], set()
    for i, x in enumerate(m["assets"]):
        if not isinstance(x, dict) or not isinstance(x.get("name"), str) or not x["name"] or "bbox" not in x:
            who = x.get("name") if isinstance(x, dict) and x.get("name") else f"#{i + 1}"
            sys.exit(f"gptkit: {mf}: asset {who} needs 'name' and 'bbox' {{x,y,w,h}}")
        n = x["name"]
        if n in seen:
            sys.exit(f"gptkit: {mf}: duplicate asset name '{n}' (rename one of them)")
        seen.add(n)
        added = []
        for k, v in ASSET_DEFAULTS:
            if k not in x:
                x[k] = list(v) if isinstance(v, list) else v
                added.append(f"{k}={v!r}")
        if "transparent" not in x:
            x["transparent"] = x["kind"] not in ("background", "photo")
            added.append(f"transparent={x['transparent']}")
        if added:
            defaults.append(f"{n}: {', '.join(added)}")
        b = x["bbox"]
        nb = _clip_bbox(b, W, H)
        if nb is None:
            sys.exit(f"gptkit: {mf}: asset '{n}' has an invalid bbox {b}: x,y,w,h must be numbers with w,h >= 1 "
                     f"overlapping the {W}x{H} mockup")
        try:
            asked = {k: int(round(float(b[k]))) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            asked = None
        if asked != nb:
            was = " ".join(f"{k}={b.get(k)}" for k in ("x", "y", "w", "h"))
            print(f"  ! {mf.name}: {n} bbox clamped to the {W}x{H} mockup (was {was})")
        elif any(not isinstance(b[k], int) for k in ("x", "y", "w", "h")):
            print(f"  ~ {mf.name}: {n} bbox normalised to ints")
        x["bbox"] = nb
    if defaults:
        print(f"  ~ {mf.name}: defaults applied: " + "; ".join(defaults))
    return d, m, mock


def _known(m):
    return ", ".join(a["name"] for a in m["assets"]) or "none"


def _select(m, only):
    """The manifest assets named by a comma list (all when empty); unknown names exit with the known ones."""
    if not only:
        return list(m["assets"])
    names = [n.strip() for n in only.split(",") if n.strip()]
    unknown = [n for n in names if n not in {a["name"] for a in m["assets"]}]
    if unknown:
        sys.exit(f"gptkit: unknown asset(s): {', '.join(unknown)} (known: {_known(m)})")
    return [a for a in m["assets"] if a["name"] in names]


def _asset(m, name):
    """One manifest asset by name; exits naming the known ones."""
    for a in m["assets"]:
        if a["name"] == name:
            return a
    sys.exit(f"gptkit: unknown asset: {name} (known: {_known(m)})")


PHONE_SIZE = "1152x2496"  # 9:19.5, modern iPhone proportions (1179x2556, 1206x2622); both sides multiples of 16
ASPECTS = {PHONE_SIZE: "tall 9:19.5 phone screen (modern iPhone proportions, about 2.17x taller than wide)",
           "1024x1536": "portrait 2:3 (phone screen, taller than wide)", "1536x1024": "landscape 3:2 (desktop/tablet, wider than tall)",
           "1024x1024": "square 1:1"}
for _record in DEVICES.values():
    ASPECTS.setdefault(_record["canvas"], _record["aspect"])
ASPECT_TOL = 0.02  # a render within 2% of the requested width/height ratio passes as is
CLASSIC_SIZES = ("1024x1536", "1536x1024", "1024x1024")  # the sizes every gpt-image model accepts
FIT_MAX = 0.15     # local fitting moves at most 15% of the height; past that the render is saved with a warning


def _size_rejected(e):
    """A 400 about the requested size: an endpoint or model that only takes the classic sizes."""
    return e.status == 400 and (e.param == "size" or bool(re.search(r"\bsize\b", str(e), re.I)))


def _default_size(platform):
    """Default canvas, preserving the historical phone canvas when no platform is selected."""
    rec = _device_record(platform) if platform else None
    return rec["canvas"] if rec else PHONE_SIZE


def _ratio(size):
    w, h = map(int, size.split("x"))
    return w / h


def _canvas_size(w, h):
    """Request size for revising an existing WxH design: the canvas with the nearest ratio (a 9:19.5 screen stays one).
    A foldable's folded and tabletop canvases are left out: they come from --device foldable --posture, never from a
    guess, so a slightly narrow phone screen is not revised as a folded cover screen."""
    postures = {r["canvas"] for k, r in DEVICES.items() if k[2] in ("folded", "tabletop")}
    return min((s for s in ASPECTS if s not in postures), key=lambda s: abs(math.log(w / h / _ratio(s))))


def _screen_meta(platform, record, device, orientation, posture, page, canvas):
    orientation = orientation or ("landscape" if device == "tablet" else None)
    posture = posture or ("flat" if device == "foldable" else None)
    page = page or ("app" if platform == "web" else None)
    return {"platform": platform, "device": device, "orientation": orientation, "posture": posture,
            "page": page, "screen_dp": record["logical_w"] if record else _screen_dp(platform),
            "logical_h": record["logical_h"] if record else None, "canvas": canvas,
            "hinge": record["hinge"] if record else None}


def _device_prompt(record):
    if not record:
        return ""
    return (f" Device system UI: {record['system_ui']}. Device components: {record['components']}. "
            f"Device metrics: {record['metrics']}. {record['rule']}")


def _hinge_hits(issues, record, width, height):
    hinge = record.get("hinge") if record else None
    if not hinge:
        return []
    axis, fraction = hinge
    extent = width if axis == "x" else height
    logical = record["logical_w"] if axis == "x" else record["logical_h"]
    band = 16 * extent / logical
    lo, hi = extent * fraction - band, extent * fraction + band
    hits = []
    for issue in issues:
        b = issue.get("bbox", {})
        start = b.get("x" if axis == "x" else "y", -1)
        end = start + b.get("w" if axis == "x" else "h", 0)
        if start < hi and end > lo:
            hits.append(issue)
    return hits


def _print_hinge_note(issues, record, width, height):
    hits = _hinge_hits(issues, record, width, height)
    if hits:
        print("  ! hinge band intersects: " + "; ".join(
            f"[{x.get('severity')}] {x.get('problem')} bbox={x.get('bbox')}" for x in hits))


def _aspect_ok(img_bytes, size):
    w, h = Image.open(io.BytesIO(img_bytes)).size
    tw, th = map(int, size.split("x"))
    return (w > h) == (tw > th) and (w < h) == (tw < th)


def _aspect_err(img, size):
    """Relative error of img's width/height ratio against the requested size's (0.02 = 2%)."""
    return abs(img.width / img.height / _ratio(size) - 1)


def _flat_gaps(img, min_h=28):
    """(start, end) row ranges at least min_h tall that are one flat colour across the width, outside the system bars
    (top 6%, bottom 12%). A thin vertical line through a gap (a timeline, a card edge) still counts as flat."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    flat = (np.abs(a - np.median(a, axis=1, keepdims=True)).max(axis=2) > 10).mean(axis=1) < 0.02
    H = len(flat)
    flat[: int(H * 0.06)] = False
    flat[int(H * 0.88):] = False
    gaps, y = [], 0
    while y < H:
        if flat[y]:
            s = y
            while y < H and flat[y]:
                y += 1
            if y - s >= min_h:
                gaps.append((s, y))
        y += 1
    return gaps


def _fit_aspect(img, size):
    """img brought to size's ratio WITHOUT scaling anything: rows are taken out of, or repeated inside, the flat gaps
    between sections (each gap keeps >= 20 px). Returns (image, note), or (None, reason) when the gaps can't absorb it."""
    W, H = img.size
    need = round(W / _ratio(size)) - H
    n = abs(need)
    if n > FIT_MAX * H:
        return None, f"needs {n} px, {n / H:.0%} of the height; local fitting stops at {FIT_MAX:.0%}"
    spare = [(s, e, e - s - 20 if need < 0 else e - s) for s, e in _flat_gaps(img)]
    total = sum(x[2] for x in spare)
    if not spare or total < (n if need < 0 else 1):
        return None, f"only {max(0, total)} px of empty gaps for {n} px"
    ks = [int(sp * n / total) for *_, sp in spare]
    for i in sorted(range(len(spare)), key=lambda i: -spare[i][2])[: n - sum(ks)]:
        ks[i] += 1
    order, prev = [], 0
    for (s, e, _), k in zip(spare, ks):
        m = (s + e) // 2
        if need < 0:
            order += range(prev, m - k // 2)
            prev = m - k // 2 + k
        else:
            order += list(range(prev, m)) + [m] * k
            prev = m
    order += range(prev, H)
    out = Image.fromarray(np.asarray(img)[np.asarray(order)])
    return out, f"{'removed' if need < 0 else 'added'} {n} px of empty space across {len(spare)} gaps, nothing scaled"


def _layout_diff(a, b):
    return round(_mean_diff(a.convert("L").resize((64, 96)), b.convert("L").resize((64, 96))), 1)


def _prep_ref(path, long_side=1536, what="reference image"):
    """PNG bytes of a reference image for /images/edits: any mode/format, downscaled to long_side (a 4000x6000 photo
    would otherwise be uploaded at full size and may exceed the endpoint's limits)."""
    img = _open_image(path, what)
    s = long_side / max(img.size)
    if s < 1:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return _png_bytes(img)


def cmd_mockup(a):
    if a.edit and a.style_ref:
        sys.exit("gptkit: use --edit (revise a screen) or --style-ref (new screen in its style), not both")
    if a.size is not None and not re.fullmatch(r"[1-9]\d*x[1-9]\d*", a.size):
        sys.exit(f"gptkit: --size must be WIDTHxHEIGHT, e.g. {PHONE_SIZE} (9:19.5 phone), 1024x1536, 1536x1024 or "
                 f"1024x1024; got {a.size!r}")
    edit_png = _prep_ref(a.edit, what="--edit mockup") if a.edit else None
    anchor_path = getattr(a, "anchor", None)
    set_devices = [x.strip() for x in (getattr(a, "set", None) or "").split(",") if x.strip()]
    if set_devices and not anchor_path:
        sys.exit("gptkit: --set requires --anchor")
    ref_path = anchor_path or a.style_ref
    ref_png = _prep_ref(ref_path, what="--anchor image" if anchor_path else "--style-ref image") if ref_path else None
    ref = Image.open(io.BytesIO(ref_png)) if ref_png else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    source_meta = _design_meta(a.edit or ref_path) if (a.edit or ref_path) else {}
    platform = a.platform or source_meta.get("platform")
    if not platform:
        print("  (no --platform: designing platform-neutral; pass --platform android|ios|web for native components)")
    requested_device = getattr(a, "device", None)
    # the reference's device, orientation, posture and page carry over only within its own platform: a phone screen used
    # as --style-ref for --platform macos or web must not make that a phone-shaped desktop window or web page
    same_platform = platform == source_meta.get("platform")
    device_arg = requested_device or (source_meta.get("device") if same_platform else None)
    inherit_device = same_platform and (not requested_device or requested_device == source_meta.get("device"))
    orientation = getattr(a, "orientation", None) or (source_meta.get("orientation") if inherit_device else None)
    posture = getattr(a, "posture", None) or (source_meta.get("posture") if inherit_device else None)
    page = getattr(a, "page", None) or (source_meta.get("page") if inherit_device else None)
    if not platform and (device_arg or orientation or posture or page or set_devices):
        sys.exit("gptkit: --device/--orientation/--posture/--page/--set require --platform android|ios|web|macos|windows|linux")
    try:
        record = _device_record(platform, device_arg, orientation, posture, page) if platform else None
        if set_devices:
            for dname in set_devices:
                _device_record(platform, dname, orientation if dname == "tablet" else None,
                               posture if dname == "foldable" else None, page if platform == "web" else None)
    except ValueError as e:
        sys.exit(f"gptkit: {e}")
    if set_devices:
        screens = {}
        for dname in set_devices:
            child = argparse.Namespace(**vars(a))
            child.set = None
            child.device = dname
            child.orientation = orientation if dname == "tablet" else None
            child.posture = posture if dname == "foldable" else None
            child.page = page if platform == "web" else None
            child.name = f"{a.name}_{dname}"
            child.variants = 1
            child.anchor = anchor_path
            child.style_ref = None
            child.exact_name = child.name  # one render per device, written straight to <name>_<device>.png
            cmd_mockup(child)
            final_path = out / f"{child.name}.png"
            rec = _device_record(platform, dname, orientation if dname == "tablet" else None,
                                 posture if dname == "foldable" else None, page if platform == "web" else None)
            screens[final_path.name] = _screen_meta(platform, rec, dname, orientation, posture, page, rec["canvas"])
        mf = out / "design.json"
        existing = json.loads(mf.read_text()) if mf.exists() else {"platform": platform}
        existing.setdefault("screens", {}).update(screens)  # other screens in this folder keep their records
        existing["platform"] = platform
        mf.write_text(json.dumps(existing, indent=2))
        (out / "set.json").write_text(json.dumps({"anchor": str(Path(anchor_path).resolve()), "devices": {
            d: str(out / f"{a.name}_{d}.png") for d in set_devices}}, indent=2))
        return
    if edit_png and not a.size:
        size = (record["canvas"] if record and (a.device or a.posture or a.orientation)
                else _canvas_size(*Image.open(io.BytesIO(edit_png)).size))
    else:
        size = a.size or (record["canvas"] if record else _default_size(platform))
    qa = not a.no_critique
    _plan(a.variants, (a.variants if qa else 0) + (ref_png is not None), image_model=a.model, size=size,
          edits=bool(a.edit or a.style_ref))
    rule = _platform_rule(platform)
    house = _house_style(a.no_house_style)
    colours = _reference_colours(ref) if ref is not None else ""

    def prompt_for(sz):
        asp = (record["aspect"] if record and sz == record["canvas"] else ASPECTS.get(sz, f"{sz} px"))
        if edit_png is not None:
            return f"{a.prompt} Keep the {asp.removesuffix(' canvas')} canvas and everything not mentioned exactly as it is."
        if a.raw:
            return a.prompt
        surface = "web page viewport" if platform == "web" else ("app window" if device_arg == "desktop" or platform in ("macos", "windows", "linux") else "app screen")
        if anchor_path and record:
            base = ADAPT_PROMPT.format(prompt=a.prompt + colours, aspect=asp, device_rule=record["rule"])
        else:
            base = (STYLE_REF_PROMPT if ref_png is not None else MOCKUP_WRAP).format(prompt=a.prompt + colours, aspect=asp,
                                                                                    surface=surface)
        return base + house + rule + _device_prompt(record)
    canvas = {"size": size}  # shared by the variants: drops to a classic size once if the endpoint rejects this one
    lock = threading.Lock()
    if platform:
        meta_path = out / "design.json"
        meta = _design_meta(meta_path) if meta_path.exists() else {}
        meta.update(_screen_meta(platform, record, device_arg or ("desktop" if platform in ("web", "macos", "windows", "linux") else "phone"),
                                 orientation, posture, page, size))
        meta_path.write_text(json.dumps(meta, indent=2))

    def one(i):
        # the backend picks aspect from the wording, not from `size`, and a style-ref edit can return the reference
        attempts = 0
        while attempts < 2:
            sz = canvas["size"]
            try:
                if edit_png is not None or ref_png is not None:
                    img, meta = edit(edit_png if edit_png is not None else ref_png, prompt_for(sz), sz, model=a.model,
                                     task="concept")
                else:
                    img, meta = generate(prompt_for(sz), sz, model=a.model)
            except ApiError as e:
                if not (_size_rejected(e) and sz not in CLASSIC_SIZES):
                    raise
                with lock:
                    if canvas["size"] == sz:
                        canvas["size"] = _size_for(*map(int, sz.split("x")))
                        print(f"  ! the endpoint rejected size {sz} ({e}); rendering on {canvas['size']} instead")
                continue
            attempts += 1
            im = Image.open(io.BytesIO(img))
            aspect_mismatch = ref is not None and abs(_ratio(sz) / (ref.width / ref.height) - 1) > 0.10
            meta["layout_diff"] = _layout_diff(ref, im) if ref is not None and not aspect_mismatch else None
            copied = ref is not None and not aspect_mismatch and meta["layout_diff"] < 10
            if copied:
                print(f"  ! variant {i + 1} copied the style reference (layout_diff {meta['layout_diff']}), retrying")
            if _aspect_err(im, sz) <= ASPECT_TOL and not copied:
                break
        p = out / (f"{a.exact_name}.png" if getattr(a, "exact_name", None) else f"{a.name}_v{i + 1}.png")
        got = f"{im.width}x{im.height}"
        fell_back = f" on {sz}, the endpoint rejected {size}" if sz != size else ""
        if _aspect_err(im, sz) <= ASPECT_TOL:
            meta["aspect"] = "ok" + (f" ({fell_back.strip()})" if fell_back else "")
        else:
            fitted, note = _fit_aspect(im, sz) if (not record or not record.get("hinge")) and _aspect_ok(img, sz) else (None, "foldable hinge geometry must not be fitted" if record and record.get("hinge") else "wrong orientation")
            if fitted is not None:
                (out / f"{p.stem}_raw.png").write_bytes(img)
                img = _png_bytes(fitted)
                meta["aspect"] = f"fitted {got} -> {fitted.width}x{fitted.height} ({note}; original in {p.stem}_raw.png)"
            else:
                meta["aspect"] = f"OFF ({got} for {sz}{fell_back})"
                print(f"  ! variant {i + 1}: backend returned {got} instead of the requested {sz} aspect after 2 attempts "
                      f"and it could not be fitted ({note}); saved anyway (re-run it, or crop/pad by hand)")
        if copied:
            print(f"  ! variant {i + 1} still copies the style reference after 2 attempts; saved anyway (reword the prompt)")
        p.write_bytes(img)
        return p, meta
    ex = cf.ThreadPoolExecutor(a.variants)
    futs = {ex.submit(one, i): i for i in range(a.variants)}
    made, last_err = [], None
    try:
        for f in cf.as_completed(futs):
            try:
                p, meta = f.result()
            except ApiError as e:
                last_err = e
                if a.variants > 1:
                    print(f"  ! variant {futs[f] + 1} failed: {e}")
                continue
            made.append(p)
            extra = f" layout_diff={meta['layout_diff']}" if meta.get("layout_diff") is not None else ""
            print(f"{p}  quality={meta.get('quality')} size={meta.get('size')} aspect={meta['aspect']}{extra}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if not made:
        if a.variants == 1 or last_err is None:
            raise last_err or RuntimeError("mockup produced no image")
        sys.exit(f"gptkit: mockup produced no image (all {a.variants} variants failed, see above; last: {last_err})")
    if qa:
        _mockup_qa(sorted(made), platform or "android", ref, record)


def _reference_colours(img):
    """Prompt words carrying the approved screen's measured background and accent. A reference image alone lets the
    accent drift; vision only locates the accent-filled elements, their colour is measured from the pixels."""
    from .measure import _fill_colour, _screen_background  # deferred: only needed with a reference
    bg = _screen_background(img)
    try:
        found = ask([ACCENT_PROMPT.format(W=img.width, H=img.height), img.convert("RGB")], ACCENT_SCHEMA, "accent_boxes")
        boxes = [b for b in (_clip_bbox(e, img.width, img.height) for e in found.get("elements") or []) if b]
    except (ApiError, AttributeError) as e:
        print(f"  ! the reference's accent colour was not located ({e}); passing only its background colour")
        boxes = []
    accent = _fill_colour(img, boxes)
    named = [f"background {bg}"] * bool(bg) + [f"accent {accent}"] * bool(accent)
    return (f" Keep the approved screen's colours ({', '.join(named)}) unless this brief names other colours or a theme."
            if named else "")


def _mockup_qa(paths, platform, ref, record=None):
    """Design QA of each new variant (same files as `critique`), in parallel; a failed call is reported, never hidden."""
    dp = record["logical_w"] if record else _screen_dp(platform)

    def one(p):
        img = _open_image(p, "mockup")
        return img, critique(img, dp, platform, ref, device_record=record)
    with cf.ThreadPoolExecutor(len(paths)) as ex:
        futs = [ex.submit(one, p) for p in paths]
    print("design QA (critique):")
    for p, f in zip(paths, futs):
        try:
            img, issues = f.result()
        except (ApiError, SystemExit) as e:
            print(f"  ! {p.name}: critique NOT run ({e}); run `critique {p}` before showing it")
            continue
        print(f"{p.name}: {len(issues)} issues, weight {_weight(issues)}")
        _print_issues(issues)
        _print_hinge_note(issues, record, img.width, img.height)
        _issue_overlay(img, issues, p.parent / f"{p.stem}_critique.png")
        log = {"mockup": str(p), "rounds": [{"image": str(p), "issues": issues, "weight": _weight(issues)}], "best": str(p)}
        (p.parent / f"{p.stem}_critique.json").write_text(json.dumps(log, indent=2))
    print("  read each *_critique.png before showing a variant; `critique <png> --apply` edits the fixes in")


def _lum(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ [0.2126, 0.7152, 0.0722]


def measured_contrast(img, bbox):
    """WCAG ratio estimate inside bbox: perimeter median as background vs the most distinct 5% of pixels."""
    x0, y0 = min(max(0, bbox["x"]), img.width - 2), min(max(0, bbox["y"]), img.height - 2)
    x1, y1 = min(img.width, max(x0 + 2, bbox["x"] + bbox["w"])), min(img.height, max(y0 + 2, bbox["y"] + bbox["h"]))
    px = np.asarray(img.convert("RGB").crop((x0, y0, x1, y1)), float)
    edge = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]])
    bg = _lum(np.median(edge, 0))
    lum = _lum(px.reshape(-1, 3))
    far = np.abs(lum - bg)
    fg = np.median(lum[far >= np.quantile(far, 0.95)])
    hi, lo = max(fg, bg), min(fg, bg)
    return round((hi + 0.05) / (lo + 0.05), 1)


def _card_surfaces(img, dp):
    """Wide filled surfaces (cards, groups) with corner radii in dp, measured along each corner's diagonal on the mask."""
    from .measure import _dominant, _lab_arr
    from .trace_rig import _label  # deferred: trace_rig is heavy
    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    k = W / dp
    y0, y1 = int(0.05 * H), int(0.92 * H)
    side = np.concatenate([rgb[y0:y1, :4].reshape(-1, 3), rgb[y0:y1, -4:].reshape(-1, 3)]).astype(np.float64)
    bg = _lab_arr(_dominant(side))
    noise = float(np.percentile(np.linalg.norm(_lab_arr(side) - bg, axis=-1), 95))
    mask = np.linalg.norm(_lab_arr(rgb.astype(np.float64)) - bg, axis=-1) > max(1.0, 2 * noise)
    holes, _ = _label(~mask)
    edge = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    lab, n = _label(mask | (~np.isin(holes, edge) & (holes > 0)))
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n + 1)
    big = np.zeros(n + 1, bool)
    big[1:] = area[1:] >= 0.85 * 0.4 * W * 40 * k
    yy, xx = np.divmod(np.arange(flat.size), W)
    sel = big[flat]
    lx, ly, hx, hy = np.full(n + 1, W), np.full(n + 1, H), np.zeros(n + 1, int), np.zeros(n + 1, int)
    np.minimum.at(lx, flat[sel], xx[sel]); np.maximum.at(hx, flat[sel], xx[sel])
    np.minimum.at(ly, flat[sel], yy[sel]); np.maximum.at(hy, flat[sel], yy[sel])
    out = []
    for i in np.nonzero(big)[0]:
        bx0, bx1, by0, by1 = int(lx[i]), int(hx[i]) + 1, int(ly[i]), int(hy[i]) + 1
        w, h = bx1 - bx0, by1 - by0
        if w < 0.4 * W or h < 40 * k or bx0 <= 2 or bx1 >= W - 2 or by0 < y0 or by1 > y1 or area[i] < 0.85 * w * h:
            continue
        m = lab[by0:by1, bx0:bx1] == i
        d = np.arange(min(h, w) // 2)
        steps = []
        for cm in (m, m[:, ::-1], m[::-1], m[::-1, ::-1]):
            hit = cm[d, d] & cm[np.minimum(d + 1, h - 1), d] & cm[d, np.minimum(d + 1, w - 1)]
            steps.append(int(np.argmax(hit)) if hit.any() else None)
        if None in steps:
            continue
        corners = [round(s / (1 - 2 ** -0.5) / k, 1) for s in steps]
        r = float(np.median(corners))
        out.append({"bbox": {"x": bx0, "y": by0, "w": w, "h": h}, "radius_dp": r, "corners_dp": corners,
                    "pill": r * k >= 0.45 * min(w, h)})
    return out


def _corner_mix_issues(img, dp):
    """A square-cornered surface (<= 3 dp) beside rounded ones (>= 8 dp) of about its width; vision QA never sees it."""
    surfaces = _card_surfaces(img, dp)
    rounded = [s for s in surfaces if s["radius_dp"] >= 8 and not s["pill"]]
    issues = []
    for s in surfaces:
        peers = [r for r in rounded if abs(s["bbox"]["w"] - r["bbox"]["w"]) <= 0.15 * r["bbox"]["w"]]
        if s["radius_dp"] <= 3 and peers:
            want = round(float(np.median([r["radius_dp"] for r in peers])))
            issues.append({"severity": "medium", "category": "shape", "bbox": s["bbox"], "scope": "local",
                           "problem": f"This surface has square corners (measured {s['radius_dp']:g} dp) while "
                                      f"{len(peers)} surface(s) of the same width are rounded (about {want} dp).",
                           "edit": f"Round this card's corners to about {want} dp to match the other cards.",
                           "measured_radius_dp": s["radius_dp"]})
    return issues


def critique(img, dp=390, platform="android", style_ref=None, note=None, device_record=None):
    rule = CRITIQUE_REF_RULE if style_ref is not None else ""
    rec = device_record or _device_record(platform)
    touch = rec["touch"] if rec else (48 if platform == "android" else 44)
    device_rule = rec["checks"] if rec else "platform-neutral layout and system UI"
    label = PLATFORMS[platform]["label"] if platform in PLATFORMS else platform
    text = CRITIQUE_PROMPT.format(W=img.width, H=img.height, dp=dp, k=img.width / dp, platform=label, touch=touch,
                                  device_rule=device_rule, ref_rule=rule) + (f"\nThe user also says: {note}" if note else "")
    parts = [text, img.convert("RGB")] + ([style_ref.convert("RGB")] if style_ref is not None else [])
    issues = _clean_issues(ask(parts, CRITIQUE_SCHEMA, "design_qa"), "issues", img)
    issues += _corner_mix_issues(img, dp)
    for i, x in enumerate(issues):
        x["id"] = i + 1
        if x["category"] == "contrast" and x["bbox"]["w"] > 2 and x["bbox"]["h"] > 2:
            x["measured_contrast"] = measured_contrast(img, x["bbox"])
            _weigh_contrast(x)
    return issues


def _contrast_target(issue):
    """The WCAG ratio the issue claims is missed: 3:1 for graphics and large text when it says so, else 4.5:1."""
    text = f"{issue.get('problem', '')} {issue.get('edit', '')}"
    return 3.0 if re.search(r"\b3(\.0)?\s*:\s*1\b", text) and not re.search(r"\b4\.5\s*:\s*1\b", text) else 4.5


def _weigh_contrast(issue):
    """A contrast issue whose measured ratio already meets its target drops to low: the estimate (perimeter background
    vs the most distinct 5% of pixels) can miss a thin element, so it is kept and flagged for a look, not deleted."""
    target = _contrast_target(issue)
    if issue["measured_contrast"] >= target:
        issue["measured_passes"] = target
        if issue.get("severity") in ("high", "medium"):
            issue["severity"] = "low"


def _clean_issues(res, key, img):
    """res[key] from a critique/verify response, each issue's bbox clipped to img (ints inside the image). Issues whose
    bbox is unreadable or lies outside the image are dropped and printed; a missing list exits (the call was wasted)."""
    items = res.get(key) if isinstance(res, dict) else None
    if not isinstance(items, list):
        sys.exit(f"gptkit: the model's design-QA response has no '{key}' list (non-conforming gateway?): {str(res)[:200]}")
    out = []
    for x in items:
        if not isinstance(x, dict):
            continue
        b = _clip_bbox(x.get("bbox"), img.width, img.height)
        if b is None:
            print(f"  ! dropped an issue whose bbox {x.get('bbox')} is not inside the {img.width}x{img.height} image: "
                  f"[{x.get('severity')}] {x.get('problem', '')}")
            continue
        scope = x.get("scope") if x.get("scope") in ("local", "screen") else "local"
        out.append({"category": "", "problem": "", "edit": "", **x, "scope": scope, "bbox": b})
    return out


_SEV_WARNED = set()


def _sev(x):
    """Weight of an issue's severity. An unknown one (lax gateway, older log) counts as medium, with one warning."""
    s = x.get("severity")
    if s in SEVERITY:
        return SEVERITY[s]
    if repr(s) not in _SEV_WARNED:
        _SEV_WARNED.add(repr(s))
        print(f"  ! unknown issue severity {s!r} from the model: counted as medium (weight 2)")
    return 2


def _weight(issues):
    return sum(_sev(x) for x in issues)


def _issue_overlay(img, issues, out):
    ov = img.convert("RGB")
    d = ImageDraw.Draw(ov)
    col = {"high": "#ff2d55", "medium": "#ff9f0a", "low": "#0a84ff"}
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for x in issues:
        b = _clip_bbox(x.get("bbox"), ov.width, ov.height)  # already clipped by critique(); older logs may not be
        if b is None:
            continue
        c = col.get(x.get("severity"), col["medium"])
        d.rectangle([b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1], outline=c, width=4)
        ty = max(0, b["y"] - 36)
        d.rectangle([b["x"], ty, b["x"] + 40, ty + 36], fill=c)
        d.text((b["x"] + 6, ty + 3), str(x.get("id", "")), fill="white", font=font)
    ov.save(out)


def _print_issues(issues):
    for x in issues:
        mc = f"  [measured {x['measured_contrast']}:1]" if "measured_contrast" in x else ""
        if "measured_passes" in x:
            mc = f"  [measured {x['measured_contrast']}:1, meets {x['measured_passes']:g}:1 here; check by eye]"
        sw = "  [screen-wide: re-run mockup]" if x.get("scope") == "screen" else ""
        print(f"  {x.get('id', ''):>3}. {str(x.get('severity')):6s} {str(x.get('category', '')):16s} {x.get('problem', '')}"
              f"{sw}{mc}\n      edit: {x.get('edit', '')}")


def cmd_critique(a):
    src = Path(a.mockup)
    out = Path(a.out) if a.out else src.parent
    cur_path, cur = src, _open_image(src, "mockup")
    ref = _open_image(a.style_ref, "--style-ref image") if a.style_ref else None
    out.mkdir(parents=True, exist_ok=True)
    a.platform = a.platform or _design_meta(src).get("platform") or "android"
    design = _design_meta(src)
    a.device = design.get("device", "phone")
    try:
        record = _device_record(a.platform, a.device, design.get("orientation"), design.get("posture"), design.get("page"))
    except ValueError:
        record = _device_record(a.platform)
    a.screen_dp = _screen_dp(a.platform, a.screen_dp or design.get("screen_dp") or (record["logical_w"] if record else None))
    issues = critique(cur, a.screen_dp, a.platform, ref, a.note, record)
    log = {"mockup": str(src), "rounds": [{"image": str(src), "issues": issues, "weight": _weight(issues)}]}
    print(f"{src.name}: {len(issues)} issues, weight {_weight(issues)}")
    _print_issues(issues)
    _print_hinge_note(issues, record, cur.width, cur.height)
    _issue_overlay(cur, issues, out / f"{src.stem}_critique.png")
    min_w = SEVERITY[a.min_severity]
    if a.apply:
        screen = [x for x in issues if _sev(x) >= min_w and x.get("scope") == "screen"]
        if screen:
            ids = ", ".join(f"#{x['id']}" for x in screen)
            print(f"  ! {len(screen)} screen-wide issue(s) ({ids}) need a re-layout an edit can't do: re-run `mockup` "
                  f"with the fix in the brief")
    for rnd in range(a.rounds if a.apply else 0):
        todo = [x for x in issues if _sev(x) >= min_w and x.get("scope") != "screen"]
        screen = [x for x in issues if _sev(x) >= min_w and x.get("scope") == "screen"]
        if not todo:
            print("nothing local at or above --min-severity to apply" if screen else
                  "nothing at or above --min-severity to apply")
            break
        aspect = ASPECTS.get(_canvas_size(*cur.size), "same")
        fixes = "\n".join(f"{i + 1}. {x['edit']}" for i, x in enumerate(todo))
        prompt = (f"Revise the attached app UI design. Apply ONLY these fixes:\n{fixes}\nKeep everything else exactly as "
                  f"it is: layout, content, copy, colours, typefaces, illustrations, icons and the {aspect} canvas.")
        raw, _ = edit(_png_bytes(cur.convert("RGB")), prompt, _canvas_size(*cur.size), model=a.model)
        p = out / f"{src.stem}_qa{rnd + 1}.png"
        p.write_bytes(raw)
        new = _rgba(p)
        listing = "\n".join(f"{x['id']}. [{x.get('severity')}] {x['problem']}" for x in todo)
        res = ask([f"FIRST image: a UI design. SECOND image: its revision, which was asked to fix these issues:\n{listing}\n"
                   "For each id say whether the SECOND image fixed it. Then list NEW defects the revision introduced "
                   "(changed layout or copy, garbled text, lost elements, style drift) in the same issue format, "
                   "bbox in the SECOND image's pixels. Do not re-list issues that exist in both.", cur.convert("RGB"),
                   new.convert("RGB")], VERIFY_SCHEMA, "design_qa_verify")
        new_issues = _clean_issues(res, "new_issues", new)
        fixed = {c.get("id") for c in res.get("checks") or [] if isinstance(c, dict) and c.get("fixed") is True
                 and isinstance(c.get("id"), int)}
        left = [x for x in todo if x["id"] not in fixed]
        before, after = _weight(todo), _weight(left) + _weight(new_issues)
        drift = round(_mean_diff(cur.resize((128, 192)), new.resize((128, 192))), 1)
        kept = after < before
        fixed &= {x["id"] for x in todo}
        print(f"round {rnd + 1}: fixed {len(fixed)}/{len(todo)}, new {len(new_issues)}, weight {before} -> {after}, "
              f"drift {drift} -> {'KEPT' if kept else 'discarded'}  ({p})")
        _print_issues([dict(x, id=f"n{i + 1}") for i, x in enumerate(new_issues)])
        log["rounds"].append({"image": str(p), "fixed": sorted(fixed), "new_issues": new_issues,
                              "weight_before": before, "weight_after": after, "drift": drift, "kept": kept})
        if not kept:
            break
        rest = [x for x in issues if _sev(x) < min_w or x.get("scope") == "screen"]
        issues = [dict(x, id=i + 1) for i, x in enumerate(left + new_issues + rest)]
        cur_path, cur = p, new
    log["best"] = str(cur_path)
    (out / f"{src.stem}_critique.json").write_text(json.dumps(log, indent=2))
    print(f"-> {out / (src.stem + '_critique.json')}, overlay {out / (src.stem + '_critique.png')}, best: {cur_path}")


MANIFEST_REQUIRED = ("name", "kind", "bbox", "format", "transparent", "animated", "description")


def _manifest_assets(res, W, H):
    """The model's asset list made safe to write: required fields present, bboxes clipped to the WxH mockup (int, inside),
    boxes mostly outside it dropped, names normalised and unique (flame, flame_2, flame_3). Every drop/clamp is printed."""
    from .svg_icons import _norm_name  # deferred: svg_icons imports this module
    items = res.get("assets") if isinstance(res, dict) else None
    if not isinstance(items, list):
        sys.exit(f"gptkit: the model's manifest response has no 'assets' list (non-conforming gateway?): {str(res)[:200]}")
    out, seen = [], set()
    for x in items:
        missing = [k for k in MANIFEST_REQUIRED if not isinstance(x, dict) or k not in x]
        if missing:
            print(f"  ! dropped an asset missing {missing} from the model's response (non-conforming gateway?): "
                  f"{(x.get('name') if isinstance(x, dict) else x)!r}")
            continue
        b0 = x["bbox"]
        b = _clip_bbox(b0, W, H)
        try:
            claimed = abs(float(b0["w"]) * float(b0["h"]))
        except (KeyError, TypeError, ValueError):
            claimed = 0
        if b is None or b["w"] * b["h"] < 0.25 * claimed:
            print(f"  ! dropped '{x['name']}': bbox mostly outside the image ({b0} on {W}x{H})")
            continue
        if any(b[k] != b0.get(k) for k in "xywh"):
            print(f"  ! '{x['name']}': bbox clamped to the image ({b0} -> {b['x']},{b['y']},{b['w']},{b['h']})")
        n = re.sub(r"_(svg|png|icon|asset)$", "", _norm_name(str(x["name"]))).strip("_") or "asset"
        base, i = n, 2
        while n in seen:
            n, i = f"{base}_{i}", i + 1
        seen.add(n)
        out.append(dict(x, name=n, bbox=b))
    return out


def cmd_manifest(a):
    from .svg_icons import _symbol_names  # deferred: svg_icons imports this module
    mock_path = Path(a.mockup).resolve()
    mock = _open_image(mock_path, "mockup")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    local = out / "mockup.png"
    if mock_path != local.resolve():
        mock.save(local)
    design_meta = _design_meta(mock_path)
    platform = a.platform or design_meta.get("platform")
    screen_dp = _screen_dp(platform, a.screen_dp or design_meta.get("screen_dp"))
    native = (f"\n- Target platform: {PLATFORMS[platform]['label']}. Native components ({PLATFORMS[platform]['components']}) "
              "are built in code with the platform toolkit: never list a component itself (tab bar, app bar, switch, slider, "
              "chip, button, sheet) as an asset, only custom glyphs/artwork inside it (e.g. the tab icons, not the tab bar).") \
        if platform else ""
    res = ask([MANIFEST_PROMPT.format(W=mock.width, H=mock.height) + native, mock], MANIFEST_SCHEMA, "asset_manifest",
              model=a.model)
    assets = _manifest_assets(res, mock.width, mock.height)
    system = re.compile(r"wifi|wi_fi|signal|cellular|battery|status_bar|clock|dynamic_island|home_indicator|gesture")
    dropped = [x["name"] for x in assets if system.search(x["name"]) and
               (x["bbox"]["y"] + x["bbox"]["h"] < 0.05 * mock.height or x["bbox"]["y"] > 0.95 * mock.height)]
    if dropped:
        print(f"  (dropped system UI the OS draws: {', '.join(dropped)})")
        assets = [x for x in assets if x["name"] not in dropped]
    for x in assets:
        x["material_symbol"], x["sf_symbol"] = _symbol_names(str(x.get("material_symbol") or ""),
                                                             str(x.get("sf_symbol") or ""))
        x["dp"] = {"w": round(x["bbox"]["w"] * screen_dp / mock.width, 1),
                   "h": round(x["bbox"]["h"] * screen_dp / mock.width, 1)}
    manifest = {"mockup": "mockup.png", "width": mock.width, "height": mock.height, "platform": platform,
                "screen_dp": screen_dp, "assets": assets}
    (out / "assets.json").write_text(json.dumps(manifest, indent=2))
    ov = mock.convert("RGB")
    d = ImageDraw.Draw(ov)
    for x in assets:
        b = x["bbox"]
        col = "#ff2d55" if x["format"] == "png" else "#34c759"
        d.rectangle([b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1], outline=col, width=3)
        d.text((b["x"] + 2, max(0, b["y"] - 12)), x["name"], fill=col)
    ov.save(out / "manifest_overlay.png")
    for x in assets:
        b = x["bbox"]
        print(f"{x['name']:34s} {str(x['kind']):12s} {x['format']}  bbox={b['x']},{b['y']},{b['w']},{b['h']}  "
              f"~{x['dp']['w']}x{x['dp']['h']}dp  transparent={x['transparent']} animated={x['animated']}")
    print(f"\n{len(assets)} assets -> {out / 'assets.json'}   overlay -> {out / 'manifest_overlay.png'} "
          f"(red=png, green=svg)")


def _platform_styles(platform):
    """The text-style names `tokens` maps onto: M3 type-scale slots on android, HIG text styles on ios, none elsewhere."""
    from .tokens import _HIG_SCALE, _M3_SCALE  # deferred: tokens imports this module
    if platform == "android":
        return tuple(n for n, _, _ in _M3_SCALE)
    if platform == "ios":
        return tuple(n for n, _ in _HIG_SCALE)
    return ()


def _spec_schema(platform):
    """SPEC_SCHEMA with `platform_style` restricted to _platform_styles(platform) when the platform has fixed names."""
    sch = json.loads(json.dumps(SPEC_SCHEMA))
    names = _platform_styles(platform)
    if names:
        sch["properties"]["typography"]["items"]["properties"]["platform_style"]["enum"] = list(names)
    return sch


def cmd_spec(a):
    if a.recalibrate:
        return cmd_recalibrate(a.dir, a.scale)
    d, m, mock = _load(a.dir)
    platform = m.get("platform")
    design_meta = _design_meta(d / m["mockup"])
    styles = _platform_styles(platform)
    target = (f" Target platform: {PLATFORMS[platform]['label']}. For every component set 'native' to the native "
              f"component that should implement it (from: {PLATFORMS[platform]['components']}; add its variant, e.g. "
              "'Card (elevated)'), 'system' for UI the OS draws (status bar, home indicator), or 'custom' when none fits. For every text style set 'platform_style' to the closest "
              f"style of {PLATFORMS[platform]['type']}" + (f", named as one of: {', '.join(styles)}." if styles else ".")) if platform else \
        " Set 'native' to 'custom' and 'platform_style' to '' (no target platform given)."
    screen_dp = design_meta.get("screen_dp") or m.get("screen_dp") or _screen_dp(platform)
    res = ask([f"This is an APPROVED UI design ({mock.width}x{mock.height}px; the screen is {screen_dp}dp wide). "
               "Extract the design spec a developer needs to rebuild it 1:1: palette (hex), every text style "
               "(role, a sample string, likely font family, size in px of THIS image, weight, letter spacing, colour), "
               "corner radii, key spacings, and the component list with pixel bboxes. Every bbox covers the WHOLE component "
               "(for text, the full block including every line, not one word)." + target, mock], _spec_schema(platform),
              "design_spec", model=a.model)
    if not isinstance(res, dict):
        sys.exit(f"gptkit: the model's spec response is not a JSON object (non-conforming gateway?): {str(res)[:200]}")
    comps = []
    for c in res.get("components") or []:
        b = _clip_bbox(c.get("bbox"), mock.width, mock.height) if isinstance(c, dict) else None
        if b is None:
            print(f"  ! dropped component {(c.get('name') if isinstance(c, dict) else c)!r}: bbox "
                  f"{c.get('bbox') if isinstance(c, dict) else None} is not inside the {mock.width}x{mock.height} mockup")
            continue
        comps.append({"name": "", "native": "", "notes": "", **c, "bbox": b})
    res["components"] = comps
    res["platform"] = platform
    res["px_to_dp"] = round(screen_dp / mock.width, 5)
    _calibrate(d, res, m, mock.width, a.scale)
    (d / "design_spec.json").write_text(json.dumps(res, indent=2))
    print(f"-> {d / 'design_spec.json'}  (multiply px values by px_to_dp={res['px_to_dp']} for dp/pt)")


# native widgets whose platform size is fixed: (pattern on spec 'native', dp, measured bbox side)
NATIVE_ANCHORS = {
    "android": [(r"navigation ?bar", 80, "h"), (r"extended ?(fab|floating)", 56, "h"),
                (r"^(?!.*extended).*(\bfab\b|floating ?action)", 56, "h"),
                (r"top ?app ?bar(?!.*(medium|large))", 64, "h"), (r"search ?bar", 56, "h"), (r"navigation ?rail", 80, "w")],
    # iOS: UISwitch 31 pt, segmented control 32 pt, tab bar 49 pt (above the home-indicator inset), inline navigation
    # bar 44 pt; a large-title bar grows with its title, so it is never an anchor
    "ios": [(r"toggle|switch", 31, "h"), (r"segmented|picker \(segmented", 32, "h"), (r"tab ?bar", 49, "h"),
            (r"^(?!.*large).*(navigation ?stack|nav(igation)? ?bar)", 44, "h")],
}


def _calibrate(d, spec, m, width, scale="auto"):
    """Scale px -> dp from native widgets of known size; GPT draws them smaller than the platform does."""
    design_k = round((m.get("screen_dp") or _screen_dp(m.get("platform"))) / width, 5)
    spec["px_to_dp_design"] = design_k
    spec["px_to_dp"] = design_k
    anchors = []
    for c in spec.get("components") or []:
        for pat, dp, side in NATIVE_ANCHORS.get(m.get("platform") or "", []):
            px = c.get("bbox", {}).get(side) if isinstance(c, dict) and isinstance(c.get("bbox"), dict) else None
            if not isinstance(px, (int, float)) or isinstance(px, bool):
                break
            if px > 4 and re.search(pat, str(c.get("native", "")), re.I):
                anchors.append({"component": c.get("name", ""), "native": c["native"], "px": px, "dp": dp, "k": dp / px})
                break
    if not anchors:
        spec["native_scale"] = {"factor": 1.0, "anchors": [], "applied": False}
        if scale == "native":
            print("  ! --scale native: no native anchor widgets (nav bar, FAB, app bar...) found; kept the design scale")
        return
    k = float(np.median([x["k"] for x in anchors]))
    factor = k / design_k
    apply = scale == "native" or (scale == "auto" and abs(factor - 1) > 0.12)
    spec["native_scale"] = {"factor": round(factor, 3), "px_to_dp_native": round(k, 5), "anchors": anchors, "applied": apply}
    if apply:
        spec["px_to_dp"] = round(k, 5)
        spec["effective_width_dp"] = round(width * k)
        for x in m.get("assets") or []:
            if isinstance(x, dict) and isinstance(x.get("bbox"), dict) and {"w", "h"} <= x["bbox"].keys():
                x["dp"] = {"w": round(x["bbox"]["w"] * k, 1), "h": round(x["bbox"]["h"] * k, 1)}
        m["px_to_dp"] = spec["px_to_dp"]
        m["effective_width_dp"] = spec["effective_width_dp"]
        (d / "assets.json").write_text(json.dumps(m, indent=2))
        print(f"  native scale: GPT drew {', '.join(a['component'] for a in anchors[:3])} at {1 / factor:.2f}x platform size; "
              f"px_to_dp {design_k} -> {spec['px_to_dp']} (design depicts a ~{spec['effective_width_dp']}dp-wide screen; "
              "asset dp estimates updated). `spec --scale design` keeps the proportional scale.")
    elif scale == "design":
        print(f"  native scale: kept the proportional px_to_dp {design_k} (--scale design; native factor {factor:.2f})")
    else:
        print(f"  native scale: design matches platform sizes (factor {factor:.2f}); kept px_to_dp {design_k}")


def cmd_recalibrate(d, scale):
    d = Path(d)
    _need(d / "design_spec.json", "design spec", "run `spec DIR` first")
    spec = _read_json(d / "design_spec.json", "design spec")
    m = _read_json(d / "assets.json", "assets.json", required=("width",))
    if not isinstance(spec, dict) or not isinstance(m["width"], (int, float)) or m["width"] <= 0:
        sys.exit(f"gptkit: {d / 'design_spec.json'} or {d / 'assets.json'} is malformed (re-run `manifest` / `spec`)")
    _calibrate(d, spec, m, m["width"], scale)
    (d / "design_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"-> {d / 'design_spec.json'}  px_to_dp={spec['px_to_dp']}")


FIX_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["differences", "edit_instruction"],
              "properties": {"differences": {"type": "array", "items": {"type": "string"}},
                             "edit_instruction": {"type": "string"}}}

_warned_format = set()  # image formats a provider returned instead of PNG, announced once per process


def _png_verified(raw):
    """(png_bytes, source_format): the image result decoded with PIL and re-encoded as PNG when it was not one
    (belt and braces with the API layer). ValueError names undecodable bytes (an HTML error page, a cut-off download)."""
    try:
        im = Image.open(io.BytesIO(raw or b""))
        im.load()
    except Exception as e:  # PIL raises UnidentifiedImageError/OSError/ValueError/SyntaxError
        head = (raw or b"")[:40]
        raise ValueError(f"the provider returned no decodable image ({type(e).__name__}; starts with {head!r})") from None
    fmt = im.format or "?"
    if fmt == "PNG":
        return raw, fmt
    if fmt not in _warned_format:
        _warned_format.add(fmt)
        print(f"  ! provider returned {fmt} instead of PNG: no alpha channel, background/output_format were ignored by "
              "this endpoint/model (the file is re-encoded as PNG)", file=sys.stderr)
    return _png_bytes(im.convert("RGBA")), fmt


def _forge_job(mock, asset, cand, model, vary=True):
    b = asset["bbox"]
    colors = f" ({', '.join(asset['colors'])})" if asset.get("colors") else ""
    if asset["kind"] in ("background", "photo") and not asset["transparent"]:
        # the region's own crop kept the design's composition closer than the full mockup did
        prompt = BACKGROUND_PROMPT.format(kind=asset["kind"], name=asset["name"], description=asset["description"],
                                          aspect=_aspect_words(b["w"], b["h"]))
        ref = ref_crop(mock, b, margin=0.02, long_side=1536)
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]), "opaque", model)
    else:
        ref = ref_crop(mock, b, center=True)
        bg = ("Transparent background (real alpha). Centered with about 10% empty padding on every side."
              if asset["transparent"] else
              "Fill the canvas edge to edge with the element's own artwork (no letterbox, no empty padding).")
        prompt = ASSET_PROMPT.format(kind=asset["kind"], name=asset["name"].replace("_", " "),
                                     description=asset["description"], colors=colors, bg=bg)
        prompt += CANDIDATE_FOCUS[cand % len(CANDIDATE_FOCUS)] if vary else ""
        img, meta = edit(_png_bytes(ref), prompt, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", model)
    img, fmt = _png_verified(img)
    meta = dict(meta or {})
    meta.setdefault("source_format", fmt)
    return asset["name"], cand, img, meta, ref
