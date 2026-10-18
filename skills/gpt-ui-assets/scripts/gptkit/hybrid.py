"""Material, state, stage (+ --compose)."""
import concurrent.futures as cf
import functools
import http.server
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .core import ApiError, _JAVA_KEYWORDS, _plan, _pos_float, _pos_int, edit, generate
from .imaging import _flag_weight, content_box, sheet, trim
from .frames import _decode_result, _opaque_share
from .previews import _chrome
from .tokens import _ident, _kotlinc_check, _kt_header, _uniq, cmd_tokens
from .fonts import cmd_fonts
from .redline import cmd_redline
from .compare import cmd_compare

# ---------------------------------------------------------------- hybrid screens: code builds the UI, GPT paints the materials
# GPT image models draw a plausible screen, not a designed one: chunky type, no hairlines, loud fills. The hybrid route
# keeps layout, type, states and motion in code (HTML/CSS/JS, measured and crisp) and asks GPT only for what code cannot
# fake: materials and real objects (foil, paper, glass, food, fabric). `material` paints them, `state` derives a state
# variant that stays pixel-identical outside the change, `stage` renders the page to stills and a scrubbed motion clip.

HYBRID_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=412, initial-scale=1">
<title>Screen</title>
<style>
/* Animate a mask radius smoothly: gradients do not interpolate, a registered length does. */
@property --hole { syntax: '<length>'; inherits: true; initial-value: 0px; }
:root { --paper: #F1EFEA; --ink: #151716; --muted: #55595A; --hair: #D9D5CC; --accent: #FF7A1A; }
* { box-sizing: border-box; margin: 0; padding: 0 }
body { width: 412px; height: 915px; overflow: hidden; position: relative; background: var(--paper); color: var(--ink);
  font-family: system-ui, sans-serif; -webkit-font-smoothing: antialiased; font-feature-settings: "tnum" 1 }
.hero { position: absolute; left: 24px; top: 72px; font-size: 34px; line-height: 40px; font-weight: 500; letter-spacing: -.035em }
.object { position: absolute; left: 24px; top: 160px; width: 160px; filter: drop-shadow(0 14px 22px rgba(60,52,40,.13)) }
.object img { display: block; width: 100% }
</style>
</head>
<body>
<div class="hero" data-in>One idea, executed quietly.</div>
<div class="object" data-in><img id="material" src="assets/material.png" alt=""></div>
<script>
// Page contract for `gptkit stage`:
//   body[data-ready="1"] once fonts, images and layout are done.
//   ?capture: lay out only; window.demo() builds the whole timeline as Web Animations and pauses it;
//   window.seek(ms) sets every animation's currentTime, so the recorder scrubs exact frames.
// Every motion is a Web Animation. Fill "forwards" by default; only the entrance fills "both", or a later
// animation's first keyframe leaks backwards into the idle frames.
const A = (el, kf, o) => el.animate(kf, Object.assign({fill: "forwards", easing: "cubic-bezier(.2,.8,.2,1)"}, o));
const params = new URLSearchParams(location.search);
function entrance(t0 = 0) {
  document.querySelectorAll("[data-in]").forEach((el, i) =>
    A(el, [{opacity: 0, transform: "translateY(10px)"}, {opacity: 1, transform: "none"}], {duration: 700, delay: t0 + 80 + i * 45, fill: "both"}));
}
window.demo = () => { entrance(0); document.getAnimations().forEach(a => a.pause()); };
window.seek = ms => document.getAnimations().forEach(a => { a.currentTime = ms; });
(async () => {
  await Promise.all([...document.images].map(i => i.decode().catch(() => {})));
  await document.fonts.ready;
  // Optional measured asset geometry. geo.js may assign window.GEO = {...}; missing file is harmless.
  try { const r = await fetch('geo.js'); if (r.ok) (0, eval)(await r.text()); } catch (_) {}
  window.GEO = window.GEO || {};
  if (!params.has("capture") && !params.has("still")) entrance(0);
  document.body.dataset.ready = "1";
})();
</script>
</body>
</html>
"""


def _merge_json(path, key, value):
    """Merge one entry into a JSON object file (re-read + temp + os.replace, so parallel runs keep each other's entries)."""
    p = Path(path)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except ValueError:
            data = {}
        data = data if isinstance(data, dict) else {}
    data[key] = value
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def _hybrid_flags(img, transparent=True):
    """Flags for a free-standing asset (no mockup to compare with): EMPTY, OPAQUE, CUT-OFF; plus its content box."""
    cb = content_box(img, 40)
    empty = cb is None or (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.002 * img.width * img.height
    flags = ["EMPTY"] if empty else []
    if transparent:
        share, corners = _opaque_share(img)
        if share > 0.95 or corners:
            flags.append("OPAQUE")
        elif not empty and (cb[0] <= 2 or cb[1] <= 2 or cb[2] >= img.width - 2 or cb[3] >= img.height - 2):
            flags.append("CUT-OFF")
    return flags, cb


def _image_bytes_checked(path, what):
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"{what}: no such file {p}")
    raw = p.read_bytes()
    return _decode_result(raw, f"{what} {p}"), raw


def cmd_material(a):
    """New material/object art for a code-built screen: N candidates in parallel, flagged and ranked; the best
    unflagged one is written to OUT/NAME.png (trimmed to its content with a small margin)."""
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    bg = "transparent" if a.transparent else "auto"
    prompt = a.prompt
    if a.transparent:
        prompt += (" One isolated object on a fully transparent background, whole and centred with clear margin on every side;"
                   " no text, no labels, no backdrop, no cast shadow outside the object.")
    ref = None
    if a.ref:
        _, ref = _image_bytes_checked(a.ref, "--ref")
        prompt = ("Use the reference only for its material, lighting and rendering quality, not its layout or content. " + prompt)
    _plan(a.candidates, 0, image_model=a.model, size=a.size, edits=bool(a.ref))

    def one(i):
        try:
            raw, meta = (edit(ref, prompt, size=a.size, background=bg, model=a.model, fidelity="low", task="concept") if ref
                         else generate(prompt, size=a.size, background=bg, model=a.model))
        except ApiError as e:
            return i, None, f"c{i} failed: {e}"
        img = _decode_result(raw, f"{a.name} c{i}")
        f = out / "raw" / f"{a.name}_c{i}.png"
        img.save(f)
        flags, cb = _hybrid_flags(img, a.transparent)
        return i, {"file": str(f.relative_to(out)), "size": list(img.size), "bbox": list(cb) if cb else None,
                   "flags": flags, "source_format": (meta or {}).get("source_format")}, None

    results = {}
    with cf.ThreadPoolExecutor(max_workers=min(a.jobs, a.candidates)) as ex:
        for fut in cf.as_completed([ex.submit(one, i) for i in range(1, a.candidates + 1)]):
            i, r, err = fut.result()
            if err:
                print(f"! {a.name} {err}")
                continue
            results[i] = r
            print(f"  {a.name} c{i}: {r['size'][0]}x{r['size'][1]} {' '.join(r['flags']) or 'ok'}")
    if not results:
        sys.exit(f"gptkit material: every candidate of {a.name} failed; re-run with --candidates 1 to see one error")
    usable = [i for i in sorted(results) if "EMPTY" not in results[i]["flags"]]
    best = min(usable, key=lambda i: (_flag_weight(results[i]["flags"]), i)) if usable else None
    tiles = [(f"c{i} {' '.join(r['flags']) or 'ok'}{' BEST' if i == best else ''}", Image.open(out / r["file"]))
             for i, r in sorted(results.items())]
    rev = sheet([tiles], out / "review" / f"material_{a.name}.png", T=360)
    entry = {"prompt": a.prompt, "ref": a.ref, "size": a.size, "transparent": a.transparent, "model": a.model or "auto",
             "candidates": {f"c{i}": r for i, r in sorted(results.items())}, "best": f"c{best}" if best else None}
    if best is None:
        _merge_json(out / "material.json", a.name, entry)
        sys.exit(f"gptkit material: no usable candidate for {a.name} (all EMPTY); see {rev}")
    img = Image.open(out / results[best]["file"]).convert("RGBA")
    final = trim(img, pad=a.margin) if a.transparent and a.trim else img
    final.save(out / f"{a.name}.png")
    entry["file"] = f"{a.name}.png"
    _merge_json(out / "material.json", a.name, entry)
    flags = results[best]["flags"]
    print(f"material {a.name}: c{best} -> {out / (a.name + '.png')} ({final.width}x{final.height})"
          f"{'  ! ' + ' '.join(flags) if flags else ''}\n  review: {rev}  (Read it: the flags rank, your eyes decide; "
          f"`gptkit material ... --name {a.name}` again re-rolls)")


def _region_mask(size, region, grow, feather):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if region[0] == "circle":
        cx, cy, r = region[1]
        r *= grow
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    else:
        x0, y0, x1, y1 = region[1]
        gx, gy = (x1 - x0) * (grow - 1) / 2, (y1 - y0) * (grow - 1) / 2
        d.rectangle((x0 - gx, y0 - gy, x1 + gx, y1 + gy), fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather else m


def _parse_region(a):
    try:
        if a.circle:
            cx, cy, r = (float(v) for v in a.circle.split(","))
            return ("circle", (cx, cy, r))
        x0, y0, x1, y1 = (float(v) for v in a.box.split(","))
        return ("box", (x0, y0, x1, y1))
    except ValueError:
        raise ValueError("state: --circle takes CX,CY,R and --box takes X0,Y0,X1,Y1, in pixels of SRC")


def _best_shift(src_l, cand_l, outside, search):
    """Integer (dx, dy) that best registers the candidate onto the source outside the changed region (coarse, then fine)."""
    def err(dx, dy, s=1):
        c = np.roll(np.roll(cand_l, dy, 0), dx, 1)
        return float(np.abs(c - src_l)[::s, ::s][outside[::s, ::s]].mean())
    best = min(((err(dx, dy, 3), dx, dy) for dx in range(-search, search + 1, 2) for dy in range(-search, search + 1, 2)))
    _, bx, by = best
    best = min(((err(dx, dy), dx, dy) for dx in range(bx - 2, bx + 3) for dy in range(by - 2, by + 3)))
    return best[1], best[2], best[0]


def cmd_state(a):
    """A state variant of an approved asset (popped, filled, lit, open...): one-reference edits, registered onto the
    source, and only the changed region composited back, so everything outside it stays pixel-identical."""
    src, raw_src = _image_bytes_checked(a.src, "state")
    region = _parse_region(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rawdir = out.parent / "raw"
    rawdir.mkdir(exist_ok=True)
    transparent = _opaque_share(src)[0] < 0.95
    prompt = ("Edit this image. Keep it identical in every pixel: the same object, position, size, framing, lighting and "
              f"texture. Change ONLY this: {a.change}. Everything else stays exactly as it is."
              + (" Transparent background." if transparent else ""))
    inner = np.asarray(_region_mask(src.size, region, 1.0, 0)) > 127
    outside = np.asarray(_region_mask(src.size, region, a.grow * 1.25, 0)) < 1
    src_l = np.asarray(src.convert("L"), dtype=np.float32)
    mask = _region_mask(src.size, region, a.grow, a.feather)

    def one(i):
        try:
            raw, _ = edit(raw_src, prompt, size=_state_size(src), background="transparent" if transparent else "auto",
                          model=a.model, fidelity="high", task="precise")
        except ApiError as e:
            return i, None, f"c{i} failed: {e}"
        c = _decode_result(raw, f"state c{i}")
        c.save(rawdir / f"{out.stem}_c{i}.png")
        if abs(c.width / c.height - src.width / src.height) > 0.03:
            return i, None, f"c{i}: returned {c.width}x{c.height}, not the source's aspect ({src.width}x{src.height})"
        c = c.resize(src.size, Image.LANCZOS) if c.size != src.size else c
        cl = np.asarray(c.convert("L"), dtype=np.float32)
        dx, dy, drift = _best_shift(src_l, cl, outside, a.search)
        c = Image.fromarray(np.roll(np.roll(np.asarray(c), dy, 0), dx, 1))
        change = float(np.abs(np.asarray(c.convert("L"), dtype=np.float32) - src_l)[inner].mean())
        comp = Image.composite(c, src, mask)
        flags = (["NO-CHANGE"] if change < a.min_change else []) + (["DRIFT"] if drift > 25 else [])
        return i, {"comp": comp, "shift": [dx, dy], "drift_outside": round(drift, 1), "change_inside": round(change, 1),
                   "flags": flags}, None

    res = {}
    with cf.ThreadPoolExecutor(max_workers=min(a.jobs, a.candidates)) as ex:
        for fut in cf.as_completed([ex.submit(one, i) for i in range(1, a.candidates + 1)]):
            i, r, err = fut.result()
            if err:
                print(f"! {err}")
                continue
            res[i] = r
            print(f"  c{i}: shift {r['shift']}, re-render drift outside {r['drift_outside']}, change inside "
                  f"{r['change_inside']} {' '.join(r['flags']) or 'ok'}")
    if not res:
        sys.exit("gptkit state: no candidate came back; re-run with --candidates 1 to see one error")
    best = min(res, key=lambda i: (1000 * ("NO-CHANGE" in res[i]["flags"]), res[i]["drift_outside"]))
    r = res[best]
    if "NO-CHANGE" in r["flags"]:
        sys.exit(f"gptkit state: no candidate changed the region (best change {r['change_inside']} < --min-change "
                 f"{a.min_change}); describe the change more concretely, or check --circle/--box")
    r["comp"].save(out)
    after = np.abs(np.asarray(r["comp"].convert("RGBA"), dtype=np.int16) - np.asarray(src.convert("RGBA"), dtype=np.int16))
    untouched = np.asarray(mask) == 0
    max_out = int(after[untouched].max()) if untouched.any() else 0
    if region[0] == "circle":
        cx, cy, rr = region[1]
        rr *= a.grow * 1.6
        box = (int(cx - rr), int(cy - rr), int(cx + rr), int(cy + rr))
    else:
        x0, y0, x1, y1 = region[1]
        gx, gy = (x1 - x0) * 0.5, (y1 - y0) * 0.5
        box = (int(x0 - gx), int(y0 - gy), int(x1 + gx), int(y1 + gy))
    tiles = [("source", src.crop(box))] + [(f"c{i} {' '.join(v['flags']) or 'ok'}{' BEST' if i == best else ''}",
                                             v["comp"].crop(box)) for i, v in sorted(res.items())]
    rev = sheet([tiles], out.parent / f"{out.stem}_state_review.png", T=320)
    _merge_json(out.parent / "state.json", out.name, {
        "src": a.src, "change": a.change, "region": {region[0]: list(region[1])}, "grow": a.grow, "feather": a.feather,
        "best": f"c{best}", "max_diff_outside_mask": max_out,
        "candidates": {f"c{i}": {k: v for k, v in c.items() if k != "comp"} for i, c in sorted(res.items())}})
    print(f"state {out.name}: c{best} -> {out}  (max difference outside the mask: {max_out}; pixel-identical there)\n"
          f"  review: {rev}")


def _state_size(img):
    w, h = img.size
    return "1536x1024" if w / h > 1.2 else "1024x1536" if h / w > 1.2 else "1024x1024"


def _wh(s, flag):
    try:
        w, h = (int(v) for v in s.lower().split("x"))
        if w > 0 and h > 0:
            return w, h
    except ValueError:
        pass
    raise ValueError(f"stage: {flag} takes WIDTHxHEIGHT in CSS px, e.g. 412x915")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        # The starter page optionally loads geo.js; a missing optional file is not a page error.
        if self.path.split("?", 1)[0].endswith("/geo.js") and not os.path.isfile(self.translate_path(self.path)):
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


def _motion_easing(value, warnings=None):
    """CSS easing to Compose expression. Returns (Kotlin expression, warning)."""
    value = (value or "linear").strip().lower()
    known = {"linear": "LinearEasing", "ease": "CubicBezierEasing(0.25f, 0.1f, 0.25f, 1f)",
             "ease-in": "CubicBezierEasing(0.42f, 0f, 1f, 1f)",
             "ease-out": "CubicBezierEasing(0f, 0f, 0.58f, 1f)",
             "ease-in-out": "CubicBezierEasing(0.42f, 0f, 0.58f, 1f)"}
    if value in known:
        return known[value], None
    m = re.fullmatch(r"cubic-bezier\(\s*([-+\d.e]+)\s*,\s*([-+\d.e]+)\s*,\s*([-+\d.e]+)\s*,\s*([-+\d.e]+)\s*\)", value)
    if m:
        vals = [float(x) for x in m.groups()]
        if all(math.isfinite(x) for x in vals):
            return "CubicBezierEasing(" + ", ".join(_kt_float(x) for x in vals) + ")", None
    if value.startswith("steps("):
        return "LinearEasing", f"unsupported easing {value}: using linear"
    return "LinearEasing", f"unknown easing {value}: using linear"


def _kt_float(value):
    """Finite Kotlin Float literal with an explicit suffix."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite motion value")
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return (s if "." in s else s + ".0") + "f"


def _transform_components(value):
    """Parse supported CSS transforms into Compose property values; matrix skew is rejected."""
    result = {"translationX": 0.0, "translationY": 0.0, "scaleX": 1.0, "scaleY": 1.0, "rotationZ": 0.0}
    if not value or value.strip().lower() in ("none", "initial"):
        return result
    def length(s):
        m = re.fullmatch(r"\s*([-+\d.e]+)(?:px)?\s*", s)
        if not m:
            raise ValueError(f"unsupported transform length {s!r}")
        return float(m.group(1))
    matrix = re.fullmatch(r"matrix\(([^)]+)\)", value.strip())
    if matrix:
        v = [float(x.strip()) for x in matrix.group(1).split(",")]
        if len(v) != 6 or not all(math.isfinite(x) for x in v):
            raise ValueError(f"invalid matrix transform {value!r}")
        a, b, c, d, tx, ty = v
        sx = math.hypot(a, b)
        if sx == 0:
            raise ValueError("matrix has zero x scale")
        det = a * d - b * c
        sy = det / sx
        skew = (a * c + b * d) / (sx * sx)
        if abs(skew) > 1e-5:
            raise ValueError("matrix contains skew (unsupported)")
        result.update(translationX=tx, translationY=ty, scaleX=sx, scaleY=sy,
                      rotationZ=math.degrees(math.atan2(b, a)))
        return result
    for name, args in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", value):
        vals = [x.strip() for x in re.split(r"\s*,\s*|\s+", args.strip()) if x.strip()]
        if name in ("translate", "translateX", "translateY"):
            if name == "translateX": result["translationX"] = length(vals[0])
            elif name == "translateY": result["translationY"] = length(vals[0])
            else:
                result["translationX"] = length(vals[0]); result["translationY"] = length(vals[1]) if len(vals) > 1 else 0.0
        elif name in ("scale", "scaleX", "scaleY"):
            if name == "scaleX": result["scaleX"] = float(vals[0])
            elif name == "scaleY": result["scaleY"] = float(vals[0])
            else:
                result["scaleX"] = float(vals[0]); result["scaleY"] = float(vals[1]) if len(vals) > 1 else float(vals[0])
        elif name == "rotate":
            m = re.fullmatch(r"\s*([-+\d.e]+)(deg|rad|turn)?\s*", vals[0])
            if not m: raise ValueError(f"unsupported rotation {vals[0]!r}")
            n, unit = float(m.group(1)), m.group(2) or "deg"
            result["rotationZ"] = math.degrees(n) if unit == "rad" else n * 360 if unit == "turn" else n
        else:
            raise ValueError(f"unsupported transform function {name}")
    if not re.fullmatch(r"\s*(?:[a-zA-Z]+\([^)]*\)\s*)+", value):
        raise ValueError(f"unsupported transform {value!r}")
    if not all(math.isfinite(x) for x in result.values()): raise ValueError("non-finite transform")
    return result


def _motion_property_values(prop, value):
    if prop == "transform":
        return _transform_components(str(value))
    if prop == "opacity":
        return {"opacity": float(value)}
    if prop.startswith("--"):
        m = re.fullmatch(r"\s*([-+\d.e]+)(?:px)?\s*", str(value))
        if not m: raise ValueError(f"unsupported custom property value {value!r}")
        return {re.sub(r"[^A-Za-z0-9]+", " ", prop).strip().title().replace(" ", ""): float(m.group(1))}
    return {}


def _emit_motion_kt(doc, prefix, package, author=None):
    """Emit a compact Compose keyframe object from the neutral extracted timeline."""
    class A: pass
    a = A(); a.author = author
    lines = [_kt_header(a, package), "import androidx.compose.animation.core.CubicBezierEasing",
             "import androidx.compose.animation.core.LinearEasing", "import androidx.compose.animation.core.keyframes",
             "import androidx.compose.animation.core.KeyframesSpec", "\n", f"object {prefix}Motion {{"]
    emitted, warnings = [], []
    property_counts = {}
    elems = {}
    for i, anim in enumerate(doc.get("animations", []), 1):
        target = anim.get("target", {})
        label = target.get("id") or target.get("dataName") or target.get("selector") or f"element{i}"
        elem = _ident(label, "camel")
        elems.setdefault(label, elem)
        timing = anim.get("timing", {})
        dur = max(0, int(round(float(timing.get("duration", 0)))))
        delay = int(round(float(timing.get("delay", 0))))
        if timing.get("iterations", 1) not in (1, 1.0):
            warnings.append(f"{label}: iterations={timing.get('iterations')} is preserved in motion.json but not repeated by Compose keyframes")
        if timing.get("direction", "normal") != "normal":
            warnings.append(f"{label}: direction={timing.get('direction')} is preserved in motion.json but not applied by Compose keyframes")
        name = _ident(f"animation {i} {label}", "camel")
        lines.append(f"    const val {name}DelayMillis = {delay}")
        lines.append(f"    const val {name}DurationMillis = {dur}")
        for frame in anim.get("keyframes", []):
            for prop, raw in frame.get("properties", {}).items():
                if prop not in ("transform", "opacity") and not prop.startswith("--"):
                    warnings.append(f"unsupported property {prop} on {label}")
        effect_easing = (timing.get("easing") or "linear").strip().lower()
        if effect_easing != "linear" and len(anim.get("keyframes", [])) > 2:
            warnings.append(f"{label}: the effect easing {effect_easing} spans the whole animation in CSS; it is applied "
                            "per keyframe segment in Compose, so the timing differs: move it onto the keyframes")
        mapped = {}
        for frame in anim.get("keyframes", []):
            offset = frame.get("offset")
            if offset is None:
                offset = frame.get("computedOffset")
            if offset is None: continue
            for prop, raw in frame.get("properties", {}).items():
                try: props = _motion_property_values(prop, raw)
                except ValueError as e:
                    warnings.append(f"{label} {prop}: {e}"); continue
                frame_easing = frame.get("easing") or "linear"
                # WAAPI reports the computed keyframe default as linear; inherit the effect easing in that case.
                easing, warn = _motion_easing(anim.get("timing", {}).get("easing")
                                              if frame_easing == "linear" else frame_easing)
                if warn: warnings.append(f"{label}: {warn}")
                for key, val in props.items(): mapped.setdefault(key, []).append((int(round(float(offset) * dur)), val, easing))
        for prop, frames in mapped.items():
            if len({round(v, 6) for _, v, _ in frames}) < 2:
                continue  # constant: transform() decomposes into all five channels, most of which do not move
            base = elems[label] + prop[:1].upper() + prop[1:]
            property_counts[base] = property_counts.get(base, 0) + 1
            fn = base if property_counts[base] == 1 else f"{base}{property_counts[base]}"
            lines.append(f"    val {fn}: KeyframesSpec<Float> get() = keyframes {{")
            lines.append(f"        durationMillis = {dur}")
            lines.append(f"        delayMillis = {delay}")
            for ms, val, easing in frames:
                lines.append(f"        {_kt_float(val)} at {max(0, ms)} using {easing}")
            lines.append("    }")
            emitted.append((label, prop))
    lines.append("}")
    return "\n".join(lines) + "\n", emitted, warnings, elems


def _emit_geo_kt(geo, prefix, package, author=None):
    class A: pass
    a = A(); a.author = author
    lines = [_kt_header(a, package)]
    def value(name, val, indent, used=None):
        used = used if used is not None else []
        ident = _uniq(used + [_ident(name, "pascal")])[-1]
        used.append(ident)
        if isinstance(val, bool): return [" " * indent + f"const val {ident} = {str(val).lower()}"]
        if isinstance(val, (int, float)):
            return [" " * indent + f"const val {ident} = {_kt_float(val)}"]
        if isinstance(val, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val):
            return [" " * indent + f"val {ident} = listOf(" + ", ".join(_kt_float(x) for x in val) + ")"]
        if isinstance(val, dict):
            rows = [" " * indent + f"object {ident} {{"]
            child_used = []
            for k, v in val.items(): rows += value(k, v, indent + 4, child_used)
            return rows + [" " * indent + "}"]
        return []
    lines.append(f"object {prefix}Geo {{")
    used = [f"{prefix}Geo"]
    for k, v in (geo or {}).items(): lines += value(k, v, 4, used)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _lint_stage_compose(files):
    findings = []
    decl = re.compile(r"^\s*(?:const\s+)?(?:val|var|fun|object|class)\s+(\w+)")
    for f in map(Path, files):
        text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), f.read_text(), flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
        if text.count("{") != text.count("}"):
            findings.append(f"{f.name}: unbalanced braces")
        names = {}
        scopes = ["<top>"]
        for n, line in enumerate(text.splitlines(), 1):
            m = decl.match(line)
            if m:
                key = (tuple(scopes), m.group(1))
                if key in names: findings.append(f"{f.name}:{n}: duplicate identifier {m.group(1)}")
                names[key] = n
                if line.lstrip().startswith("object "):
                    scopes.append(m.group(1))
            if line.strip() == "}" and len(scopes) > 1:
                scopes.pop()
            if re.search(r"(?<![A-Za-z])(?:NaN|Infinity)(?![A-Za-z])", line): findings.append(f"{f.name}:{n}: non-finite literal")
            for lit in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", line):
                if re.search(re.escape(lit) + r"\s*(?:at\s+\d+|\n|$)", line) and "Millis =" not in line:
                    findings.append(f"{f.name}:{n}: float literal {lit} requires f suffix")
    return findings


def _stage_compose_files(doc, out, a, page):
    out.mkdir(parents=True, exist_ok=True)
    kt, properties, warnings, _ = _emit_motion_kt(doc, a.prefix, a.package, a.author)
    motion = out / f"{a.prefix}Motion.kt"; motion.write_text(kt)
    files = [motion]
    if isinstance(doc.get("geo"), dict):
        geo = out / f"{a.prefix}Geo.kt"; geo.write_text(_emit_geo_kt(doc["geo"], a.prefix, a.package, a.author)); files.append(geo)
    # capture where the motion changes: every keyframe of every animation, delay included (absolute ms on the
    # page's timeline), capped so the template stays a handful of screenshots rather than one per video frame
    times = sorted({int(round(float(x.get("timing", {}).get("delay", 0) or 0)
                              + float(k.get("offset") if k.get("offset") is not None else k.get("computedOffset") or 0)
                              * float(x.get("timing", {}).get("duration", 0) or 0)))
                    for x in doc.get("animations", []) for k in x.get("keyframes", [])} | {0})
    if len(times) > 24:
        times = [times[round(i * (len(times) - 1) / 23)] for i in range(24)]
    class A: pass
    h = A(); h.author = a.author
    test = [_kt_header(h, a.package), "import androidx.compose.ui.test.junit4.createComposeRule",
            "import com.github.takahirom.roborazzi.captureRoboImage",
            "import org.junit.Rule", "import org.junit.Test", "import androidx.compose.runtime.Composable", "",
            f"class {a.prefix}MotionScreenshotTest {{", "    @get:Rule val composeTestRule = createComposeRule()", "",
            f"    @Composable private fun {a.prefix}Screen() {{ /* TODO: port the approved screen to Compose */ }}", "",
            "    @Test fun captureTimeline() {", "        composeTestRule.mainClock.autoAdvance = false", f"        composeTestRule.setContent {{ {a.prefix}Screen() }}"]
    prev = 0
    for ms in times:  # advanceTimeBy takes a delta, the names carry the absolute time
        if ms > prev:
            test.append(f"        composeTestRule.mainClock.advanceTimeBy({ms - prev})")
        test.append(f'        composeTestRule.onRoot().captureRoboImage("{a.prefix.lower()}_{ms}.png")')
        prev = ms
    test += ["    }", "}"]
    tf = out / f"{a.prefix}MotionScreenshotTest.kt"; tf.write_text("\n".join(test) + "\n"); files.append(tf)
    (out / "motion.json").write_text(json.dumps(doc, indent=2))
    lint = _lint_stage_compose(files)
    print(f"  Compose lint: {'FAILED: ' + '; '.join(lint[:5]) if lint else 'passed'}")
    verdict = _kotlinc_check(files, a.package, [])
    print(f"  Compose compiler: {verdict}")
    return properties, warnings, files


def cmd_stage(a):
    """Render a code-built page: stills at --dpr, and with --motion a scrubbed clip (demo()/seek(ms) page contract),
    encoded to MP4 with ffmpeg (animated WebP without it) plus a labelled keyframe sheet."""
    page = Path(a.page).resolve()
    if a.compose:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", a.prefix or ""):
            sys.exit(f"gptkit stage: --prefix {a.prefix!r} is not a Kotlin identifier prefix")
        pkg_re = r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
        kt_keywords = _JAVA_KEYWORDS | {"as", "fun", "in", "is", "object", "typealias", "typeof", "val", "var", "when"}
        if not re.fullmatch(pkg_re, a.package or "") or any(x in kt_keywords for x in (a.package or "").split(".")):
            sys.exit(f"gptkit stage: --package {a.package!r} is not a valid Kotlin package")
    if a.init:
        if page.exists():
            sys.exit(f"gptkit stage: {page} already exists; --init only writes a new page")
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(HYBRID_PAGE)
        print(f"wrote {page}: the page contract (data-ready, ?capture, demo(), seek(ms)) and the animation helper are in it")
        return
    if not page.is_file():
        raise RuntimeError(f"stage: no such page {page} (create one with --init)")
    try:
        from playwright.sync_api import sync_playwright, Error as PwError
    except ImportError:
        sys.exit("gptkit stage: needs Playwright: pip install playwright (it drives your installed Chrome; "
                 "without one, also run `playwright install chromium`)")
    w, h = _wh(a.size, "--size")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stills = []
    for s in a.still or ["screen="]:
        name, _, query = s.partition("=")
        if not name or "/" in name:
            raise ValueError(f"stage: --still takes NAME=QUERY (e.g. after=state=after), got {s!r}")
        stills.append((name, query))
    root = page.parent
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root)))
    server_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    server_thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/{page.name}"
    errors, report = [], {"page": str(page), "size": [w, h], "dpr": a.dpr, "stills": {}}
    try:
        with sync_playwright() as p:
            exe = _chrome()
            b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()

            def open_page(dpr, query):
                pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=dpr)
                pg.on("pageerror", lambda e: errors.append(f"page error: {e}"))
                pg.on("console", lambda m: m.type == "error" and not m.text.startswith("Failed to load resource")
                      and errors.append(f"console: {m.text}"))
                pg.on("response", lambda r: r.status >= 400 and not r.url.endswith("/favicon.ico")
                      and errors.append(f"HTTP {r.status} {r.url.split(str(srv.server_address[1]), 1)[-1]}"))
                pg.goto(base + ("?" + query if query else ""))
                try:
                    pg.wait_for_selector("body[data-ready='1']", state="attached", timeout=a.timeout * 1000)
                except PwError:
                    raise RuntimeError(f"stage: {page.name}{'?' + query if query else ''} never set body[data-ready='1'] "
                                       f"within {a.timeout}s (see the page contract: `gptkit stage PAGE --init`)")
                return pg

            if a.compose:
                pg = open_page(1, "capture")
                if not pg.evaluate("typeof window.demo === 'function'"):
                    raise RuntimeError("stage: --compose needs window.demo() on the page (?capture); see `gptkit stage PAGE --init`")
                pg.evaluate("window.demo()")
                doc = pg.evaluate("""() => {
                  const animations = [...document.getAnimations()].map((a, i) => {
                    const e = a.effect, t = e && e.target, r = t && t.getBoundingClientRect();
                    let selector = null;
                    if (t && !t.id && !t.dataset.name) {
                      const parts = []; let n = t;
                      while (n && n.nodeType === 1 && parts.length < 4) {
                        let p = n.tagName.toLowerCase();
                        if (n.classList.length) p += '.' + [...n.classList].slice(0, 2).join('.');
                        const siblings = n.parentElement ? [...n.parentElement.children].filter(x => x.tagName === n.tagName) : [];
                        if (siblings.length > 1) p += `:nth-of-type(${siblings.indexOf(n) + 1})`;
                        parts.unshift(p); n = n.parentElement;
                      }
                      selector = parts.join(' > ');
                    }
                    const frames = e ? e.getKeyframes().map(k => {
                      const properties = {};
                      for (const [key, value] of Object.entries(k))
                        if (!['offset','computedOffset','easing','composite'].includes(key)) properties[key] = value;
                      // offset is null when the page did not set one; computedOffset is where Chrome places it
                      return {offset: k.offset ?? k.computedOffset, easing: k.easing || 'linear', properties};
                    }) : [];
                    const timing = e ? e.getTiming() : {};
                    return {index: i + 1, target: {id: t && t.id || null, dataName: t && t.dataset.name || null,
                      selector, warning: t && !t.id ? 'animation target has no id' : null,
                      rect: r ? {x:r.x,y:r.y,width:r.width,height:r.height} : null},
                      keyframes: frames, timing: {delay: timing.delay || 0, duration: typeof timing.duration === 'number' ? timing.duration : 0,
                        iterations: Number.isFinite(timing.iterations) ? timing.iterations : 'Infinity',
                        direction: timing.direction, fill: timing.fill, easing: timing.easing || 'linear'}};
                  });
                  return {schema: 'gptkit-motion-v1', viewport: {width: innerWidth, height: innerHeight},
                    geo: window.GEO && Object.keys(window.GEO).length ? window.GEO : null, animations};
                }""")
                warnings = []
                for anim in doc["animations"]:
                    if anim["target"].get("warning"):
                        warnings.append(f"animation {anim['index']} target has no id: {anim['target'].get('dataName') or anim['target'].get('selector')}")
                props, compose_warnings, files = _stage_compose_files(doc, Path(a.compose), a, page)
                warnings.extend(compose_warnings)
                pg.close()
                b.close()
                print(f"stage compose: {len(doc['animations'])} animation(s) on {len({x['target'].get('id') or x['target'].get('dataName') or x['target'].get('selector') for x in doc['animations']})} element(s); "
                      f"properties {', '.join(sorted({p for _, p in props})) or 'none'}; wrote {', '.join(f.name for f in files)} and motion.json")
                for warning in sorted(set(warnings)): print(f"! {warning}")
                return

            for name, query in stills:
                pg = open_page(a.dpr, query)
                pg.wait_for_timeout(a.settle)
                f = out / f"{name}.png"
                pg.screenshot(path=str(f))
                pg.close()
                report["stills"][name] = {"query": query, "file": f.name}
                print(f"  still {name}: {f} ({w * a.dpr}x{h * a.dpr})")
            if a.motion:
                pg = open_page(a.motion_dpr, "capture" + ("&" + a.query if a.query else ""))
                if not pg.evaluate("typeof window.demo === 'function' && typeof window.seek === 'function'"):
                    raise RuntimeError("stage: --motion needs window.demo() and window.seek(ms) on the page (?capture); "
                                       "see `gptkit stage PAGE --init`")
                pg.evaluate("window.demo()")
                frames = out / "frames"
                shutil.rmtree(frames, ignore_errors=True)
                frames.mkdir()
                n = max(1, int(round(a.motion * a.fps)))
                for i in range(n):
                    pg.evaluate(f"window.seek({i * 1000 / a.fps})")
                    pg.screenshot(path=str(frames / f"{i:04d}.png"))
                pg.close()
                report["motion"] = {"fps": a.fps, "frames": n, "seconds": a.motion, "dpr": a.motion_dpr}
                ff = shutil.which("ffmpeg")
                if ff:
                    mp4 = out / "motion.mp4"
                    r = subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "%04d.png"),
                                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                        "-crf", "16", "-movflags", "+faststart", str(mp4)], capture_output=True, text=True)
                    if r.returncode:
                        raise RuntimeError(f"stage: ffmpeg failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
                    report["motion"]["file"] = mp4.name
                else:
                    ims = [Image.open(frames / f"{i:04d}.png").convert("RGB") for i in range(n)]
                    durs = [round((i + 1) * 1000 / a.fps) - round(i * 1000 / a.fps) for i in range(n)]
                    webp = out / "motion.webp"
                    ims[0].save(webp, save_all=True, append_images=ims[1:], duration=durs, loop=0, quality=90, method=6)
                    report["motion"]["file"] = webp.name
                    print("  (no ffmpeg: wrote an animated WebP instead of MP4)")
                picks = sorted({min(n - 1, round(k * (n - 1) / max(1, a.sheet - 1))) for k in range(a.sheet)})
                rows, row = [], []
                for k in picks:
                    row.append((f"{round(k * 1000 / a.fps)} ms", Image.open(frames / f"{k:04d}.png")))
                    if len(row) == 6:
                        rows.append(row)
                        row = []
                if row:
                    rows.append(row)
                sheet(rows, out / "motion_sheet.png", T=300)
                print(f"  motion: {n} frames at {a.fps} fps -> {out / report['motion']['file']}; keyframes: {out / 'motion_sheet.png'}")
            b.close()
    finally:
        srv.shutdown()
        srv.server_close()
        server_thread.join(timeout=2)
    report["errors"] = errors
    (out / "stage.json").write_text(json.dumps(report, indent=2))
    for e in errors:
        print(f"! {e}")
    print(f"stage: {len(stills)} still(s){' + motion' if a.motion else ''} -> {out}")


def _hybrid_parsers(sub):
    p = sub.add_parser("material", help="new material/object art for a code-built screen (transparent candidates, flagged)")
    p.add_argument("prompt", help="the object and its material, light and camera; no UI")
    p.add_argument("--out", required=True, help="assets dir: NAME.png, raw/, review/, material.json")
    p.add_argument("--name", required=True, help="asset name (file stem)")
    p.add_argument("--size", default="1536x1024", help="WIDTHxHEIGHT canvas: 1536x1024, 1024x1536 or 1024x1024")
    p.add_argument("--candidates", type=_pos_int, default=2, help="images generated in parallel")
    p.add_argument("--jobs", type=_pos_int, default=4)
    p.add_argument("--ref", help="one approved image to match in material and rendering quality (not layout)")
    p.add_argument("--opaque", dest="transparent", action="store_false", help="a full-bleed texture/background instead")
    p.add_argument("--no-trim", dest="trim", action="store_false", help="keep the full canvas instead of trimming to content")
    p.add_argument("--margin", type=_pos_float, default=0.02, help="margin kept around the trimmed content (share of size)")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_material)
    p = sub.add_parser("state", help="state variant of an approved asset, pixel-identical outside the changed region")
    p.add_argument("src", help="the approved asset PNG")
    p.add_argument("change", help='what changes, e.g. "the third cavity has been pushed through: foil torn, empty"')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--circle", help="changed region CX,CY,R in SRC pixels")
    g.add_argument("--box", help="changed region X0,Y0,X1,Y1 in SRC pixels")
    p.add_argument("--out", required=True, help="output PNG (review sheet, raw/ and state.json go next to it)")
    p.add_argument("--candidates", type=_pos_int, default=2); p.add_argument("--jobs", type=_pos_int, default=4)
    p.add_argument("--grow", type=_pos_float, default=1.16, help="composite radius/size as a multiple of the region")
    p.add_argument("--feather", type=float, default=7, help="mask feather in px")
    p.add_argument("--search", type=_pos_int, default=8, help="max registration shift in px")
    p.add_argument("--min-change", type=float, default=4, help="mean grey change inside the region below which nothing changed")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_state)
    p = sub.add_parser("stage", help="render a code-built page: stills, scrubbed motion, or Compose motion files")
    p.add_argument("page", help="the HTML page (its folder is served over a local http server)")
    p.add_argument("--out", default="shots", help="output dir (default shots)")
    p.add_argument("--size", default="412x915", help="viewport WIDTHxHEIGHT in CSS px (412x915 phone; 1600x1200 a presentation shot)")
    p.add_argument("--dpr", type=_pos_int, default=3, help="device pixel ratio of the stills")
    p.add_argument("--still", action="append", help="NAME=QUERY, repeatable (e.g. before=still --still after=state=after)")
    p.add_argument("--motion", type=_pos_float, help="seconds of the demo() timeline to record")
    p.add_argument("--fps", type=_pos_int, default=30); p.add_argument("--motion-dpr", type=_pos_int, default=2)
    p.add_argument("--query", help="extra query string for the motion page (added after capture)")
    p.add_argument("--sheet", type=_pos_int, default=12, help="keyframes on the motion sheet")
    p.add_argument("--settle", type=int, default=300, help="ms to wait after ready before a still")
    p.add_argument("--timeout", type=_pos_int, default=30, help="seconds to wait for body[data-ready]")
    p.add_argument("--init", action="store_true", help="write a starter page with the page contract, then stop")
    p.add_argument("--compose", metavar="OUTDIR", help="extract the Web Animation timeline and emit Compose Kotlin")
    p.add_argument("--package", default="com.acme.app.ui.hybrid", help="Kotlin package for generated Compose files")
    p.add_argument("--prefix", default="Hybrid", help="Kotlin identifier prefix")
    p.add_argument("--author", help="adds an @author/@since header to generated Kotlin files")
    p.set_defaults(fn=cmd_stage)


def _design_to_code_parsers(sub):
    p = sub.add_parser("tokens", help="design_spec.json -> Compose/SwiftUI/CSS/Tailwind theme code (palette snapped to pixels)")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/tokens")
    p.add_argument("--package", default="com.example.app.ui.theme", help="Kotlin package for the Compose files")
    p.add_argument("--r-package", help="package of the app's R class: the module namespace in AGP 8 (default: --package minus .ui.theme)")
    p.add_argument("--author", help="adds an @author/@since header to Kotlin files")
    p.add_argument("--prefix", default="App", help="identifier prefix for the generated objects (AppColors, AppTypography, ...)")
    p.add_argument("--platforms", default="compose,swiftui,web",
                   help="comma-separated subset of compose, swiftui, web (aliases: android, ios)")
    p.add_argument("--fonts", help="fonts.json from `fonts` (default DIR/fonts/fonts.json when it exists)")
    p.add_argument("--no-snap", action="store_true", help="keep the spec's hex guesses")
    p.add_argument("--dtcg", action="store_true", help="also write tokens.tokens.json (W3C Design Tokens, Style Dictionary v4 subset)")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.add_argument("--verify", action="store_true", help="lint the emitted code; kotlinc when on PATH, swiftc on macOS, "
                                                         "Tailwind config in node, CSS in headless Chrome")
    p.set_defaults(fn=cmd_tokens)
    p = sub.add_parser("fonts", help="match typography to Google Fonts by rendering + measuring; download winners")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/fonts")
    p.add_argument("--candidates", type=int, default=24, help="similar families per group besides the guessed ones (0 = named only)")
    p.add_argument("--weights", type=_pos_int, default=3, help="weights tried per family and style")
    p.add_argument("--extra", help="comma-separated Google Fonts families to add to every shortlist")
    p.add_argument("--family", help="force this Google Fonts family for the groups of its category (every group when none matches)")
    p.add_argument("--tie", type=float, default=0.02, help="keep the spec's named family if it scores within this of the best")
    p.add_argument("--show", type=_pos_int, default=4, help="candidates per row on the review sheet")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.set_defaults(fn=cmd_fonts)
    p = sub.add_parser("compare", help="implemented-screen screenshot vs approved mockup: heatmap, score, ranked fixes")
    p.add_argument("dir", help="build dir with assets.json / design_spec.json"); p.add_argument("screenshot")
    p.add_argument("--mockup", help="default: the dir's mockup"); p.add_argument("--out", help="default DIR/compare")
    p.add_argument("--fonts", help="fonts.json from `fonts` (default DIR/fonts/fonts.json when it exists)")
    p.add_argument("--top", type=_pos_int, default=10, help="regions listed and boxed on the heatmap")
    p.add_argument("--vision", action="store_true", help="one gpt vision call describing the top regions")
    p.add_argument("--crop-top", type=int, default=0, help="screenshot px to drop at the top (status bar)")
    p.add_argument("--crop-bottom", type=int, default=0, help="screenshot px to drop at the bottom (nav bar)")
    p.add_argument("--shot-width-dp", type=_pos_int, help="dp width of the screen in the screenshot (warns when it differs "
                                                          "from the design's effective width by more than 5%%)")
    p.add_argument("--density", type=_pos_float, help="screenshot px per dp (alternative to --shot-width-dp)")
    p.add_argument("--fail-below", type=float, help="exit 1 when the overall score is below this (CI gate; default: no gate)")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("redline", help="pixel-true geometry: refined edges, corner radii, insets, gaps (dp) -> DIR/redline")
    p.add_argument("dir"); p.add_argument("--out", help="default DIR/redline")
    p.add_argument("--vision", action="store_true", help="locate text the spec components don't cover with one vision call")
    p.set_defaults(fn=cmd_redline)
