"""Forge, fix, pick, score."""
import base64
import concurrent.futures as cf
import functools
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .core import _chrome_err, _chrome_flags, _chrome_probe, _plan, _png_bytes, _read_json, _size_for, ask, edit
from .imaging import _backdrop, _lab, _rank, _rgba, _unusable, color_match, score, sheet, trim
from .design import FIX_SCHEMA, _asset, _forge_job, _load, _png_verified, _select

# ---- forge_report.json: dir-relative paths, re-read + merge + atomic replace (parallel forge/fix/pick runs are safe)

def _rep_rel(d, p):
    """A path as stored in forge_report.json: relative to the project dir when it lies inside it."""
    try:
        return Path(p).resolve().relative_to(Path(d).resolve()).as_posix()
    except ValueError:
        return str(p)


def _rep_abs(d, p):
    """A stored report path resolved: d/path (current reports), as-is (older cwd-relative reports), d/raw/<basename>."""
    d = Path(d)
    for c in (d / p, Path(p), d / "raw" / Path(p).name):
        if c.exists():
            return c
    return d / p


def _read_report(d):
    p = Path(d) / "forge_report.json"
    if not p.exists():
        return {}
    r = _read_json(p, "forge report")
    r = r if isinstance(r, dict) else {}
    ok = lambda e: isinstance(e, dict) and isinstance(e.get("candidates", []), list) and all(
        isinstance(c, dict) and isinstance(c.get("path"), str) for c in e.get("candidates", []))
    bad = [n for n, e in r.items() if not ok(e)]
    if bad:  # hand-edited or truncated entries: treated as never forged instead of crashing forge/fix/pick
        print(f"  ! {p}: ignoring damaged entries: {', '.join(bad)} (forge {d} --only {','.join(bad)} --force redoes them)")
    return {n: e for n, e in r.items() if n not in bad}


def _update_report(d, entries):
    """Merge entries into forge_report.json: re-read, replace only these names, write a temp file, os.replace."""
    p = Path(d) / "forge_report.json"
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text())
            cur = cur if isinstance(cur, dict) else {}
        except ValueError:
            print(f"  ! {p} was not valid JSON; rewritten with this run's entries")
    cur.update(entries)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, p)
    return cur


def _entry(d, report, name):
    """The report entry of a forged asset; exits when forge never produced a chosen candidate for it."""
    e = report.get(name)
    if isinstance(e, dict) and e.get("chosen") and not (isinstance(e["chosen"], dict) and isinstance(e["chosen"].get("path"), str)):
        sys.exit(f"gptkit: {Path(d) / 'forge_report.json'}: the '{name}' entry's chosen candidate is damaged "
                 f"({e['chosen']!r:.60}); run pick {d} {name} N or forge {d} --only {name} --force")
    if not isinstance(e, dict) or not e.get("chosen"):
        sys.exit(f"gptkit: no forged candidate for '{name}' in {Path(d) / 'forge_report.json'}: run forge {d} --only {name} first")
    return e


def _current(d, name, entry):
    """Path of the image behind an asset's final file: the colour-matched copy when one was kept, else the chosen
    candidate; falls back to assets/<name>.png with a warning when the raw file is gone."""
    cm = entry.get("color_match") or {}
    p = _rep_abs(d, cm["path"]) if cm.get("applied") and cm.get("path") else _rep_abs(d, entry["chosen"]["path"])
    if p.exists():
        return p
    alt = Path(d) / "assets" / f"{name}.png"
    if alt.exists():
        print(f"  ! {name}: {p} is gone (raw/ cleaned?); using {alt}")
        return alt
    sys.exit(f"gptkit: {name}: neither {p} nor {alt} exists; run forge {d} --only {name} --force")


def _color_final(mock, asset, path, sc, strength, d):
    """Colour-match a chosen candidate into d/raw/<stem>_cm.png, kept only if it scores better without new flags."""
    # opaque backgrounds are skipped: UI drawn over them skews the stats and desaturated the accents in tests
    if strength <= 0 or not asset["transparent"]:
        return None
    if sc["flags"]:
        print(f"  ~ {asset['name']}: colour match skipped (flagged {' '.join(sc['flags'])})")
        return None
    src = _rgba(path)
    img = color_match(mock, asset["bbox"], src, strength)
    if img is src:
        print(f"  ~ {asset['name']}: colour match skipped (too few opaque pixels to match)")
        return None
    new = score(mock, asset["bbox"], img, None, True)
    info = {"strength": strength, "fid_before": sc["fid"], "fid_after": new["fid"],
            "applied": new["fid"] < sc["fid"] and not new["flags"]}
    if new["flags"]:
        info["note"] = "introduced " + " ".join(new["flags"])
    if info["applied"]:
        p = Path(d) / "raw" / f"{Path(path).stem}_cm.png"
        img.save(p)
        info.update(path=_rep_rel(d, p), score=new)
    return info


def _finalize(asset, img):
    if asset["transparent"]:
        return trim(img, pad=0.02)
    b = asset["bbox"]
    tw, th = img.size
    target = b["w"] / b["h"]
    if tw / th > target:
        nw = round(th * target)
        return img.crop(((tw - nw) // 2, 0, (tw - nw) // 2 + nw, th))
    nh = round(tw / target)
    return img.crop((0, (th - nh) // 2, tw, (th - nh) // 2 + nh))


def _design_tile(mock, bbox, T=220):
    """The design element with 25% context for the review sheet: tiny elements upscaled NEAREST, the bbox outlined."""
    b = bbox
    pad = int(max(b["w"], b["h"]) * 0.25)
    box = (max(0, b["x"] - pad), max(0, b["y"] - pad), min(mock.width, b["x"] + b["w"] + pad), min(mock.height, b["y"] + b["h"] + pad))
    t = mock.crop(box).convert("RGBA")
    k = 1
    if max(t.size) < T // 2:
        k = max(1, (T // 2) // max(t.size))
        t = t.resize((t.width * k, t.height * k), Image.NEAREST)
    ImageDraw.Draw(t).rectangle([(b["x"] - box[0]) * k, (b["y"] - box[1]) * k, (b["x"] + b["w"] - box[0]) * k - 1,
                                 (b["y"] + b["h"] - box[1]) * k - 1], outline="#FF00FF", width=max(1, k // 2))
    return t


def _mid(s, n):
    return s if len(s) <= n else s[:n // 2 - 1] + "~" + s[-(n - n // 2):]


def _cand_label(c, best):
    return f"{'BEST ' if c is best else ''}c{c.get('cand', '?')} fid={c['score']['fid']} {' '.join(c['score']['flags'])}".rstrip()


def _incomplete(e):
    return isinstance(e, dict) and not e.get("complete", bool(e.get("chosen")))


def cmd_forge(a):
    d, m, mock = _load(a.dir)
    todo = [x for x in _select(m, a.only) if x["format"] == "png" or a.include_svg]
    if not todo:
        sys.exit(f"forge: no png-format assets{' among ' + a.only if a.only else ''} in {d / 'assets.json'} "
                 "(svg-format ones: `svg`, or forge --include-svg)")
    report = _read_report(d)
    keep = {}  # name -> already scored candidates of an interrupted run (their raw files still exist)
    if not a.force:
        rest = []
        for x in todo:
            e = report.get(x["name"]) or {}
            if (d / "assets" / f"{x['name']}.png").exists() and not _incomplete(e) and not e.get("needs_attention"):
                continue
            if _incomplete(e):
                keep[x["name"]] = [c for c in e.get("candidates", [])
                                   if isinstance(c.get("cand"), int) and _rep_abs(d, c["path"]).exists()]
            rest.append(x)
        if not rest:
            print("nothing to forge: every asset has a final file without flags (use --force to redo, --only NAME for one)")
            return
        todo = rest
    (d / "raw").mkdir(exist_ok=True)
    (d / "assets").mkdir(exist_ok=True)
    by_name = {x["name"]: x for x in todo}
    results = {x["name"]: list(keep.get(x["name"], [])) for x in todo}
    jobs = [(x, c) for x in todo for c in range(a.candidates) if c + 1 not in {k["cand"] for k in results[x["name"]]}]
    _plan(len(jobs), 0, image_model=a.model, task="precise", background="transparent", edits=True,
          size=[_size_for(x["bbox"]["w"], x["bbox"]["h"]) for x, _ in jobs] or "1024x1024")
    if getattr(a, "dry_run", False):
        print("would render: " + (", ".join(f"{x['name']} c{c + 1}" for x, c in jobs) or "nothing"))
        return
    pending = {x["name"]: sum(1 for j, _ in jobs if j is x) for x in todo}
    reused = sum(len(v) for v in results.values())
    print(f"forging {len(todo)} assets x {a.candidates} candidates = {len(jobs)} image calls, {a.jobs} parallel"
          + (f" ({reused} scored candidates of an interrupted run reused)" if reused else ""))
    rows, nocand, unusable, flagged = [], [], [], []

    def finish(name):
        asset, cands = by_name[name], results[name]
        entry = dict(report.get(name) or {}, candidates=cands, complete=True, dp=asset.get("dp"))
        cm = None
        if not cands:
            entry.update(chosen=None, color_match=None, final=None, needs_attention=True)
            nocand.append(name)
        else:
            cands.sort(key=lambda c: _rank(c["score"], c["meta"]))
            best = cands[0]
            entry.update(chosen=best, picked_manually=False)
            bad = _unusable(best["score"])
            if bad:
                entry.update(color_match=None, final=None, needs_attention=True, unusable=bad)
                print(f"  ! {name}: no usable candidate (the best is {bad}) - forge {d} --only {name} --force --candidates 3")
                unusable.append(name)
            else:
                entry.pop("unusable", None)
                cm = _color_final(mock, asset, _rep_abs(d, best["path"]), best["score"], a.color_match, d)
                entry["color_match"] = cm
                if cm:
                    print(f"  {name}: colour match {cm['strength']} fid {cm['fid_before']} -> {cm['fid_after']} "
                          f"({'kept' if cm['applied'] else 'discarded'}{'; ' + cm['note'] if cm.get('note') else ''})")
                final = _finalize(asset, _rgba(_current(d, name, entry)))
                out = d / "assets" / f"{name}.png"
                final.save(out)
                entry.update(final=_rep_rel(d, out), final_size=list(final.size), needs_attention=bool(best["score"]["flags"]))
                if entry["needs_attention"]:
                    flagged.append(name)
            rows.append([(f"{_mid(name, 24)} (design)", _design_tile(mock, asset["bbox"]))]
                        + [(_cand_label(c, best), _rgba(_rep_abs(d, c["path"]))) for c in cands]
                        + ([(f"BEST+colour fid={cm['fid_after']}", _rgba(_rep_abs(d, cm["path"])))] if cm and cm["applied"] else []))
        report[name] = entry
        _update_report(d, {name: entry})

    for name in [n for n, k in pending.items() if k == 0]:
        finish(name)
    ex = cf.ThreadPoolExecutor(max(1, min(a.jobs, len(jobs))))
    futs = {ex.submit(_forge_job, mock, x, c, a.model, not a.same_prompt): (x["name"], c + 1) for x, c in jobs}
    try:
        for f in cf.as_completed(futs):
            name, cn = futs[f]
            try:
                name, cand, raw, meta, ref = f.result()
                asset = by_name[name]
                p = d / "raw" / f"{name}_c{cand + 1}.png"
                p.write_bytes(raw)
                sc = score(mock, asset["bbox"], _rgba(p), ref, asset["transparent"])
                results[name].append({"cand": cand + 1, "path": _rep_rel(d, p), "score": sc,
                                      "meta": {k: meta.get(k) for k in ("quality", "size", "source_format")}})
                print(f"  {name} c{cand + 1}: fid={sc['fid']} (diff={sc['diff']} iou={sc['iou']} de={sc['de']} col={sc['col']}) "
                      f"flags={' '.join(sc['flags']) or '-'} warn={' '.join(sc['warn']) or '-'} quality={meta.get('quality')}")
            except Exception as e:  # ApiError (carries the prompt after the API-layer fix), undecodable bytes, scoring
                print(f"  ! {name} c{cn} failed: {e}")
            pending[name] -= 1
            if pending[name] == 0:
                finish(name)
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        partial = {n: dict(report.get(n) or {}, candidates=results[n], complete=False) for n, k in pending.items() if k > 0}
        if partial:
            _update_report(d, partial)
        print(f"\n  ! interrupted: {sum(len(v['candidates']) for v in partial.values())} scored candidates of "
              f"{', '.join(partial) or 'no asset'} kept; re-run forge {d} to draw only the missing ones")
        raise  # main() exits 130 without waiting for the calls still in flight
    except BaseException:  # a fatal error in finish(): do not start the queued image calls before exiting
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    if rows:
        (d / "review").mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i in range(0, len(rows), 6):
            print("review sheet ->", sheet(rows[i:i + 6], d / "review" / f"forge_{stamp}_{i // 6 + 1}.png"))
    bad = nocand + unusable
    ok = [n for n in results if n not in bad]
    print(f"forged {len(ok)}/{len(results)} assets" + (f"; failed: {', '.join(bad)} (re-run: forge {d} --only {','.join(bad)})" if bad else ""))
    if flagged:
        print(f"NEEDS ATTENTION (every candidate flagged): {', '.join(flagged)} -> read review/forge_*.png, then "
              f"pick {d} <name> <N>, fix {d} <name>, or forge {d} --only {','.join(flagged)} --force --candidates 3")
    if bad:
        sys.exit(1)


def cmd_fix(a):
    d, m, mock = _load(a.dir)
    asset = _asset(m, a.name)
    report = _read_report(d)
    entry = _entry(d, report, a.name)
    cur = _rgba(_current(d, a.name, entry))
    cur_score = score(mock, asset["bbox"], cur, None, asset["transparent"])
    b = asset["bbox"]
    is_bg = asset["kind"] in ("background", "photo") and not asset["transparent"]
    for rnd in range(a.rounds):
        T = 512
        pad = 6
        tgt = mock.crop((max(0, b["x"] - pad), max(0, b["y"] - pad), min(mock.width, b["x"] + b["w"] + pad),
                         min(mock.height, b["y"] + b["h"] + pad)))
        tgt = tgt.resize((max(1, round(tgt.width * T / tgt.height)), T), Image.LANCZOS).convert("RGB")
        g = trim(cur)
        cv = Image.new("RGBA", g.size, _backdrop(mock, b) + (255,))
        cv.alpha_composite(g)
        cv = cv.resize((max(1, round(cv.width * T / cv.height)), T), Image.LANCZOS).convert("RGB")
        sbs = Image.new("RGB", (tgt.width + cv.width + 30, T), "white")
        sbs.paste(tgt, (0, 0))
        sbs.paste(cv, (tgt.width + 30, 0))
        extra = f" The user also says: {a.note}" if a.note else ""
        left = ("LEFT is the design region with UI (text, buttons, cards) drawn on top: ignore all UI; compare only the "
                "artwork behind it. RIGHT is the recreated full-bleed artwork. " if is_bg else
                "LEFT is the target graphic from an approved design (low-res screenshot on its backdrop). RIGHT is a "
                "recreated asset on a similar backdrop. ")
        res = ask([left + "List every visible difference RIGHT must fix to match LEFT 1:1 (silhouette, proportions, "
                   "parts, colours with hex, gradients, glow, relative sizes). Ignore resolution/sharpness"
                   + (" and the backdrop" if not is_bg else " and the UI") + ". Then write ONE concise corrective edit "
                   f"instruction (<=100 words) limited to the real differences; both empty if nothing differs.{extra}",
                   sbs], FIX_SCHEMA, "fix")
        diffs = [str(x).strip() for x in (res.get("differences") or []) if str(x).strip()]
        instruction = str(res.get("edit_instruction") or "").strip()
        if not diffs or not instruction:
            print(f"round {rnd + 1}: no visible differences - nothing to fix")
            break
        print(f"round {rnd + 1} differences:", *[f"  - {x}" for x in diffs], sep="\n")
        head = "Keep everything that already matches; change only what is listed. "
        if is_bg:
            prompt = ("Revise the attached full-bleed background artwork. " + head +
                      "Clean high-resolution redraw; keep it full-bleed, no UI, no text. Corrections: ")
        elif asset["transparent"]:
            prompt = ("Revise the attached standalone asset. " + head +
                      "Clean high-resolution redraw, transparent background, no backdrop, no text. Corrections: ")
        else:
            prompt = ("Revise the attached standalone asset. " + head + "Clean high-resolution redraw. Fill the canvas "
                      "edge to edge with the element's own artwork (no letterbox), no text. Corrections: ")
        raw, meta = edit(_png_bytes(cur), prompt + instruction, _size_for(b["w"], b["h"]),
                         "transparent" if asset["transparent"] else "opaque", a.model)
        raw, fmt = _png_verified(raw)
        n = 1 + sum(1 for c in entry.get("candidates", []) if str(c.get("cand", "")).startswith("fix"))
        p = d / "raw" / f"{a.name}_fix{n}.png"
        while p.exists():
            n += 1
            p = d / "raw" / f"{a.name}_fix{n}.png"
        p.write_bytes(raw)
        new = _rgba(p)
        sc = score(mock, b, new, None, asset["transparent"])
        cm = _color_final(mock, asset, p, sc, a.color_match, d)
        final_img, final_sc = (_rgba(_rep_abs(d, cm["path"])), cm["score"]) if cm and cm["applied"] else (new, sc)
        bad = _unusable(final_sc)
        better = _rank(final_sc, meta) < _rank(cur_score, None)
        kept = (better or a.accept) and not bad
        cand = {"cand": f"fix{n}", "path": _rep_rel(d, p), "score": sc,
                "meta": {"quality": (meta or {}).get("quality"), "size": (meta or {}).get("size"), "source_format": fmt}}
        entry.setdefault("candidates", []).append(cand)
        why = "KEPT" if kept else (f"discarded ({bad})" if bad else "discarded (worse)")
        print(f"round {rnd + 1}: fid {cur_score['fid']} -> {final_sc['fid']} (diff {cur_score['diff']} -> {final_sc['diff']}"
              f"{', colour-matched' if final_img is not new else ''}) flags={' '.join(sc['flags']) or '-'} -> {why}  ({p})")
        if kept:
            cur, cur_score = final_img, final_sc
            entry.update(chosen=cand, color_match=cm, picked_manually=False)
            final = _finalize(asset, final_img)
            out = d / "assets" / f"{a.name}.png"
            final.save(out)
            entry.update(final=_rep_rel(d, out), final_size=list(final.size), needs_attention=bool(sc["flags"]), complete=True)
            entry.pop("unusable", None)
        _update_report(d, {a.name: entry})


_warned_render = False


@functools.cache
def _cairosvg():
    try:
        import cairosvg
        return cairosvg
    except Exception:  # missing package or missing libcairo
        return None


@functools.cache
def _renderer_name():
    """The SVG rasteriser that works here: chrome (only when the probe render succeeds), cairosvg, rsvg-convert,
    qlmanage, or None."""
    from .previews import _chrome  # deferred: previews imports this module
    if _chrome() and _chrome_probe()[0]:
        return "chrome"
    if _cairosvg():
        return "cairosvg"
    return next((t for t in ("rsvg-convert", "qlmanage") if shutil.which(t)), None)


def _ensure_xmlns(txt):
    """SVG text with xmlns="http://www.w3.org/2000/svg" on the root (browsers ignore an <img> SVG without it)."""
    m_ = re.search(r"<svg\b[^>]*>", txt)
    if m_ and not re.search(r"""\sxmlns\s*=\s*["']http://www\.w3\.org/2000/svg["']""", m_.group(0)):
        return txt[:m_.start()] + m_.group(0).replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1) + txt[m_.end():]
    return txt


def _sized_svg(txt, size):
    """The SVG with width/height set to `size` (ElementTree round-trip: single quotes and duplicate attributes are safe)."""
    root = ET.fromstring(_ensure_xmlns(txt))
    root.set("width", str(size))
    root.set("height", str(size))
    return ET.tostring(root, encoding="unicode")


_warned_chrome = False


def _chrome_render(svgs, size):
    """Rasterise sized SVG texts with headless Chrome, one grid page per <= 256 icons; [RGBA Image or None]."""
    from .previews import _chrome  # deferred: previews imports this module
    global _warned_chrome
    out = []
    per = max(1, min(256, (2048 // size) * (8192 // size)))  # keep each page under ~2048 x 8192 px
    for k in range(0, len(svgs), per):
        chunk = svgs[k:k + per]
        cols = max(1, min(len(chunk), 2048 // size))
        rows = math.ceil(len(chunk) / cols)
        W, H = cols * size, rows * size
        cells = "".join(f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}" style="position:absolute;'
                        f'left:{i % cols * size}px;top:{i // cols * size}px;width:{size}px;height:{size}px">'
                        for i, s in enumerate(chunk))
        grid = None
        with tempfile.TemporaryDirectory() as td:
            html, png = Path(td) / "r.html", Path(td) / "r.png"
            html.write_text(f"<!doctype html><style>html,body{{margin:0;background:transparent}}</style>{cells}")
            try:
                r = subprocess.run([_chrome(), *_chrome_flags(W, H), "--virtual-time-budget=3000", f"--screenshot={png}",
                                    html.as_uri()], capture_output=True, text=True, timeout=120)
                err = _chrome_err(r) if not png.exists() else ""
            except (OSError, subprocess.TimeoutExpired) as e:
                err = f"{type(e).__name__}: {e}"
            if png.exists():
                grid = Image.open(png).convert("RGBA").crop((0, 0, W, H))
        if grid is None:
            if not _warned_chrome:
                print(f"  ! headless Chrome produced no screenshot: {err}")
                _warned_chrome = True
            out += [None] * len(chunk)
            continue
        out += [grid.crop((i % cols * size, i // cols * size, i % cols * size + size, i // cols * size + size))
                for i in range(len(chunk))]
    return out


def _matte(on_white, on_black):
    """Recover RGBA from two opaque renders on white and on black."""
    w = np.asarray(on_white.convert("RGB").resize(on_black.size), dtype=float)
    b = np.asarray(on_black.convert("RGB"), dtype=float)
    a = np.clip(255 - (w - b).mean(2), 0, 255)
    rgb = np.clip(b * 255 / np.maximum(a, 1)[..., None], 0, 255)
    return Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")


def _render_one(sized, size):
    if _cairosvg():
        try:
            return _rgba(io.BytesIO(_cairosvg().svg2png(bytestring=sized.encode(), output_width=size, output_height=size)))
        except Exception as e:
            print(f"  ! cairosvg failed: {str(e).splitlines()[0]}")
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "i.svg", Path(td) / "o.png"
        if shutil.which("rsvg-convert"):
            src.write_text(sized)
            subprocess.run(["rsvg-convert", "-w", str(size), "-h", str(size), "-o", str(out), str(src)], capture_output=True)
            return _rgba(out) if out.exists() else None
        if shutil.which("qlmanage"):
            # Quick Look has no alpha and pads/clips its thumbnail: render smaller on white and black, matte, re-frame
            vb = re.search(r'viewBox="([^"]+)"', sized)
            x, y, w, h = [float(v) for v in re.split(r"[\s,]+", vb.group(1).strip())] if vb else (0, 0, size, size)
            small = _sized_svg(sized, int(size * 0.75))
            shots = []
            for bg in ("#ffffff", "#000000"):
                src.write_text(re.sub(r"(<svg\b[^>]*>)", rf'\1<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{bg}"/>',
                                      small, count=1))
                subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", td, str(src)], capture_output=True)
                ql = Path(td) / "i.svg.png"
                if not ql.exists():
                    return None
                shots.append(Image.open(ql).convert("RGB"))
                ql.unlink()
            frame = ImageChops.difference(*shots).convert("L").point(lambda v: 255 if v > 128 else 0).getbbox()
            if not frame:
                return None
            frame = (frame[0] + 1, frame[1] + 1, frame[2] - 1, frame[3] - 1)
            return _matte(shots[0].crop(frame), shots[1].crop(frame)).resize((size, size), Image.LANCZOS)
    return None


def render_svgs(svgs, size=256):
    """Rasterise SVG texts with real alpha (headless Chrome, cairosvg, rsvg-convert, or Quick Look + matting);
    one Image or None per input (unparseable SVG, or no working renderer: said once)."""
    global _warned_render
    sized = []
    for s in svgs:
        try:
            sized.append(_sized_svg(s, size) if s else None)
        except (ET.ParseError, AttributeError, TypeError):
            sized.append(None)
    ok = [i for i, s in enumerate(sized) if s]
    out = [None] * len(svgs)
    if not ok:
        return out
    name = _renderer_name()
    if name == "chrome":
        for i, im in zip(ok, _chrome_render([sized[i] for i in ok], size)):
            out[i] = im
        return out
    if not name:
        if not _warned_render:
            print(f"  (no SVG renderer: {_chrome_probe()[1]}; install Chrome/Chromium (GPTKIT_CHROME) or pip install cairosvg)")
            _warned_render = True
        return out
    for i in ok:
        out[i] = _render_one(sized[i], size)
    return out


def _render_svg(svg_path, out_dir, size=256):
    """Rasterise one SVG file to out_dir/<stem>.png with transparency; returns the path or None."""
    svg_path, out_dir = Path(svg_path), Path(out_dir)
    im = render_svgs([svg_path.read_text()], size)[0]
    if im is None:
        return None
    final = out_dir / f"{svg_path.stem}.png"
    im.save(final)
    return final




def _element_mask(mock, bbox):
    """Segment a design element from its backdrop: (soft alpha, backdrop rgb, crop rgb) for bbox plus a margin."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    m = max(3, round(max(w, h) * 0.12))
    c = np.asarray(mock.crop((max(0, x - m), max(0, y - m), min(mock.width, x + w + m),
                              min(mock.height, y + h + m))).convert("RGB"), dtype=float)
    ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    bg = np.median(ring, 0)
    dist = np.linalg.norm(c - bg, axis=2)
    noise = np.percentile(np.linalg.norm(ring - bg, axis=1), 90)
    core = max(np.percentile(dist, 99), noise + 20)
    return np.clip((dist - noise) / (core - noise), 0, 1), bg, c


def _abox(a, thr=0.3):
    ys, xs = np.nonzero(a > thr)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def svg_fidelity(mock, bbox, render, visual=False):
    """Registered comparison of an SVG render with the design element (rel: 0 = perfect, 1 = as bad as empty)."""
    ad, bg, crop = _element_mask(mock, bbox)
    z = max(1.0, 96 / max(ad.shape))
    if z > 1:
        size = (round(ad.shape[1] * z), round(ad.shape[0] * z))
        crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize(size, Image.BICUBIC), dtype=float)
        ad = np.asarray(Image.fromarray((ad * 255).astype(np.uint8)).resize(size, Image.BICUBIC), dtype=float) / 255
    db, rb = _abox(ad), _abox(np.asarray(render.getchannel("A"), dtype=float) / 255)
    if not db or not rb:
        sc = {"fit": 0.0, "iou": 0.0, "ink": 0.0, "dE": 99.0, "diff": 99.0, "rel": 1.0, "aspect_err": 1.0,
              "design_hex": "", "svg_hex": "", "backdrop_hex": ""}
        return (sc, Image.fromarray(crop.astype(np.uint8))) if visual else sc
    dw, dh = db[2] - db[0], db[3] - db[1]
    g0 = render.crop(rb)
    s0 = min(dw / g0.width, dh / g0.height)
    H, W = ad.shape
    r = max(1, round(z))
    best = None
    for f in (0.95, 0.975, 1.0, 1.025, 1.05):
        g = g0.resize((max(1, round(g0.width * s0 * f)), max(1, round(g0.height * s0 * f))), Image.BOX)
        ga = g.getchannel("A")
        x0, y0 = round((db[0] + db[2] - g.width) / 2), round((db[1] + db[3] - g.height) / 2)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                m_ = Image.new("L", (W, H), 0)
                m_.paste(ga, (x0 + dx, y0 + dy))
                ar = np.asarray(m_, dtype=float) / 255
                iou = np.minimum(ad, ar).sum() / max(np.maximum(ad, ar).sum(), 1e-6)
                if best is None or iou > best[0]:
                    best = (iou, g, (x0 + dx, y0 + dy))
    iou, g, pos = best
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(g, pos) if pos[0] >= 0 and pos[1] >= 0 else layer.paste(g, pos)
    lr = np.asarray(layer, dtype=float)
    ar = lr[..., 3] / 255
    k = 2 * max(1, round(max(dw, dh) / 48)) + 1
    grow = lambda a: np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k)), dtype=float) / 255
    prec = np.minimum(ar, grow(ad)).sum() / max(ar.sum(), 1e-6)
    rec = np.minimum(ad, grow(ar)).sum() / max(ad.sum(), 1e-6)
    fit = 2 * prec * rec / max(prec + rec, 1e-6)
    comp = lr[..., :3] * ar[..., None] + bg * (1 - ar[..., None])
    core_d, core_r = ad > 0.85, ar > 0.85
    if core_d.sum() >= 3 and core_r.sum() >= 3:
        cd, cr = np.median(crop[core_d], 0), np.median(lr[..., :3][core_r], 0)
    else:
        cd, cr = np.average(crop.reshape(-1, 3), 0, ad.ravel() + 1e-6), np.average(lr[..., :3].reshape(-1, 3), 0, ar.ravel() + 1e-6)
    hexc = lambda c: "#" + "".join(f"{int(round(v)):02X}" for v in c)
    err = np.abs(comp - crop).mean()
    sc = {"fit": round(float(fit), 3), "iou": round(float(iou), 3), "ink": round(float(ar.sum() / max(ad.sum(), 1e-6)), 2),
          "dE": round(float(np.linalg.norm(_lab(cd) - _lab(cr))), 1), "diff": round(float(err), 1),
          "rel": round(float(err / max(np.abs(bg - crop).mean(), 1e-6)), 3),
          "aspect_err": round(float(abs((rb[2] - rb[0]) / (rb[3] - rb[1]) / (dw / dh) - 1)), 3),
          "design_hex": hexc(cd), "svg_hex": hexc(cr), "backdrop_hex": hexc(bg)}
    if not visual:
        return sc
    T = 256
    left, right = (Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)) for x in (crop, comp))
    left, right = (im.resize((round(im.width * T / im.height), T), Image.LANCZOS) for im in (left, right))
    sbs = Image.new("RGB", (left.width * 2 + 16, T), "white")
    sbs.paste(left, (0, 0))
    sbs.paste(right, (left.width + 16, 0))
    return sc, sbs
