"""Offline checks for the vector animation builders: python3 tests/test_vector.py (no API calls)."""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("gptkit", Path(__file__).parent.parent / "scripts" / "gptkit.py")
gk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gk)

HEART = ("M22 40.5 8.2 27.1C3.9 22.8 1.75 19.2 1.75 14.1 1.75 7.6 6.1 3.5 11.7 3.5 16.1 3.5 19.4 6.2 22 9.7 "
         "24.6 6.2 27.9 3.5 32.3 3.5 37.9 3.5 42.25 7.6 42.25 14.1 42.25 19.2 40.1 22.8 35.8 27.1Z")
MOTION = {"anchor_x": 22, "anchor_y": 22, "keys": [
    {"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "ease_in"},
    {"t": 0.25, "dx": 0, "dy": -2, "sx": 1.25, "sy": 1.25, "rotate": 8, "opacity": 0.6, "ease": "overshoot"},
    {"t": 0.6, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"}]}


def vec():
    return {"viewBox": "0 0 44 44", "defs": "", "layer_order": ["halo", "heart", "blob"],
            "static_layers": [{"name": "heart", "motion": MOTION, "paths": [
                {"d": HEART, "fill": "none", "opacity": 1, "stroke": "#8D8DB4", "stroke_width": 3.5}]}],
            "animated_layers": [{"name": "blob", "fill": "#FFB15E", "opacity": 1, "stroke": "none", "stroke_width": 0,
                                 "motion": {"anchor_x": 0, "anchor_y": 0, "keys": []},
                                 "keyframes": ["M10 10 L20 10 L15 20 Z", "M10 12 C12 8 18 8 20 12 L15 22 Z",
                                               "M9 10 L21 10 L15 18 Z", "M10 10 L20 10 L15 20 Z"]}],
            "glows": [{"name": "halo", "cx": 22, "cy": 22, "rx": 20, "ry": 14, "color": "#B9B6FF", "opacity": 0.5,
                       "motion": {"anchor_x": 22, "anchor_y": 22, "keys": []}}]}


def test_paths():
    subs = gk._subpaths("m10 10 h10 v10 a5 5 0 0 1 -10 0 q0 -5 5 -5 t5 0 s5 5 0 5 z")
    assert all(s[0] in "LC" for sp in subs for s in sp["segs"]) and subs[0]["closed"]
    arc = gk._subpaths("M0 0 A10 10 0 0 1 20 0")[0]["segs"]
    assert len(arc) == 4 and all(abs(math.dist(s[3], (10, 0)) - 10) < 1e-6 for s in arc)
    assert gk._subpaths(gk._d(subs)) == gk._subpaths(gk._d(gk._subpaths(gk._d(subs))))


def test_build():
    v = vec()
    assert not gk._validate_vec(v), gk._validate_vec(v)
    gk._normalize_vec(v)
    assert len(v["animated_layers"][0]["keyframes"]) == 3, "closing duplicate keyframe should be dropped"
    rep = gk.motion_report(v)
    assert rep["layers"]["heart"]["move"] > 0.05 and rep["layers"]["heart"]["fade"] >= 0.4
    lot = gk.build_lottie(v, 512, 1.2, 30)
    assert [lyr["nm"] for lyr in lot["layers"]] == ["blob", "heart", "halo"]
    heart = lot["layers"][1]["ks"]
    assert heart["s"]["a"] == 1 and heart["r"]["a"] == 1 and heart["a"]["k"][:2] == [22, 22]
    assert all(k["o"]["y"][0] <= 1 for k in heart["o"]["k"][:-1]), "opacity must not overshoot"
    fit = gk._fit_viewbox(v)
    assert fit and gk._vec_vb(dict(v, viewBox=fit))[2] > 44
    root = ET.fromstring(gk.build_avd(v, 1.2, 24))
    names = [t.get("{http://schemas.android.com/apk/res/android}name") for t in root.findall("target")]
    assert "heart" in names and {"blob_p0", "blob_p1", "blob_p2"} <= set(names), names


def _chrome_available():
    """True when headless Chrome renders here (the shared flags make it start as root); tests then MUST pass."""
    ok, _ = gk._chrome_probe()
    return ok or gk._chrome() is not None


def _patched(**kw):
    """Context manager: temporarily replace module attributes of gk."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        saved = {k: getattr(gk, k) for k in kw}
        try:
            for k, v in kw.items():
                setattr(gk, k, v)
            yield
        finally:
            for k, v in saved.items():
                setattr(gk, k, v)
    return cm()


def test_chrome_shot_full_height():
    """G6-1: --headless=new lays out ~87 px less than --window-size; the padded capture is cropped to (0, 0, w, h)."""
    with tempfile.TemporaryDirectory() as td:
        if not _chrome_available():
            assert gk._chrome_shot("<body>", Path(td) / "s.png", 64, 64) is None and gk._SHOT_REASON
            print("  (skipped the Chrome part: no Chrome/Chromium)")
            return
        for w, h in ((256, 256), (2048, 256), (512, 512)):
            im = gk._chrome_shot(f"<body style='margin:0'><div style='width:{w}px;height:{h}px;background:red'></div>",
                                 Path(td) / "s.png", w, h)
            assert im is not None, gk._SHOT_REASON
            a = np.asarray(im)[..., 3]
            assert im.size == (w, h) and int((a.min(1) == 255).sum()) == h, (w, h, im.size, int((a.min(1) == 255).sum()))
        # a failing binary: None, a reason naming the binary, and the warning printed once
        with _patched(_chrome=lambda: "/bin/false", _warned_shot=False):
            assert gk._chrome_shot("<body>", Path(td) / "s.png", 64, 64) is None
            assert "/bin/false" in gk._SHOT_REASON


def test_review_sheet_how_string():
    """G6-1: keys_review.png says how it was rendered and never leaves key_*.svg behind."""
    v = vec()
    v["static_layers"].append({"name": "plate", "motion": {"anchor_x": 0, "anchor_y": 0, "keys": []}, "paths": [
        {"d": "M0 0 L44 0 L44 44 L0 44 Z", "fill": "#123456", "opacity": 1, "stroke": "none", "stroke_width": 0}]})
    v["layer_order"].insert(0, "plate")
    gk._normalize_vec(v)
    lot = gk.build_lottie(v, 256, 1.2, 30)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "key_3.svg").write_text("<svg/>")  # stale file from an older run
        # no Chrome at all and no other renderer -> NOT RENDERED, nothing written
        with _patched(_chrome=lambda: None, _renderer_name=lambda: None, _warned_shot=False):
            how, px = gk._review_sheet(v, lot, None, out, "#1b1d33", 4)
        assert how.startswith("NOT RENDERED (lottie-web unavailable: headless Chrome failed (no Chrome") and px is None, how
        assert "GPTKIT_CHROME_NO_SANDBOX=1" in how and "cairosvg" in how
        assert not list(out.glob("key_*.svg")) and not (out / "keys_review.png").exists()
        if not _chrome_available():
            print("  (skipped the lottie-web part: no Chrome/Chromium)")
            return
        # lottie-web missing but Chrome present -> the SVG fallback names the reason
        with _patched(_lottie_js=lambda: None, _LOTTIE_JS_REASON="could not fetch (test)"):
            how, px = gk._review_sheet(v, lot, None, out, "#1b1d33", 4)
        assert how == "svg fallback (lottie-web unavailable: could not fetch (test))" and px is None, how
        assert (out / "keys_review.png").exists() and not list(out.glob("key_*.svg"))
        if gk._lottie_js() is None:
            print(f"  (skipped the lottie-web render: {gk._LOTTIE_JS_REASON})")
            return
        how, px = gk._review_sheet(v, lot, None, out, "#1b1d33", 4)
        assert how == "lottie-web" and px is not None and 0 < px <= 1, (how, px)
        frames = gk.render_lottie(lot, [0, 0.5], out / "strip.png")
        assert frames is not None and frames[0].size == (256, 256), gk._LOTTIE_REASON
        rows = np.where(np.asarray(frames[0])[..., 3].max(1) > 0)[0]
        assert rows.min() == 0 and rows.max() == 255, "the full-height plate must reach the bottom row"


NS = 'xmlns="http://www.w3.org/2000/svg"'
LUCIDE_HEART = (f'<svg {NS} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 '
                '0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>')
A = "{http://schemas.android.com/apk/res/android}"
KIT = str(Path(__file__).parent.parent / "scripts" / "gptkit.py")


def _import(svg, name="icon"):
    vb, defs, layers = gk._svg_layers(svg, name, f"{name}.svg")
    return gk._import_vec((vb, defs, layers))


def _pts(d):
    return gk._points(gk._subpaths(d))


def test_svg_layers_table():
    """G6-2: transforms are baked, fill-rule/opacities/caps/joins kept, hidden elements skipped, unsupported ones rejected."""
    # translate(12 0) moves every point by 12
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><g transform="translate(12 0)"><rect width="10" height="24" fill="#00AA00"/></g></svg>')
    assert not gk._validate_vec(v) and {round(x) for x, _ in _pts(v["static_layers"][0]["paths"][0]["d"])} == {12, 22}
    # scale on the path itself, rotate about a pivot
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><path transform="scale(0.5)" d="M0 0H48V48H0Z" fill="#0000FF" stroke="#000" stroke-width="2"/></svg>')
    p = v["static_layers"][0]["paths"][0]
    assert {round(x) for x, _ in _pts(p["d"])} == {0, 24} and abs(p["stroke_width"] - 1) < 1e-6, p
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><rect x="8" y="8" width="8" height="8" fill="#F00" transform="rotate(45 12 12)"/></svg>')
    xs = sorted(x for x, _ in _pts(v["static_layers"][0]["paths"][0]["d"]))
    assert abs(xs[0] - (12 - 4 * math.sqrt(2))) < 0.02 and abs(xs[-1] - (12 + 4 * math.sqrt(2))) < 0.02, xs
    # evenodd reaches Lottie (r: 2) and the AVD (fillType="evenOdd")
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><path fill-rule="evenodd" fill="#FF0000" d="M2 2H22V22H2Z M7 7H17V17H7Z"/></svg>')
    lot = gk.build_lottie(v, 128)
    assert lot["layers"][0]["shapes"][0]["it"][-2]["r"] == 2
    assert 'android:fillType="evenOdd"' in gk.build_avd(v, 1.2, 24)
    # display:none is dropped; visibility hidden too; a nested opacity multiplies
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#FF0000" display="none"/>'
                f'<rect width="4" height="4" fill="#000000" style="visibility:hidden"/><g opacity=".5"><rect width="24" height="24" fill="#FF0000" opacity=".5"/></g></svg>')
    assert len(v["static_layers"]) == 1 and v["static_layers"][0]["paths"][0]["opacity"] == 0.25, v["static_layers"]
    # stroke-opacity stays separate from fill-opacity; caps/joins map to the VD/Lottie sets
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><path d="M2 2L12 22L22 2" fill="#FF0000" fill-opacity="0.5" stroke="#000" stroke-width="2" '
                'stroke-opacity="0.2" stroke-linejoin="miter-clip" stroke-linecap="square"/></svg>')
    p = v["static_layers"][0]["paths"][0]
    assert p["fill_opacity"] == 0.5 and p["stroke_opacity"] == 0.2 and p["join"] == "miter" and p["cap"] == "square", p
    lot = gk.build_lottie(v, 128)
    st, fl = lot["layers"][0]["shapes"][0]["it"][1:3]
    assert (st["ty"], st["o"]["k"], st["lj"], st["lc"]) == ("st", 20, 1, 3) and (fl["ty"], fl["o"]["k"]) == ("fl", 50), (st, fl)
    avd = gk.build_avd(v, 1.2, 24)
    assert 'android:fillAlpha="0.5"' in avd and 'android:strokeAlpha="0.2"' in avd and 'strokeLineJoin="miter"' in avd, avd
    # gradients: percentages, the objectBoundingBox default, radial defaults and href chains become explicit userSpaceOnUse
    for g in ('<linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">@S@</linearGradient>', '<linearGradient id="g">@S@</linearGradient>',
              '<linearGradient id="s">@S@</linearGradient><linearGradient id="g" xlink:href="#s" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="24" y2="0"/>'):
        g = g.replace("@S@", '<stop offset="0" stop-color="#FF0000"/><stop offset="1" stop-color="#0000FF"/>')
        v = _import(f'<svg {NS} xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24"><defs>{g}</defs><rect width="24" height="24" fill="url(#g)"/></svg>')
        assert not gk._validate_vec(v), gk._validate_vec(v)
        lin, s0, e0, stops = gk._gradient(gk._defs_root(v), "g")
        assert lin and s0 == [0, 0] and e0 == [24, 0] and len(stops) == 2, (s0, e0, stops)
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><defs><radialGradient id="g"><stop offset="0" stop-color="#FFF"/><stop offset="1" stop-color="#000"/></radialGradient></defs>'
                '<circle cx="12" cy="12" r="12" fill="url(#g)"/></svg>')
    lin, s0, e0, _ = gk._gradient(gk._defs_root(v), "g")
    assert not lin and [round(x, 2) for x in s0] == [12, 12] and abs(e0[0] - 24) < 0.05, (s0, e0)
    # a userSpaceOnUse gradient under a transformed group moves with the shape
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="12" y2="0">'
                '<stop offset="0" stop-color="#FF0000"/><stop offset="1" stop-color="#0000FF"/></linearGradient></defs>'
                '<g transform="translate(12 0)"><rect width="12" height="24" fill="url(#g)"/></g></svg>')
    assert gk._gradient(gk._defs_root(v), v["static_layers"][0]["paths"][0]["fill"][5:-1])[1:3] == ([12, 0], [24, 0])
    # duplicate ids after sanitising get unique names; no viewBox falls back to width/height
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><g id="a-b"><circle cx="6" cy="6" r="5" fill="#F00"/></g><g id="a_b"><circle cx="18" cy="18" r="5" fill="#00F"/></g></svg>')
    assert [L["name"] for L in v["static_layers"]] == ["a_b", "a_b_2"]
    assert _import(f'<svg {NS} width="48px" height="48px"><circle cx="24" cy="24" r="20" fill="#F00"/></svg>')["viewBox"] == "0 0 48 48"
    # unsupported features: one ValueError naming the file, before any API call
    for bad, needle in ((f'<svg {NS} viewBox="0 0 24 24"><defs><circle id="c" r="5" fill="#F00"/></defs><use href="#c"/></svg>', "<use>"),
                        (f'<svg {NS} viewBox="0 0 24 24"><style>.a{{fill:#0F0}}</style><circle class="a" cx="12" cy="12" r="10"/></svg>', "<style>"),
                        (f'<svg {NS} viewBox="0 0 24 24"><clipPath id="c"><rect width="12" height="24"/></clipPath><circle clip-path="url(#c)" cx="12" cy="12" r="10" fill="#F00"/></svg>', "clip-path"),
                        (f'<svg {NS} viewBox="0 0 24 24"><text x="2" y="12">Hi</text></svg>', "<text>"),
                        (f'<svg {NS} viewBox="0 0 24 24"><g transform="translate()"><rect width="24" height="24" fill="#F00"/></g></svg>', "translate()"),
                        (f'<svg {NS} viewBox="0 0 0 0"><rect width="24" height="24" fill="#F00"/></svg>', "viewBox")):
        try:
            _import(bad, "star")
            assert False, f"expected a ValueError for {needle}"
        except ValueError as e:
            assert str(e).startswith("star.svg: ") and needle in str(e) and "\n" not in str(e), str(e)


def test_colours_normalised():
    """G6-2: every colour form becomes #RRGGBB with alpha folded into the opacities; bad ones are errors, never tracebacks."""
    def one(fill):
        v = vec()
        v["static_layers"][0]["paths"][0]["fill"] = fill
        errs = gk._canon_vec(v)
        return errs, v["static_layers"][0]["paths"][0]
    for fill, want, alpha in (("#abc", "#AABBCC", None), ("#aabbcc", "#AABBCC", None), ("#aabbccdd", "#AABBCC", 0.8667),
                              ("#F00A", "#FF0000", 0.6667), ("white", "#FFFFFF", None), ("rgb(1,2,3)", "#010203", None),
                              ("rgba(255, 0, 0, 0.5)", "#FF0000", 0.5), ("currentColor", "#000000", None)):
        errs, p = one(fill)
        assert not errs and p["fill"] == want, (fill, errs, p)
        assert (alpha is None and "fill_opacity" not in p) or abs(p["fill_opacity"] - alpha) < 3e-3, (fill, p)  # alpha is 8-bit
        assert not gk._validate_vec(vec() | {"static_layers": [dict(vec()["static_layers"][0], paths=[p])]})
    for fill in ("#FFB15", "reddish", "url(#nope)"):
        v = vec()
        v["static_layers"][0]["paths"][0]["fill"] = fill
        errs = gk._check_vec(v)
        assert len(errs) == 1 and "static heart" in errs[0] and "fill" in errs[0], (fill, errs)
    # 8-digit hex alpha reaches Lottie and the AVD identically
    v = _import(f'<svg {NS} viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#FF000080"/></svg>')
    assert gk.build_lottie(v, 128)["layers"][0]["shapes"][0]["it"][-2]["o"]["k"] == 50.2
    assert 'android:fillAlpha="0.502"' in gk.build_avd(v, 1.2, 24)
    # glow colour alpha folds into the glow opacity; layer names are sanitised and deduped with layer_order following
    v = vec()
    v["glows"][0]["color"] = "#B9B6FF80"
    v["static_layers"][0]["name"] = "Heart"
    v["animated_layers"][0]["name"] = "heart"
    v["layer_order"] = ["halo", "Heart", "heart"]
    assert not gk._check_vec(v)
    assert abs(v["glows"][0]["opacity"] - 0.251) < 1e-3 and v["layer_order"] == ["halo", "heart", "heart_2"]
    assert [L["name"] for L in v["static_layers"] + v["animated_layers"]] == ["heart", "heart_2"]


def test_validate_vec_shapes():
    """G6-2/G6-6: viewBox, empty answers, motion keys, garbage path data, identical morph keys and layer_order are all caught."""
    good = vec()
    assert not gk._validate_vec(good)
    bad = [(dict(good, viewBox="0 0 0 0"), "viewBox"), (dict(good, viewBox="0,0,44"), "viewBox"),
           (dict(good, static_layers=[], animated_layers=[], glows=[], layer_order=[]), "nothing to draw"),
           (dict(good, layer_order=["halo", "heart", "blob", "ghost"]), "layer_order"),
           (json.loads(json.dumps(good)) | {"static_layers": [dict(good["static_layers"][0], motion={"anchor_x": 0, "anchor_y": 0, "keys": [{"dx": 1}]})]}, "no t"),
           (json.loads(json.dumps(good)) | {"static_layers": [dict(good["static_layers"][0], motion={"anchor_x": 0, "anchor_y": 0, "keys": [{"t": 0, "ease": "bouncy"}]})]}, "ease"),
           (json.loads(json.dumps(good)) | {"static_layers": [dict(good["static_layers"][0], motion={"anchor_x": 0, "anchor_y": 0, "keys": [{"t": 0, "sx": "big"}]})]}, "sx"),
           (json.loads(json.dumps(good)) | {"static_layers": [dict(good["static_layers"][0], paths=[dict(good["static_layers"][0]["paths"][0], d="M1 1 X 2 2")])]}, "unexpected characters"),
           (json.loads(json.dumps(good)) | {"static_layers": [dict(good["static_layers"][0], paths=[dict(good["static_layers"][0]["paths"][0], d="")])]}, "empty path"),
           (json.loads(json.dumps(good)) | {"animated_layers": [dict(good["animated_layers"][0], keyframes=["M0 0 L10 0 L5 5 Z"])]}, "2 keyframes"),
           (json.loads(json.dumps(good)) | {"animated_layers": [dict(good["animated_layers"][0], keyframes=["M0 0 L10 0 L5 5 Z"] * 3)]}, "identical"),
           (json.loads(json.dumps(good)) | {"animated_layers": [dict(good["animated_layers"][0], keyframes=["M0 0 L10 0 L5 5 Z", "M0 0 L10 0 Z"])]}, "command sequence"),
           (json.loads(json.dumps(good)) | {"glows": [dict(good["glows"][0], rx=0)]}, "rx/ry"),
           (json.loads(json.dumps(good)) | {"glows": [dict(good["glows"][0], name="heart")]}, "unique")]
    for v, needle in bad:
        errs = gk._validate_vec(v)
        assert errs and any(needle in e for e in errs), (needle, errs)
        assert all("\n" not in e for e in errs)
    # the Lottie portability lint passes for the kit's own output and catches what it must
    lot = gk.build_lottie(gk._normalize_vec(vec()), 256)
    assert not gk._lottie_lint(lot)
    lot["layers"][0]["shapes"][0]["it"].insert(0, {"ty": "mm", "mm": 1})
    lot["layers"][1]["ks"]["r"]["x"] = "time * 10"
    errs = gk._lottie_lint(lot)
    assert any("merge path" in e for e in errs) and any("expression" in e for e in errs), errs


def _run(*args, env=None):
    e = dict(os.environ, GPTKIT_CHROME="none", ANDROID_HOME=os.environ.get("ANDROID_HOME", "/opt/android-sdk"))
    e.update(env or {})
    return subprocess.run([sys.executable, KIT, *args], capture_output=True, text=True, env=e)


def _mock_answer(names, keys=None):
    keys = keys or [{"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "ease_in_out"},
                    {"t": 0.5, "dx": 0, "dy": 0, "sx": 1.25, "sy": 1.25, "rotate": 0, "opacity": 1, "ease": "ease_in_out"}]
    return {"viewBox": "0 0 24 24", "defs": "", "layer_order": list(names),
            "static_layers": [{"name": n, "paths": [], "motion": {"anchor_x": 12, "anchor_y": 12, "keys": keys}} for n in names],
            "animated_layers": [], "glows": []}


def test_validate_before_api():
    """G6-2/G6-3: a Lucide icon (currentColor) costs one call; a broken SVG exits before any call; --svg needs no assets.json."""
    calls = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "heart.svg").write_text(LUCIDE_HEART)
        ns = argparse.Namespace(dir=td, name="heart", moving="the heart beats", static=None, motion=None, keyframes=4, size=256,
                                seconds=1.2, fps=30, strength="normal", min_motion=None, svg=str(d / "heart.svg"), dp=None,
                                rebuild=False, no_fit=False, review_frames=4, preview_bg="#1b1d33")
        with _patched(ask=lambda parts, schema, name: calls.append(parts) or json.loads(json.dumps(_mock_answer(["heart"]))),
                      _svg_image=lambda *a, **k: None, _review_sheet=lambda *a, **k: ("lottie-web", 0.5)):
            gk.cmd_vectorize(ns)
        assert len(calls) == 1, "the import must be validated before the API call and never blamed on the model"
        out = d / "vector" / "heart"
        vec = json.loads((out / "layers.json").read_text())
        assert vec["static_layers"][0]["paths"] and vec["static_layers"][0]["paths"][0]["stroke"] == "#000000"
        assert vec["static_layers"][0]["motion"]["keys"][1]["sx"] == 1.25
        assert (out / "heart.json").exists() and (out / "heart_avd.xml").exists() and (out / "preview.html").exists()
        assert "</" not in json.dumps(json.loads((out / "heart.json").read_text()))[:0]  # (structure only)
        # the source image sent to the model sits on a white background for dark ink
        calls.clear()
        black = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        black.paste((0, 0, 0, 255), (16, 16, 48, 48))
        with _patched(ask=lambda parts, schema, name: calls.append(parts) or json.loads(json.dumps(_mock_answer(["heart"]))),
                      _svg_image=lambda *a, **k: black, _review_sheet=lambda *a, **k: ("lottie-web", 0.5)):
            gk.cmd_vectorize(ns)
        sent = next(p for p in calls[0] if isinstance(p, Image.Image))
        assert sent.getpixel((0, 0))[:3] == (255, 255, 255), sent.getpixel((0, 0))
        # unsupported SVG: exit 1 with the file name, zero calls, and 'no API call was made'
        (d / "use.svg").write_text(f'<svg {NS} viewBox="0 0 24 24"><defs><circle id="c" r="5" fill="#F00"/></defs><use href="#c"/></svg>')
        calls.clear()
        with _patched(ask=lambda *a, **k: calls.append(a)):
            try:
                gk.cmd_vectorize(argparse.Namespace(**{**vars(ns), "name": "use", "svg": str(d / "use.svg")}))
                assert False
            except SystemExit as e:
                assert "use.svg" in str(e) and "<use>" in str(e) and "no API call" in str(e), str(e)
        assert not calls
        (d / "empty.svg").write_text(f'<svg {NS} viewBox="0 0 24 24"></svg>')
        with _patched(ask=lambda *a, **k: calls.append(a)):
            try:
                gk.cmd_vectorize(argparse.Namespace(**{**vars(ns), "name": "empty", "svg": str(d / "empty.svg")}))
                assert False
            except SystemExit as e:
                assert "empty.svg" in str(e) and "no API call" in str(e), str(e)
        assert not calls


def test_invalid_exit_codes():
    """G6-2: an invalid answer or layers.json exits non-zero naming layers.json; no empty Lottie is written."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "heart.svg").write_text(LUCIDE_HEART)
        ns = argparse.Namespace(dir=td, name="heart", moving="beats", static=None, motion=None, keyframes=4, size=256, seconds=1.2,
                                fps=30, strength="normal", min_motion=None, svg=str(d / "heart.svg"), dp=None, rebuild=False,
                                no_fit=False, review_frames=4, preview_bg="#1b1d33")
        calls = []
        empty = _mock_answer([])
        with _patched(ask=lambda *a, **k: calls.append(a) or json.loads(json.dumps(empty)), _svg_image=lambda *a, **k: None):
            try:
                gk.cmd_vectorize(ns)
                assert False
            except SystemExit as e:
                assert e.code and "layers.json" in str(e) and "animated nothing" in str(e), str(e)
        assert len(calls) == 2, "an answer without motion is retried once with the reason"
        assert "animates nothing" in json.dumps([str(x) for x in calls[1][0] if isinstance(x, str)])
        assert not (d / "vector" / "heart" / "heart.json").exists(), "no Lottie for an invalid answer"
        (d / "vector" / "heart").mkdir(parents=True, exist_ok=True)
        lp = d / "vector" / "heart" / "layers.json"
        for text, needle in ((json.dumps(dict(vec(), viewBox="0 0 0 0")), "viewBox"), ("{not json", "invalid JSON at line 1"),
                             (json.dumps(vec() | {"static_layers": [dict(vec()["static_layers"][0], paths=[dict(vec()["static_layers"][0]["paths"][0], fill="reddish")])]}), "reddish")):
            lp.write_text(text)
            r = _run("vectorize", td, "heart", "--rebuild")
            assert r.returncode == 1 and "layers.json" in r.stderr and needle in r.stderr and "Traceback" not in r.stderr, r.stderr
        lp.unlink()
        r = _run("vectorize", td, "heart", "--rebuild")
        assert r.returncode == 1 and "run without --rebuild first" in r.stderr and r.stderr.count("\n") <= 1, r.stderr


def test_missing_inputs():
    """G6-3: missing dirs/files and bad flags are one-line exits (or argparse exit 2), never tracebacks."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cases = [(["vectorize", f"{td}/nodir", "x", "--moving", "y"], 1, "assets.json"),
                 (["vectorize", td, "x", "--moving", "y", "--svg", f"{td}/nope.svg"], 1, "nope.svg"),
                 (["vectorize", td, "x", "--moving", "y", "--seconds", "0"], 2, "--seconds"),
                 (["vectorize", td, "x", "--moving", "y", "--fps", "0"], 2, "--fps"),
                 (["vectorize", td, "x", "--moving", "y", "--review-frames", "0"], 2, "--review-frames"),
                 (["vectorize", td, "x", "--moving", "y", "--keyframes", "x"], 2, "--keyframes"),
                 (["vectorize", td, "x", "--moving", "y", "--dp", "-1"], 2, "--dp")]
        (d / "bad.svg").write_text("<svg><path d='M0 0")
        cases.append((["vectorize", td, "bad", "--moving", "y", "--svg", f"{td}/bad.svg"], 1, "not valid SVG XML"))
        (d / "svg").mkdir()
        (d / "svg" / "one.svg").write_text(LUCIDE_HEART)
        cases.append((["vectorize", td, "one", "--moving", "y", "--keyframes", "1"], 1, "--keyframes"))
        Image.new("RGBA", (8, 8), (9, 9, 9, 255)).save(d / "mock.png")
        (d / "assets.json").write_text(json.dumps({"mockup": "mock.png", "width": 8, "height": 8, "screen_dp": 360, "assets": []}))
        cases.append((["vectorize", td, "x", "--moving", "y"], 1, "no source"))
        for args, code, needle in cases:
            r = _run(*args)
            assert r.returncode == code and needle in r.stderr and "Traceback" not in r.stderr, (args, r.returncode, r.stderr)
            if code == 1:
                assert r.stderr.strip().count("\n") == 0, r.stderr
        assert gk._pixel_motion([]) == 0.0 and gk._pixel_motion([Image.new("RGBA", (4, 4))]) == 0.0


def test_join_cap_mapping():
    """G6-4: SVG caps/joins outside the VD/Lottie sets map to the nearest legal value (aapt2 link rejects miter-clip)."""
    assert gk._join("miter-clip") == "miter" and gk._join("arcs") == "miter" and gk._join("bevel") == "bevel"
    assert gk._join(None) == "round" and gk._join("nonsense") == "round" and gk._join("Round") == "round"
    assert gk._cap("square") == "square" and gk._cap(None) == "round" and gk._cap("odd", "butt") == "butt"
    v = vec()
    v["static_layers"][0]["paths"][0].update(cap="butt", join="miter-clip")
    avd = gk.build_avd(v, 1.2, 24)
    assert 'strokeLineCap="butt"' in avd and 'strokeLineJoin="miter"' in avd and "miter-clip" not in avd
    lot = gk.build_lottie(v, 256)
    st = next(it for lyr in lot["layers"] for sh in lyr["shapes"] for it in sh["it"] if it["ty"] == "st")
    assert (st["lc"], st["lj"]) == (1, 1)


def test_avd_names_unique():
    """G6-4: 'heart' and 'heart 0' (or a-b / a_b) never share an android:name; every target names exactly one node."""
    v = vec()
    other = json.loads(json.dumps(v["static_layers"][0]))
    other["name"] = "heart 0"
    v["static_layers"].append(other)
    v["layer_order"].append("heart 0")
    root = ET.fromstring(gk.build_avd(v, 1.2, 24))
    names = [e.get(A + "name") for e in root.iter() if e.get(A + "name") and e.tag != "target"]
    assert len(names) == len(set(names)) and all(gk._NAME_RE.match(n) for n in names), names
    targets = [t.get(A + "name") for t in root.findall("target")]
    assert len(targets) == len(set(targets)) and set(targets) <= set(names), (targets, names)
    assert {"heart", "heart_0"} <= set(names), names
    v2 = _import(f'<svg {NS} viewBox="0 0 24 24"><g id="a-b"><circle cx="6" cy="6" r="5" fill="#F00"/></g><g id="a_b"><circle cx="18" cy="18" r="5" fill="#00F"/></g></svg>')
    names = [e.get(A + "name") for e in ET.fromstring(gk.build_avd(v2, 1.2, 24)).iter() if e.get(A + "name")]
    assert len(names) == len(set(names)) == 2, names


def _avd_fixtures():
    """Vecs covering what build_avd emits: motion + 3-key morph + glow, gradients, negative viewBox, 2-key morph, evenodd."""
    grad = _import(f'<svg {NS} viewBox="0 0 24 24"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#FF0000"/>'
                   '<stop offset="1" stop-color="#0000FF" stop-opacity="0.5"/></linearGradient></defs><rect width="24" height="24" fill="url(#g)" '
                   'stroke="#000" stroke-width="1" stroke-linejoin="miter-clip"/></svg>')
    two = vec()
    two["animated_layers"][0]["keyframes"] = two["animated_layers"][0]["keyframes"][:2]
    neg = dict(vec(), viewBox="-10 -10 64 64")
    eo = _import(f'<svg {NS} viewBox="0 0 24 24"><path fill-rule="evenodd" fill="#FF0000" fill-opacity=".5" d="M2 2H22V22H2Z M7 7H17V17H7Z"/></svg>')
    eo["static_layers"][0]["motion"] = json.loads(json.dumps(MOTION))
    return {"test": gk._normalize_vec(vec()), "grad": grad, "two_key": two, "neg_vb": neg, "evenodd": eo}


def test_avd_aapt2_compile():
    """G6-4: every AVD the kit produces compiles with aapt2 (link too when platforms/android-*/android.jar exists)."""
    import shutil
    if not shutil.which("aapt2") and not gk._android_sdk():
        print("  (skipped: no aapt2)")
        return
    with tempfile.TemporaryDirectory() as td:
        files = []
        for n, v in _avd_fixtures().items():
            assert not gk._validate_vec(v), (n, gk._validate_vec(v))
            p = Path(td) / f"{n}_avd.xml"
            p.write_text(gk.build_avd(v, 1.2, 24))
            files.append(p)
        res = gk.aapt2_check(files)
        assert res == {}, res


def test_avd_matches_motion_model():
    """G6-7: the AVD's group/path attributes at loop time t equal _motion_at / _morph_at (evaluated offline, no device)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import avd_eval
    v = gk._normalize_vec(vec())
    avd = gk.build_avd(v, 1.2, 24)
    L = {x["name"]: x for x in gk._layers(v)}
    for t in (0.1, 0.3, 0.55, 0.8):
        st = avd_eval.at(gk, avd, t, 1200)
        m = gk._motion_at(L["heart"]["ks"], t)
        g = st["heart"]
        assert abs(g.get("translateY", 0) - m["dy"]) < 1e-3 and abs(g.get("scaleX", 1) - m["sx"]) < 1e-3, (t, g, m)
        assert abs(g.get("rotation", 0) - m["rotate"]) < 1e-3, (t, g, m)
        assert abs(st["heart_p0"].get("strokeAlpha", 1) - m["opacity"]) < 2e-3, (t, st["heart_p0"], m)
        want = gk._d(gk._morph_at(L["blob"]["keys"], t))
        vis = [n for n in st if n.startswith("blob_p") and st[n].get("fillAlpha", 1) > 0.5]
        assert len(vis) == 1, (t, {n: st[n].get("fillAlpha") for n in st if n.startswith("blob_p")})
        got = [float(x) for x in avd_eval.NUM.findall(st[vis[0]]["pathData"])]
        exp = [float(x) for x in avd_eval.NUM.findall(want)]
        assert len(got) == len(exp) and max(abs(a - b) for a, b in zip(got, exp)) < 0.02, (t, st[vis[0]]["pathData"], want)
    # windows hand over without double alpha: the copies' fillAlpha never sums above 1.001
    for t in (0.3333, 0.33335, 0.3334, 0.6666, 0.66665, 0.6667):
        st = avd_eval.at(gk, avd, t, 1200)
        assert sum(st[n].get("fillAlpha", 1) for n in st if n.startswith("blob_p")) < 1.001, (t, st)


def test_motion_gate_fade():
    """G6-6: a pure fade is not 'too subtle'; a big static flame with a tiny hyperactive spark is."""
    fade = [{"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "ease_in_out"},
            {"t": 0.5, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 0, "ease": "ease_in_out"}]
    v = vec()
    v["static_layers"][0]["motion"] = {"anchor_x": 22, "anchor_y": 22, "keys": fade}
    v["animated_layers"] = []
    v["layer_order"] = ["halo", "heart"]
    rep = gk.motion_report(v)
    assert rep["layers"]["heart"]["move"] < 0.001 and rep["layers"]["heart"]["fade"] == 1 and rep["motion"] >= gk.MOTION_MIN["normal"], rep
    sq = lambda x, y, s: f"M{x} {y} L{x + s} {y} L{x + s} {y + s} L{x} {y + s} Z"
    flame = [{"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"},
             {"t": 0.5, "dx": 0.4, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"}]
    spark = [{"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"},
             {"t": 0.5, "dx": 0, "dy": -14, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"}]
    paint = {"fill": "#FF8800", "opacity": 1, "stroke": "none", "stroke_width": 0}
    v = {"viewBox": "0 0 44 44", "defs": "", "layer_order": ["flame", "spark"], "animated_layers": [], "glows": [],
         "static_layers": [{"name": "flame", "paths": [dict(paint, d=sq(2, 2, 40))], "motion": {"anchor_x": 22, "anchor_y": 22, "keys": flame}},
                           {"name": "spark", "paths": [dict(paint, d=sq(21, 21, 2))], "motion": {"anchor_x": 22, "anchor_y": 22, "keys": spark}}]}
    rep = gk.motion_report(v)
    assert rep["layers"]["flame"]["move"] < 0.011 and rep["layers"]["spark"]["move"] > 4, rep
    assert rep["motion"] < gk.MOTION_MIN["normal"], rep


def test_preview_background():
    """G6-6: dark ink is shown to the model on white, light art on the preview colour."""
    black = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    black.paste((10, 10, 10, 255), (8, 8, 24, 24))
    white = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    white.paste((250, 250, 250, 255), (8, 8, 24, 24))
    assert gk._contrast_bg(black) == "#FFFFFF" and gk._contrast_bg(white) == "#1b1d33" and gk._contrast_bg(Image.new("RGBA", (4, 4))) == "#1b1d33"


def test_loop_lengths():
    """G6-6: the Lottie loop is exactly --seconds even when seconds * fps is not an integer; the AVD uses milliseconds."""
    v = gk._normalize_vec(vec())
    lot = gk.build_lottie(v, 256, 1.25, 30)
    assert lot["op"] == 38 and abs(lot["op"] / lot["fr"] - 1.25) < 1e-9, (lot["op"], lot["fr"])
    assert all(l["ip"] == 0 and l["op"] == 38 for l in lot["layers"])
    lot = gk.build_lottie(v, 256, 1.2, 30)
    assert lot["op"] == 36 and lot["fr"] == 30
    assert 'android:duration="1250"' in gk.build_avd(v, 1.25, 24)


def test_lottie_structure():
    """G6-7: morph keyframes keep vertex counts, animated flags are set where needed, ip/op/ind are consistent."""
    v = gk._normalize_vec(vec())
    lot = gk.build_lottie(v, 512, 1.2, 30)
    assert [l["ind"] for l in lot["layers"]] == [1, 2, 3] and all(l["ty"] == 4 and l["ip"] == 0 and l["op"] == 36 for l in lot["layers"])
    blob = next(l for l in lot["layers"] if l["nm"] == "blob")
    sh = blob["shapes"][0]["it"][0]
    assert sh["ty"] == "sh" and sh["ks"]["a"] == 1 and len(sh["ks"]["k"]) == 4  # 3 keys + the wrap back to the first
    counts = {len(k["s"][0]["v"]) for k in sh["ks"]["k"]}
    assert len(counts) == 1 and all(len(k["s"][0]["i"]) == len(k["s"][0]["o"]) == len(k["s"][0]["v"]) for k in sh["ks"]["k"]), counts
    assert [k["t"] for k in sh["ks"]["k"]] == [0, 12, 24, 36] and sh["ks"]["k"][0]["s"] == sh["ks"]["k"][-1]["s"]
    assert blob["shapes"][0]["it"][-1]["ty"] == "tr" and blob["ks"]["p"]["a"] == 0
    heart = next(l for l in lot["layers"] if l["nm"] == "heart")
    assert heart["ks"]["p"]["a"] == 1 and heart["ks"]["o"]["a"] == 1 and heart["ks"]["a"]["k"] == [22, 22, 0]
    halo = next(l for l in lot["layers"] if l["nm"] == "halo")
    gf = halo["shapes"][0]["it"][1]
    assert gf["ty"] == "gf" and gf["t"] == 2 and gf["g"]["p"] == 5 and len(gf["g"]["k"]["k"]) == 5 * 4 + 5 * 2
    assert not gk._lottie_lint(lot)
    # every closed subpath drops its duplicated closing vertex, open ones keep both ends
    assert len(gk._lottie_shape(gk._subpaths("M0 0 L10 0 L10 10 Z")[0])["v"]) == 3
    assert len(gk._lottie_shape(gk._subpaths("M0 0 L10 0 L10 10")[0])["v"]) == 3 and not gk._lottie_shape(gk._subpaths("M0 0 L10 0")[0])["c"]


def test_gradient_resolution():
    """G6-2/G6-7: _gradient follows href chains, applies SVG defaults and explains what it cannot resolve."""
    defs = ('<linearGradient id="s"><stop offset="0" stop-color="red"/><stop offset="100%" stop-color="#00F" stop-opacity="50%"/></linearGradient>'
            '<linearGradient id="g" xlink:href="#s" gradientUnits="userSpaceOnUse" x1="10%" y1="0" x2="100%" y2="0"/>'
            '<radialGradient id="r" gradientUnits="userSpaceOnUse" cx="12" cy="12" r="6" gradientTransform="scale(2)"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></radialGradient>'
            '<linearGradient id="one"><stop offset="0" stop-color="#fff"/></linearGradient>'
            '<linearGradient id="obb" x1="0" x2="1"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>'
            '<radialGradient id="zero" gradientUnits="userSpaceOnUse" r="0"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></radialGradient>'
            '<pattern id="pat"/>')
    root = gk._defs_root({"viewBox": "0 0 40 20", "defs": defs})
    lin, s0, e0, stops = gk._gradient(root, "g")
    assert lin and s0 == [4, 0] and e0 == [40, 0] and stops == [(0, "#FF0000", 1.0), (1, "#0000FF", 0.5)], (s0, e0, stops)
    lin, s0, e0, stops = gk._gradient(root, "r")
    assert not lin and s0 == [24, 24] and abs(e0[0] - 36) < 1e-9, (s0, e0)
    lin, s0, e0, _ = gk._gradient(root, "obb", bbox=(10, 10, 30, 20))
    assert lin and s0 == [10, 10] and e0 == [30, 10], (s0, e0)
    for gid, needle in (("one", "2 stops"), ("obb", "objectBoundingBox"), ("zero", "radius"), ("pat", "not a gradient"), ("nope", "not found")):
        try:
            gk._gradient(root, gid)
            assert False, gid
        except ValueError as e:
            assert needle in str(e), (gid, str(e))


def test_ask_vector_retry():
    """G6-7: a weak first answer is retried with the numbers and the stronger one kept; the retry carries the model's own
    answer (imported geometry elided), never the restored paths."""
    weak = _mock_answer(["heart"], [{"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"},
                                    {"t": 0.5, "dx": 0, "dy": 0, "sx": 1.02, "sy": 1.02, "rotate": 0, "opacity": 1, "ease": "linear"}])
    strong = _mock_answer(["heart"])
    answers = [json.loads(json.dumps(weak)), json.loads(json.dumps(strong))]
    calls = []
    imported = gk._svg_layers(LUCIDE_HEART, "heart", "heart.svg")
    a = argparse.Namespace(min_motion=None, strength="normal", moving="beats", static=None, keyframes=4, seconds=1.2, motion=None)
    with _patched(ask=lambda parts, schema, name: calls.append(parts) or answers.pop(0)):
        got = gk._ask_vector(a, None, None, None, imported, "#1b1d33")
    assert len(calls) == 2 and got["static_layers"][0]["motion"]["keys"][1]["sx"] == 1.25
    fb = next(p for p in calls[1] if isinstance(p, str) and p.startswith("Your previous answer moves too little"))
    assert "heart" in fb and "Previous answer" in fb
    sent = json.loads(fb.split("Previous answer:\n", 1)[1])
    assert sent["static_layers"][0]["paths"] == [], "the retry must not echo the restored geometry"
    # an invalid first answer: the feedback names the problem, the second (valid) answer wins
    bad = json.loads(json.dumps(strong))
    bad["static_layers"][0]["motion"]["keys"][1]["ease"] = "bouncy"
    answers = [bad, json.loads(json.dumps(strong))]
    calls.clear()
    with _patched(ask=lambda parts, schema, name: calls.append(parts) or answers.pop(0)):
        got = gk._ask_vector(a, None, None, None, imported, "#1b1d33")
    assert len(calls) == 2 and not gk._validate_vec(got)
    fb = next(p for p in calls[1] if isinstance(p, str) and p.startswith("Your previous answer was invalid"))
    assert "bouncy" in fb and "currentColor" not in fb, fb


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
