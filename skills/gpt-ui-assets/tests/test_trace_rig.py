"""Offline checks for trace / rig / animate (no API). Run: python3 skills/gpt-ui-assets/tests/test_trace_rig.py"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gptkit as g  # noqa: E402


def synthetic(size=128):
    """Radial-gradient disc with a soft glow and a highlight: small stand-in for a forged illustration."""
    yy, xx = np.mgrid[:size, :size] + 0.5
    r = np.hypot(xx - size / 2, yy - size / 2)
    R = size * 0.3
    t = np.clip(r / R, 0, 1)[..., None]
    rgb = (1 - t) * np.array([255, 220, 120]) + t * np.array([230, 110, 40])
    alpha = np.where(r <= R, 1.0, np.clip(0.45 * np.exp(-(r - R) / (size * 0.06)), 0, 1))
    hl = np.hypot(xx - size * 0.42, yy - size * 0.4) < size * 0.06
    rgb[hl] = [255, 250, 235]
    return Image.fromarray(np.dstack([rgb, alpha * 255]).astype(np.uint8), "RGBA")


def test_ring_winding():
    m = np.zeros((64, 64), bool)
    yy, xx = np.mgrid[:64, :64]
    d = np.hypot(xx - 32, yy - 32)
    m[(d < 24) & (d > 12)] = True
    subs = g._subpaths(g.mask_to_path(m))
    assert len(subs) == 2
    t = np.linspace(0, 1, 12)
    areas = []
    for s in subs:
        pts, cur = [], np.array(s["pts"][0])
        for seg in s["segs"]:
            b = np.array([cur, seg[1], seg[1], seg[1]] if seg[0] == "L" else [cur, *seg[1:]], float)
            pts.append(g._bez(b, t))
            cur = b[3]
        areas.append(g._area(np.vstack(pts)))
    assert areas[0] * areas[1] < 0, "hole must wind opposite to the outline (nonzero fill)"


def test_trace_and_vd():
    src = synthetic()
    res = g.trace_image(src, "illustration", work=128)
    roles = {L["role"] for L in res["layers"]}
    assert roles == {"glow", "region"}, roles
    assert any(L["fill"]["type"] in ("radial", "linear") for L in res["layers"] if L["role"] == "region")
    svg = g.layers_svg(res["layers"], res["size"], src.size)
    ET.fromstring(svg)
    vd = g.svg_to_vd(svg, 48)
    assert "<vector" in vd and "transform" not in vd
    ET.fromstring(g.vd_to_svg(vd))
    for L in res["layers"]:
        g._subpaths(L["d"])  # only absolute M/L/C/Z
        assert "e-" not in L["d"] and " ." not in L["d"], "lint InvalidVectorPath: no exponents / bare decimals"


def test_bake_translate():
    assert g._bake_translate("M0 0 C1 2 3 4 5 6 Z", 10, 20) == "M 10 20 C 11 22 13 24 15 26 Z"


def test_procedural_keys():
    d = "M10 90 C20 50 40 20 50 5 C60 20 80 50 90 90 L50 95Z M40 80 L60 80 L50 60Z"
    rest = g.map_path(d, lambda x, y: (x, y))
    for kind in g.TR_MOTIONS:
        keys = g.procedural_keyframes([d], kind, 6, phase=0.3)[0]
        assert keys[0] == rest, f"{kind}: key 0 must be the rest pose"
        assert len({"".join(t for t in g._TOK.findall(k) if t.isalpha()) for k in keys}) == 1, kind
        assert len(set(keys)) > 1, f"{kind}: must move"


def test_animate_to_lottie():
    res = g.trace_image(synthetic(), "illustration", work=128)
    vec = g.animate_layers(g.layers_vec(res), g._parse_motions("art=flicker; glow=flicker@art"), 6)
    assert not g._validate_vec(vec)
    assert not vec["static_layers"] and vec["animated_layers"]
    lot = g._tidy_lottie(g.build_lottie(vec, 256, 1.2, 30))
    json.dumps(lot)
    assert lot["layers"] and all(L["ty"] == 4 for L in lot["layers"])


def test_animate_traced_file_without_chrome():
    import argparse
    import os
    import tempfile
    res = g.trace_image(synthetic(), "illustration", work=96)
    old = os.environ.get("GPTKIT_CHROME")
    os.environ["GPTKIT_CHROME"] = "none"
    g._chrome.cache_clear()
    try:
        assert g.chrome_shot("<p>x</p>", 16, 16) is None
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "drop.layers.json"
            src.write_text(json.dumps(g.layers_vec(res)))
            g.cmd_animate(argparse.Namespace(layers=str(src), part="art=bob", keyframes=4, seconds=1.0, fps=24, size=128,
                                             ease="linear", out=None, name=None, preview_bg="#000000"))
            assert (Path(td) / "anim" / "drop" / "drop.json").exists()
    finally:
        if old is None:
            os.environ.pop("GPTKIT_CHROME")
        else:
            os.environ["GPTKIT_CHROME"] = old
        g._chrome.cache_clear()


def test_blank_filmstrip_is_not_rendered():
    import tempfile
    vec = g.animate_layers(g.layers_vec(g.trace_image(synthetic(), "illustration", work=64)), g._parse_motions("art=bob"), 4)
    saved = g.lottie_filmstrip, g._tr_render_svgs
    g.lottie_filmstrip = lambda lot, frames, size=256, bg="#1b1d33": Image.new("RGB", (size * len(frames), size), bg)
    g._tr_render_svgs = lambda svgs, sizes: None
    try:
        with tempfile.TemporaryDirectory() as td:
            assert g.write_animation(vec, td, "t", 1.0, 24, 128)["lottie_rendered"] is False
    finally:
        g.lottie_filmstrip, g._tr_render_svgs = saved


def _chrome_ok():
    """False only when no Chrome exists at all; when one exists the shared flags must make it work."""
    ok, why = g._chrome_probe()
    if not ok and g._chrome() is None:
        print(f"  (skip: {why})")
        return False
    assert ok, why
    return True


def test_render_fidelity():
    if not _chrome_ok():
        return
    src = synthetic()
    res = g.trace_image(src, "illustration", work=128)
    rens = g._tr_render_svgs([g.layers_svg(res["layers"], res["size"], src.size)], [src.size])
    if rens is None:
        print(f"  (skip: {g._tr_chrome_reason()})")
        return
    m = g.trace_metrics(src, rens[0])
    assert m["iou"] > 0.97 and m["diff"] < 4, m


def test_chrome_shot_full_height():
    """--headless=new lays out less than --window-size: the padded window + crop must paint every row."""
    if not _chrome_ok():
        return
    im = g.chrome_shot('<body style="margin:0"><div style="position:absolute;left:0;top:0;width:200px;height:300px;'
                       'background:#f00"></div></body>', 200, 300)
    assert im is not None and im.size == (200, 300), im
    a = np.asarray(im)
    painted = (a[..., 3] > 200).all(axis=1)
    assert painted.all(), f"{int(painted.sum())}/300 rows painted"
    assert (a[..., 0] > 200).all() and (a[..., 1] < 30).all()


class _Cap:
    """Collect stdout inside a with-block."""
    def __enter__(self):
        import io
        self.buf, self.old = io.StringIO(), sys.stdout
        sys.stdout = self.buf
        return self

    def __exit__(self, *a):
        sys.stdout = self.old

    @property
    def text(self):
        return self.buf.getvalue()


def _cli(*argv):
    """Run gptkit's CLI in-process; returns (exit code, stdout)."""
    old = sys.argv
    sys.argv = ["gptkit", *argv]
    try:
        with _Cap() as cap:
            try:
                g.main()
                code = 0
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv = old
    return code, cap.text


def test_trace_without_chrome_is_honest():
    import tempfile
    import subprocess
    noop = lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", "")
    saved, g.subprocess.run = g.subprocess.run, noop
    g._CHROME_FAILED_REASON = None
    try:
        with tempfile.TemporaryDirectory() as td:
            synthetic(96).save(f"{td}/disc.png")
            code, out = _cli("trace", f"{td}/disc.png", "--out", f"{td}/o", "--preset", "auto", "--work", "96")
            assert code == 0, out
            rep = json.loads(Path(td, "o", "disc.trace.json").read_text())
            assert rep["advice"]["prefer"] == "unknown", rep["advice"]
            assert rep["advice"]["reasons"][0].startswith("fidelity not measured: "), rep["advice"]
            assert len(rep["candidates"]) == 1, "only one preset may be traced when none can be compared"
            assert rep["candidates"][0]["metrics"] is None
            assert "produced no screenshot" in out and "prefer unknown (fidelity not measured" in out, out
            assert not (Path(td) / "o" / "review" / "disc_trace.png").exists()
    finally:
        g.subprocess.run = saved
        g._CHROME_FAILED_REASON = None


def _pil_render(svgs, sizes):
    """Stand-in for _tr_render_svgs: rasterise the kit's own SVG paths with PIL polygons (no Chrome)."""
    import re
    from PIL import ImageDraw
    out = []
    for svg, (w, h) in zip(svgs, sizes):
        vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
        s = w / vb[2]
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for m in re.finditer(r'<path d="([^"]+)"[^>]*?fill="([^"]+)"', svg):
            col = m.group(2) if m.group(2).startswith("#") and len(m.group(2)) == 7 else "#808080"
            for sp in g._subpaths(m.group(1)):
                pts = [(x * s, y * s) for x, y in g._curve_pts([sp])]
                if len(pts) >= 3:
                    d.polygon(pts, fill=col)
        out.append(im)
    return out


def test_tear_metric():
    import tempfile
    two = {"viewBox": "0 0 100 100", "defs": "", "animated_layers": [],
           "static_layers": [{"name": "base", "paths": [{"d": "M10 50 L90 50 L90 90 L10 90Z", "fill": "#884422"}]},
                             {"name": "top", "paths": [{"d": "M10 10 L90 10 L90 50 L10 50Z", "fill": "#ffaa00"}]}],
           "layer_order": ["base", "top"]}
    saved = g._tr_render_svgs, g.lottie_filmstrip
    g._tr_render_svgs, g.lottie_filmstrip = _pil_render, lambda *a, **k: None
    try:
        with tempfile.TemporaryDirectory() as td:
            rep = g.write_animation(g.animate_layers(two, g._parse_motions("top=bob:5"), 6), td, "two", 1.0, 24, 128)
            assert rep["tear_px"][0] == 0 and max(rep["tear_px"]) > 500, rep["tear_px"]
            assert rep["tear_frac_max"] > 0.01
            vec = g.animate_layers(g.layers_vec(g.trace_image(synthetic(), "illustration", work=128)),
                                   g._parse_motions("art=flicker; glow=flicker@art"), 6)
            rep = g.write_animation(vec, td, "disc", 1.0, 24, 128)
            assert rep["tear_px"] == [0] * 6, rep["tear_px"]
            assert rep["keys_review"].startswith("not rendered") if "keys_review" in rep else True
    finally:
        g._tr_render_svgs, g.lottie_filmstrip = saved


def test_trace_names_and_aapt2_message():
    import shutil
    import tempfile
    assert g._tr_path_error([{"name": "r", "d": "M0 0 Lnan nan Z"}]) == "r: pathData contains nan/inf"
    assert g._tr_path_error([{"name": "r", "d": "M0 0 L1"}]).startswith("r: ")
    assert g._tr_path_error([{"name": "r", "d": "M0 0 L1 1 L0 1Z"}]) is None
    with tempfile.TemporaryDirectory() as td:
        synthetic(64).save(f"{td}/My Ring-2.png")
        synthetic(48).save(f"{td}/b.png")
        code, out = _cli("trace", f"{td}/My Ring-2.png", f"{td}/b.png", "--out", f"{td}/o", "--name", "same")
        assert code and "--name needs a single input file" in str(code), (code, out)
        code, out = _cli("trace", f"{td}/My Ring-2.png", f"{td}/missing.png", "--out", f"{td}/o", "--preset", "flat",
                         "--work", "64")
        assert code == "trace: 1 of 2 input(s) not traced (see the lines above)", (code, out)
        assert (Path(td) / "o" / "img_my_ring_2.xml").exists(), out
        assert "-> img_my_ring_2" in out and "missing.png: cannot read image" in out, out
        rep = json.loads(Path(td, "o", "My Ring-2.trace.json").read_text())
        assert rep["vd"]["res_name"] == "img_my_ring_2"
        sdk = g._android_sdk()
        if not sdk:
            assert rep["vd"]["aapt2"] == "aapt2 not found (set ANDROID_HOME): not validated" and rep["vd"]["aapt2_ok"] is None
            print("  (no aapt2: message checked, compile not run)")
            return
        level = "link" if sdk[1] else "compile"
        assert rep["vd"]["aapt2_ok"] is True and rep["vd"]["aapt2_level"] == level, rep["vd"]
        assert rep["vd"]["aapt2"].startswith("aapt2 compile+link ok (android-" if level == "link" else
                                             "aapt2 compile ok; link skipped (no $ANDROID_HOME/platforms"), rep["vd"]
        assert "valid" not in rep["vd"]["aapt2"]
        bad = '<vector xmlns:android="http://schemas.android.com/apk/res/android" android:width="24dp">\n<path'
        v = g._tr_aapt2(bad, "img_bad")
        assert v["aapt2_ok"] is False and "unclosed token" in v["aapt2"], v
        # a transparent input writes no VD and is reported
        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(f"{td}/clear.png")
        code, out = _cli("trace", f"{td}/clear.png", "--out", f"{td}/o2", "--preset", "flat")
        assert code and "nothing opaque to trace" in out and not list(Path(td, "o2").glob("*.xml")), (code, out)


def test_animate_vectorize_layers():
    """layers.json from vectorize: relative commands, arcs, strokes and a rigid motion must survive animate."""
    vec = {"viewBox": "0 0 100 100", "defs": "", "animated_layers": [],
           "static_layers": [{"name": "rel", "paths": [{"d": "M10 10 h50 v30 h-50 z", "fill": "#336699"}],
                              "motion": {"anchor_x": 35, "anchor_y": 25, "keys": [{"t": 0, "rotate": 0}, {"t": 0.5, "rotate": 4}]}},
                             {"name": "arc", "paths": [{"d": "M20 80 a15 15 0 1 0 30 0 z", "fill": "#cc3333"}]},
                             {"name": "line", "paths": [{"d": "M5 95 L95 95", "fill": "none", "stroke": "#112233",
                                                         "stroke_width": 2, "cap": "round", "join": "round"}]}],
           "layer_order": ["rel", "arc", "line"]}
    out = g.animate_layers(vec, g._parse_motions("rel=bob; arc=breathe; line=wave:1.5"), 6)
    assert not g._validate_vec(out), g._validate_vec(out)
    by = {al["name"]: al for al in out["animated_layers"]}
    assert set(by) == {"rel", "arc", "line"} and not out["static_layers"]
    assert by["line"]["stroke"] == "#112233" and by["line"]["stroke_width"] == 2 and by["line"]["cap"] == "round"
    assert by["line"]["fill"] == "none"
    assert by["rel"]["motion"]["keys"][1]["rotate"] == 4, "the rigid motion must be kept"
    st = g._motion_stats(out)
    assert all(v["max"] > 0 for v in st.values()), st
    assert st["line"]["max"] > 1, "a zero-height box must still move (H floor)"
    for al in out["animated_layers"]:
        for k in al["keyframes"]:
            assert set(t for t in g._TOK.findall(k) if t.isalpha()) <= {"M", "L", "C", "Z"}, k
    lot = g.build_lottie(out, 128, 1.0, 24)
    assert lot["layers"] and lot["op"] == 24
    svg = g._vec_render(out, (100, 100), 2)
    assert 'stroke="#112233"' in svg


def test_motion_parse_errors():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        v = {"viewBox": "0 0 10 10", "defs": "", "animated_layers": [],
             "static_layers": [{"name": "art", "paths": [{"d": "M1 1 L9 1 L9 9 L1 9Z", "fill": "#123456"}]}]}
        Path(td, "a.layers.json").write_text(json.dumps(v))
        for part, want in (("art=spin", "unknown motion 'spin'"), ("art=bob:abc", "bad amp/phase in 'art=bob:abc'"),
                           ("nope=bob", "no static layer 'nope'"), ("art=bob@ghost", "unknown anchor layer 'ghost'"),
                           ("", "give --part")):
            code, out = _cli("animate", f"{td}/a.layers.json", "--part", part, "--out", f"{td}/o")
            assert isinstance(code, str) and code.startswith("animate: ") and want in code and "\n" not in code, (part, code)
        code, out = _cli("animate", f"{td}/missing.layers.json", "--part", "art=bob")
        assert code == f"gptkit: layers.json not found: {td}/missing.layers.json", code
        Path(td, "bad.layers.json").write_text("{not json")
        code, out = _cli("animate", f"{td}/bad.layers.json", "--part", "art=bob")
        assert "is not valid JSON" in str(code), code
        Path(td, "old.layers.json").write_text(json.dumps({"layers": []}))
        code, out = _cli("animate", f"{td}/old.layers.json", "--part", "art=bob")
        assert "missing viewBox, static_layers" in str(code), code
        for argv in (("--keyframes", "1"), ("--fps", "0"), ("--seconds", "0"), ("--size", "8"), ("--size", "nan")):
            code, out = _cli("animate", f"{td}/a.layers.json", "--part", "art=bob", *argv)
            assert code == 2, (argv, code)


def campfire(size=192):
    """Three-part stand-in for a forged illustration: two logs, an outer flame and an inner flame."""
    from PIL import ImageDraw
    S = size
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([S * 0.3, S * 0.3, S * 0.7, S * 0.8], fill=(240, 120, 40, 255))
    d.polygon([(S * 0.32, S * 0.5), (S * 0.5, S * 0.05), (S * 0.68, S * 0.5)], fill=(240, 120, 40, 255))
    d.ellipse([S * 0.42, S * 0.5, S * 0.58, S * 0.76], fill=(255, 245, 200, 255))
    d.polygon([(S * 0.43, S * 0.6), (S * 0.5, S * 0.32), (S * 0.57, S * 0.6)], fill=(255, 245, 200, 255))
    d.rounded_rectangle([S * 0.15, S * 0.74, S * 0.85, S * 0.84], radius=S * 0.03, fill=(120, 70, 40, 255))
    d.rounded_rectangle([S * 0.22, S * 0.82, S * 0.78, S * 0.9], radius=S * 0.03, fill=(150, 90, 50, 255))
    return im


def _build_dir(td, name="fire", img=None):
    (Path(td) / "assets").mkdir(parents=True, exist_ok=True)
    (img or campfire()).save(f"{td}/assets/{name}.png")
    return td


PARTS = "outer=the outer flame; inner=the inner flame; logs=the two logs"


def test_trace_rig_argv_and_degenerate():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _build_dir(td, "a", synthetic(48))
        synthetic(48).save(f"{td}/a.png")
        for argv in (("--work", "0"), ("--work", "8"), ("--colors", "0"), ("--tol", "0"), ("--bands", "-1"),
                     ("--tol", "inf"), ("--dp", "-3"), ("--display", "nan")):
            code, out = _cli("trace", f"{td}/a.png", "--out", f"{td}/o", *argv)
            assert code == 2, (argv, code)
        for argv in (("--keyframes", "1"), ("--fps", "0"), ("--seconds", "-1"), ("--min-detail", "2")):
            code, out = _cli("rig", td, "a", "--parts", PARTS, *argv)
            assert code == 2, (argv, code)
        # rig needs only DIR/assets/<name>.png: no assets.json, no mockup
        code, out = _cli("rig", td, "ghost", "--parts", PARTS, "--namer", "none")
        assert code == f"gptkit: asset not found: {td}/assets/ghost.png (forge it first or copy the PNG to DIR/assets/)", code
        _build_dir(td, "clear", Image.new("RGBA", (40, 40), (0, 0, 0, 0)))
        code, out = _cli("rig", td, "clear", "--parts", PARTS, "--namer", "none")
        assert "nothing opaque to rig" in str(code), code
        code, out = _cli("rig", td, "a", "--parts", PARTS, "--namer", "none", "--order", "outer,nope")
        assert "--order names nope" in str(code), code
        code, out = _cli("rig", td, "a", "--parts", PARTS, "--namer", "none")
        assert code == 0 and "regions ->" in out and (Path(td) / "rig" / "a" / "regions.png").exists(), (code, out)
        # _asset_dp: dict, dict without h, list, number, garbage
        Image.new("RGBA", (100, 50), (255, 0, 0, 255)).save(f"{td}/assets/wide.png")
        for dp, want in (({"w": 48, "h": 20}, (48, 20)), ({"w": 48}, (48, 24)), ([64, 30], (64, 30)), ([64], (64, 32)),
                         (80, (80, 40)), (None, None), ("x", None)):
            Path(td, "assets.json").write_text(json.dumps({"assets": [{"name": "wide", "dp": dp}]}))
            with _Cap():
                got = g._asset_dp(f"{td}/assets/wide.png")
            assert got == want, (dp, got)


def test_rig_assign_diagnostics():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _build_dir(td)
        res = g.trace_image(campfire(), "illustration")
        n = sum(L["role"] == "region" for L in res["layers"])
        assert n >= 3, n
        assign, issues = g.parse_assign("outer=99; inner=c7; typo=3", res)
        assert assign == {3: "typo"} and issues == [f"--assign: 99, c7 do not exist (regions 1..{n}, components c1..c1)"], (assign, issues)
        assign, issues = g.parse_assign("a=c1", res)
        assert len(assign) == n and not issues
        code, out = _cli("rig", td, "fire", "--parts", PARTS, "--assign", "outer=99; inner=c7; typo=3")
        assert "! --assign: 99, c7 do not exist" in out and "! part 'typo' is not in --parts" in out, out
        assert "! parts with no regions: outer, inner, logs" in out, out
        assert str(code).startswith("rig: outer, inner, logs got no region"), code
        # partial assignment without 'rest': the leftovers are merged, reported and written to assign.json
        comps = g.region_components(res)
        regs = [L for L in res["layers"] if L["role"] == "region"]
        by_col = {L["color"]: i + 1 for i, L in enumerate(regs)}
        outer = max(range(1, n + 1), key=lambda i: regs[i - 1]["area"] if regs[i - 1]["color"] < "#f0a000" or True else 0)
        # pick one region per part by colour family
        pick = {}
        for i, L in enumerate(regs):
            r, gch, b = int(L["color"][1:3], 16), int(L["color"][3:5], 16), int(L["color"][5:7], 16)
            fam = "inner" if b > 150 else "outer" if r > 200 else "logs"
            if fam not in pick or L["area"] > regs[pick[fam] - 1]["area"]:
                pick[fam] = i + 1
        assert set(pick) == {"outer", "inner", "logs"}, pick
        spec = "; ".join(f"{p}={i}" for p, i in pick.items())
        code, out = _cli("rig", td, "fire", "--parts", PARTS, "--assign", spec)
        assert code == 0, (code, out)
        final = json.loads(Path(td, "rig", "fire", "assign.json").read_text())
        assert set(final) == {str(i) for i in range(1, n + 1)}
        merged = [i for i, v in final.items() if v.get("merged")]
        if n > 3:
            assert merged and "merged into their neighbours" in out, out
        for i, v in final.items():
            assert v["part"] in ("outer", "inner", "logs"), v
        vec = json.loads(Path(td, "rig", "fire", "layers.json").read_text())
        assert {l["name"] for l in vec["static_layers"]} >= {"outer", "inner", "logs"}
        assert not g._validate_vec(vec)


def test_rig_api_partial():
    import io
    import tempfile
    calls = []

    def fake_edit(ref_png, prompt, size="1024x1024", background="auto", model=None, fidelity="high"):
        calls.append(prompt)
        if "inner flame" in prompt.split("Keep ONLY")[1][:30]:
            raise g.ApiError(400, "HTTP 400 /images/edits: {\"error\": {\"code\": \"moderation_blocked\"}}")
        im = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        from PIL import ImageDraw
        ImageDraw.Draw(im).ellipse([300, 300, 700, 800], fill=(240, 120, 40, 255))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue(), {}
    saved_edit, saved_ask = g.edit, g.ask
    g.edit = fake_edit
    g.ask = lambda *a, **k: (_ for _ in ()).throw(ValueError("Expecting value: line 1 column 1 (char 0)"))
    try:
        with tempfile.TemporaryDirectory() as td:
            _build_dir(td)
            code, out = _cli("rig", td, "fire", "--parts", PARTS, "--route", "hybrid")
            assert len(calls) == 3
            assert "! part 'inner': HTTP 400 /images/edits" in out and "moderation_blocked" in out, out
            assert 'prompt "Keep ONLY the inner flame' in out, out
            assert str(code).startswith("rig: 1 of 3 part isolation(s) failed; 2 succeeded and are cached"), code
            cached = sorted(p.name for p in Path(td, "rig", "fire", "parts_gpt").glob("*_raw.png"))
            assert cached == ["logs_raw.png", "outer_raw.png"], cached
            # the vision namer failing falls back to the --assign hint (exit 1, no traceback)
            code, out = _cli("rig", td, "fire", "--parts", PARTS)
            assert "! region naming failed: Expecting value" in out, out
            assert str(code).startswith("regions -> ") and "--assign" in str(code), code
    finally:
        g.edit, g.ask = saved_edit, saved_ask


def test_trace_perf_and_slivers():
    import time
    rng = np.random.default_rng(3)
    noise = Image.fromarray(np.dstack([rng.integers(0, 256, (256, 256, 3)), np.full((256, 256), 255)]).astype(np.uint8), "RGBA")
    t0 = time.time()
    res = g.trace_image(noise, "illustration")
    secs = time.time() - t0
    assert secs < 5, secs
    assert res.get("skipped", "").startswith("photographic:") or len(res["layers"]) <= 400, (res.get("skipped"), len(res["layers"]))
    # two crisp rects (fullbleed2): the resampling seam must not become sliver regions
    im = Image.new("RGBA", (200, 100), (30, 90, 200, 255))
    im.paste((120, 200, 60, 255), (40, 20, 160, 80))
    res = g.trace_image(im, "flat")
    assert len(res["layers"]) <= 3, len(res["layers"])
    assert res["size"] == (200, 100), "no upscaling: work is clamped to the source size"
    # an aliased ring: no glow bands from Lanczos ringing, compact paths
    yy, xx = np.mgrid[:256, :256]
    d = np.hypot(xx - 128, yy - 128)
    ring = np.zeros((256, 256, 4), np.uint8)
    ring[(d < 100) & (d > 60)] = (200, 40, 40, 255)
    res = g.trace_image(Image.fromarray(ring, "RGBA"), "illustration")
    assert not any(L["role"] == "glow" for L in res["layers"]), [L["name"] for L in res["layers"]]
    assert max(len(L["d"]) for L in res["layers"]) < 3000
    assert g._alpha_bands(np.zeros((8, 8)), np.zeros((8, 8, 3)), np.zeros((8, 8), bool), 0) == []
    # cropped masks reassemble to the region map
    full = g._lmask(res["layers"][0], res["size"][::-1])
    assert full.shape == (256, 256) and full.sum() == res["layers"][0]["area"]
    # the vector advice thresholds
    adv = g.vector_advice({"display": {"diff": 1.0, "iou": 0.99}}, [{"d": "M0 0" * 500, "fill": {"type": "linear"}}], (240, 240))
    assert adv["prefer"] == "svg/lottie" and not adv["vd_ok"] and "240dp > 200dp" in adv["why_not_vd"]
    assert any("over 800 chars" in r for r in adv["reasons"]) and any("API 24+" in r for r in adv["reasons"])
    adv = g.vector_advice({"display": {"diff": 6.0, "iou": 0.99}}, [{"d": "M0 0Z", "fill": {"type": "flat"}}])
    assert adv["prefer"] == "png/webp" and adv["fidelity"] == "lossy"
    adv = g.vector_advice(None, [{"d": "M0 0Z", "fill": {"type": "flat"}}] * 70)
    assert adv["prefer"] == "unknown" and "kit heuristic" in adv["reasons"][1]
    for r in adv["reasons"]:
        assert not r.endswith("; ")


def test_flat_fill_no_gradient():
    """A flat-colour disc gets a flat fill under every preset (no invented gradient -> no API 24 dependency)."""
    yy, xx = np.mgrid[:160, :160]
    disc = np.zeros((160, 160, 4), np.uint8)
    disc[np.hypot(xx - 80, yy - 80) < 60] = (250, 200, 40, 255)
    src = Image.fromarray(disc, "RGBA")
    for preset in g.TRACE_PRESETS:
        res = g.trace_image(src, preset)
        regs = [L for L in res["layers"] if L["role"] == "region"]
        assert regs and all(L["fill"]["type"] == "flat" for L in regs), (preset, [L["fill"]["type"] for L in regs])
    # while a real gradient is still detected
    res = g.trace_image(synthetic(128), "illustration")
    assert any(L["fill"]["type"] == "radial" for L in res["layers"] if L["role"] == "region")


def test_gradient_follows_part():
    """The radial highlight must ride with a bobbing part: sampled colour constant in the SVG keys and in lottie-web."""
    src = synthetic(256)
    vec = g.animate_layers(g.layers_vec(g.trace_image(src, "illustration"), "disc"), g._parse_motions("art=bob:4"), 6)
    grads = [al for al in vec["animated_layers"] if al.get("grad_keys")]
    assert grads and grads[0]["grad_kind"] == "radial" and len(grads[0]["grad_keys"]) == 6
    assert grads[0]["grad_keys"][0] != grads[0]["grad_keys"][3], "the gradient must move"
    lot = g._tr_animate_gradients(g._tidy_lottie(g.build_lottie(vec, 256, 1.0, 24)), vec)
    gf = [it for L in lot["layers"] for gr in L["shapes"] for it in gr["it"] if it["ty"] == "gf"]
    assert gf and all(it["s"]["a"] == 1 and len(it["s"]["k"]) == 7 and it["e"]["a"] == 1 for it in gf)
    assert gf[0]["s"]["k"][-1]["t"] == 24 and gf[0]["s"]["k"][-1]["s"] == gf[0]["s"]["k"][0]["s"]
    if not _chrome_ok():
        return
    pts = np.vstack([g._path_points(al["keyframes"][0]) for al in vec["animated_layers"]])
    box = (*pts.min(0), *pts.max(0))
    p0 = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 30)
    moving = [g.motion_field("bob", box, k / 6, 4.0, 0.0, sum(map(ord, "art")))(*p0) for k in range(6)]
    keys = g._tr_render_svgs([g._vec_render(vec, (256, 256), k) for k in range(6)], [(256, 256)] * 6)
    if keys is None:
        print(f"  (skip render: {g._tr_chrome_reason()})")
        return
    cols = np.array([np.asarray(keys[k])[int(y), int(x), :3].astype(int) for k, (x, y) in enumerate(moving)])
    assert np.abs(cols - cols[0]).max() <= 6, cols
    film = g.lottie_filmstrip(lot, [4 * k for k in range(6)], 256, "#000000")
    assert film is not None, g._tr_chrome_reason()
    if film is not None:
        cols = np.array([np.asarray(film)[int(y), 256 * k + int(x), :3].astype(int) for k, (x, y) in enumerate(moving)])
        assert np.abs(cols - cols[0]).max() <= 6, cols


def test_rig_internals():
    """region_components / region_overlay / build_rig modes / _assign_by_masks / register_part / name_regions / vtracer."""
    src = campfire(192)
    res = g.trace_image(src, "illustration")
    regs = [L for L in res["layers"] if L["role"] == "region"]
    comps = g.region_components(res)
    assert set(comps) == set(range(1, len(regs) + 1)) and set(comps.values()) == {1}, comps
    ov = g.region_overlay(src, res)
    assert ov.mode == "RGB" and ov.width > ov.height * 2.5
    # a part mask per family (PIL-drawn), as the hybrid route's registered GPT parts
    masks = {}
    for name, box, col in (("flame", (50, 5, 142, 155), (240, 120, 40)), ("logs", (20, 140, 172, 175), (120, 70, 40))):
        im = Image.new("RGBA", src.size, (0, 0, 0, 0))
        from PIL import ImageDraw
        ImageDraw.Draw(im).rectangle(box, fill=col + (255,))
        masks[name] = im
    assign = g._assign_by_masks(res, masks, ["logs", "flame"])
    assert set(assign) == set(comps) and set(assign.values()) <= {"flame", "logs"}
    assert "logs" in assign.values() and "flame" in assign.values(), assign
    vec = g.build_rig(src, res, assign, ["logs", "flame"], min_detail=0.0, glow_mode="part")
    assert not g._validate_vec(vec) and set(vec["layer_order"]) >= {"logs", "flame"}
    assert vec["rig"]["parts"]["flame"]["amodal_px"] >= 0 and vec["rig"]["boxes"]["logs"][3] > vec["rig"]["boxes"]["flame"][1]
    n_all = sum(len(l["paths"]) for l in vec["static_layers"])
    vec2 = g.build_rig(src, res, assign, ["logs", "flame"], min_detail=0.5, glow_mode="part")
    assert sum(len(l["paths"]) for l in vec2["static_layers"]) <= n_all
    # the glow modes on an asset with a real glow
    disc = synthetic(128)
    dres = g.trace_image(disc, "illustration")
    one = {i + 1: "disc" for i in range(sum(L["role"] == "region" for L in dres["layers"]))}
    assert "glow_disc" in g.build_rig(disc, dres, one, ["disc"], glow_mode="part")["layer_order"]
    assert g.build_rig(disc, dres, one, ["disc"], glow_mode="shared")["layer_order"][0] == "glow"
    # register_part recovers a 0.85x scale + shift within 0.03 / a few px
    sq, off = g._square(src)
    part = Image.new("RGBA", sq.size, (0, 0, 0, 0))
    ImageDraw.Draw(part).ellipse([58, 58, 134, 154], fill=(240, 120, 40, 255))
    small = part.resize((int(192 * 0.85), int(192 * 0.85)), Image.LANCZOS)
    drift = Image.new("RGBA", sq.size, (0, 0, 0, 0))
    drift.alpha_composite(small, (20, 10))
    _, info = g.register_part(drift, src)
    assert abs(info["scale"] - 1 / 0.85) <= 0.03, info
    # name_regions: out-of-range region numbers from the model are dropped, the schema enum holds the parts + other
    seen = {}

    def fake_ask(parts, schema=None, name="result", model=None):
        seen["enum"] = schema["properties"]["assignments"]["items"]["properties"]["part"]["enum"]
        return {"assignments": [{"region": 1, "part": "outer"}, {"region": 0, "part": "logs"}, {"region": 999, "part": "inner"},
                                {"region": 2, "part": "other"}]}
    saved, g.ask = g.ask, fake_ask
    try:
        got = g.name_regions(src, res, {"outer": "o", "inner": "i", "logs": "l"})
    finally:
        g.ask = saved
    assert got == {1: "outer", 2: "other"} and seen["enum"] == ["outer", "inner", "logs", "other"], (got, seen)
    # vtracer route (when installed): flat regions with absolute M/L/C/Z only
    if g._vtracer():
        vres = g.trace_vtracer(src, work=192, bands=0)
        vregs = [L for L in vres["layers"] if L["role"] == "region"]
        assert vregs and vres["size"] == (192, 192)
        for L in vregs:
            g._subpaths(L["d"])
            assert set(t for t in g._TOK.findall(L["d"]) if t.isalpha()) <= {"M", "L", "C", "Z"}
    # _tidy_lottie: 2-decimal vertices, linear easing, op > 0
    avec = g.animate_layers(g.layers_vec(dres), g._parse_motions("art=bob"), 4)
    lot = g._tidy_lottie(g.build_lottie(avec, 128, 1.0, 24))
    assert lot["op"] == 24 and lot["layers"]
    ks = next(it["ks"] for L in lot["layers"] for gr in L["shapes"] for it in gr["it"] if it["ty"] == "sh" and it["ks"]["a"] == 1)
    assert ks["k"][0]["i"] == {"x": [1], "y": [1]} and ks["k"][0]["o"] == {"x": [0], "y": [0]}
    assert all(round(c, 2) == c for pt in ks["k"][0]["s"][0]["v"] for c in pt)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
