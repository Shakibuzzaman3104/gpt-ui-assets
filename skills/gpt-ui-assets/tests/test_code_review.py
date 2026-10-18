"""Regressions found by the Phase C code review (offline, no API calls): python3 tests/test_code_review.py."""
import argparse
import ast
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
KIT = HERE.parent / "scripts" / "gptkit.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
import gptkit as gk  # noqa: E402


def test_python39_grammar():
    """README promises Python 3.9+: the kit must parse under the 3.9 grammar (no match, except*, parenthesised with)."""
    src = KIT.read_text()
    ast.parse(src, feature_version=(3, 9))
    assert not re.search(r"zip\([^)]*strict\s*=", src), "zip(strict=) is Python 3.10+"
    for feature in (".bit_count(", "itertools.pairwise", "ExceptionGroup", "hashlib.file_digest", "datetime.UTC"):
        assert feature not in src, f"{feature} is not in Python 3.9"


def test_clip_bbox_infinite_values():
    """json.loads('1e400') is inf; a hostile model response must be dropped, not raise OverflowError."""
    assert json.loads('{"x": 1e400}')["x"] == float("inf")
    for b in ({"x": float("inf"), "y": 0, "w": 10, "h": 10}, {"x": 0, "y": 0, "w": 1e400, "h": 10},
              {"x": float("-inf"), "y": 0, "w": 10, "h": 10}, {"x": 0, "y": float("nan"), "w": 10, "h": 10}):
        assert gk._clip_bbox(b, 100, 100) is None, b
    assert gk._clip_bbox({"x": 10, "y": 10, "w": -5, "h": -5}, 100, 100) == {"x": 5, "y": 5, "w": 5, "h": 5}


def test_css_duplicate_selectors_later_rule_wins():
    """Two <style> rules for the same selector (common in exported and hand-written CSS) must not crash the sort."""
    dup = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><style>.st0{fill:#ff0000}.st0{fill:#00ff00;stroke:#0000ff}'
           'path{stroke:#333333}path{stroke-width:3}</style><rect class="st0" x="2" y="2" width="20" height="20"/>'
           '<path d="M2 2h20v20z"/></svg>')
    vd = gk.svg_to_vd(dup, 24)
    assert 'android:fillColor="#00FF00"' in vd and 'android:strokeColor="#0000FF"' in vd, vd  # later .st0 rule won
    assert 'android:strokeColor="#333333"' in vd and 'android:strokeWidth="3"' in vd, vd  # both path rules apply
    root = gk.ET.fromstring(dup)
    rules = gk._css_rules(root)
    rect = next(el for el in root.iter() if el.tag.endswith("rect"))
    assert gk._css_props(rect, rules) == {"fill": "#00ff00", "stroke": "#0000ff"}


def test_obb_radial_default_radius():
    """SVG default radialGradient (objectBoundingBox, r=50%) on a 24 x 24 shape is a 12-unit radius, not 12/sqrt(2)."""
    base = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><defs><radialGradient id="g"{attrs}>'
            '<stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/></radialGradient></defs>'
            '<rect width="24" height="24" fill="url(#g)"/></svg>')
    radius = lambda svg: float(re.search(r'gradientRadius="([^"]+)"', gk.svg_to_vd(svg, 24)).group(1))
    assert radius(base.format(attrs="")) == 12
    assert radius(base.format(attrs=' r="50%"')) == 12
    assert radius(base.format(attrs=' r="0.25"')) == 6
    assert radius(base.format(attrs=' gradientUnits="userSpaceOnUse" cx="12" cy="12"')) == 12
    assert radius(base.format(attrs=' gradientUnits="userSpaceOnUse" cx="12" cy="12" r="25%"')) == 6
    # the vectorize importer resolves the same default to the same radius
    root = gk._defs_root({"viewBox": "0 0 24 24", "defs": '<radialGradient id="g"><stop offset="0" stop-color="#ff0000"/>'
                                                          '<stop offset="1" stop-color="#0000ff"/></radialGradient>'})
    linear, s, e, _ = gk._gradient(root, "g", (0, 0, 24, 24))
    assert not linear and s == [12.0, 12.0] and e == [24.0, 12.0]
    if gk._renderer_name():  # the VD must render like the SVG (a wrong radius measured mean 16.7 / 2400 px far off)
        rt = gk.vd_roundtrip([(base.format(attrs=""), gk.svg_to_vd(base.format(attrs=""), 24))])[0]
        assert rt["mean"] < 1 and rt["strong"] == 0, rt


def test_write_anim_keeps_sibling_animations():
    """`frames build rain` after `frames build rain --procedural pulse`: the rain_* cleanup must leave rain_pulse_* alone."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "frames" / "rain"
        (out / "android" / "drawable-nodpi").mkdir(parents=True)
        (out / "android" / "drawable").mkdir(parents=True)
        for i in (1, 2):
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(out / "android" / "drawable-nodpi" / f"rain_pulse_frame_{i:02d}.png")
        (out / "android" / "drawable" / "rain_pulse_anim.xml").write_text("<x/>")
        Image.new("RGBA", (8, 8)).save(out / "android" / "drawable-nodpi" / "rain_frame_07.png")  # stale, from a longer run
        frames = [Image.new("RGBA", (16, 16), (0, 255, 0, 255)) for _ in range(3)]
        gk._write_anim(out, "rain", frames, 12, argparse.Namespace(formats="webp", px=16, preview_bg="#000000"))
        names = sorted(p.name for p in (out / "android").rglob("*.*"))
        assert names == ["rain_anim.xml", "rain_frame_01.png", "rain_frame_02.png", "rain_frame_03.png",
                         "rain_pulse_anim.xml", "rain_pulse_frame_01.png", "rain_pulse_frame_02.png"], names


def test_load_rejects_non_numeric_sizes_before_any_api_call():
    """A hand-edited width/height/screen_dp used to surface as a TypeError traceback after the svg API calls."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        Image.new("RGBA", (200, 300), (30, 30, 60, 255)).save(d / "mockup.png")
        base = {"mockup": "mockup.png", "width": 200, "height": 300, "screen_dp": 200,
                "assets": [{"name": "a", "kind": "icon", "format": "svg", "bbox": {"x": 20, "y": 20, "w": 60, "h": 60}}]}
        for k, v in (("width", "abc"), ("screen_dp", "abc"), ("screen_dp", 0), ("height", True), ("width", -5)):
            (d / "assets.json").write_text(json.dumps({**base, k: v}))
            try:
                gk._load(d)
            except SystemExit as e:
                assert f"'{k}' must be a positive number" in str(e) and "\n" not in str(e), (k, v, e)
            else:
                raise AssertionError(f"_load accepted {k}={v!r}")
        (d / "assets.json").write_text(json.dumps({**base, "width": 100, "height": 100}))  # a replaced mockup
        _, m, mock = gk._load(d)
        assert (m["width"], m["height"]) == mock.size == (200, 300)


def test_affine_helpers_agree():
    """_tf_affine is _matrix (skews included): the converter and the normaliser share one affine implementation."""
    for tf in ("skewX(30)", "skewY(-15) translate(2 3)", "rotate(45 12 12) scale(2 0.5)", "matrix(1 0 0.5 1 0 0)"):
        assert gk._tf_affine(tf) == gk._matrix(tf), tf
    a, b, c, d, e, f = gk._tf_affine("skewX(45)")
    assert abs(c - 1.0) < 1e-9 and (a, b, d, e, f) == (1, 0, 1, 0, 0)
    skewed = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect x="4" y="4" width="8" height="16" '
              'transform="skewX(30)" fill="#123456"/></svg>')
    warns = []
    vd = gk.svg_to_vd(skewed, 24, warns)
    assert "skew transform baked into path coordinates" in warns and "<group" not in vd and "M" in vd


def test_responses_text_non_dict_body():
    """A gateway answering with a JSON list or string is an ApiError (one line from main), not an AttributeError."""
    for body in ([{"output": []}], "text", None, 42):
        try:
            gk._responses_text(body)
        except gk.ApiError as e:
            assert e.code == "empty", (body, e.code)
        else:
            raise AssertionError(f"no ApiError for {body!r}")


def test_forge_cancels_queued_calls_on_fatal_error():
    """A fatal error while finishing one asset must not let the queued image calls of the next asset start."""
    calls = []

    def fake_edit(ref, prompt, size, background="auto", model=None, fidelity="high"):
        calls.append(prompt)
        time.sleep(0.4 if '"b"' in prompt else 0.02)  # b's first call is in flight when a's finish fails; its second is queued
        im = Image.new("RGBA", (128, 128))
        ImageDraw.Draw(im).ellipse([20, 20, 108, 108], fill=(220, 90, 40, 255))
        return gk._png_bytes(im), {"quality": "high", "size": size}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mock = Image.new("RGBA", (200, 200), (20, 20, 40, 255))
        ImageDraw.Draw(mock).ellipse([20, 20, 80, 80], fill=(220, 90, 40, 255))
        ImageDraw.Draw(mock).ellipse([120, 120, 180, 180], fill=(40, 200, 90, 255))
        mock.save(d / "mockup.png")
        (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": 200, "height": 200, "screen_dp": 200, "assets": [
            {"name": n, "kind": "illustration", "format": "png", "transparent": True, "bbox": {"x": x, "y": x, "w": 60, "h": 60}}
            for n, x in (("a", 20), ("b", 120))]}))
        old_edit, old_fin = gk.edit, gk._finalize
        gk.edit = fake_edit
        gk._finalize = lambda asset, img: (_ for _ in ()).throw(RuntimeError("disk full"))
        try:
            a = argparse.Namespace(dir=str(d), only=None, candidates=2, jobs=1, force=True, include_svg=False, same_prompt=False,
                                   color_match=0.0, model=None)
            try:
                gk.cmd_forge(a)
            except RuntimeError as e:
                assert "disk full" in str(e)
            else:
                raise AssertionError("cmd_forge did not raise")
            time.sleep(1.0)  # without cancel_futures the worker would have drawn b's second candidate by now
            assert sum('"b"' in p for p in calls) <= 1, [p[p.find("Recreate"):][:40] for p in calls]
        finally:
            gk.edit, gk._finalize = old_edit, old_fin



def test_package_state_is_registered():
    """Every module global a function rebinds with `global` is listed in gptkit.STATE (else gptkit.<name> goes stale)."""
    pkg = Path(gk.__file__).parent
    rebound = set()
    for mod in pkg.glob("*.py"):
        for node in ast.walk(ast.parse(mod.read_text())):
            if isinstance(node, ast.Global):
                rebound.update(node.names)
    assert rebound <= set(gk.STATE), f"add to gptkit.STATE: {sorted(rebound - set(gk.STATE))}"
    for name, mod in gk.STATE.items():
        assert name in vars(mod), f"gptkit.STATE[{name!r}] names the wrong module"

if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            t = time.time()
            fn()
            print(f"ok {k} ({time.time() - t:.1f}s)")
