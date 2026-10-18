"""Hybrid screens (material / state / stage), offline: python3 tests/test_hybrid.py

material: candidates are flagged EMPTY / OPAQUE / CUT-OFF and the best unflagged one is trimmed to NAME.png.
state: an edit that re-renders the whole image and comes back shifted is registered, and only the changed region is
composited back, so the output is pixel-identical to the source outside the mask; an edit that changes nothing is refused.
stage: a real page is rendered to stills and a scrubbed motion clip (skipped without Playwright + Chrome).
The image API is mocked at gptkit.generate / gptkit.edit; no network."""
import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def _run(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    K._hybrid_parsers(sub)
    a = ap.parse_args(argv)
    a.fn(a)


def _object(w=1536, h=1024, box=(300, 300, 1200, 700), alpha=True):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0) if alpha else (230, 228, 222, 255))
    ImageDraw.Draw(im).rounded_rectangle(box, 40, fill=(200, 201, 196, 255))
    return im


def test_material_flags_and_best():
    renders = iter([_object(alpha=False), _object(box=(0, 300, 900, 700)), _object()])
    K.generate = lambda prompt, **kw: (_png(next(renders)), {"source_format": "PNG"})
    with tempfile.TemporaryDirectory() as d:
        _run(["material", "a foil card", "--out", d, "--name", "foil", "--candidates", "3", "--jobs", "1"])
        meta = json.loads((Path(d) / "material.json").read_text())["foil"]
        flags = {k: v["flags"] for k, v in meta["candidates"].items()}
        assert flags == {"c1": ["OPAQUE"], "c2": ["CUT-OFF"], "c3": []}, flags
        assert meta["best"] == "c3", meta["best"]
        out = Image.open(Path(d) / "foil.png")
        assert out.width < 1536 and out.height < 1024, out.size  # trimmed to content
        assert (Path(d) / "review" / "material_foil.png").exists()


def test_material_all_empty_exits():
    K.generate = lambda prompt, **kw: (_png(Image.new("RGBA", (1536, 1024), (0, 0, 0, 0))), {})
    with tempfile.TemporaryDirectory() as d:
        try:
            _run(["material", "x", "--out", d, "--name", "e", "--candidates", "1"])
        except SystemExit as e:
            assert "no usable candidate" in str(e), e
        else:
            raise AssertionError("an all-EMPTY run must exit")


def _scene(seed, w=900, h=500):
    """A textured card with fixed structure (bars and dots), so a re-render keeps the layout but not the texture."""
    rng = np.random.default_rng(seed)
    im = Image.fromarray(np.dstack([np.full((h, w, 3), 190, np.uint8), np.full((h, w), 255, np.uint8)]))
    d = ImageDraw.Draw(im)
    for i in range(9):
        d.rectangle((40 + i * 95, 40, 80 + i * 95, 120), fill=(70, 72, 70, 255))
        d.ellipse((60 + i * 90, 380, 90 + i * 90, 410), fill=(120, 110, 100, 255))
    arr = np.asarray(im).astype(np.int16)
    arr[..., :3] += rng.integers(-8, 9, (h, w, 1), dtype=np.int16)
    return np.clip(arr, 0, 255).astype(np.uint8)


def test_state_registers_and_keeps_outside_identical():
    src = Image.fromarray(_scene(1))
    ImageDraw.Draw(src).ellipse((400, 200, 500, 300), fill=(250, 250, 250, 255))
    edited = Image.fromarray(_scene(2))  # a full re-render: same structure, every pixel's texture differs
    ImageDraw.Draw(edited).ellipse((400, 200, 500, 300), fill=(90, 90, 92, 255))  # the change
    shifted = Image.fromarray(np.roll(np.roll(np.asarray(edited), -2, 0), 3, 1))
    unchanged = src.copy()
    calls = iter([shifted, unchanged])
    K.edit = lambda ref, prompt, **kw: (_png(next(calls)), {})
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "src.png"
        src.save(sp)
        out = Path(d) / "popped.png"
        _run(["state", str(sp), "the dot is dark", "--circle", "450,250,50", "--out", str(out), "--candidates", "2", "--jobs", "1"])
        rep = json.loads((Path(d) / "state.json").read_text())["popped.png"]
        assert rep["best"] == "c1", rep
        assert rep["candidates"]["c1"]["shift"] == [-3, 2], rep["candidates"]["c1"]
        assert "NO-CHANGE" in rep["candidates"]["c2"]["flags"], rep["candidates"]["c2"]
        assert rep["max_diff_outside_mask"] == 0, rep
        o = np.asarray(Image.open(out).convert("L"), np.float32)
        assert o[250, 450] < 140, o[250, 450]  # the change is in


def test_state_refuses_when_nothing_changed():
    src = Image.fromarray(_scene(3))
    K.edit = lambda ref, prompt, **kw: (_png(src), {})
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "s.png"
        src.save(sp)
        try:
            _run(["state", str(sp), "x", "--box", "100,100,200,200", "--out", str(Path(d) / "o.png"), "--candidates", "1"])
        except SystemExit as e:
            assert "no candidate changed" in str(e), e
        else:
            raise AssertionError("an unchanged edit must be refused")


def test_stage_renders_stills_and_motion():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("  skip stage: no playwright")
        return
    if not K._chrome():
        print("  skip stage: no Chrome")
        return
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "screen.html"
        _run(["stage", str(page), "--init"])
        (Path(d) / "assets").mkdir()
        _object(400, 260, (20, 20, 380, 240)).save(Path(d) / "assets" / "material.png")
        out = Path(d) / "shots"
        _run(["stage", str(page), "--out", str(out), "--still", "screen=still", "--motion", "0.6", "--fps", "10", "--sheet", "4"])
        assert Image.open(out / "screen.png").size == (412 * 3, 915 * 3)
        rep = json.loads((out / "stage.json").read_text())
        assert rep["motion"]["frames"] == 6 and (out / rep["motion"]["file"]).exists(), rep
        assert not rep["errors"], rep["errors"]
        f0, f5 = (np.asarray(Image.open(out / "frames" / f"000{i}.png").convert("L"), np.float32) for i in (0, 5))
        assert np.abs(f5 - f0).mean() > 1, "the scrubbed frames must differ (entrance animation)"
        assert (out / "motion_sheet.png").exists()


def test_compose_converters_and_emitters():
    assert K._transform_components("translate(12px, -4px) scale(2, .5) rotate(90deg)") == {
        "translationX": 12.0, "translationY": -4.0, "scaleX": 2.0, "scaleY": .5, "rotationZ": 90.0}
    m = K._transform_components("matrix(0, 2, -3, 0, 8, 9)")
    assert m["translationX"] == 8 and m["translationY"] == 9
    assert m["scaleX"] == 2 and m["scaleY"] == 3 and m["rotationZ"] == 90
    try:
        K._transform_components("matrix(1, 0, 1, 1, 0, 0)")
    except ValueError as e:
        assert "skew" in str(e)
    else:
        raise AssertionError("matrix skew must be rejected")
    assert K._motion_easing("linear")[0] == "LinearEasing"
    assert "0.42f, 0f, 1f, 1f" in K._motion_easing("ease-in")[0]
    assert "CubicBezierEasing(0.1f, 0.2f, 0.3f, 0.4f)" == K._motion_easing("cubic-bezier(.1,.2,.3,.4)")[0]
    assert K._motion_easing("steps(4)") == ("LinearEasing", "unsupported easing steps(4): using linear")
    assert K._motion_property_values("--hole", "12px") == {"Hole": 12.0}
    doc = {"animations": [{"target": {"id": "pill"}, "timing": {"duration": 600, "delay": 40,
            "easing": "ease-out"}, "keyframes": [
            {"offset": 0, "easing": "ease-in", "properties": {"transform": "translate(0px, 2px) scale(.5)", "opacity": 0, "--hole": "0px"}},
            {"offset": 1, "easing": "linear", "properties": {"transform": "translate(20px, 0px) scale(1)", "opacity": 1, "--hole": "18px"}}]}]}
    kt, props, warns, _ = K._emit_motion_kt(doc, "Hybrid", "com.acme.app.ui.hybrid")
    assert "keyframes {" in kt and "durationMillis = 600" in kt and "delayMillis = 40" in kt and "DelayMillis = 40" in kt
    assert "TranslationX" in kt and "Opacity" in kt and "Hole" in kt
    assert {p for _, p in props} == {"translationX", "translationY", "scaleX", "scaleY", "opacity", "Hole"}  # rotation stays 0
    assert not warns
    geo = K._emit_geo_kt({"radius": 12, "centers": [1, 2], "hero": {"x": 4.5}}, "Hybrid", "a.b")
    assert "12.0f" in geo and "listOf(1.0f, 2.0f)" in geo and "object Hero" in geo
    original_read_text = Path.read_text
    Path.read_text = lambda self, *args, **kwargs: "object X {\n  val x = 1f\n  val x = NaN\n" \
        if self.name == "bad.kt" else original_read_text(self, *args, **kwargs)
    try:
        lint = K._lint_stage_compose([Path("bad.kt")])
    finally:
        Path.read_text = original_read_text
    assert any("unbalanced braces" in x for x in lint)
    assert any("duplicate identifier x" in x for x in lint)
    assert any("non-finite" in x for x in lint)


def test_stage_compose_integration():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("  skip stage compose: Playwright is not installed for this Python")
        return
    if not K._chrome():
        print("  skip stage compose: installed Chrome was not found")
        return
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "screen.html"
        _run(["stage", str(page), "--init"])
        page.write_text('''<!doctype html><meta name="viewport" content="width=412, initial-scale=1"><style>
          @property --hole { syntax: '<length>'; inherits: false; initial-value: 0px }
          #pill { position:absolute; left:30px; top:40px; width:80px; height:40px; --hole:0px }
        </style><div id="pill"></div><script>
          window.GEO = {asset:{center:[12, 18]}, radius:7};
          document.body.dataset.ready = '1';
          window.demo = () => { document.getElementById('pill').animate([
            {transform:'translate(0px, 0px) scale(.5)', opacity:0, '--hole':'0px'},
            {transform:'translate(20px, 8px) scale(1)', opacity:1, '--hole':'16px'}
          ], {duration:500, delay:50, easing:'ease-out', fill:'forwards'}).pause(); };
          window.seek = ms => document.getAnimations().forEach(a => a.currentTime = ms);
        </script>''')
        out = Path(d) / "compose"
        _run(["stage", str(page), "--compose", str(out)])
        motion = json.loads((out / "motion.json").read_text())
        assert motion["schema"] == "gptkit-motion-v1" and len(motion["animations"]) == 1
        assert motion["animations"][0]["target"]["id"] == "pill"
        assert motion["animations"][0]["target"]["rect"]["width"] == 80
        assert motion["geo"]["radius"] == 7
        kt = (out / "HybridMotion.kt").read_text()
        assert all(f"val pill{p}:" in kt for p in ("TranslationX", "TranslationY", "ScaleX", "Opacity", "Hole")), kt
        assert "pillRotationZ" not in kt                                   # rotate never changes: not emitted
        assert "20.0f at 500 using CubicBezierEasing(0f, 0f, 0.58f, 1f)" in kt and "delayMillis = 50" in kt
        test_kt = (out / "HybridMotionScreenshotTest.kt").read_text()
        assert "advanceTimeBy(50)" in test_kt and "advanceTimeBy(500)" in test_kt and 'hybrid_550.png' in test_kt, test_kt
        assert (out / "HybridGeo.kt").exists() and (out / "HybridMotionScreenshotTest.kt").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        gen, ed = K.generate, K.edit
        try:
            t()
            print(f"ok   {t.__name__}")
        finally:
            K.generate, K.edit = gen, ed
    print(f"{len(tests)} hybrid tests passed")
