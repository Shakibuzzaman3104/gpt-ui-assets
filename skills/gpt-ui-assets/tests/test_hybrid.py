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
