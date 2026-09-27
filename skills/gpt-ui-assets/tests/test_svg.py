"""SVG -> VectorDrawable and 24-grid normalisation checks. Run: python3 tests/test_svg.py (or pytest)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402

FIX = HERE / "svg"
A = "{http://schemas.android.com/apk/res/android}"


def vd(name, warns=None):
    return K.svg_to_vd((FIX / f"{name}.svg").read_text(), None, warns)


def test_path_parser():
    segs = K.parse_path("M4 12a8 8 0 1016 0 8 8 0 10-16 0zm4 0h-1.5e1")
    assert [c for c, _ in segs] == ["M", "a", "a", "z", "m", "h"]
    assert segs[1][1] == [8, 8, 0, 1, 0, 16, 0] and segs[5][1] == [-15]
    assert K.path_str(K.transform_path(K.parse_path("m2 2l1 1"), 2, 1, 1)) == "M5 5 l2 2"


def test_root_attrs_inherited():
    paths = ET.fromstring(vd("root_attrs")).findall("path")
    assert paths and all(p.get(A + "strokeColor") == "#5B4FC4" and p.get(A + "fillColor") is None for p in paths)
    assert all(p.get(A + "strokeLineCap") == "round" for p in paths)


def test_group_opacity_multiplies():
    alphas = [float(p.get(A + "fillAlpha")) for p in ET.fromstring(vd("group_opacity")).findall("path")]
    assert alphas == [0.25, 0.5]


def test_colors_and_transforms():
    fills = [p.get(A + "fillColor") for p in ET.fromstring(vd("colors")).findall("path")]
    assert fills == ["#0A7CFF", "#FF4020", "#FFA500", "#8800FF00"]
    assert ET.fromstring(vd("transforms")).find("group/group") is not None


def test_dasharray_warns():
    warns = []
    vd("rects_dash", warns)
    assert any("dasharray" in w for w in warns)


def test_normalize_to_grid():
    svg = (FIX / "root_attrs.svg").read_text()
    out = K.normalize_svg(svg, (2, 1, 22, 23), 24, 2)
    root = ET.fromstring(out)
    assert root.get("viewBox") == "0 0 24 24" and "transform" not in out
    assert abs(float(root.get("stroke-width")) - 2 * 20 / 22) < 1e-3


def test_roundtrip_and_aapt2():
    names = sorted(p.stem for p in FIX.glob("*.svg"))
    pairs = [((FIX / f"{n}.svg").read_text(), vd(n)) for n in names]
    diffs = K.vd_roundtrip(pairs)
    for n, d in zip(names, diffs):
        assert d is None or d < (2 if n != "rects_dash" else 3), (n, d)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        files = []
        for n, (_, x) in zip(names, pairs):
            files.append(Path(td) / f"ic_{n}.xml")
            files[-1].write_text(x)
        errs = K.aapt2_check(files)
        assert not errs, errs


def test_icons_mixed_sizes_without_renderer():
    import argparse
    import json
    import tempfile
    from PIL import Image, ImageDraw
    icon = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" '
            'stroke-linecap="round"><path d="M4 12h16"/></svg>')
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "svg").mkdir()
        im = Image.new("RGB", (400, 400), "#101830")
        ImageDraw.Draw(im).rectangle([40, 40, 60, 60], fill="white")
        ImageDraw.Draw(im).rectangle([200, 200, 300, 300], fill="white")
        im.save(d / "mockup.png")
        assets = [{"name": n, "kind": "icon", "format": "svg", "bbox": b} for n, b in
                  (("small", {"x": 40, "y": 40, "w": 20, "h": 20}), ("big", {"x": 200, "y": 200, "w": 100, "h": 100}))]
        (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": 400, "height": 400, "screen_dp": 390,
                                                   "assets": assets}))
        for n in ("small", "big"):
            (d / "svg" / f"{n}.svg").write_text(icon)
        saved = K.ask, K.render_svgs
        K.ask = lambda *a_, **k_: {"files": [{"name": "bell", "svg": icon}]}
        K.render_svgs = lambda svgs, size=256: [None] * len(svgs)
        try:
            K.cmd_icons(argparse.Namespace(dir=td, concepts="bell", out=None, treatment="auto", stroke=None, color=None,
                                           snap=False, vd=None, dp=None))
        finally:
            K.ask, K.render_svgs = saved
        assert (d / "icons" / "bell.svg").exists() and (d / "icons" / "preview.png").exists()


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
