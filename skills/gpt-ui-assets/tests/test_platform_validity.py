"""Platform-output validity regressions (phase C, lens c1): every file the kit emits must pass a real Android/iOS/web build.
Run: python3 tests/test_platform_validity.py (or pytest). No API calls; aapt2/Chrome checks skip honestly when absent."""
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger
import gptkit as g  # noqa: E402

A = "{http://schemas.android.com/apk/res/android}"
NS = 'xmlns="http://www.w3.org/2000/svg"'


def _renderer():
    return g._chrome_probe()[0] or (g._renderer_name() not in (None, "chrome"))


def test_obb_radial_default_radius():
    """An objectBoundingBox radial gradient (the SVG default) with r=50% (the default) spans half the shape's box: on a
    40x40 rect the VD radius is 20, not 20/sqrt(2)."""
    svg = (f'<svg {NS} viewBox="0 0 48 48"><defs><radialGradient id="rg"><stop offset="0" stop-color="#fff"/>'
           '<stop offset="1" stop-color="#000"/></radialGradient></defs><rect x="4" y="4" width="40" height="40" fill="url(#rg)"/></svg>')
    grad = ET.fromstring(g.svg_to_vd(svg)).find(".//gradient")
    assert abs(float(grad.get(A + "gradientRadius")) - 20) < 1e-6, grad.attrib
    assert abs(float(grad.get(A + "centerX")) - 24) < 1e-6
    explicit = ET.fromstring(g.svg_to_vd(svg.replace('id="rg"', 'id="rg" r="0.25"'))).find(".//gradient")
    assert abs(float(explicit.get(A + "gradientRadius")) - 10) < 1e-6
    if _renderer():
        rt = g.vd_roundtrip([(svg, g.svg_to_vd(svg))])[0]
        assert rt is not None and rt["mean"] < 1 and rt["strong"] <= 16, rt


def test_hsl_hue_units_and_percent_lengths():
    assert g._color("hsl(0.5turn, 50%, 50%)") == g._color("hsl(180, 50%, 50%)") == "#40BFBF"
    assert g._color("hsl(200grad 100% 50%)") == g._color("hsl(180deg 100% 50%)")
    assert g._color("hsl(3.14159265rad, 100%, 50%)") == "#00FFFF"
    svg = f'<svg {NS} viewBox="0 0 24 24"><rect x="10%" y="25%" width="50%" height="50%" fill="#000"/><circle cx="50%" cy="50%" r="25%" fill="#f00"/></svg>'
    x = g.svg_to_vd(svg)
    assert 'android:pathData="M2.4 6 h12 v12 h-12 Z"' in x, x
    assert "M6 12 a6 6" in x, x  # r = 25% of the normalised diagonal (24) = 6
    if _renderer():
        rt = g.vd_roundtrip([(svg, x)])[0]
        assert rt is not None and rt["strong"] <= 16, rt


def test_no_viewbox_is_warned():
    warns = []
    g.svg_to_vd(f'<svg {NS}><rect width="10" height="10"/></svg>', None, warns)
    assert any("no viewBox" in w and "24x24" in w for w in warns), warns
    warns = []
    g.svg_to_vd(f'<svg {NS} width="48" height="48"><rect width="10" height="10"/></svg>', None, warns)
    assert not any("no viewBox" in w for w in warns), warns


def test_evenodd_clip_keeps_its_holes():
    """A <clip-path> is always nonzero on Android: an evenodd clip with a hole is rewound (not filled in)."""
    svg = (f'<svg {NS} viewBox="0 0 24 24"><defs><clipPath id="c" clip-rule="evenodd"><path clip-rule="evenodd" '
           'd="M0 0h24v24H0z M6 6h12v12H6z"/></clipPath></defs><rect width="24" height="24" fill="#00f" clip-path="url(#c)"/></svg>')
    warns = []
    x = g.svg_to_vd(svg, None, warns)
    assert any("rewound" in w for w in warns) and not any("approximated as nonzero" in w for w in warns), warns
    clip = ET.fromstring(x).find(".//clip-path").get(A + "pathData")
    assert clip.startswith("M0,0 L24,0 L24,24 L0,24 Z M6,18") , clip  # the hole runs the other way round
    assert not g._vd_lint(x)
    if _renderer():
        rt = g.vd_roundtrip([(svg, x)])[0]
        assert rt is not None and rt["strong"] == 0 and rt["mean"] < 0.5, rt
    # a single-subpath evenodd clip is identical either way: no warning
    warns = []
    g.svg_to_vd(svg.replace(" M6 6h12v12H6z", ""), None, warns)
    assert not any("evenodd" in w for w in warns), warns
    # curves: a ring of two circles rewinds the inner one (control points swap with the direction)
    ring = "M2 12a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z M6 12a6 6 0 1 0 12 0 6 6 0 1 0-12 0Z"
    subs = g._subpaths(g._rewind_evenodd(ring))
    assert len(subs) == 2 and (g._area(np.array(g._curve_pts([subs[0]]))) > 0) != (g._area(np.array(g._curve_pts([subs[1]]))) > 0)


def test_export_names_are_url_and_path_safe():
    import subprocess
    kit = Path(__file__).resolve().parents[1] / "scripts" / "gptkit.py"
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "a.png"
        Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(src)
        env = {**os.environ, "GPTKIT_CONFIG": "/nonexistent"}
        r = subprocess.run([sys.executable, str(kit), "export", str(src), "--dp", "16", "--res", f"{td}/web", "--platform", "web",
                            "--name", "My Icon/2"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (Path(td) / "web" / "My-Icon-2.webp").exists() and (Path(td) / "web" / "My-Icon-2@2x.webp").exists(), r.stdout
        assert 'srcset="My-Icon-2.webp 1x, My-Icon-2@2x.webp 2x"' in r.stdout and "renamed" in r.stdout, r.stdout
        r = subprocess.run([sys.executable, str(kit), "export", str(src), "--dp", "16", "--res", f"{td}/A.xcassets", "--platform", "ios",
                            "--name", "a/b:c"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and (Path(td) / "A.xcassets" / "a-b-c.imageset" / "Contents.json").exists(), r.stdout + r.stderr
        c = json.loads((Path(td) / "A.xcassets" / "a-b-c.imageset" / "Contents.json").read_text())
        assert [i["filename"] for i in c["images"]] == ["a-b-c@1x.png", "a-b-c@2x.png", "a-b-c@3x.png"]


def test_negative_radius_is_clamped():
    """A negative spec radius has no meaning on any platform (CSS drops the declaration): tokens emit 0 and say so."""
    import contextlib
    import io
    spec = {"px_to_dp": 0.5, "palette": [], "typography": [], "spacing": [],
            "radii": [{"element": "chip", "px": -3}, {"element": "card", "px": 12}]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tm = g._token_model(spec, [], {}, None)
    assert [r["dp"] for r in tm["radii"]] == [0, 6], tm["radii"]
    assert "negative" in buf.getvalue() and "chip" in buf.getvalue(), buf.getvalue()


def test_markers_on_an_invisible_carrier_are_reported():
    """flag-icons' US flag draws its stars as marker-mid on a fill=none path: the VD loses them and must say so."""
    svg = (f'<svg {NS} viewBox="0 0 24 24"><defs><marker id="m" markerWidth="4" markerHeight="4"><circle cx="2" cy="2" r="2" fill="#fff"/>'
           '</marker></defs><path fill="none" marker-mid="url(#m)" d="M2 2h10 10"/><rect width="24" height="24" fill="#00f"/></svg>')
    warns = []
    g.svg_to_vd(svg, None, warns)
    assert any("marker" in w for w in warns), warns


def test_lottie_lint_accepts_image_sequences():
    """frames' image-sequence Lottie shows each frame for one composition frame: a legal ty=2 window, not a lint error."""
    frames = [Image.new("RGBA", (8, 8), (255, 0, 0, 255)) for _ in range(3)]
    lot = g.lottie_image_sequence(frames, 12, "seq")
    assert g._lottie_lint(lot) == [], g._lottie_lint(lot)
    lot["layers"][0]["op"] = lot["op"] + 1
    assert any("ip/op" in e for e in g._lottie_lint(lot))
    shape = {"v": "5.7.0", "fr": 30, "ip": 0, "op": 30, "w": 8, "h": 8, "assets": [],
             "layers": [{"ty": 4, "nm": "s", "ip": 0, "op": 15, "ks": {}, "shapes": []}]}
    assert any("ip/op" in e for e in g._lottie_lint(shape))


def test_icons_vd_failure_is_an_error_exit():
    """icons --vd exits 1 when a VectorDrawable fails its checks (it used to print the failure and exit 0)."""
    import contextlib
    import io
    import subprocess
    from PIL import ImageDraw
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "svg").mkdir()
        mock = Image.new("RGB", (200, 200), "#fff")
        ImageDraw.Draw(mock).rectangle([20, 20, 44, 44], outline="#123456", width=2)
        mock.save(d / "mockup.png")
        (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": 200, "height": 200, "platform": "android", "screen_dp": 360,
            "assets": [{"name": "home", "kind": "icon", "format": "svg", "bbox": {"x": 20, "y": 20, "w": 24, "h": 24}, "transparent": True,
                        "animated": False, "description": "home", "colors": []}]}))
        (d / "svg" / "home.svg").write_text(f'<svg {NS} viewBox="0 0 24 24" fill="none" stroke="#123456" stroke-width="2"><path d="M3 10l9-7 9 7v10H3z"/></svg>')
        calls = []
        # the model's SVG converts (svg_to_vd accepts it) but the VD fails aapt2/lint: force that by breaking convert_vd's view
        answer = {"files": [{"name": "bell", "svg": f'<svg {NS} viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" fill="none" stroke="#123456" stroke-width="2"/></svg>'}]}
        saved = {k: getattr(g, k) for k in ("ask", "_config", "_vd_lint", "_call")}
        g.ask = lambda *a, **k: answer
        g._config = lambda: {"base": "http://mock.invalid/v1", "key": "sk-test", "image_model": "gpt-image-2", "text_model": "gpt-6", "config_path": "-"}
        g._vd_lint = lambda xml: ["forced lint failure"]
        g._call = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
        old_argv, buf, code = sys.argv, io.StringIO(), 0
        sys.argv = ["gptkit", "icons", td, "bell", "--vd", f"{td}/res/drawable"]
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    g.main()
                except SystemExit as e:
                    code = e.code
        finally:
            sys.argv = old_argv
            for k, v in saved.items():
                setattr(g, k, v)
        out = buf.getvalue()
        assert code == 1 and "forced lint failure" in out and "failed validation" in out, (code, out)
        assert (d / "res" / "drawable" / "ic_bell.xml").exists() and (d / "icons" / "bell.svg").exists()


def test_package_keyword_segment_and_frame_count():
    """`package com.example.class` does not compile; a 1-frame procedural loop is not an animation."""
    import contextlib
    import io
    from PIL import ImageDraw
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "assets").mkdir()
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([8, 8, 56, 56], fill=(255, 0, 0, 255))
        im.save(d / "assets" / "dot.png")
        (d / "mockup.png").write_bytes((d / "assets" / "dot.png").read_bytes())
        (d / "design_spec.json").write_text(json.dumps({"palette": [], "typography": [], "radii": [], "spacing": [], "components": [], "px_to_dp": 1}))
        for argv, needle in ((["tokens", td, "--package", "com.example.class", "--platforms", "compose"], "keyword"),
                             (["tokens", td, "--r-package", "com.fun.app", "--platforms", "compose"], "keyword"),
                             (["frames", td, "dot", "--procedural", "spin", "--count", "1"], "at least 2")):
            old_argv, buf, code = sys.argv, io.StringIO(), None
            sys.argv = ["gptkit", *argv]
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    try:
                        g.main()
                    except SystemExit as e:
                        code = e.code
            finally:
                sys.argv = old_argv
            assert code is not None and needle in (str(code) + buf.getvalue()), (argv, code, buf.getvalue())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
