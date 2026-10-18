"""SVG -> VectorDrawable and 24-grid normalisation checks. Run: python3 tests/test_svg.py (or pytest)."""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger

FIX = HERE / "svg"
FX = FIX / "fx"  # real-world edge cases: converted individually, not part of the round-trip corpus
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
    import shutil
    import tempfile
    names = sorted(p.stem for p in FIX.glob("*.svg"))
    pairs = [((FIX / f"{n}.svg").read_text(), vd(n)) for n in names]
    diffs = K.vd_roundtrip(pairs)
    for n, d in zip(names, diffs):
        if K._chrome_probe()[0]:
            assert d is not None, (n, "Chrome exists but the round-trip was not rendered")
        if d is not None:
            assert d["mean"] < (2 if n != "rects_dash" else 3), (n, d)
            assert d["strong"] <= (16 if n != "rects_dash" else 400), (n, d)
    with tempfile.TemporaryDirectory() as td:
        files = []
        for n, (_, x) in zip(names, pairs):
            files.append(Path(td) / f"ic_{n}.xml")
            files[-1].write_text(x)
            assert K._vd_lint(x) == [], (n, K._vd_lint(x))
        errs = K.aapt2_check(files)
        if shutil.which("aapt2") or K._android_sdk():
            assert errs is not None and K.aapt2_level() in ("compile", "link")
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


RECT = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" fill="#f00"/></svg>'


def _chrome_ok():
    """True when a Chrome binary exists (then the shared flags must make it work; the test fails otherwise)."""
    return K._chrome() is not None


def test_chrome_grid_geometry():
    if not _chrome_ok():
        print("  (skipped: no Chrome binary)")
        return
    assert K._chrome_probe()[0], K._chrome_probe()
    assert K._renderer_name() == "chrome"
    r = K.render_svgs([RECT], 128)[0]
    assert r is not None and r.getchannel("A").getbbox() == (0, 0, 128, 128)
    rows = K.render_svgs([RECT] * 9, 256)
    assert all(im is not None for im in rows)
    assert rows[-1].getchannel("A").getbbox()[3] >= 250
    assert rows[-1].getchannel("A").getbbox()[1] == 0


def test_chrome_failure_message():
    import io
    import contextlib
    import subprocess
    if not _chrome_ok():
        print("  (skipped: no Chrome binary)")
        return
    K._chrome_probe()  # cached before subprocess.run is patched, so the renderer stays "chrome"

    class R:
        returncode, stdout = 1, ""
        stderr = ("[0928/120000.000000:ERROR:zygote_host_impl_linux.cc(105)] Running as root without --no-sandbox "
                  "is not supported. See https://crbug.com/638180.\n")
    saved, K._warned_chrome = subprocess.run, False
    subprocess.run = lambda *a_, **k_: R()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out = K.render_svgs([RECT] * 3, 64)
            out2 = K.render_svgs([RECT], 64)
    finally:
        subprocess.run = saved
    assert out == [None, None, None] and out2 == [None]
    printed = buf.getvalue()
    assert "no screenshot" in printed and "no-sandbox" in printed
    assert printed.count("no screenshot") == 1, printed


def test_sized_svg_quotes():
    sq = "<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'><circle cx='12' cy='12' r='9' fill='#c0c'/></svg>"
    out = K._sized_svg(sq, 64)
    root = ET.fromstring(out)
    assert root.get("width") == "64" and root.get("height") == "64" and out.count("width=") == 1
    bare = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/></svg>'
    assert 'xmlns="http://www.w3.org/2000/svg"' in K._ensure_xmlns(bare)
    assert K._ensure_xmlns(sq) == sq
    assert K.render_svgs(["<svg"], 32) == [None]
    if _chrome_ok():
        ims = K.render_svgs([sq, bare], 64)
        assert all(im is not None and im.getchannel("A").getbbox() for im in ims)


def test_aapt2_level_and_lint():
    import os
    import shutil
    import tempfile
    import io
    import contextlib
    aapt2 = (K._android_sdk() or (shutil.which("aapt2"), None))[0]
    if not aapt2:
        print("  (skipped aapt2 part: no aapt2)")
    else:
        saved = {k: os.environ.get(k) for k in ("ANDROID_HOME", "ANDROID_SDK_ROOT")}
        with tempfile.TemporaryDirectory() as td:
            bt = Path(td) / "sdk" / "build-tools" / "34.0.0"
            bt.mkdir(parents=True)
            os.symlink(aapt2, bt / "aapt2")
            os.environ["ANDROID_HOME"] = str(bt.parent.parent)
            os.environ.pop("ANDROID_SDK_ROOT", None)
            K._android_sdk.cache_clear()
            try:
                assert K._android_sdk() == (str(bt / "aapt2"), None)
                assert K.aapt2_level() == "compile"
                line = K._aapt2_line(1)
                assert "link skipped" in line and "NOT checked" in line and "valid" not in line
                (Path(td) / "a.svg").write_text(RECT)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    failed = K.convert_vd([Path(td) / "a.svg"], Path(td) / "out", check=True)
                assert failed == 0 and "link skipped" in buf.getvalue(), buf.getvalue()
                assert "valid" not in buf.getvalue()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                K._android_sdk.cache_clear()
    good = vd("root_attrs")
    assert K._vd_lint(good) == []
    assert any("evenOdd|nonZero" in e for e in K._vd_lint(good.replace("<path", '<path android:fillType="evenodd"', 1)))
    assert any("unit" in e for e in K._vd_lint(good.replace('android:width="24dp"', 'android:width="24"')))
    assert any("unknown attribute" in e for e in K._vd_lint(good.replace("android:strokeColor", "android:strokeColour")))
    assert any("nan/inf" in e for e in K._vd_lint(good.replace('android:pathData="', 'android:pathData="M nan 0 ')))
    assert any("viewportWidth" in e for e in K._vd_lint(good.replace('android:viewportWidth="24"', 'android:viewportWidth="0"')))
    rad = ('<vector xmlns:android="http://schemas.android.com/apk/res/android" xmlns:aapt="http://schemas.android.com/aapt" '
           'android:width="24dp" android:height="24dp" android:viewportWidth="24" android:viewportHeight="24"><path '
           'android:pathData="M0 0h24v24H0z"><aapt:attr name="android:fillColor"><gradient android:type="radial" '
           'android:centerX="12" android:centerY="12" android:gradientRadius="0"><item android:offset="0" android:color="#fff"/>'
           '</gradient></aapt:attr></path></vector>')
    errs = K._vd_lint(rad)
    assert any("gradientRadius" in e for e in errs) and any("two colours" in e for e in errs), errs
    assert K._vd_lint("<vector") and K._vd_lint('<shape xmlns:android="http://schemas.android.com/apk/res/android"/>')


def test_vd_error_paths():
    import argparse
    import io
    import contextlib
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "empty").mkdir()
        (d / "bad_vb.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24"><circle cx="12" cy="12" r="7"/></svg>')
        (d / "good.svg").write_text(RECT)
        (d / "tr.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g transform="translate()"><circle cx="12" cy="12" r="7"/></g></svg>')
        a = argparse.Namespace(svgs=[str(d / "missing.svg"), str(d / "empty"), str(d / "bad_vb.svg"), str(d / "good.svg"),
                                     str(d / "tr.svg")], out=str(d / "out"), dp=None, prefix="ic_", no_check=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                K.cmd_vd(a)
                code = 0
            except SystemExit as e:
                code = e.code
        out = buf.getvalue()
        assert code == 1, out
        assert (d / "out" / "ic_good.xml").exists() and not (d / "out" / "ic_bad_vb.xml").exists()
        lines = [l for l in out.splitlines() if l.startswith("  ! ")]
        assert len(lines) == 4, out
        assert "missing.svg: cannot read" in out and "no .svg files" in out and "viewBox must be 4 numbers" in out
        assert "bad translate() arguments" in out and "IndexError" not in out and "Traceback" not in out
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                K.cmd_vd(argparse.Namespace(svgs=[str(d / "empty")], out=str(d / "out"), dp=None, prefix="ic_", no_check=True))
            except SystemExit as e:
                assert "no SVG files" in str(e.code)
    try:
        K.svg_to_vd('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 0 0"><rect width="1" height="1"/></svg>')
        assert False, "zero viewBox accepted"
    except ValueError as e:
        assert "viewBox" in str(e)


def fx(name):
    return (FX / f"{name}.svg").read_text()


def test_vd_names():
    import io
    import contextlib
    import re
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "sub").mkdir()
        for n in ("Arrow-Left", "2x"):
            (d / f"{n}.svg").write_text(fx(n))
        (d / "sub" / "arrow_left.svg").write_text(fx("Arrow-Left"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            failed = K.convert_vd([d / "Arrow-Left.svg", d / "2x.svg", d / "sub" / "arrow_left.svg"], d / "out", prefix="ic_", check=False)
        names = sorted(p.name for p in (d / "out").glob("*.xml"))
        assert names == ["ic_2x.xml", "ic_arrow_left.xml", "ic_arrow_left_2.xml"], (names, buf.getvalue())
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", p[:-4]) for p in names)
        assert "renamed Arrow-Left.svg -> ic_arrow_left.xml" in buf.getvalue()
        assert "already used" in buf.getvalue() and failed == 0
        with contextlib.redirect_stdout(io.StringIO()):
            K.convert_vd([d / "2x.svg"], d / "out2", prefix="", check=False)
        assert (d / "out2" / "ic_2x.xml").exists()
    assert K._res_name("ic_" + "Arrow-Left", "ic_") == "ic_arrow_left" and K._res_name("2x", "ic_") == "ic_2x"


def test_default_dp():
    root = ET.fromstring(K.svg_to_vd(fx("material_symbol")))
    assert root.get(A + "width") == "24dp" and root.get(A + "height") == "24dp"
    assert root.get(A + "viewportWidth") == "960" and root.find("group") is not None  # offset viewBox -> translate group
    root = ET.fromstring(K.svg_to_vd(fx("material_symbol").replace('width="24"', 'width="48"')))
    assert root.get(A + "width") == "48dp"
    root = ET.fromstring(K.svg_to_vd(fx("wh_mismatch")))
    assert root.get(A + "width") == "24dp" and root.get(A + "height") == "12dp"  # viewBox aspect kept
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="18pt" viewBox="0 0 24 24"><rect width="24" height="24"/></svg>'
    assert ET.fromstring(K.svg_to_vd(svg)).get(A + "width") == "24dp"
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 512 512"><rect width="24" height="24"/></svg>'
    warns = []
    root = ET.fromstring(K.svg_to_vd(svg, None, warns))
    assert root.get(A + "width") == "512dp" and any("> 200dp" in w and "--dp" in w for w in warns), warns
    root = ET.fromstring(K.svg_to_vd(svg, 24))
    assert root.get(A + "width") == "24dp"
    assert ET.fromstring(K.svg_to_vd('<svg xmlns="http://www.w3.org/2000/svg"><rect width="24" height="24"/></svg>')).get(A + "width") == "24dp"


def test_gradient_defaults_and_one_stop():
    import re
    x = K.svg_to_vd(fx("grad_radial_nor"))
    g = ET.fromstring(x).find(".//gradient")
    assert g.get(A + "gradientRadius") == "12" and g.get(A + "centerX") == "12"
    x = K.svg_to_vd(fx("grad_one_stop"))
    assert 'android:fillColor="#00CC66"' in x and "aapt:attr" not in x
    g = ET.fromstring(K.svg_to_vd(fx("grad_pct"))).find(".//gradient")
    assert (g.get(A + "startX"), g.get(A + "endX"), g.get(A + "endY")) == ("0", "24", "0")
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><defs><radialGradient id="g" gradientUnits="userSpaceOnUse" '
           'cx="12" cy="12" r="{r}"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></radialGradient></defs>'
           '<rect width="24" height="24" fill="url(#g)"/></svg>')
    try:
        K.svg_to_vd(svg.format(r="0"))
        assert False, "radius 0 accepted"
    except ValueError as e:
        assert "radius 0" in str(e)
    g = ET.fromstring(K.svg_to_vd(svg.format(r="50%"))).find(".//gradient")
    assert abs(float(g.get(A + "gradientRadius")) - 12) < 1e-6
    lin = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><defs><linearGradient id="g" gradientUnits="userSpaceOnUse">'
           '{stops}</linearGradient></defs><rect width="24" height="24" fill="url(#g)"/></svg>')
    x = K.svg_to_vd(lin.format(stops='<stop offset="0.8" stop-color="#f00"/><stop offset="0.2" stop-color="#0f0"/>'))
    assert re.findall(r'android:offset="([^"]*)"', x) == ["0.8", "0.8"]  # offsets clamped monotonic
    x = K.svg_to_vd(lin.format(stops=""))
    assert "<path" not in x  # zero stops paints nothing; the rect is dropped
    x = K.svg_to_vd(fx("tabler_dead"))
    assert x.count("<path") == 1 and "M0 0" not in x
    warns = []
    K.svg_to_vd((FIX / "gradient.svg").read_text(), None, warns)
    assert any(w.startswith("needs API 24+") and "gradient" in w for w in warns), warns


ST = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#C9B8FF" stroke-width="1.75" '
      'stroke-linecap="round" stroke-linejoin="round">')
ICONS = {"home": ST + '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/></svg>',
         "chevron": ST + '<path d="M9 5l7 7-7 7"/></svg>',
         "bell": ST + '<path d="M6 16V11a6 6 0 0 1 12 0v5l2 2H4z"/><path d="M10 20a2 2 0 0 0 4 0"/></svg>',
         "heart": ST + '<g transform="translate(0 .5)"><path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/></g></svg>'}


def _proj(td, names, sizes=None, write_svgs=()):
    """A synthetic project dir: dark mockup with one light square per asset, assets.json, optional DIR/svg files."""
    import json
    from PIL import Image, ImageDraw
    d = Path(td)
    (d / "svg").mkdir(exist_ok=True)
    im = Image.new("RGB", (900, 400), "#101830")
    assets = []
    for i, n in enumerate(names):
        sz = (sizes or {}).get(n, 40)
        x, y = 40 + (i % 6) * 140, 60 + (i // 6) * 160
        ImageDraw.Draw(im).rectangle([x, y, x + sz - 1, y + sz - 1], fill="#C9B8FF")
        assets.append({"name": n, "kind": "icon", "format": "svg", "description": f"{n} icon", "colors": ["#C9B8FF"],
                       "bbox": {"x": x, "y": y, "w": sz, "h": sz}, "dp": {"w": sz / 2, "h": sz / 2}})
    im.save(d / "mockup.png")
    (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": 900, "height": 400, "screen_dp": 450,
                                               "assets": assets}))
    for n in write_svgs:
        (d / "svg" / f"{n}.svg").write_text(ICONS.get(n, ICONS["home"]))
    return d


def _run(fn, *a_, **k_):
    """(exit code or 0, printed text) of a command function."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*a_, **k_)
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if not isinstance(e.code, int):
                buf.write(f"{e.code}\n")  # what sys.exit("...") prints to stderr
    return code, buf.getvalue()


def test_icon_set_report_bad_path():
    rows, notes = K.icon_set_report([{"name": "home", "svg": ICONS["home"], "size": 40},
                                     {"name": "broken", "svg": ST + '<path d="M1 X"/></svg>', "size": 40},
                                     {"name": "cut", "svg": ST + '<path d="M1 1"/>', "size": 40}])
    assert [r["name"] for r in rows] == ["home"]
    assert any(n.startswith("broken: unreadable path data") for n in notes), notes
    assert any(n.startswith("cut: invalid XML") for n in notes), notes


def test_icons_malformed_and_missing():
    import argparse
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = _proj(td, ["home", "chevron"], write_svgs=["home", "chevron"])
        (d / "svg" / "chevron.svg").write_text(ST + '<path d="M9 5 X"/></svg>')  # unreadable existing icon
        calls = []

        def ask(parts, schema=None, name="result", model=None):
            calls.append(schema["properties"]["files"]["items"]["properties"]["name"].get("enum"))
            return {"files": [{"name": "bell", "svg": '<svg viewBox="0 0 24 24"><path d="M1 1L5 5"/>'},
                              {"name": "extra", "svg": ICONS["home"]}]}
        saved, K.ask = K.ask, ask
        try:
            args = argparse.Namespace(dir=td, concepts="bell, sleep timer", out=None, treatment="auto", stroke=None, color=None,
                                      snap=False, vd=None, dp=None)
            code, out = _run(K.cmd_icons, args)
        finally:
            K.ask = saved
        assert code == 1 and "Traceback" not in out, out
        assert calls == [["bell", "sleep_timer"], ["bell", "sleep_timer"]], calls
        assert "chevron.svg: unreadable" in out
        assert not (d / "icons" / "bell.svg").exists() and (d / "icons" / "raw" / "bell_try1.svg").exists()
        assert "unrequested icon 'extra'" in out and not (d / "icons" / "extra.svg").exists()
        assert "missing: bell, sleep_timer" in out and "invalid XML" in out, out

        def ask2(parts, schema=None, name="result", model=None):
            return {"files": [{"name": "Sleep Timer", "svg": ST.replace(' xmlns="http://www.w3.org/2000/svg"', "") +
                               '<circle cx="12" cy="13" r="8"/><path d="M12 13V8"/></svg>'},
                              {"name": "bell", "svg": ST + '<path d="M6 16V11" stroke-dasharray="2 1"/></svg>'}]}
        K.ask = ask2
        try:
            args.concepts = "bell, sleep timer"
            code, out = _run(K.cmd_icons, args)
        finally:
            K.ask = saved
        assert code == 0, out
        assert 'xmlns="http://www.w3.org/2000/svg"' in (d / "icons" / "sleep_timer.svg").read_text()
        assert "(kept out of --vd)" not in out  # dasharray is a warning, not a conversion failure
        code, out = _run(K.cmd_icons, argparse.Namespace(dir=td, concepts=" , ", out=None, treatment="auto", stroke=None,
                                                         color=None, snap=False, vd=None, dp=None))
        assert code == 1 and "no concepts" in out or code == 1


def test_svg_missing_and_failed_batch():
    import argparse
    import json
    import tempfile
    names = ["home", "chevron", "bell", "heart", "play", "stop", "zeta", "menu"]
    with tempfile.TemporaryDirectory() as td:
        d = _proj(td, names)
        seen = {}

        def ask(parts, schema=None, name="result", model=None):
            import re
            asked = re.findall(r'Asset "([^"]+)"', "\n".join(p for p in parts if isinstance(p, str)))
            enum = schema["properties"]["files"]["items"]["properties"]["name"]["enum"]
            assert sorted(enum) == sorted(asked)
            for n in asked:
                seen[n] = seen.get(n, 0) + 1
            if "zeta" in asked and seen["zeta"] == 1:
                raise K.ApiError(400, "HTTP 400 /responses [moderation_blocked]: blocked")
            files = [{"name": n, "svg": ICONS.get(n, ICONS["home"])} for n in asked if n != "bell"]
            if "home" in asked:
                files.append({"name": "extra_icon", "svg": ICONS["home"]})
            return {"files": files}
        saved, K.ask = K.ask, ask
        try:
            code, out = _run(K.cmd_svg, argparse.Namespace(dir=td, only=None, retry=False, grid=24, pad=2, check=False))
        finally:
            K.ask = saved
        assert code == 1 and "Traceback" not in out, out
        written = sorted(p.stem for p in (d / "svg").glob("*.svg"))
        assert written == sorted(n for n in names if n != "bell"), (written, out)
        assert "svg request failed for zeta, menu" in out and "moderation_blocked" in out
        assert "bell: model returned no SVG" in out and "unrequested file 'extra_icon'" in out
        assert seen["bell"] == 2 and seen["zeta"] == 2
        rep = json.loads((d / "svg_report.json").read_text())
        assert rep["bell"] == {"error": "missing"} and rep["home"]["kept_try"] == 1
        code, out = _run(K.cmd_svg, argparse.Namespace(dir=td, only="nope", retry=False, grid=24, pad=2, check=True))
        assert code == 1 and "unknown asset(s): nope" in out
        (d / "assets.json").write_text(json.dumps({**json.loads((d / "assets.json").read_text()), "assets": []}))
        code, out = _run(K.cmd_svg, argparse.Namespace(dir=td, only=None, retry=False, grid=24, pad=2, check=True))
        assert code == 1 and "no svg-format assets" in out
    with tempfile.TemporaryDirectory() as td:
        d = _proj(td, ["home"])
        code, out = _run(K.cmd_svg, argparse.Namespace(dir=td, only=None, retry=False, grid=24, pad=2, check=True))
        assert code == 1 and "no svg files to check" in out and "run svg first" in out
        (d / "svg" / "home.svg").write_text(ST + '<path d="M2.9 X 10.483 L12"/></svg>')  # the sed-broken file
        code, out = _run(K.cmd_svg, argparse.Namespace(dir=td, only=None, retry=False, grid=24, pad=2, check=True))
        assert code == 0 and "Traceback" not in out and "unreadable path data" in out, out


def test_snap_style():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000">'
           '<circle cx="12" cy="12" r="8" style="fill:none;stroke:#000;stroke-width:2.694"/>'
           '<path d="M12 4v8" stroke-width="3"/><path d="M4 12h8"/></svg>')
    out = K._snap_stroke(svg, 1.75)
    st = K.svg_style(out)
    assert st["strokes"] == [1.75, 1.75, 1.75], st["strokes"]
    assert "stroke-width:1.75" in out and ET.fromstring(out).get("stroke-width") == "1.75"


def test_group_normalisation_keeps_stroke():
    if K._renderer_name() is None:
        print("  (skipped: no SVG renderer)")
        return
    res = K._grid_normalize([ICONS["chevron"], ICONS["home"]], 24, 2, sizes=[32, 40], dps=[16, 20], with_dp=True)
    (chev, dp1), (home, dp2) = res
    assert chev and home
    s1, s2 = K._stroke24(K.svg_style(chev)), K._stroke24(K.svg_style(home))
    assert abs(s1 - s2) <= 0.02, (s1, s2)
    assert dp1 == dp2 == 24.0
    box = K.content_box(K.render_svgs([home], 240)[0], 25)
    assert 15 <= box[0] <= 25 and 215 <= box[2] <= 225, box  # the largest icon fills 2..22 of 24
    boxc = K.content_box(K.render_svgs([chev], 240)[0], 25)
    assert (boxc[3] - boxc[1]) < (box[3] - box[1]) * 0.9  # the smaller icon stays smaller
    per_icon = K._grid_normalize([ICONS["chevron"], ICONS["home"]], 24, 2)
    assert all(per_icon) and abs(K._stroke24(K.svg_style(per_icon[0])) - s1) > 0.1  # per-icon fit would inflate the chevron


def test_transform_baked_or_reported():
    baked = K._bake_transforms(ICONS["heart"])
    assert baked and "transform" not in baked and "A" not in ET.fromstring(baked).find(f".//{K.SVG_NS}path").get("d")
    out = K.normalize_svg(ICONS["heart"], (5, 5.5, 19, 20.5), 24, 2)
    assert out and "transform" not in out and ET.fromstring(out).get("viewBox") == "0 0 24 24"
    rot = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect x="8" y="8" width="8" height="8" '
           'transform="rotate(45 12 12)" fill="none" stroke="#000" stroke-width="2"/></svg>')
    b = K._bake_transforms(rot)
    assert b and "rotate" not in b and 'stroke-width="2"' in b
    if K._renderer_name():
        d = K.vd_roundtrip([(rot, K.svg_to_vd(b))])[0]
        assert d and d["mean"] < 2 and d["strong"] <= 16, d
    sc = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g transform="scale(2)"><path d="M1 6h10" '
          'stroke="#000" style="stroke-width:1;stroke-dasharray:1 1"/></g></svg>')
    b = K._bake_transforms(sc)
    assert b and "stroke-width:2" in b and "stroke-dasharray:2 2" in b and "M2,12 L22,12" in b, b
    skew = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g transform="scale(2 1)"><path d="M2 8h8" '
            'stroke="#000"/></g></svg>')
    assert K._bake_transforms(skew) is None and K.normalize_svg(skew, (2, 8, 22, 8), 24, 2) is None
    import argparse
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = _proj(td, ["home"], write_svgs=["home"])
        odd = ST.replace('viewBox="0 0 24 24"', 'viewBox="0 0 48 48"') + '<g transform="scale(2 1)"><path d="M2 8h8"/></g></svg>'
        saved, K.ask = K.ask, lambda *a_, **k_: {"files": [{"name": "bell", "svg": odd}]}
        try:
            code, out = _run(K.cmd_icons, argparse.Namespace(dir=td, concepts="bell", out=None, treatment="auto", stroke=None,
                                                             color=None, snap=False, vd=None, dp=None))
        finally:
            K.ask = saved
        assert code == 0 and "bell: kept as drawn" in out and "not on the 24 grid" in out, out


def test_converter_real_world_fixtures():
    import shutil
    import tempfile
    clean = ["clip_figma", "clip_real", "css_class", "use_symbol", "named_hsl", "grad_url_quoted", "inherit_fill", "doctype",
             "grad_obb", "grad_transform_figma", "skew", "visibility_child", "single_quote", "no_xmlns", "bom", "inkscape"]
    lossy = {"nested_svg": "<svg> ignored", "marker": "marker", "paint_order": "paint-order", "nonscaling": "vector-effect",
             "grad_focal": "focal", "nonuniform_stroke": "non-uniform"}
    xmls, warned = {}, {}
    for n in clean + list(lossy):
        warns = []
        xmls[n] = K.svg_to_vd((FX / f"{n}.svg").read_text(encoding="utf-8-sig"), None, warns)
        warned[n] = [w for w in warns if not w.startswith("needs API")]
        assert K._vd_lint(xmls[n]) == [], (n, K._vd_lint(xmls[n]))
    for n, word in lossy.items():
        assert any(word in w for w in warned[n]), (n, warned[n])
    for n in clean:
        assert not warned[n] or n in ("skew", "grad_obb", "clip_real"), (n, warned[n])
    assert "clip-path" in xmls["clip_real"] and "clip-path" not in xmls["clip_figma"]  # full-frame Figma clip dropped
    assert 'android:fillColor="#FFD700"' in xmls["named_hsl"] and 'android:fillColor="#1AA2E6"' in xmls["named_hsl"]
    assert 'android:fillColor="#112233"' in xmls["inherit_fill"] and 'android:fillColor="#EE3333"' in xmls["css_class"]
    assert xmls["use_symbol"].count("<path") == 2 and "#00AA00" in xmls["use_symbol"] and "#AA0000" in xmls["use_symbol"]
    assert 'android:fillAlpha="0.5"' in K.svg_to_vd((FX / "opacity_pct.svg").read_text())
    assert K._color("hsla(0, 100%, 50%, 0.5)") == "#80FF0000" and K._color("azure") == "#F0FFFF"
    assert K._color("#fff !important") == "#FFFFFF" and K._color("rebeccapurple") == "#663399"
    g = ET.fromstring(xmls["grad_transform_figma"]).find(".//gradient")
    assert abs(float(g.get(A + "centerX")) - 12) < 1e-6 and abs(float(g.get(A + "gradientRadius")) - 10) < 1e-6
    g = ET.fromstring(xmls["grad_obb"]).find(".//gradient")
    assert (g.get(A + "startX"), g.get(A + "endX")) == ("2", "22")
    if K._renderer_name():
        xmls["opacity_pct"] = K.svg_to_vd((FX / "opacity_pct.svg").read_text())
        pairs = [((FX / f"{n}.svg").read_text(encoding="utf-8-sig"), xmls[n]) for n in clean + ["opacity_pct"]]
        for n, rt in zip(clean + ["opacity_pct"], K.vd_roundtrip(pairs)):
            assert rt is not None and rt["mean"] < (1 if n == "opacity_pct" else 0.5) and rt["strong"] <= 16, (n, rt)
    if shutil.which("aapt2") or K._android_sdk():
        with tempfile.TemporaryDirectory() as td:
            files = []
            for n, x in xmls.items():
                files.append(Path(td) / f"ic_{n.lower()}.xml")
                files[-1].write_text(x)
            errs = K.aapt2_check(files)
            assert errs == {}, errs
    for bad in ("mask", "text"):
        try:
            K.svg_to_vd(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><{bad} id="m"/><g {bad}="url(#m)"/></svg>')
            assert False, bad
        except ValueError as e:
            assert bad in str(e)


def test_roundtrip_strong_metric():
    if K._renderer_name() is None:
        print("  (skipped: no SVG renderer)")
        return
    S = 'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"'
    bad = {  # conversions that lose content but stay under mean 2 (from the audit's metric_cases.py)
        "dropped_dot_r1": f'<svg {S}><path d="M4 4h16v12H4z" fill="none" stroke="#000" stroke-width="2"/><svg x="11" y="18" width="2" height="2" viewBox="0 0 2 2"><circle cx="1" cy="1" r="1" fill="#000"/></svg></svg>',
        "focal_small": f'<svg {S}><defs><radialGradient id="g" gradientUnits="userSpaceOnUse" cx="12" cy="12" r="4" fx="10" fy="10"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#036"/></radialGradient></defs><circle cx="12" cy="12" r="4" fill="url(#g)"/></svg>',
    }
    good = {
        "visibility_small": f'<svg {S}><g visibility="hidden"><circle cx="12" cy="20" r="1.5" fill="#000" visibility="visible"/></g><path d="M4 4h16v12H4z" fill="#000"/></svg>',
        "group_opacity": (FIX / "group_opacity.svg").read_text(),
    }
    names = list(bad) + list(good)
    rts = K.vd_roundtrip([(s, K.svg_to_vd(s)) for s in list(bad.values()) + list(good.values())])
    for n, rt in zip(names, rts):
        assert rt is not None and rt["mean"] < 2, (n, rt)
        flagged = rt["strong"] > 16
        assert flagged == (n in bad), (n, rt)


def test_svg_rules_mention_forbidden_features():
    rules = K.SVG_RULES
    for word in ("xmlns", "transform attribute on ANY element", "gradientTransform", "stroke-dasharray", "%", "units",
                 "marker", "vector-effect", "paint-order", "fx/fy", "cx cy r", "two stops", "userSpaceOnUse", "clipPath",
                 "use", "style", "nested svg"):
        assert word in rules, word
    assert "VectorDrawable rule" in K.SVG_FIX_PROMPT
    sch = K._svg_schema(["a", "b"])
    assert sch["properties"]["files"]["items"]["properties"]["name"]["enum"] == ["a", "b"]
    assert "enum" not in K.SVG_SCHEMA["properties"]["files"]["items"]["properties"]["name"]


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
