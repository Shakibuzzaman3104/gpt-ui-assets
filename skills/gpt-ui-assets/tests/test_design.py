"""Offline checks for mockup / critique / manifest / spec / iconspec (API mocked). Run: python3 tests/test_design.py (or pytest)."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402

W, H = 400, 600


def _mock_png(d, w=W, h=H, name="mockup.png"):
    im = Image.new("RGBA", (w, h), (250, 250, 245, 255))
    dr = ImageDraw.Draw(im)
    dr.rectangle((20, 40, 380, 120), fill=(30, 30, 30, 255))
    dr.ellipse((300, 480, 360, 540), fill=(0, 120, 90, 255))
    p = Path(d) / name
    im.save(p)
    return p


@contextlib.contextmanager
def patched(**kw):
    saved = {k: getattr(K, k) for k in kw}
    for k, v in kw.items():
        setattr(K, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(K, k, v)


def run(fn, a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(a)
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            print(e.code)
    return code, buf.getvalue()


def _asset(name, bbox, **kw):
    x = {"name": name, "kind": "icon", "bbox": bbox, "format": "svg", "transparent": True, "animated": False,
         "description": "d", "colors": [], "material_symbol": "", "sf_symbol": ""}
    x.update(kw)
    return x


def _manifest(td, assets, **kw):
    mock = _mock_png(td, name="src.png")
    a = argparse.Namespace(mockup=str(mock), out=str(Path(td) / "build"), platform="android", screen_dp=None, model=None)
    vars(a).update(kw)
    with patched(ask=lambda *a_, **k_: {"assets": assets}):
        code, out = run(K.cmd_manifest, a)
    m = json.loads((Path(a.out) / "assets.json").read_text()) if (Path(a.out) / "assets.json").exists() else None
    return code, out, m


def test_manifest_bbox_and_fields():
    fmtless = _asset("nofmt", {"x": 10, "y": 10, "w": 20, "h": 20})
    del fmtless["format"]
    assets = [_asset("good", {"x": 10, "y": 200, "w": 40, "h": 40}),
              fmtless,
              _asset("far", {"x": 5000, "y": 5000, "w": 9999, "h": 9999}),
              _asset("mostly_out", {"x": 390, "y": 100, "w": 100, "h": 100}),
              _asset("inverted", {"x": 60, "y": 300, "w": -20, "h": -30}),
              _asset("edge", {"x": 370, "y": 580, "w": 40, "h": 30}, format="png")]
    with tempfile.TemporaryDirectory() as td:
        code, out, m = _manifest(td, assets)
        assert code == 0, out
        names = [x["name"] for x in m["assets"]]
        assert names == ["good", "inverted", "edge"], names
        assert "dropped an asset missing ['format']" in out and "'nofmt'" in out, out
        assert "dropped 'far': bbox mostly outside the image" in out and "dropped 'mostly_out'" in out, out
        for x in m["assets"]:
            b = x["bbox"]
            assert all(isinstance(b[k], int) for k in "xywh"), b
            assert 0 <= b["x"] < W and 0 <= b["y"] < H and b["w"] >= 1 and b["h"] >= 1, b
            assert b["x"] + b["w"] <= W and b["y"] + b["h"] <= H, b
        assert m["assets"][1]["bbox"] == {"x": 40, "y": 270, "w": 20, "h": 30}
        assert m["assets"][2]["bbox"] == {"x": 370, "y": 580, "w": 30, "h": 20} and "'edge': bbox clamped" in out
        assert m["assets"][0]["dp"] == {"w": round(40 * 412 / W, 1), "h": round(40 * 412 / W, 1)}
        assert (Path(td) / "build" / "manifest_overlay.png").exists()


def test_manifest_non_conforming_response():
    with tempfile.TemporaryDirectory() as td:
        mock = _mock_png(td)
        a = argparse.Namespace(mockup=str(mock), out=str(Path(td) / "b"), platform=None, screen_dp=None, model=None)
        with patched(ask=lambda *a_, **k_: {"items": []}):
            code, out = run(K.cmd_manifest, a)
        assert code == 1 and "no 'assets' list" in out, out
        assert not (Path(td) / "b" / "assets.json").exists()
        code, out = run(K.cmd_manifest, argparse.Namespace(mockup=str(Path(td) / "nope.png"), out=str(Path(td) / "c"),
                                                           platform=None, screen_dp=None, model=None))
        assert code == 1 and "mockup not found" in out and "nope.png" in out, out


def test_manifest_dedup():
    b = {"x": 10, "y": 200, "w": 30, "h": 30}
    with tempfile.TemporaryDirectory() as td:
        code, out, m = _manifest(td, [_asset("Flame", b), _asset("flame_icon", b), _asset("flame.svg", b),
                                      _asset("Wi-Fi", {"x": 350, "y": 2, "w": 20, "h": 20}), _asset("--", b)])
        assert code == 0, out
        assert [x["name"] for x in m["assets"]] == ["flame", "flame_2", "flame_3", "asset"], m["assets"]
        assert "dropped system UI the OS draws: wi_fi" in out, out


def _critique_args(td, mock, **kw):
    a = argparse.Namespace(mockup=str(mock), out=str(Path(td) / "qa"), style_ref=None, platform="android", screen_dp=None,
                           note=None, apply=False, min_severity="medium", rounds=1, model=None)
    vars(a).update(kw)
    return a


def test_critique_overlay_hostile_bboxes():
    issues = [{"severity": "high", "category": "text", "bbox": {"x": 20, "y": 40, "w": 360, "h": 80},
               "problem": "garbled", "edit": "fix it"},
              {"severity": "critical", "category": "overlap", "bbox": {"x": 10, "y": 10, "w": -5, "h": -5},
               "problem": "inverted", "edit": "e"},
              {"severity": "low", "category": "alignment", "bbox": {"x": 5000, "y": 5000, "w": 10, "h": 10},
               "problem": "way outside", "edit": "e"},
              {"severity": "medium", "category": "contrast", "bbox": {"x": 280, "y": 460, "w": 500, "h": 500},
               "problem": "low contrast", "edit": "e"}]
    K._SEV_WARNED.clear()  # the unknown-severity warning is once per process
    with tempfile.TemporaryDirectory() as td:
        mock = _mock_png(td)
        with patched(ask=lambda *a_, **k_: {"issues": [dict(x) for x in issues]}):
            code, out = run(K.cmd_critique, _critique_args(td, mock))
        assert code == 0, out
        assert (Path(td) / "qa" / "mockup_critique.png").exists()
        log = json.loads((Path(td) / "qa" / "mockup_critique.json").read_text())
        got = log["rounds"][0]["issues"]
        assert [x["problem"] for x in got] == ["garbled", "inverted", "low contrast"], got
        assert got[1]["bbox"] == {"x": 5, "y": 5, "w": 5, "h": 5}
        assert got[2]["bbox"] == {"x": 280, "y": 460, "w": 120, "h": 140} and "measured_contrast" in got[2]
        assert "dropped an issue whose bbox" in out and "way outside" in out, out
        assert "unknown issue severity 'critical'" in out, out
        assert log["rounds"][0]["weight"] == 3 + 2 + 2
        # the overlay helper itself tolerates an inverted box and an unknown severity (older logs)
        K._issue_overlay(Image.open(mock), [{"id": 1, "severity": "urgent", "bbox": {"x": 50, "y": 50, "w": -20, "h": -20}}],
                         Path(td) / "ov.png")
        assert (Path(td) / "ov.png").exists()


def test_critique_apply_round_hostile_verify():
    issues = [{"severity": "high", "category": "text", "bbox": {"x": 20, "y": 40, "w": 360, "h": 80},
               "problem": "garbled", "edit": "fix it"}]
    calls = []

    def ask(parts, schema=None, name="", model=None):
        calls.append(name)
        if name == "design_qa":
            return {"issues": [dict(x) for x in issues]}
        return {"checks": [{"id": 1, "fixed": True}, {"id": [9], "fixed": True}],
                "new_issues": [{"severity": "bogus", "category": "text", "bbox": {"x": -50, "y": -50, "w": 10, "h": 10},
                                "problem": "outside", "edit": "e"}]}

    with tempfile.TemporaryDirectory() as td:
        mock = _mock_png(td)
        fake_edit = lambda *a_, **k_: (K._png_bytes(Image.new("RGB", (W, H), "white")), {"quality": "high", "size": "x"})
        with patched(ask=ask, edit=fake_edit):
            code, out = run(K.cmd_critique, _critique_args(td, mock, apply=True))
        assert code == 0, out
        assert calls == ["design_qa", "design_qa_verify"]
        assert "round 1: fixed 1/1, new 0, weight 3 -> 0" in out and "KEPT" in out, out


def test_critique_missing_inputs():
    with tempfile.TemporaryDirectory() as td:
        calls = []
        with patched(ask=lambda *a_, **k_: calls.append(1)):
            code, out = run(K.cmd_critique, _critique_args(td, Path(td) / "missing.png"))
            assert code == 1 and "mockup not found" in out, out
            bad = Path(td) / "bad.png"
            bad.write_bytes(b"not a png")
            code, out = run(K.cmd_critique, _critique_args(td, bad))
            assert code == 1 and "cannot read mockup" in out, out
        assert calls == []


def _spec_dir(td, **m_extra):
    d = Path(td)
    _mock_png(d)
    m = {"mockup": "mockup.png", "width": W, "height": H, "platform": "android", "screen_dp": 412,
         "assets": [{"name": "home", "kind": "icon", "format": "svg", "bbox": {"x": 10, "y": 10, "w": 24, "h": 24},
                     "description": "house", "transparent": True}]}
    m.update(m_extra)
    (d / "assets.json").write_text(json.dumps(m))
    return d


def test_spec_clips_component_bboxes():
    comps = [{"name": "Nav", "native": "NavigationBar (4 destinations)", "notes": "",
              "bbox": {"x": -10, "y": 540, "w": 500, "h": 60}},
             {"name": "Ghost", "native": "Card (filled)", "notes": "", "bbox": {"x": 900, "y": 900, "w": 10, "h": 10}},
             {"name": "Neg", "native": "Card (filled)", "notes": "", "bbox": {"x": 100, "y": 300, "w": -50, "h": -40}}]
    res = {"palette": [], "typography": [], "radii": [], "spacing": [], "components": comps}
    with tempfile.TemporaryDirectory() as td:
        d = _spec_dir(td)
        seen = {}

        def ask(parts, schema=None, name="", model=None):
            seen["model"] = model
            return json.loads(json.dumps(res))
        with patched(ask=ask):
            code, out = run(K.cmd_spec, argparse.Namespace(dir=str(d), scale="auto", recalibrate=False, model="m-x"))
        assert code == 0, out
        assert seen["model"] == "m-x"
        spec = json.loads((d / "design_spec.json").read_text())
        assert [c["name"] for c in spec["components"]] == ["Nav", "Neg"], spec["components"]
        assert spec["components"][0]["bbox"] == {"x": 0, "y": 540, "w": 400, "h": 60}
        assert spec["components"][1]["bbox"] == {"x": 50, "y": 260, "w": 50, "h": 40}
        assert "dropped component 'Ghost'" in out, out
        assert spec["native_scale"]["anchors"][0]["px"] == 60


def test_calibrate_tolerates_missing_keys():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        m = {"mockup": "mockup.png", "width": 1024, "height": 1536, "platform": "android"}  # no assets, no screen_dp
        spec = {"components": [{"name": "Nav", "native": "NavigationBar", "bbox": {"x": 0, "y": 1400, "w": 1024, "h": 136}},
                               {"name": "broken", "native": "FloatingActionButton", "bbox": None},
                               "not a component"]}
        with contextlib.redirect_stdout(io.StringIO()):
            K._calibrate(d, spec, m, 1024)
        assert spec["native_scale"]["applied"] and abs(spec["px_to_dp"] - round(80 / 136, 5)) < 1e-9
        assert spec["px_to_dp_design"] == round(412 / 1024, 5)
        # recalibrate on a dir with no design_spec.json: one-line exit naming the fix
        code, out = run(lambda a: K.cmd_recalibrate(a, "auto"), d)
        assert code == 1 and "design spec not found" in out and "run `spec DIR` first" in out, out


def _main(argv):
    """Run gptkit's CLI in-process with a POST counter in place of the network; returns (exit code, output, posts)."""
    posts = []

    def post(*a_, **k_):
        posts.append(a_[0] if a_ else k_.get("url"))
        raise AssertionError("no API call expected")
    buf = io.StringIO()
    saved = sys.argv, K.requests.post
    sys.argv, K.requests.post = ["gptkit.py", *argv], post
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                K.main()
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
                print(e.code)
    finally:
        sys.argv, K.requests.post = saved
    return code, buf.getvalue(), posts


def _png(size, color="white"):
    return K._png_bytes(Image.new("RGB", size, color))


def _mockup_args(td, **kw):
    a = argparse.Namespace(prompt="habit tracker", out=str(Path(td) / "out"), name="mockup", size="1024x1536", variants=1,
                           edit=None, raw=False, style_ref=None, platform="android", model=None)
    vars(a).update(kw)
    return a


class _FakeApiError(K.ApiError):
    """An ApiError that does not depend on the constructor signature of the API layer."""
    def __init__(self, msg):
        RuntimeError.__init__(self, msg)
        self.status, self.code, self.param, self.body, self.prompt, self.etype = 400, "moderation_blocked", None, None, "", ""

    def __str__(self):
        return self.args[0]


def test_mockup_validation():
    with tempfile.TemporaryDirectory() as td:
        for size in ("auto", "1024X1536", "0x100", ""):
            code, out, posts = _main(["mockup", "x", "--out", td, "--size", size])
            assert code == 1 and "--size must be WIDTHxHEIGHT" in out and posts == [], (size, out)
        code, out, posts = _main(["mockup", "x", "--out", td, "--edit", str(Path(td) / "missing.png")])
        assert code == 1 and "--edit mockup not found" in out and "missing.png" in out and posts == [], out
        code, out, posts = _main(["mockup", "x", "--out", td, "--style-ref", str(Path(td) / "gone.png")])
        assert code == 1 and "--style-ref image not found" in out and posts == [], out
        bad = Path(td) / "bad.png"
        bad.write_bytes(b"GIF89a nope")
        code, out, posts = _main(["mockup", "x", "--out", td, "--edit", str(bad)])
        assert code == 1 and "cannot read --edit mockup" in out and posts == [], out
        for v in ("0", "-2", "two"):
            code, out, posts = _main(["mockup", "x", "--out", td, "--variants", v])
            assert code == 2 and "--variants" in out and posts == [], (v, out)
        assert not (Path(td) / "design.json").exists()


def test_prep_ref():
    with tempfile.TemporaryDirectory() as td:
        big = Path(td) / "big.png"
        Image.new("RGB", (4000, 6000), "teal").save(big)
        assert Image.open(io.BytesIO(K._prep_ref(big))).size == (1024, 1536)
        small = Path(td) / "small.jpg"
        Image.new("CMYK", (300, 200)).save(small)
        im = Image.open(io.BytesIO(K._prep_ref(small)))
        assert im.size == (300, 200) and im.format == "PNG" and im.mode == "RGBA"
        sent = []

        def fake_edit(ref, prompt, size="1024x1024", **k_):
            sent.append(Image.open(io.BytesIO(ref)).size)
            return _png((1024, 1536)), {"quality": "high", "size": size}
        with patched(edit=fake_edit):
            code, out = run(K.cmd_mockup, _mockup_args(td, edit=str(big)))
            assert code == 0 and sent == [(1024, 1536)], (out, sent)
            code, out = run(K.cmd_mockup, _mockup_args(td, style_ref=str(big), name="p"))
            assert code == 0 and sent[1:] == [(1024, 1536)], (out, sent)


def test_mockup_wrong_aspect_warns():
    calls = []

    def fake_generate(prompt, size="1024x1024", **k_):
        calls.append(size)
        return _png((1536, 1024)), {"quality": "high", "size": "1536x1024"}
    with tempfile.TemporaryDirectory() as td, patched(generate=fake_generate):
        code, out = run(K.cmd_mockup, _mockup_args(td))
        assert code == 0 and len(calls) == 2, (out, calls)
        assert "! variant 1: backend returned 1536x1024 instead of the requested 1024x1536 aspect after 2 attempts" in out, out
        assert (Path(td) / "out" / "mockup_v1.png").exists()
        assert json.loads((Path(td) / "out" / "design.json").read_text()) == {"platform": "android", "screen_dp": 412}
        calls.clear()
        with patched(generate=lambda p, size="1024x1024", **k_: (calls.append(size), (_png((1024, 1536)), {"quality": "high", "size": size}))[1]):
            code, out = run(K.cmd_mockup, _mockup_args(td))
        assert code == 0 and len(calls) == 1 and "instead of the requested" not in out, out


def test_mockup_variant_failures():
    n = []

    def flaky(prompt, size="1024x1024", **k_):
        n.append(1)
        if len(n) == 1:
            raise _FakeApiError("HTTP 400 /images/generations [moderation_blocked]: blocked")
        return _png((1024, 1536)), {"quality": "high", "size": size}
    with tempfile.TemporaryDirectory() as td:
        with patched(generate=flaky):
            code, out = run(K.cmd_mockup, _mockup_args(td, variants=2))
        assert code == 0 and "failed: HTTP 400 /images/generations [moderation_blocked]" in out, out
        assert len(list((Path(td) / "out").glob("mockup_v*.png"))) == 1

        def always(*a_, **k_):
            raise _FakeApiError("HTTP 400 [moderation_blocked]: no")
        with patched(generate=always):
            code, out = run(K.cmd_mockup, _mockup_args(td, variants=2, name="z"))
        assert code == 1 and "mockup produced no image" in out, out


def _icon_spec(icons):
    return {"style": "outlined", "stroke_px": 3, "stroke_24": 1.5, "linecap": "round", "linejoin": "round",
            "corner_radius_24": 2, "traits": "rounded", "states": [{"state": "active", "treatment": "filled", "color": "#000"}],
            "icons": icons}


def test_iconspec_missing_names():
    def asset(n, fmt="svg"):
        return {"name": n, "kind": "icon", "format": fmt, "bbox": {"x": 10, "y": 10, "w": 30, "h": 30},
                "description": n, "transparent": True, "material_symbol": "", "sf_symbol": ""}
    with tempfile.TemporaryDirectory() as td:
        d = _spec_dir(td, assets=[asset("home"), asset("search"), asset("profile"), asset("hero", "png")])
        calls = []

        def ask(parts, schema=None, name="", model=None):
            calls.append((name, model, sum(isinstance(x, Image.Image) for x in parts)))
            return _icon_spec([{"name": "Home", "state": "tab active", "treatment": "filled", "material_symbol": "home",
                                "sf_symbol": "house.fill"},
                               {"name": "bell", "state": "x", "treatment": "outlined", "material_symbol": "<b>",
                                "sf_symbol": "bell"}])
        with patched(ask=ask):
            code, out = run(K.cmd_iconspec, argparse.Namespace(dir=str(d), model="m-y"))
        assert code == 0, out
        assert calls == [("icon_spec", "m-y", 4)], calls  # the mockup + 3 svg crops
        assert "! the model didn't describe: profile, search" in out, out
        assert "not in assets.json: bell" in out, out
        m = json.loads((d / "assets.json").read_text())
        assert [(x["material_symbol"], x["sf_symbol"]) for x in m["assets"]][:2] == [("home", "house.fill"), ("", "")]
        spec = json.loads((d / "icon_spec.json").read_text())
        assert spec["icons"][1]["material_symbol"] == "" and spec["icons"][1]["sf_symbol"] == "bell"

    with tempfile.TemporaryDirectory() as td:
        d = _spec_dir(td, assets=[asset("hero", "png")])
        calls = []
        with patched(ask=lambda *a_, **k_: calls.append(1)):
            code, out = run(K.cmd_iconspec, argparse.Namespace(dir=str(d), model=None))
        assert code == 1 and 'no format "svg" assets' in out and "run `manifest` first" in out and calls == [], out
        assert not (d / "icon_spec.json").exists()


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
