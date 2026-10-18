"""Offline checks for `gptkit.py appicon` (no API calls; the API is mocked in-process).
Run: python3 tests/test_appicon.py (or pytest / python3 -m unittest discover skills/gpt-ui-assets/tests)"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

K = Path(__file__).resolve().parents[1] / "scripts" / "gptkit.py"
sys.path.insert(0, str(K.parent))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger
import gptkit as g  # noqa: E402

KEYED = {"base": "http://mock.invalid/v1", "key": "sk-test", "image_model": "gpt-image-2", "image_model_pinned": True,
         "text_model": "gpt-6", "retries": 0, "debug": False, "moderation": None, "config_path": "-"}
CIRCLE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108"><circle cx="54" cy="54" r="24" fill="#000"/></svg>'


def _no_network(*a, **k):
    raise AssertionError("test tried to reach the API")


def run(argv, **patch):
    """gptkit.main() in-process with gptkit attributes patched (the API is always blocked).
    Returns (exit code, stdout+stderr+exit message); any other exception propagates and fails the test."""
    patch = {"_call": _no_network, **patch}
    saved = {k: getattr(g, k) for k in patch}
    old_argv, buf, code = sys.argv, io.StringIO(), 0
    try:
        for k, v in patch.items():
            setattr(g, k, v)
        sys.argv = ["gptkit.py", *map(str, argv)]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            g.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if isinstance(e.code, str):
            buf.write(e.code + "\n")
    finally:
        sys.argv = old_argv
        for k, v in saved.items():
            setattr(g, k, v)
    return code, buf.getvalue()


def glyph_build(tmp, svg, **patch):
    """appicon build --mono glyph with a keyed config and a mocked gpt-6 glyph answer."""
    src, out = Path(tmp) / "c.png", Path(tmp) / "g"
    concept(src, 260)
    code, log = run(["appicon", "build", src, "--out", out, "--separate", "local", "--bg", "color", "--no-validate"],
                    _config=lambda: KEYED, ask=lambda *a_, **k_: {"svg": svg}, **patch)
    return code, log, out, json.loads((out / "appicon.json").read_text())


def faint_ring(*_a, **_k):
    """A glyph render whose only ink is a thin value-80 ring: non-empty raw bbox, empty at the 128 threshold."""
    yy, xx = np.mgrid[0:432, 0:432]
    return Image.fromarray(np.where(abs(np.hypot(xx - 216, yy - 216) - 100) < 3, 80, 0).astype(np.uint8), "L")


def concept(path, r):
    S = 1024
    yy, xx = np.mgrid[0:S, 0:S] / S
    d = np.sqrt((xx - 0.5) ** 2 + (yy - 0.45) ** 2)
    bg = np.stack([40 - 30 * d, 45 - 30 * d, 110 - 70 * d], -1).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(bg)
    m = Image.new("L", (S, S), 0)
    dm = ImageDraw.Draw(m)
    dm.ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], fill=255)
    dm.ellipse([S / 2 - r * 0.55, S / 2 - r * 1.25, S / 2 + r * 1.45, S / 2 + r * 0.75], fill=0)
    im.paste(Image.new("RGB", (S, S), (255, 219, 205)), (0, 0), m)
    im.save(path)


def opened(p):
    with Image.open(p) as im:
        return im.size, im.mode, dict(im.info)


def build(tmp, r, *extra):
    src = Path(tmp) / f"c{r}.png"
    concept(src, r)
    out = Path(tmp) / f"b{r}"
    code, log = run(["appicon", "build", src, "--out", out, "--mono", "silhouette", "--no-validate", *extra],
                    _config=lambda: {**KEYED, "key": None})
    assert code == 0, log
    return out, json.loads((out / "appicon.json").read_text())


def renderer():
    """True when an SVG renderer actually works here: Chrome that renders with the shared flags, or another backend."""
    return g._chrome_probe()[0] or (g._renderer_name() not in (None, "chrome"))


_SHARED = {}


def shared_build():
    """One validated CLI build of the r=260 concept, shared by the read-only tests below (saves ~4 s per test)."""
    if not _SHARED:
        _SHARED["tmp"] = tempfile.TemporaryDirectory()
        src, out = Path(_SHARED["tmp"].name) / "c260.png", Path(_SHARED["tmp"].name) / "b260"
        concept(src, 260)
        env = {"GPTKIT_CONFIG": "/nonexistent", **{k: v for k, v in os.environ.items() if k in ("ANDROID_HOME", "PATH")}}
        subprocess.run([sys.executable, str(K), "appicon", "build", str(src), "--out", str(out), "--mono", "silhouette"],
                       check=True, capture_output=True, env=env)
        _SHARED["out"], _SHARED["rep"] = out, json.loads((out / "appicon.json").read_text())
    return _SHARED["out"], _SHARED["rep"]


class AppIconBuild(unittest.TestCase):
    def test_small_subject_is_kept_and_exact(self):
        if True:
            out, rep = shared_build()
            self.assertEqual(rep["background"]["kind"], "gradient")
            self.assertEqual(rep["safe_zone"]["shrink"], 1.0)
            self.assertTrue(rep["safe_zone"]["ok"])
            self.assertLess(rep["composite_diff_vs_concept"], 3)
            self.assertTrue(rep["ios"]["opaque"])
            res = out / "android" / "res"
            for dens, px in (("mdpi", 108), ("xxxhdpi", 432)):
                self.assertEqual(opened(res / f"mipmap-{dens}" / "ic_launcher_foreground.webp")[0], (px, px))
            self.assertEqual(opened(res / "mipmap-xxhdpi" / "ic_launcher_round.webp")[0], (144, 144))
            xml = (res / "mipmap-anydpi-v26" / "ic_launcher.xml").read_text()
            self.assertIn("@drawable/ic_launcher_background", xml)
            self.assertIn("@mipmap/ic_launcher_monochrome", xml)
            self.assertEqual(opened(out / "ios" / "AppIcon.appiconset" / "AppIcon-1024.png")[1], "RGB")
            self.assertEqual(opened(out / "web" / "favicon.ico")[2]["sizes"], {(16, 16), (32, 32), (48, 48)})

    def test_validation_reports_instead_of_crashing(self):
        _, rep = shared_build()
        self.assertNotIn("FAILED", rep["aapt2"])
        sdk = g._android_sdk()
        if sdk is None:
            self.assertEqual(rep["aapt2"], "aapt2 not found (set ANDROID_HOME): not validated")
        elif sdk[1] is None:  # never claim a link that did not run (contract C4)
            self.assertTrue(rep["aapt2"].startswith("aapt2 compile ok; link skipped (no $ANDROID_HOME/platforms"), rep)
            self.assertNotIn("compile+link", rep["aapt2"])
        else:
            self.assertTrue(rep["aapt2"].startswith("aapt2 compile+link ok ("), rep)
        self.assertTrue(rep["ios"]["actool"].startswith(("ok", "skipped")), rep["ios"]["actool"])

    def test_android_sdk_shape(self):
        """_android_sdk() -> (aapt2 path str, android.jar str | None) | None: a Path here once crashed appicon validation."""
        sdk = g._android_sdk()
        if sdk is None:
            self.skipTest("no aapt2 / Android SDK")
        self.assertEqual(len(sdk), 2)
        self.assertIsInstance(sdk[0], str)
        self.assertTrue(sdk[1] is None or isinstance(sdk[1], str), sdk)

    def test_glyph_multicolour_is_flagged(self):
        if not renderer():
            self.skipTest("no SVG renderer")
        M = 1536
        sil = Image.new("L", (M, M), 0)
        ImageDraw.Draw(sil).ellipse([500, 500, 1036, 1036], fill=255)
        two = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108"><circle cx="54" cy="54" r="20" fill="#000"/>'
               '<rect x="40" y="40" width="8" height="8" fill="#ff0000"/></svg>')
        saved = g.ask
        try:
            with tempfile.TemporaryDirectory() as tmp:
                res, work = Path(tmp) / "res", Path(tmp) / "work"
                res.mkdir(), work.mkdir()
                a = argparse.Namespace(name="ic_launcher", subject="a dot")
                for svg, single in ((two, False), (CIRCLE, None)):
                    g.ask = lambda *a_, **k_: {"svg": svg}
                    glyph, info = g._icon_glyph(a, Image.new("RGB", (1024, 1024)), sil, 33.0, res, work, M)
                    self.assertIsNotNone(glyph)
                    self.assertEqual(info.get("single_colour"), single, info)
        finally:
            g.ask = saved

    def test_appicon_validation(self):
        """Bad invocations exit 1 (one line naming the flag/file) or 2 (argparse, naming the flag); never a traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            src, mock = Path(tmp) / "c.png", Path(tmp) / "m.png"
            concept(src, 260)
            Image.new("RGB", (300, 200), "navy").save(mock)
            (Path(tmp) / "junk.png").write_bytes(b"not a png")
            cases = [
                (["build", Path(tmp) / "missing.png"], 1, "concept not found"),
                (["build", Path(tmp) / "junk.png"], 1, "cannot read concept"),
                (["build", src, "--fg-scale", "0"], 2, "--fg-scale"),
                (["build", src, "--fg-scale", "-1"], 2, "--fg-scale"),
                (["build", src, "--fg-scale", "nan"], 2, "--fg-scale"),
                (["build", src, "--name", "Ic-Launcher"], 1, "use --name ic_launcher"),
                (["build", src, "--name", "2x"], 1, "--name"),
                (["build", src, "--bg", "color", "--bg-color", "blue"], 1, "--bg-color"),
                (["concepts", "b", "--variants", "0"], 2, "--variants"),
                (["concepts", "b", "--jobs", "-2"], 2, "--jobs"),
                (["concepts", "b", "--variants", "two"], 2, "--variants"),
                (["concepts", "b", "--crop", "1,2,3,4"], 1, "--crop needs --mockup"),
                (["concepts", "b", "--mockup", mock, "--crop", "1,2,3"], 1, "--crop must be x,y,w,h"),
                (["concepts", "b", "--mockup", mock, "--crop", "a,b,c,d"], 1, "--crop must be x,y,w,h"),
                (["concepts", "b", "--mockup", mock, "--crop", "900,0,10,10"], 1, "no area inside"),
                (["concepts", "b", "--mockup", Path(tmp) / "none.png"], 1, "mockup not found"),
            ]
            for args, want, text in cases:
                out = Path(tmp) / "o"
                code, log = run(["appicon", args[0], *args[1:], "--out", out])
                self.assertEqual(code, want, (args, log))
                self.assertIn(text, log, args)
                self.assertNotIn("Traceback", log)
                self.assertFalse(out.exists(), f"{args}: wrote output before validating")

    def test_build_warnings_and_api_fallbacks(self):
        """A transparent concept is flagged; with --separate api / --bg image a failed edit falls back to the local
        matte / fitted bitmap, named in warnings (one build covers both to keep the file fast)."""
        def refused(*a_, **k_):
            raise g.ApiError(400, 'HTTP 400 /images/edits: {"error": {"code": "moderation_blocked"}}')
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "t.png"
            concept(src, 260)
            im = Image.open(src).convert("RGBA")
            im.putalpha(Image.new("L", im.size, 0))
            im.save(src)
            code, log = run(["appicon", "build", src, "--out", Path(tmp) / "o", "--separate", "api", "--bg", "image",
                             "--mono", "none", "--no-validate"], _config=lambda: KEYED, edit=refused)
            self.assertEqual(code, 0, log)
            rep = json.loads((Path(tmp) / "o" / "appicon.json").read_text())
            self.assertTrue(any("transparency" in w for w in rep["warnings"]), rep.get("warnings"))
            self.assertIn("  ! concept has transparency", log)
            self.assertTrue(rep["ios"]["had_alpha_discarded"])
            self.assertEqual(rep["background"]["kind"], "model")
            self.assertEqual(rep["separation"]["used"], "local")
            self.assertEqual(sum("moderation_blocked" in w for w in rep["warnings"]), 2, rep["warnings"])
            self.assertIn("  ! foreground edit failed", log)
            self.assertIn("  ! background plate edit failed", log)
            self.assertTrue((Path(tmp) / "o" / "android" / "res" / "mipmap-xxxhdpi" / "ic_launcher_background.webp").exists())

    def test_concepts_partial_failure(self):
        buf = io.BytesIO()
        with tempfile.TemporaryDirectory() as tmp:
            concept(Path(tmp) / "src.png", 260)
            Image.open(Path(tmp) / "src.png").save(buf, "PNG")
            png = buf.getvalue()

            def gen(prompt, *a_, **k_):
                if "beta" in prompt:
                    raise g.ApiError(400, 'HTTP 400 /images/generations: {"error": {"code": "moderation_blocked"}}')
                return png, {"quality": "high"}
            out = Path(tmp) / "c"
            code, log = run(["appicon", "concepts", "Hush", "--out", out, "--variants", "3", "--directions",
                             "alpha|beta|gamma"], _config=lambda: KEYED, generate=gen)
            self.assertEqual(code, 0, log)
            self.assertEqual(sorted(p.name for p in out.glob("concept_v*.png")), ["concept_v1.png", "concept_v3.png"])
            self.assertIn("! variant 2 (beta) failed", log)
            self.assertIn("moderation_blocked", log)
            self.assertEqual(sorted(json.loads((out / "concepts.json").read_text())), ["concept_v1.png", "concept_v3.png"])
            self.assertEqual(len(list((out / "review").glob("concepts_*.png"))), 1)

            def blocked(*a_, **k_):
                raise g.ApiError(429, "HTTP 429 /images/generations: insufficient_quota")
            code, log = run(["appicon", "concepts", "Hush", "--out", Path(tmp) / "none", "--variants", "2"],
                            _config=lambda: KEYED, generate=blocked)
            self.assertEqual(code, 1, log)
            self.assertIn("all 2 variants failed", log)

    def test_glyph_mask_is_the_glyph_not_the_canvas(self):
        if not renderer():
            self.skipTest("no SVG renderer")
        styled = CIRCLE.replace('fill="#000"', 'style="fill: #ffee00; stroke:none"')  # a light, CSS-styled fill
        for svg in (CIRCLE, styled):
            with tempfile.TemporaryDirectory() as tmp:
                bb = g._svg_raster(svg, tmp, 432).point(lambda v: 255 if v > 128 else 0).getbbox()
            self.assertTrue(bb and all(abs(a - b) <= 3 for a, b in zip(bb, (120, 120, 312, 312))), (svg, bb))

    def test_glyph_faint_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, log, out, rep = glyph_build(tmp, CIRCLE, _svg_raster=faint_ring)
            self.assertEqual(code, 0, log)
            self.assertEqual(rep["monochrome"], "@mipmap/ic_launcher_monochrome")
            self.assertFalse(rep["glyph"]["ok"])
            self.assertIn("faint", rep["glyph"]["note"])
            self.assertIn("glyph placement not verified/empty", log)
            res = out / "android" / "res"
            self.assertFalse((res / "drawable" / "ic_launcher_monochrome.xml").exists())
            self.assertTrue((res / "mipmap-xxxhdpi" / "ic_launcher_monochrome.webp").exists())
            self.assertIn("@mipmap/ic_launcher_monochrome", (res / "mipmap-anydpi-v26" / "ic_launcher.xml").read_text())

    def test_glyph_empty_not_referenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, log, out, rep = glyph_build(tmp, '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108"></svg>')
            self.assertEqual(code, 0, log)
            self.assertEqual(rep["monochrome"], "@mipmap/ic_launcher_monochrome")
            self.assertFalse(rep["glyph"]["ok"])
            self.assertIn("glyph placement not verified/empty", log)
            self.assertNotIn("@drawable/ic_launcher_monochrome",
                             (out / "android" / "res" / "mipmap-anydpi-v26" / "ic_launcher_round.xml").read_text())
            if renderer():
                self.assertRegex(rep["glyph"]["note"], "empty|did not render")

    def test_glyph_placement_verified_or_rejected(self):
        """Direct _icon_glyph: a real glyph is placed inside the safe zone; shapes outside the viewBox are rejected."""
        if not renderer():
            self.skipTest("no SVG renderer")
        M = 1536
        sil = Image.new("L", (M, M), 0)
        ImageDraw.Draw(sil).ellipse([500, 500, 1036, 1036], fill=255)
        off = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108"><rect x="200" y="200" width="10" height="10"/></svg>'
        saved = g.ask
        try:
            with tempfile.TemporaryDirectory() as tmp:
                res, work = Path(tmp) / "res", Path(tmp) / "work"
                res.mkdir(), work.mkdir()
                a = argparse.Namespace(name="ic_launcher", subject="a dot")
                g.ask = lambda *a_, **k_: {"svg": off}
                glyph, info = g._icon_glyph(a, Image.new("RGB", (1024, 1024)), sil, 33.0, res, work, M)
                self.assertIsNone(glyph)
                self.assertIn("empty", info["note"])
                self.assertFalse((res / "drawable" / "ic_launcher_monochrome.xml").exists())
                g.ask = lambda *a_, **k_: {"svg": CIRCLE}
                glyph, info = g._icon_glyph(a, Image.new("RGB", (1024, 1024)), sil, 33.0, res, work, M)
                self.assertIsNotNone(glyph)
                self.assertLessEqual(info["radius_dp"], 33.0)
                self.assertGreater(info["iou_vs_silhouette"], 0.9)
                self.assertTrue((res / "drawable" / "ic_launcher_monochrome.xml").exists())
        finally:
            g.ask = saved

    def test_copied_illustration_defensive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            mock = d / "mockup.png"
            Image.new("RGB", (200, 300), (30, 40, 90)).save(mock)
            good = {"mockup": "mockup.png", "assets": [{"name": "hero", "format": "png", "kind": "illustration",
                                                        "bbox": {"x": 20, "y": 20, "w": 120, "h": 120}}]}
            img = Image.new("RGB", (256, 256), (30, 40, 90))
            ImageDraw.Draw(img).ellipse([80, 80, 176, 176], fill=(250, 220, 200))
            bad = [{k: v for k, v in good.items() if k != "mockup"}, {k: v for k, v in good.items() if k != "assets"},
                   {**good, "assets": [{**good["assets"][0], "bbox": {"x": 1, "y": 1, "w": 5}}]},
                   {**good, "assets": {"hero": 1}}, {**good, "assets": [{"format": "png", "bbox": good["assets"][0]["bbox"]}]},
                   {**good, "mockup": "gone.png"}, "not json {"]
            for m in bad:
                (d / "assets.json").write_text(m if isinstance(m, str) else json.dumps(m))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(g._copied_illustration(img, mock), m)
            (d / "assets.json").write_text(json.dumps(good))
            r = g._copied_illustration(img, mock)
            self.assertTrue(r is None or (r[0] == "hero" and isinstance(r[1], float)), r)

    def test_ios_tinted_alpha(self):
        out, rep = shared_build()
        ios = out / "ios" / "AppIcon.appiconset"
        self.assertEqual(opened(ios / "AppIcon-1024.png")[1], "RGB")  # light stays opaque
        with Image.open(ios / "AppIcon-1024-tinted.png") as t:
            self.assertIn(t.mode, ("LA", "RGBA"))
            self.assertEqual(t.size, (1024, 1024))
            lo, hi = t.getchannel("A").getextrema()
            self.assertEqual((lo, hi), (0, 255))  # transparent where there is no subject, solid on it
            self.assertEqual(t.getpixel((5, 5))[-1], 0)
            self.assertEqual(t.getchannel("A").getpixel((330, 512)), 255)
            if t.mode == "RGBA":
                r_, g_, b_ = (np.asarray(t.getchannel(c), int) for c in "RGB")
                self.assertTrue((r_ == g_).all() and (g_ == b_).all())
        with Image.open(ios / "AppIcon-1024-dark.png") as d:
            self.assertEqual(d.mode, "RGBA")
            self.assertEqual(d.getpixel((5, 5))[3], 0)  # the system supplies the dark backing
        self.assertTrue(rep["ios"]["tinted_has_alpha"] and rep["ios"]["dark_has_alpha"] and rep["ios"]["opaque"])
        self.assertFalse(rep["ios"]["had_alpha_discarded"])
        self.assertLess(rep["play_store"]["kb"], 1024)
        self.assertEqual(opened(out / "android" / "play_store_512.png")[:2], ((512, 512), "RGB"))
        images = json.loads((ios / "Contents.json").read_text())["images"]
        self.assertEqual([i.get("appearances", [{}])[0].get("value") for i in images], [None, "dark", "tinted"])

    def test_oversized_subject_is_shrunk_into_safe_zone(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rep = build(tmp, 470, "--bg", "color")
            self.assertLess(rep["safe_zone"]["shrink"], 1.0)
            self.assertLessEqual(rep["safe_zone"]["subject_radius_dp"], 33.0)
            self.assertTrue(rep["web_maskable"]["ok"])


if __name__ == "__main__":
    unittest.main()
