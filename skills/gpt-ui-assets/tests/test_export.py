"""Offline checks for `export`: python3 tests/test_export.py (no API calls, no Chrome)."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger
import gptkit as gk  # noqa: E402
KIT = str(HERE.parent / "scripts" / "gptkit.py")


def _ns(**kw):
    base = dict(file=None, dp=48.0, hdp=None, name=None, res=None, platform="android", format=None, quality=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _icon(size=256):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([size // 12, size // 12, size - size // 12, size - size // 12], fill=(30, 144, 255, 255))
    d.rectangle([size // 3, size // 3, size * 2 // 3, size * 2 // 3], fill=(255, 255, 255, 255))
    return im


def _photo(size=256):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    im = Image.fromarray(a, "RGB").convert("RGBA")
    return im


def test_names_and_densities():
    """G6-3: Android names go through _res_name in every density bucket; sizes round like Android (int(px + 0.5))."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "My Icon-2.png"
        _icon().save(src)
        gk.cmd_export(_ns(file=str(src), res=f"{td}/res"))
        for dens, sc in gk.DENSITIES.items():
            p = Path(td) / "res" / f"drawable-{dens}" / "my_icon_2.webp"
            assert p.exists(), p
            assert Image.open(p).size == (int(48 * sc + 0.5),) * 2
        assert not list(Path(td).glob("res/*/My Icon-2*"))
        gk.cmd_export(_ns(file=str(src), name="9lives", dp=23, res=f"{td}/res"))
        assert (Path(td) / "res" / "drawable-hdpi" / "img_9lives.webp").exists()
        assert Image.open(Path(td) / "res" / "drawable-hdpi" / "img_9lives.webp").size == (35, 35), "23 * 1.5 rounds to 35 on Android"
        # iOS keeps the name and writes Xcode's imageset shape; a new .xcassets gets its root Contents.json
        gk.cmd_export(_ns(file=str(src), name="Hero-Art", dp=44, platform="ios", res=f"{td}/App/Assets.xcassets"))
        cat = Path(td) / "App" / "Assets.xcassets"
        assert json.loads((cat / "Contents.json").read_text()) == {"info": {"version": 1, "author": "xcode"}}
        cj = json.loads((cat / "Hero-Art.imageset" / "Contents.json").read_text())
        assert cj == {"images": [{"idiom": "universal", "filename": f"Hero-Art@{s}x.png", "scale": f"{s}x"} for s in (1, 2, 3)],
                      "info": {"version": 1, "author": "xcode"}}
        assert [Image.open(cat / "Hero-Art.imageset" / f"Hero-Art@{s}x.png").size for s in (1, 2, 3)] == [(44, 44), (88, 88), (132, 132)]
        # web: 1x + @2x (the name goes into src/srcset URLs: no spaces)
        gk.cmd_export(_ns(file=str(src), dp=320, platform="web", res=f"{td}/public"))
        assert Image.open(Path(td) / "public" / "My-Icon-2.webp").size == (320, 320)
        assert Image.open(Path(td) / "public" / "My-Icon-2@2x.webp").size == (640, 640)


def test_export_argparse_and_inputs():
    """G6-3: --dp 0 is an argparse error (exit 2); a missing or undecodable file is a one-line exit (no traceback)."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "a.png"
        _icon(64).save(src)
        for args, code, needle in ((["--dp", "0"], 2, "--dp"), (["--dp", "nan"], 2, "--dp"), (["--dp", "-5"], 2, "--dp")):
            r = subprocess.run([sys.executable, KIT, "export", str(src), *args, "--res", f"{td}/res"], capture_output=True, text=True)
            assert r.returncode == code and needle in r.stderr and "Traceback" not in r.stderr, r.stderr
        r = subprocess.run([sys.executable, KIT, "export", f"{td}/nope.png", "--dp", "48", "--res", f"{td}/res"], capture_output=True, text=True)
        assert r.returncode == 1 and "not found" in r.stderr and "nope.png" in r.stderr and "Traceback" not in r.stderr, r.stderr
        (Path(td) / "bad.png").write_text("not an image")
        r = subprocess.run([sys.executable, KIT, "export", f"{td}/bad.png", "--dp", "48", "--res", f"{td}/res"], capture_output=True, text=True)
        assert r.returncode == 1 and "cannot read" in r.stderr and r.stderr.count("\n") <= 1 and "Traceback" not in r.stderr, r.stderr
        r = subprocess.run([sys.executable, KIT, "export", str(src), "--dp", "48", "--platform", "ios", "--format", "png",
                            "--res", f"{td}/A.xcassets"], capture_output=True, text=True)
        assert r.returncode == 1 and "--format" in r.stderr and "ios" in r.stderr, r.stderr


def test_export_quality_and_formats():
    """G6-5: flat art is lossless and exact; photo-like art is lossy by default (smaller than --quality 100); --format png works for web."""
    with tempfile.TemporaryDirectory() as td:
        icon, photo = Path(td) / "icon.png", Path(td) / "photo.png"
        _icon().save(icon)
        _photo().save(photo)
        gk.cmd_export(_ns(file=str(icon), dp=48, res=f"{td}/r1"))
        got = Image.open(Path(td) / "r1" / "drawable-xxxhdpi" / "icon.webp").convert("RGBA")
        want = gk._fit(gk._rgba(icon), 192, 192)
        assert np.array_equal(np.asarray(got), np.asarray(want)), "flat art must round-trip losslessly"
        gk.cmd_export(_ns(file=str(photo), dp=48, res=f"{td}/r2"))
        gk.cmd_export(_ns(file=str(photo), dp=48, res=f"{td}/r3", quality=100))
        gk.cmd_export(_ns(file=str(photo), dp=48, res=f"{td}/r4", quality=60))
        lossy = (Path(td) / "r2" / "drawable-xxxhdpi" / "photo.webp").stat().st_size
        lossless = (Path(td) / "r3" / "drawable-xxxhdpi" / "photo.webp").stat().st_size
        q60 = (Path(td) / "r4" / "drawable-xxxhdpi" / "photo.webp").stat().st_size
        assert q60 < lossy < lossless, (q60, lossy, lossless)
        gk.cmd_export(_ns(file=str(icon), dp=100, platform="web", format="png", res=f"{td}/web"))
        assert (Path(td) / "web" / "icon.png").exists() and (Path(td) / "web" / "icon@2x.png").exists()
        assert Image.open(Path(td) / "web" / "icon@2x.png").format == "PNG"
        gk.cmd_export(_ns(file=str(icon), dp=100, format="png", res=f"{td}/r5"))
        assert Image.open(Path(td) / "r5" / "drawable-mdpi" / "icon.png").format == "PNG"


def test_export_warnings(capsys=None):
    """G6-5: a --res that is itself a drawable folder, a tiny source and a fully transparent source are warned about."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as td:
        tiny, empty = Path(td) / "tiny.png", Path(td) / "empty.png"
        _icon(16).save(tiny)
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(empty)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gk.cmd_export(_ns(file=str(tiny), dp=48, res=f"{td}/app/src/main/res/drawable"))
        out = buf.getvalue()
        assert "! --res" in out and "did you mean --res" in out and out.count("drawable/drawable-") == 5, out
        assert "! source 16x16 px is smaller than the largest target 192x192" in out, out
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gk.cmd_export(_ns(file=str(empty), dp=8, res=f"{td}/res"))
        out = buf.getvalue()
        assert "! source is fully transparent" in out and "smaller than" not in out, out


def test_export_modes_and_exif():
    """G6-7: P, L, LA, RGB JPEG, CMYK JPEG sources export; EXIF orientation is honoured once _rgba applies it (G3-1)."""
    from PIL import ImageOps
    with tempfile.TemporaryDirectory() as td:
        base = _icon(64)
        for mode, ext in (("P", "png"), ("L", "png"), ("LA", "png"), ("RGB", "jpg"), ("CMYK", "jpg")):
            p = Path(td) / f"m_{mode}.{ext}"
            base.convert(mode).save(p)
            gk.cmd_export(_ns(file=str(p), dp=10, res=f"{td}/res_{mode}"))
            assert Image.open(Path(td) / f"res_{mode}" / "drawable-xxxhdpi" / f"m_{mode.lower()}.webp").size == (40, 40)
        # EXIF orientation 6: a 100x200 portrait stored as 200x100 landscape
        ex = Path(td) / "exif.jpg"
        im = Image.new("RGB", (200, 100), "red")
        exif = im.getexif()
        exif[0x0112] = 6
        im.save(ex, exif=exif)
        if gk._rgba(ex).size != ImageOps.exif_transpose(Image.open(ex)).size:
            print("  (skipped: _rgba does not apply EXIF orientation yet; g3's G3-1 lands ImageOps.exif_transpose)")
            return
        gk.cmd_export(_ns(file=str(ex), dp=50, res=f"{td}/res_exif"))
        assert Image.open(Path(td) / "res_exif" / "drawable-mdpi" / "exif.webp").size == (50, 100)


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
