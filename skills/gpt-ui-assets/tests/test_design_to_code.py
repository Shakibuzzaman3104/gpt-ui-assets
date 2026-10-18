"""Offline checks for the design-to-code commands (run: python3 -m pytest skills/gpt-ui-assets/tests, or as a script).

Fixtures are drawn with Pillow's bundled default font (no downloads); the API is never called (`ask` is monkeypatched
where a vision path is exercised); Google Fonts is replaced by a tiny metadata list and the OFL-licensed subsetted
TTFs in tests/fonts/ (Inter 300/400/500/700, Roboto 400 and SerifTest 700, a renamed Lora subset; Basic Latin only)."""
import contextlib
import io
import json
import os
import plistlib
import shutil
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
import gptkit as g  # noqa: E402

FONTS = Path(__file__).resolve().parent / "fonts"
W, H, SS = 720, 1440, 4
TRUTH = dict(bg="#0E1428", card="#171E36", border="#26304F", title="#F8F7FB", body="#A7ADC4", accent="#A69BF7",
             label="#0E1428", avatar="#F4A261", chip="#2B3558")


def _font(size, weight=400):
    p = FONTS / f"Inter-{weight}.ttf"
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default(size=size)


def _rr(d, box, r, fill, outline=None, width=0, corners=None):
    x, y, w, h = box
    d.rounded_rectangle([x * SS, y * SS, (x + w) * SS - 1, (y + h) * SS - 1], r * SS, fill=fill, outline=outline,
                        width=width * SS, corners=corners)


def draw_design(title_size=56, body=TRUTH["body"], accent=TRUTH["accent"], chip_dy=0, avatar=True, status=0, nav=0,
                scale=1.0, body_weight=400, width=W, height=H, font_scale=1.0, body_font=None):
    """The synthetic 720x1440 dark design (card r=32 with 2px border, pill r=56, chip r=16, avatar d=80, three text styles)
    with optional planted deviations; returns an RGB image."""
    Hh = height + status + nav
    big = Image.new("RGB", (width * SS, Hh * SS), TRUTH["bg"])
    d = ImageDraw.Draw(big)
    _rr(d, (40, 200 + status, 640, 360), 32, TRUTH["card"], TRUTH["border"], 2)
    _rr(d, (60, 1200 + status, 600, 112), 56, accent)
    _rr(d, (72, 440 + status + chip_dy, 200, 64), 16, TRUTH["chip"])
    if avatar:
        d.ellipse([600 * SS, (60 + status) * SS, 680 * SS - 1, (140 + status) * SS - 1], fill=TRUTH["avatar"])
    if status:
        d.rectangle([0, 0, width * SS, status * SS], fill="#000000")
    if nav:
        d.rectangle([0, (height + status) * SS, width * SS, Hh * SS], fill="#000000")
    im = big.resize((width, Hh), Image.BOX)
    dr = ImageDraw.Draw(im)
    fl, fi, fb = _font(round(title_size * font_scale), 700), _font(round(28 * font_scale), body_weight), _font(round(32 * font_scale), 700)
    if body_font:
        fi = ImageFont.truetype(str(FONTS / body_font), round(28 * font_scale))
    dr.text((40, 80 + status), "Good evening", font=fl, fill=TRUTH["title"])
    dr.text((80, 240 + status), "Your plan for today", font=fi, fill=body)
    dr.text((80, 290 + status), "Three short sessions", font=fi, fill=body)
    tw = dr.textlength("Start session", font=fb)
    dr.text((360 - tw / 2, 1256 + status), "Start session", font=fb, fill=TRUTH["label"], anchor="lm")
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return im


def design_spec():
    return {"palette": [
        {"name": "Screen background", "hex": "#0F1529", "usage": "app background canvas"},
        {"name": "Card surface", "hex": "#181F38", "usage": "plan card fill"},
        {"name": "Card border", "hex": "#28324F", "usage": "1px outline around the plan card"},
        {"name": "Primary text", "hex": "#FAF8FC", "usage": "greeting title"},
        {"name": "Secondary text", "hex": "#A0A8C0", "usage": "card body copy"},
        {"name": "Accent button fill", "hex": "#A89CF5", "usage": "start button"},
        {"name": "Button label", "hex": "#0E1428", "usage": "text on accent"},
        {"name": "Avatar", "hex": "#F2A060", "usage": "profile avatar disc"}],
        "typography": [
        {"role": "Greeting title", "sample": "Good evening", "family_guess": "Inter", "size_px": 56, "weight": 700,
         "letter_spacing_em": 0, "color": "#FAF8FC", "platform_style": "headlineLarge"},
        {"role": "Card body", "sample": "Your plan for today\nThree short sessions", "family_guess": "Inter", "size_px": 28,
         "weight": 400, "letter_spacing_em": 0, "color": "#A0A8C0", "platform_style": "bodyLarge"},
        {"role": "Button label", "sample": "Start session", "family_guess": "Inter", "size_px": 32, "weight": 700,
         "letter_spacing_em": 0, "color": "#0E1428", "platform_style": "labelLarge"}],
        "radii": [{"element": "plan card", "px": 30}, {"element": "start button", "px": 56}, {"element": "chip", "px": 14}],
        "spacing": [{"between": "screen edge to card", "px": 40}, {"between": "card padding", "px": 40}],
        "components": [
        {"name": "Greeting", "bbox": {"x": 36, "y": 70, "w": 400, "h": 90}, "native": "Text", "notes": "Title 'Good evening'"},
        {"name": "Plan card", "bbox": {"x": 42, "y": 198, "w": 636, "h": 364}, "native": "Card (outlined)",
         "notes": "Card with body lines Your plan for today / Three short sessions"},
        {"name": "Chip", "bbox": {"x": 70, "y": 438, "w": 204, "h": 68}, "native": "AssistChip", "notes": "empty chip"},
        {"name": "Start button", "bbox": {"x": 58, "y": 1198, "w": 604, "h": 116}, "native": "Button (filled)",
         "notes": "Pill button labelled Start session"},
        {"name": "Avatar", "bbox": {"x": 598, "y": 58, "w": 84, "h": 84}, "native": "custom", "notes": "orange profile avatar disc"}],
        "platform": "android", "px_to_dp": 0.5, "px_to_dp_design": 0.5}


_CACHE = {}


def make_design(d, spec=None, assets=True):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    if "mockup" not in _CACHE:  # drawn once per process: the 4x supersampled render is the slow part
        _CACHE["mockup"] = _tmp() / "mockup.png"
        draw_design().save(_CACHE["mockup"])
    shutil.copy(_CACHE["mockup"], d / "mockup.png")
    (d / "design_spec.json").write_text(json.dumps(spec or design_spec(), indent=1))
    if assets:
        (d / "assets.json").write_text(json.dumps({
            "mockup": "mockup.png", "width": W, "height": H, "platform": "android", "screen_dp": 360,
            "assets": [{"name": "avatar_disc", "kind": "illustration", "format": "png", "bbox": {"x": 598, "y": 58, "w": 84, "h": 84},
                        "transparent": True, "animated": False, "description": "orange avatar disc", "colors": ["#F4A261"]}]}))
    return d


def run(*argv, expect_exit=None):
    """Run a gptkit command in-process; returns captured stdout (and asserts the exit message when expect_exit is set)."""
    buf = io.StringIO()
    saved = sys.argv
    sys.argv = ["gptkit", *map(str, argv)]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                g.main()
            except SystemExit as e:
                code = e.code
                msg = str(code) if isinstance(code, str) else ""
                if expect_exit is None:
                    raise AssertionError(f"unexpected exit {code!r}; output:\n{buf.getvalue()}")
                assert expect_exit in msg + buf.getvalue(), f"exit {code!r} lacks {expect_exit!r}; output:\n{buf.getvalue()}"
                return buf.getvalue() + msg
    finally:
        sys.argv = saved
    assert expect_exit is None, f"expected an exit containing {expect_exit!r}; output:\n{buf.getvalue()}"
    return buf.getvalue()


@contextlib.contextmanager
def patched(**kw):
    saved = {k: getattr(g, k) for k in kw}
    for k, v in kw.items():
        setattr(g, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(g, k, v)


@contextlib.contextmanager
def env(**kw):
    saved = {k: os.environ.get(k) for k in kw}
    clear = lambda: [f.cache_clear() for f in (g._chrome, g._chrome_probe, g._gf_meta, g._config) if hasattr(f, "cache_clear")]
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        clear()


def _tmp():
    return Path(tempfile.mkdtemp(prefix="gptkit_d2c_"))


# ---------------------------------------------------------------- colour maths (unchanged from the original suite, extended)

SHARMA = [  # the 34 CIEDE2000 reference pairs (Sharma, Wu, Dalal 2005), (Lab1, Lab2, dE00)
    ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425), ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
    ((50, 2.8361, -74.02), (50, 0, -82.7485), 3.4412), ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0),
    ((50, -1.1848, -84.8006), (50, 0, -82.7485), 1.0), ((50, -0.9009, -85.5211), (50, 0, -82.7485), 1.0),
    ((50, 0, 0), (50, -1, 2), 2.3669), ((50, -1, 2), (50, 0, 0), 2.3669), ((50, 2.49, -0.001), (50, -2.49, 0.0009), 7.1792),
    ((50, 2.49, -0.001), (50, -2.49, 0.001), 7.1792), ((50, 2.49, -0.001), (50, -2.49, 0.0011), 7.2195),
    ((50, 2.49, -0.001), (50, -2.49, 0.0012), 7.2195), ((50, -0.001, 2.49), (50, 0.0009, -2.49), 4.8045),
    ((50, -0.001, 2.49), (50, 0.001, -2.49), 4.8045), ((50, -0.001, 2.49), (50, 0.0011, -2.49), 4.7461),
    ((50, 2.5, 0), (73, 25, -18), 27.1492), ((50, 2.5, 0), (61, -5, 29), 22.8977), ((50, 2.5, 0), (56, -27, -3), 31.903),
    ((50, 2.5, 0), (58, 24, 15), 19.4535), ((50, 2.5, 0), (50, 3.1736, 0.5854), 1.0), ((50, 2.5, 0), (50, 3.2972, 0), 1.0),
    ((50, 2.5, 0), (50, 1.8634, 0.5757), 1.0), ((50, 2.5, 0), (50, 3.2592, 0.335), 1.0),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644), ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.263),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.248, -4.962), 1.8731), ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.694), (23.0331, 14.973, -42.5619), 2.0373), ((36.4612, 47.858, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.441), (91.1528, -1.6435, 0.0447), 1.4441), ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377), ((2.0776, 0.0795, -1.135), (0.9033, -0.0636, -0.5514), 0.9082),
    ((50, 2.5, 0), (50, 0, -2.5), 4.3065)]


def test_ciede2000_reference_pairs():
    for a, b, want in SHARMA:
        assert abs(float(g._de00(np.array(a), np.array(b))) - want) < 1e-3, (a, b, want)
        assert abs(float(g._de00(np.array(b), np.array(a))) - want) < 1e-3


def test_surface_geometry_recovers_rounded_rect():
    ss, (x, y, w, h, r) = 4, (60, 50, 300, 140, 24)
    big = Image.new("RGB", (420 * ss, 260 * ss), "#0E1428")
    ImageDraw.Draw(big).rounded_rectangle([x * ss, y * ss, (x + w) * ss - 1, (y + h) * ss - 1], r * ss, fill="#171E36",
                                          outline="#26304F", width=ss)
    rgb = np.asarray(big.resize((420, 260), Image.BOX))
    geo = g._surface_geometry(rgb, {"x": x + 3, "y": y - 2, "w": w - 5, "h": h + 3})
    assert geo and geo["shape"] == "rect"
    assert max(abs(a - b) for a, b in zip(geo["box"], (x, y, w, h))) <= 1.5
    assert abs(geo["radius"] - r) <= 1.5


def test_locate_text_finds_the_ink_box():
    font = ImageFont.load_default(size=40)
    im = Image.new("RGB", (500, 120), "#0E1428")
    ImageDraw.Draw(im).text((40, 30), "Quiet Lake", font=font, fill="#F8F7FB")
    layer = Image.new("L", im.size, 0)
    ImageDraw.Draw(layer).text((40, 30), "Quiet Lake", font=font, fill=255)
    truth = layer.point(lambda v: 255 if v > 90 else 0).getbbox()
    got = g._locate_text(np.asarray(im), [20, 15, 330, 90], "Quiet Lake", 40, 0, "#F8F7FB")
    x, y, w, h = got["box"]
    assert max(abs(a - b) for a, b in zip((x, y, x + w, y + h), truth)) <= 2
    assert g._de_hex(got["ink"], "#F8F7FB") < 3


# ---------------------------------------------------------------- G9-1: generated theme code compiles

def test_norm_hex():
    assert g._norm_hex("#fff") == ("#FFFFFF", 1.0)
    assert g._norm_hex("#FFFFFF80")[0] == "#FFFFFF" and abs(g._norm_hex("#FFFFFF80")[1] - 128 / 255) < 1e-9
    assert g._norm_hex("rgba(0,0,0,.5)") == ("#000000", 0.5)
    assert g._norm_hex("rgb(255, 128, 0)") == ("#FF8000", 1.0)
    assert g._norm_hex("#abc") == ("#AABBCC", 1.0)
    for bad in ("bogus", "#12345", "rgb(1,2)", "", None):
        try:
            g._norm_hex(bad)
            raise AssertionError(bad)
        except (ValueError, TypeError):
            pass
    assert g._hexi("#fff") == (255, 255, 255) and g._hexi("#1A1A1A80") == (26, 26, 26)
    d = _tmp()
    spec = design_spec()
    spec["palette"] = [{"name": "White", "hex": "#fff", "usage": "x"}, {"name": "Scrim", "hex": "#FFFFFF80", "usage": "y"},
                       {"name": "Half black", "hex": "rgba(0,0,0,.5)", "usage": "z"}]
    make_design(d, spec)
    run("tokens", d, "--no-snap", "--platforms", "compose,swiftui,web")
    kt = (d / "tokens" / "compose" / "Color.kt").read_text()
    assert "val White = Color(0xFFFFFFFF)" in kt and "val Scrim = Color(0x80FFFFFF)" in kt and "val HalfBlack = Color(0x80000000)" in kt
    css = (d / "tokens" / "web" / "tokens.css").read_text()
    assert "--color-white: #FFFFFF;" in css and "--color-scrim: #FFFFFF80;" in css
    sw = (d / "tokens" / "swiftui" / "AppTokens.swift").read_text()
    assert "opacity: 0.502" in sw
    spec["palette"].append({"name": "Odd one", "hex": "bogus", "usage": "w"})
    (d / "design_spec.json").write_text(json.dumps(spec))
    run("tokens", d, "--no-snap", expect_exit="palette 'Odd one'")
    shutil.rmtree(d)


def test_compose_namespace_and_weights():
    d = _tmp()
    spec = design_spec()
    spec["typography"][2]["weight"] = 450  # "Button label" role + "Button label" palette entry
    make_design(d, spec)
    out = run("tokens", d, "--package", "com.acme.app.ui.theme", "--prefix", "Acme", "--platforms", "android", "--verify")
    ty = (d / "tokens" / "compose" / "Type.kt").read_text()
    col = (d / "tokens" / "compose" / "Color.kt").read_text()
    assert "object AcmeColors {" in col and "val ButtonLabel = Color(0xFF" in col
    assert "primary = AcmeColors." in col and "darkColorScheme(" in col
    body = ty.split("object AcmeTypography {", 1)[1]
    assert "color = AcmeColors." in body and "color = ButtonLabel," not in body
    assert "FontWeight(450)" in ty and "FontWeight.W450" not in ty and "FontWeight.W700" in ty
    assert not g._lint_tokens(sorted((d / "tokens" / "compose").glob("*.kt")))
    assert "verify lint    : ok" in out and "css" not in json.loads((d / "tokens" / "tokens.json").read_text())["verify"]
    # the lint catches the three original compile failures
    bad = d / "bad"
    bad.mkdir()
    (bad / "Type.kt").write_text("package x\n\nval ButtonLabel = Color(0xFF0E1428)\n\nobject AcmeTypography {\n"
                                 "    val ButtonLabel = TextStyle(\n        fontWeight = FontWeight.W450,\n        color = ButtonLabel,\n    )\n}\n")
    (bad / "Color.kt").write_text("package x\n\nval Primary = Color(0xFFFFF)\nval Primary = Color(0xFF1A1A1A80)\n")
    (bad / "T.swift").write_text("import SwiftUI\n\nextension Color {\n    static let primary = Color(red: 1, green: 1, blue: 1)\n"
                                 "    static let primary = Color(red: 0, green: 0, blue: 0)\n}\n\nenum AppRadius {\n    static let default: CGFloat = 8\n}\n")
    (bad / "t.css").write_text(":root {\n  --color-primary: #fff;\n  --color-primary: #000;\n}\n")
    findings = "\n".join(g._lint_tokens(sorted(bad.iterdir())))
    for needle in ("FontWeight.W450", "names a sibling", "0xFFFFF is not", "0xFF1A1A1A80", "duplicate identifier Primary",
                   "duplicate declaration primary", "`default` is a Swift keyword", "duplicate custom property --color-primary"):
        assert needle in findings, (needle, findings)
    shutil.rmtree(d)


def test_swift_css_names():
    d = _tmp()
    spec = design_spec()
    spec["palette"] = [{"name": "Primary", "hex": "#FFFFFF", "usage": "a"}, {"name": "Primary", "hex": "#F8F7FB", "usage": "b"},
                       {"name": "Secondary", "hex": "#A7ADC4", "usage": "c"}, {"name": "Import", "hex": "#0E1428", "usage": "d"}]
    spec["typography"][0].update(role="Default", color="#FFFFFF")
    spec["radii"].append({"element": "default", "px": 8})
    make_design(d, spec)
    run("tokens", d, "--no-snap")
    sw = (d / "tokens" / "swiftui" / "AppTokens.swift").read_text()
    assert "static let primaryToken = " in sw and "static let primary2 = " in sw and "static let secondaryToken = " in sw
    assert "static let primary = " not in sw and "static let `default` = " in sw and "static let `import` = " in sw
    assert "color: .primaryToken" in sw
    css = (d / "tokens" / "web" / "tokens.css").read_text()
    kt = (d / "tokens" / "compose" / "Color.kt").read_text()
    assert "--color-primary:" in css and "--color-primary-2:" in css and "val Primary2 = " in kt
    tw = (d / "tokens" / "web" / "tailwind.tokens.cjs").read_text()
    assert '"primary-2"' in tw and "presets: [require('./tailwind.tokens.cjs')]" in tw
    assert not g._lint_tokens([*sorted((d / "tokens" / "compose").glob("*.kt")), d / "tokens" / "swiftui" / "AppTokens.swift",
                               d / "tokens" / "web" / "tokens.css"])
    shutil.rmtree(d)


def _fake_exe(path, body):
    Path(path).write_text("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return str(path)


def test_platforms_and_verify():
    d = make_design(_tmp())
    run("tokens", d, "--platforms", "bogus", expect_exit="--platforms: unknown 'bogus'")
    run("tokens", d, "--prefix", "my-app", expect_exit="--prefix 'my-app' is not an identifier")
    run("tokens", d, "--package", "com.acme-app", expect_exit="--package")
    fake = _fake_exe(d / "chrome", "echo 'boom: sandbox failure' >&2\nexit 1\n")
    with env(GPTKIT_CHROME=fake):
        out = run("tokens", d, "--platforms", "compose", "--verify")
        rep = json.loads((d / "tokens" / "tokens.json").read_text())
        assert rep["platforms"] == ["compose"] and set(rep["verify"]) == {"lint", "compose"}
        assert not (d / "tokens" / "web").exists() and not (d / "tokens" / "swiftui").exists()
        out = run("tokens", d, "--platforms", "web", "--verify")
        rep = json.loads((d / "tokens" / "tokens.json").read_text())
        assert rep["verify"]["css"].startswith("FAILED") and "sandbox failure" in rep["verify"]["css"], rep["verify"]
        assert "headless Chrome produced no screenshot" in out
    # the fake Chrome receives the shared flags (padding + --no-sandbox as root)
    argv_log = d / "argv.txt"
    fake2 = _fake_exe(d / "chrome2", f"echo \"$@\" > {argv_log}\nexit 0\n")
    with env(GPTKIT_CHROME=fake2, GPTKIT_CHROME_NO_SANDBOX="1"):
        run("tokens", d, "--platforms", "web", "--verify")
    argv = argv_log.read_text()
    assert "--headless=new" in argv and "--no-sandbox" in argv and "--dump-dom" in argv
    ok, why = g._chrome_probe()
    if ok:
        with env(GPTKIT_CHROME=None):
            out = run("tokens", d, "--platforms", "web", "--verify")
            assert "verify css     : ok (Chrome parsed" in out, out
    shutil.rmtree(d)


# ---------------------------------------------------------------- G9-2: measurements that match ground truth

GEO = {"Sheet": (0, 1100, 720, 500, 28, "top-only"), "Search": (32, 80, 656, 56, 28, "pill"), "Tile A": (32, 180, 200, 200, 12, "rect"),
       "Tile B": (260, 180, 200, 200, 12, "rect"), "Low card": (32, 420, 656, 160, 20, "rect"), "Fab": (600, 960, 88, 88, 44, "circle"),
       "Square": (32, 640, 300, 180, 0, "rect"), "Shadow card": (372, 640, 316, 180, 16, "rect")}


def draw_geo():
    """Light geometry board: full-bleed bottom sheet (top corners only), pill, two tiles, a low-contrast card, a circle, a
    square and a card with a drop shadow (offset 8 px, blur 8)."""
    Wg, Hg = 720, 1600
    big = Image.new("RGB", (Wg * SS, Hg * SS), "#F6F1E7")
    d = ImageDraw.Draw(big)
    _rr(d, GEO["Sheet"][:4], 28, "#FFFFFF", corners=(True, True, False, False))
    _rr(d, GEO["Search"][:4], 28, "#E9E1D3")
    _rr(d, GEO["Tile A"][:4], 12, "#D9573B")
    _rr(d, GEO["Tile B"][:4], 12, "#3B7DD9")
    _rr(d, GEO["Low card"][:4], 20, "#FFFDF9")
    d.ellipse([600 * SS, 960 * SS, 688 * SS - 1, 1048 * SS - 1], fill="#2E2A24")
    _rr(d, GEO["Square"][:4], 0, "#8FB996")
    im = big.resize((Wg, Hg), Image.BOX)
    sh = Image.new("L", (Wg, Hg), 0)
    ImageDraw.Draw(sh).rounded_rectangle([372, 648, 372 + 316, 648 + 180], 16, fill=90)
    im = Image.composite(Image.new("RGB", (Wg, Hg), "#6b6150"), im, sh.filter(ImageFilter.GaussianBlur(8)))
    big2 = im.resize((Wg * SS, Hg * SS), Image.NEAREST)
    _rr(ImageDraw.Draw(big2), GEO["Shadow card"][:4], 16, "#FFFFFF")
    return big2.resize((Wg, Hg), Image.BOX)


def make_geo(d, radii=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    draw_geo().save(d / "mockup.png")
    comps = [{"name": n, "bbox": {"x": v[0] + 2, "y": v[1] - 2, "w": v[2] - 3, "h": v[3] + 3}, "native": "custom", "notes": ""}
             for n, v in GEO.items()]
    spec = {"palette": [], "typography": [], "radii": radii if radii is not None else [{"element": n.lower(), "px": v[4]} for n, v in GEO.items()],
            "spacing": [], "components": comps, "px_to_dp": 0.5}
    (d / "design_spec.json").write_text(json.dumps(spec, indent=1))
    return d


def test_multiline_width():
    d = make_design(_tmp())
    layer = Image.new("L", (W, H), 0)
    ImageDraw.Draw(layer).text((80, 290), "Three short sessions", font=_font(28, 400), fill=255)
    tx0, ty0, tx1, ty1 = layer.point(lambda v: 255 if v > 90 else 0).getbbox()
    texts = g._text_targets(d, json.loads((d / "design_spec.json").read_text()), Image.open(d / "mockup.png").convert("RGBA"))
    inst = texts["Card body"]["instances"]
    assert len(inst) == 2 and inst[1]["sample"] == "Three short sessions"
    x, y, w, h = inst[1]["box"]
    assert abs(w - (tx1 - tx0)) <= 2 and abs(x - tx0) <= 2, (inst[1]["box"], (tx0, ty0, tx1, ty1))
    draw_design().save(d / "shot.png")
    run("compare", d, d / "shot.png")
    rep = json.loads((d / "compare" / "report.json").read_text())
    text_hints = [h_ for r in rep["regions"] for h_ in r["hints"] if h_.startswith("'")]
    assert not text_hints and rep["overall"]["score"] >= 99.5, (text_hints, rep["overall"])
    shutil.rmtree(d)


def test_object_box_ring():
    im = Image.new("RGB", (92, 92), TRUTH["bg"])
    ImageDraw.Draw(im).ellipse([6, 6, 85, 85], fill=TRUTH["avatar"])  # d = 80 in a 92 px crop (84 px bbox padded by 4)
    x, y, w, h = g._object_box(np.asarray(im, dtype=np.float64))
    assert abs(w - 80) <= 1 and abs(h - 80) <= 1 and abs(x - 6) <= 1 and abs(y - 6) <= 1, (x, y, w, h)
    d = make_design(_tmp())
    run("redline", d)
    av = next(i for i in json.loads((d / "redline" / "redline.json").read_text())["items"] if i["name"] == "Avatar")
    assert av["kind"] == "object" and abs(av["px"][2] - 80) <= 1 and abs(av["px"][3] - 80) <= 1, av
    shutil.rmtree(d)


def test_radius_linking():
    three = [{"name": "Chip", "notes": ""}, {"name": "Plan card", "notes": ""}, {"name": "Start button", "notes": ""}]
    assert [r["name"] for r in g._link_regions("chip", three)] == ["Chip"]
    assert [r["name"] for r in g._link_regions("chip", three[:2])] == ["Chip"]
    tiles = [{"name": "Tile A", "notes": ""}, {"name": "Tile B", "notes": ""}]
    assert [r["name"] for r in g._link_regions("tile b", tiles)] == ["Tile B"]
    assert [r["name"] for r in g._link_regions("tile a", tiles)] == ["Tile A"]
    assert g._link_regions("shadow card", [{"name": "Low card", "notes": ""}], scored=True)[0][0] < 10
    d = make_geo(_tmp(), radii=[{"element": "chip", "px": 14}, {"element": "tile a", "px": 12}, {"element": "tile b", "px": 12},
                                {"element": "shadow card", "px": 16}, {"element": "low card", "px": 20}, {"element": "search", "px": 28}])
    spec = json.loads((d / "design_spec.json").read_text())
    spec["components"] = [c for c in spec["components"] if c["name"] in ("Tile A", "Tile B", "Low card", "Search")]
    spec["components"][0]["name"] = "Chip"  # the pill is the chip: with 4 surfaces the spec radius "chip" must still link
    (d / "design_spec.json").write_text(json.dumps(spec))
    out = run("redline", d)
    got = {r["element"]: r for r in json.loads((d / "redline" / "redline.json").read_text())["radii"]}
    assert got["chip"]["component"] == "Chip" and got["chip"]["match_score"] == "exact"
    assert got["tile a"]["component"] == "Tile A" and got["tile b"]["component"] == "Tile B"
    assert got["shadow card"]["measured_px"] is None and got["shadow card"]["component"] is None
    assert got["low card"]["component"] == "Low card" and "match exact" in out
    shutil.rmtree(d)


def test_fullbleed_and_shadow():
    d = make_geo(_tmp())
    out = run("redline", d)
    rl = json.loads((d / "redline" / "redline.json").read_text())
    items = {i["name"]: i for i in rl["items"]}
    assert all(i["kind"] == "surface" for i in items.values()), {n: i["kind"] for n, i in items.items()}
    sheet = items["Sheet"]
    assert sheet["px"] == [0.0, 1100.0, 720.0, 500.0] and sheet["full_bleed"] == ["left", "right", "bottom"]
    tl, tr, bl, br = sheet["corner_radii_px"]
    assert abs(tl - 28) <= 1 and abs(tr - 28) <= 1 and bl is None and br is None
    sc = items["Shadow card"]
    assert sc["shadow"] is True and abs(sc["radius_px"] - 16) <= 1
    assert max(abs(a - b) for a, b in zip(sc["px"], GEO["Shadow card"][:4])) <= 1, sc["px"]
    for name, (x, y, w, h, r, shape) in GEO.items():
        it = items[name]
        assert max(abs(a - b) for a, b in zip(it["px"], (x, y, w, h))) <= 1, (name, it["px"])
        assert abs(it["radius_px"] - r) <= 1, (name, it["radius_px"], r)
    assert items["Fab"]["shape"] == "circle" and items["Search"]["shape"] == "pill"
    assert "drop shadow detected" in out and "full-bleed at left/right/bottom" in out
    # an unmeasurable component is kept and named instead of dropped
    spec = json.loads((d / "design_spec.json").read_text())
    spec["components"].append({"name": "Ghost", "bbox": {"x": 400, "y": 900, "w": 120, "h": 40}, "native": "custom", "notes": ""})
    (d / "design_spec.json").write_text(json.dumps(spec))
    out = run("redline", d)
    ghost = next(i for i in json.loads((d / "redline" / "redline.json").read_text())["items"] if i["name"] == "Ghost")
    assert ghost["kind"] == "unmeasured" and "reason" in ghost and "1 unmeasured: Ghost" in out
    shutil.rmtree(d)


def test_palette_snap_exact():
    d = make_design(_tmp())
    run("tokens", d, "--platforms", "web")
    pal = {p["name"]: p for p in json.loads((d / "tokens" / "tokens.json").read_text())["palette"]}
    assert pal["Screen background"]["hex"] == TRUTH["bg"], pal["Screen background"]
    for name, key in (("Card surface", "card"), ("Card border", "border"), ("Accent button fill", "accent"), ("Avatar", "avatar")):
        assert g._de_hex(pal[name]["hex"], TRUTH[key]) < 1.0, (name, pal[name]["hex"], TRUTH[key])
    assert all(p["snapped"] for p in pal.values())
    shutil.rmtree(d)


# ---------------------------------------------------------------- G9-3 / G9-4: fonts without the network

FAKE_META = [{"family": "Roboto", "category": "Sans Serif", "subsets": ["latin"], "fonts": {"300": {}, "400": {}, "500": {}, "700": {}},
              "popularity": 1, "isNoto": False, "isOpenSource": True, "classifications": [], "stroke": None},
             {"family": "Inter", "category": "Sans Serif", "subsets": ["latin"], "fonts": {"300": {}, "400": {}, "500": {}, "700": {}},
              "popularity": 2, "isNoto": False, "isOpenSource": True, "classifications": [], "stroke": None},
             {"family": "Lora", "category": "Serif", "subsets": ["latin"], "fonts": {"400": {}, "700": {}, "400i": {}},
              "popularity": 3, "isNoto": False, "isOpenSource": True, "classifications": [], "stroke": None},
             {"family": "Roboto Slab", "category": "Serif", "subsets": ["latin"], "fonts": {"400": {}, "700": {}},
              "popularity": 4, "isNoto": False, "isOpenSource": True, "classifications": [], "stroke": "Slab Serif"}]
OFL = "Copyright 2020 The Inter Project Authors\n\nThis Font Software is licensed under the SIL Open Font License, Version 1.1.\n"


class _NoNet(Exception):
    pass


def _no_net(*a, **k):
    raise g.requests.ConnectionError("network disabled in tests")


LOCAL_FILE = {"Lora": "SerifTest"}  # the Lora subset is renamed: Lora's licence reserves its name for unmodified fonts


def _local_ttf(fam, w, it=False):
    p = FONTS / f"{LOCAL_FILE.get(fam, fam)}-{w}.ttf"
    if p.exists() and not it:
        return p
    g._GF_TTF_ERRORS[(fam, w, it)] = "ConnectionError: network disabled in tests"
    return None


@contextlib.contextmanager
def fake_gf(ttf=_local_ttf, license_text=OFL):
    """Google Fonts replaced by FAKE_META, the local subsetted TTFs and a canned licence; every requests call raises."""
    lic = lambda fam: {"license": "OFL", "dir": f"ofl/{fam.lower()}", "text": license_text} if license_text else \
        {"license": "unknown (see fonts.google.com)", "dir": None, "text": None, "error": "ConnectionError: offline"}
    saved_get = g.requests.get
    g.requests.get = _no_net
    try:
        with patched(_gf_meta=lambda: FAKE_META, _gf_ttf=ttf, _gf_license=lic):
            yield
    finally:
        g.requests.get = saved_get


def _fonts_fixture(candidates="0"):
    """A design dir with a finished `fonts` run (computed once per process, then copied)."""
    key = ("fonts", candidates)
    if key not in _CACHE:
        d = make_design(_tmp())
        with fake_gf():
            out = run("fonts", d, "--candidates", candidates)
        _CACHE[key] = (d, out)
    src, out = _CACHE[key]
    d = _tmp()
    shutil.rmtree(d)
    shutil.copytree(src, d)
    return d, out


def test_plist_and_font_xml():
    with env(ANDROID_HOME=os.environ.get("ANDROID_HOME") or "/opt/android-sdk"):
        d, out = _fonts_fixture()
    fj = json.loads((d / "fonts" / "fonts.json").read_text())
    assert [f["family"] for f in fj["families"]] == ["Inter"], fj["families"]
    names = [Path(x["ttf"]).name for x in fj["families"][0]["files"]]
    assert plistlib.loads((d / "fonts" / "ios" / "UIAppFonts.plist").read_bytes())["UIAppFonts"] == names
    assert names == ["Inter-Regular.ttf", "Inter-Bold.ttf"] and all(not Path(x["ttf"]).is_absolute() for x in fj["families"][0]["files"])
    assert all((d / "fonts" / x["ttf"]).exists() for x in fj["families"][0]["files"])
    root = ET.parse(d / "fonts" / "android" / "res" / "font" / "inter.xml").getroot()
    A, APP = "{http://schemas.android.com/apk/res/android}", "{http://schemas.android.com/apk/res-auto}"
    fonts = list(root)
    assert len(fonts) == 2
    for f in fonts:
        for attr in ("font", "fontStyle", "fontWeight"):
            assert f.get(A + attr) and f.get(A + attr) == f.get(APP + attr), f.attrib
    assert (d / "fonts" / "android" / "res" / "font" / "inter_regular.ttf").exists()
    css = (d / "fonts" / "web" / "fonts.css").read_text()
    assert "font-display: swap" in css and 'url("../ttf/Inter-Bold.ttf")' in css and "system-ui" in css
    assert fj["families"][0]["css"].endswith("&display=swap") and "rel=\"preconnect\"" in fj["link_html"]
    assert (d / "fonts" / "licenses" / "Inter-OFL.txt").read_text() == OFL
    assert "PlistBuddy" in out and "Merge" in out
    sdk = g._android_sdk()
    if sdk:
        assert fj["aapt2"].startswith("aapt2 compile ok; link skipped" if not sdk[1] else "aapt2 compile+link ok"), fj["aapt2"]
        assert "valid" not in fj["aapt2"]
    else:
        assert fj["aapt2"].startswith("aapt2 not found")
    # no licence text: the OFL note is printed and no licence file is written
    d2 = make_design(_tmp())
    with fake_gf(license_text=None):
        out2 = run("fonts", d2, "--candidates", "0")
    assert "license text for Inter not fetched" in out2 and "OFL requires" in out2 and not (d2 / "fonts" / "licenses").exists()
    shutil.rmtree(d)
    shutil.rmtree(d2)


def test_weight_tie():
    d, out = _fonts_fixture()
    fj = json.loads((d / "fonts" / "fonts.json").read_text())
    st = {s["role"]: s for s in fj["styles"]}
    assert st["Card body"]["weight"] == 400 and st["Card body"]["family"] == "Inter", st["Card body"]
    assert st["Greeting title"]["weight"] == 700 and st["Button label"]["weight"] == 700
    assert abs(st["Card body"]["size_px"] - 28) <= 1.5 and abs(st["Greeting title"]["size_px"] - 56) <= 2
    assert "weight_tie" in st["Card body"]
    # a hand-edited spec without letter_spacing_em still gets its summary table
    spec = json.loads((d / "design_spec.json").read_text())
    for t in spec["typography"]:
        t.pop("letter_spacing_em", None)
    (d / "design_spec.json").write_text(json.dumps(spec))
    with fake_gf():
        out = run("fonts", d, "--candidates", "0")
    assert "Card body" in out
    shutil.rmtree(d)


def test_fonts_download_failures():
    d = make_design(_tmp())
    (d / "fonts").mkdir()
    (d / "fonts" / "fonts.json").write_text('{"families": [], "styles": [], "keep": "me"}')
    cache = _tmp()
    with env(GPTKIT_CACHE=str(cache)):
        saved = g.requests.get
        g.requests.get = _no_net
        try:
            with patched(_gf_meta=lambda: FAKE_META, _gf_license=lambda fam: {"license": "OFL", "dir": "ofl/x", "text": OFL}):
                out = run("fonts", d, "--candidates", "0", expect_exit="could not download any candidate TTF")
        finally:
            g.requests.get = saved
    assert json.loads((d / "fonts" / "fonts.json").read_text())["keep"] == "me"
    assert "fonts.json left untouched" in out
    # one weight missing: the run continues and says which candidates failed
    def partial(fam, w, it=False):
        return None if (fam, w) == ("Inter", 500) else _local_ttf(fam, w, it)
    with fake_gf(ttf=partial):
        out = run("fonts", d, "--candidates", "0")
    assert "candidate fonts could not be downloaded: Inter 500" in out
    assert [f["family"] for f in json.loads((d / "fonts" / "fonts.json").read_text())["families"]] == ["Inter"]
    shutil.rmtree(d)
    shutil.rmtree(cache)


class _Resp:
    def __init__(self, text="", status=200):
        self.text, self.status_code, self.ok = text, status, status < 400

    def raise_for_status(self):
        if not self.ok:
            raise g.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


def test_gf_cache_validation():
    cache = _tmp()
    meta_file = cache / "gf_metadata.json"
    good = json.dumps({"familyMetadataList": FAKE_META})
    with env(GPTKIT_CACHE=str(cache)):
        meta_file.write_text('{"nope": 1}')
        saved = g.requests.get
        try:
            g.requests.get = lambda url, **k: _Resp(")]}'" + good) if url == g.GF_METADATA else _Resp("", 404)
            g._gf_meta.cache_clear()
            assert [f["family"] for f in g._gf_meta()] == [f["family"] for f in FAKE_META]
            assert json.loads(meta_file.read_text())["familyMetadataList"][0]["family"] == "Roboto"
            # corrupt cache + no network: the error names the file, the corrupt file is gone
            meta_file.write_text("{trunc")
            g.requests.get = _no_net
            g._gf_meta.cache_clear()
            try:
                g._gf_meta()
                raise AssertionError("expected RuntimeError")
            except RuntimeError as e:
                assert str(meta_file) in str(e) and "corrupt" in str(e), e
            assert not meta_file.exists()
            d = make_design(_tmp())
            meta_file.write_text("{trunc")
            g._gf_meta.cache_clear()
            run("fonts", d, expect_exit="fonts: cached Google Fonts metadata at")
            # no cache, no network, with an API key: both hosts are named, the key is never printed
            g._gf_meta.cache_clear()
            with env(GOOGLE_FONTS_API_KEY="SECRET123"):
                out = run("fonts", d, expect_exit="fonts.google.com")
            assert "www.googleapis.com" in out and "SECRET123" not in out
            # the Developer API fallback maps variants to the metadata shape
            api = json.dumps({"items": [{"family": "Inter", "category": "sans-serif", "variants": ["300", "regular", "italic", "700"],
                                         "subsets": ["latin"]}]})
            g.requests.get = lambda url, **k: _Resp(api) if url == g.GF_WEBFONTS else _Resp("", 403)
            g._gf_meta.cache_clear()
            with env(GOOGLE_FONTS_API_KEY="SECRET123"):
                fams = g._gf_meta()
            assert fams[0]["family"] == "Inter" and fams[0]["category"] == "Sans Serif" and set(fams[0]["fonts"]) == {"300", "400", "400i", "700"}
            assert json.loads(meta_file.read_text())["source"] == "webfonts/v1"
            # licence: a failed lookup is not cached; the next (successful) one is
            g.requests.get = _no_net
            lic = g._gf_license("Inter")
            assert lic["text"] is None and not (cache / "licenses" / "inter.json").exists()
            g.requests.get = lambda url, **k: _Resp('name: "Inter"\nlicense: "OFL"\n' if url.endswith("METADATA.pb") else OFL) \
                if "/ofl/inter/" in url else _Resp("", 404)
            lic = g._gf_license("Inter")
            assert lic["license"] == "OFL" and lic["text"] == OFL and (cache / "licenses" / "inter.json").exists()
            g.requests.get = _no_net
            assert g._gf_license("Inter")["text"] == OFL
            shutil.rmtree(d)
        finally:
            g.requests.get = saved
            g._gf_meta.cache_clear()
    shutil.rmtree(cache)


def test_family_flags():
    d = make_design(_tmp())
    with fake_gf():
        run("fonts", d, "--family", "Robotto", expect_exit="--family 'Robotto' is not a Google Fonts family (close: Roboto")
        run("fonts", d, "--extra", "Nope Sans", expect_exit="--extra 'Nope Sans' is not a Google Fonts family")
        out = run("fonts", d, "--candidates", "0", "--family", "lora")
    fj = json.loads((d / "fonts" / "fonts.json").read_text())
    assert fj["groups"] and all(gr["winner"] == "Lora" and gr["decided_by"] == "forced" for gr in fj["groups"]), fj["groups"]
    assert "[forced]" in out and [f["family"] for f in fj["families"]] == ["Lora"]
    shutil.rmtree(d)


def test_compare_without_metadata():
    d, _ = _fonts_fixture()
    draw_design(title_size=60).save(d / "shot.png")
    saved = g.requests.get
    g.requests.get = _no_net
    try:
        with patched(_gf_meta=_no_net, _gf_ttf=_local_ttf):
            out = run("compare", d, d / "shot.png")
    finally:
        g.requests.get = saved
    assert "(font verdict skipped: Google Fonts metadata unavailable" in out
    rep = json.loads((d / "compare" / "report.json").read_text())
    assert rep["overall"]["score"] > 0 and any("'Good evening'" in h_ for r in rep["regions"] for h_ in r["hints"])
    shutil.rmtree(d)


# ---------------------------------------------------------------- G9-5: compare hints, coverage, scale, vision, errors, gate

def test_compare_hints():
    d, _ = _fonts_fixture()
    run("tokens", d, "--prefix", "Acme", "--platforms", "compose")
    draw_design(title_size=60, body="#8A90A8", accent="#9A8BE6", chip_dy=16, avatar=False, status=48, nav=60).save(d / "dev.png")
    with fake_gf():
        out = run("compare", d, d / "dev.png", "--fail-below", "99", expect_exit="1")
    assert "gate: score" in out and "< --fail-below 99: FAIL" in out
    rep = json.loads((d / "compare" / "report.json").read_text())
    regs = {r["name"]: r for r in rep["regions"]}
    assert rep["overall"]["status_bar_px"] == 48 and rep["overall"]["extra_bottom_px"] == 60
    title = "; ".join(regs["Greeting"]["hints"])
    assert "renders like ~30." in title and "design 28sp" in title and "AcmeTypography.GreetingTitle" in title, title
    assert "try " not in title
    assert len(regs["Avatar"]["hints"]) == 1 and "missing in the implementation" in regs["Avatar"]["hints"][0], regs["Avatar"]["hints"]
    assert regs["Avatar"]["shift_dp"] == [0.0, 0.0] and "impl_bbox" not in regs["Avatar"]
    assert any("+8dp vertically" in h_ for h_ in regs["Chip"]["hints"]), regs["Chip"]["hints"]
    assert abs(regs["Start button"]["fill"][2] - 4.8) <= 0.3 and any("fill colour off" in h_ for h_ in regs["Start button"]["hints"])
    assert any("colour off by dE 8.9" in h_ for h_ in regs["Plan card"]["hints"]), regs["Plan card"]["hints"]
    # body set in Roboto instead of Inter: the Android default font is named
    draw_design(body_font="Roboto-400.ttf").save(d / "swap.png")
    with fake_gf():
        out = run("compare", d, d / "swap.png", "--out", d / "cmp_swap")
    assert "looks like Roboto 400" in out and "rather than the specified Inter 400" in out and "fontFamily not set?" in out, out
    shutil.rmtree(d)


def test_compare_coverage_and_scale():
    d = make_design(_tmp())
    draw_design(font_scale=0.8).save(d / "small.png")
    out = run("compare", d, d / "small.png", "--shot-width-dp", "411")
    assert "! the screenshot shows a ~411dp-wide screen, the design is 360dp" in out
    assert "! all text is ~" in out and "smaller" in out and "per-text size hints suppressed" in out, out
    rep = json.loads((d / "compare" / "report.json").read_text())
    assert 0.72 <= rep["overall"]["text_scale"] <= 0.86 and rep["screenshot_width_dp"] == 411
    assert not any("renders like" in h_ for r in rep["regions"] for h_ in r["hints"]), [r["hints"] for r in rep["regions"]]
    draw_design().crop((0, 0, W, 800)).save(d / "short.png")
    out = run("compare", d, d / "short.png", "--out", d / "cmp_short", "--density", "2")
    rep = json.loads((d / "compare" / "report.json").read_text()) if False else json.loads((d / "cmp_short" / "report.json").read_text())
    assert abs(rep["overall"]["coverage"] - 800 / 1440) < 0.02 and "compared 56% of the design" in out
    st = {r["name"]: r.get("status") for r in rep["regions"]}
    assert st["Start button"] == "outside screenshot" and st["Greeting"] is None and not next(
        r for r in rep["regions"] if r["name"] == "Start button")["hints"]
    assert "1 outside the screenshot" in out and "! the screenshot shows" not in out
    shutil.rmtree(d)


def test_vision_fallback():
    d = make_design(_tmp())
    spec = json.loads((d / "design_spec.json").read_text())
    spec["components"] = [c for c in spec["components"] if c["name"] != "Start button"]  # the label can only come from vision
    (d / "design_spec.json").write_text(json.dumps(spec))
    out = run("tokens", d, "--platforms", "web", "--no-snap")
    assert "missing: Button label (add --vision)" in out
    vcache = d / "text_boxes_vision.json"
    answers = {"ok": {"boxes": [{"index": 1, "text": "Start session", "x": 258, "y": 1243, "w": 203, "h": 25}]},
               "case": {"boxes": [{"index": 9, "text": "Start Session", "x": 258, "y": 1243, "w": 203, "h": 25}]},
               "offscreen": {"boxes": [{"index": 1, "text": "Start session", "x": 900, "y": 3000, "w": 203, "h": 25}]},
               "empty": {}}
    calls = []

    def fake_ask(parts, schema=None, name="result", model=None):
        calls.append(name)
        if mode == "err":
            raise g.ApiError(400, "HTTP 400 /responses [moderation_blocked]: blocked")
        return answers[mode]
    for mode in ("offscreen", "empty", "err", "case", "ok"):
        vcache.unlink(missing_ok=True)
        with patched(ask=fake_ask):
            out = run("tokens", d, "--vision", "--no-snap", "--platforms", "web", "--out", d / f"tok_{mode}")
        if mode in ("case", "ok"):
            assert "text samples located: 3/3" in out and json.loads(vcache.read_text())["Start session"] == [258, 1243, 203, 25], (mode, out)
        else:
            assert "(vision couldn't find it: check the sample text in design_spec.json)" in out and "(add --vision)" not in out, (mode, out)
            assert not vcache.exists() or "Start session" not in json.loads(vcache.read_text()), mode
        if mode == "offscreen":
            assert "outside the mockup" in out
        if mode in ("empty", "err"):
            assert "vision text location failed" in out and "continuing without it" in out
        if mode == "err":
            assert "moderation_blocked" in out
    assert calls == ["text_boxes"] * 5
    # redline accepts --vision and uses the cached box without a call
    with patched(ask=fake_ask):
        out = run("redline", d, "--vision")
    assert calls == ["text_boxes"] * 5 and "text: Start session" in json.dumps(json.loads((d / "redline" / "redline.json").read_text()))
    # compare --vision: a bad row shape or an API error never aborts the run
    draw_design().save(d / "same.png")
    for ans in ({"rows": [{"index": 1, "differences": ["avatar missing"]}, {"index": 99, "differences": ["x"]}, "garbage"]}, None):
        def cmp_ask(parts, schema=None, name="result", model=None, ans=ans):
            if ans is None:
                raise g.ApiError(429, "HTTP 429 /responses: rate limited")
            return ans
        with patched(ask=cmp_ask):
            out = run("compare", d, d / "same.png", "--vision", "--out", d / "cmp_vis")
        assert "report.json" in out and (d / "cmp_vis" / "vision_rows.png").exists()
        if ans is None:
            assert "! vision pass failed: HTTP 429" in out
        else:
            assert "vision: avatar missing" in out
    shutil.rmtree(d)


def test_compare_errors_and_gate():
    d = make_design(_tmp())
    draw_design().save(d / "same.png")
    run("compare", d, d / "nope.png", expect_exit="gptkit: screenshot not found")
    run("compare", d / "nodir", d / "same.png", expect_exit="gptkit: mockup not found")
    run("compare", d, d / "same.png", "--crop-top", "800", "--crop-bottom", "800", expect_exit="leaves nothing of the 1440px screenshot")
    run("compare", d, d / "same.png", "--fonts", d / "missing.json", expect_exit="fonts.json (--fonts) not found")
    run("compare", d, d / "same.png", "--fail-below", "101", expect_exit="between 0 and 100")
    (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "assets": []}))  # an older manifest: no screen_dp/width
    spec = json.loads((d / "design_spec.json").read_text())
    spec.pop("px_to_dp")
    (d / "design_spec.json").write_text(json.dumps(spec))
    for cmd in ("redline", "tokens", "fonts"):
        with fake_gf():
            out = run(cmd, d, *(["--platforms", "web"] if cmd == "tokens" else []))
        assert "no px_to_dp in design_spec.json" in out and "1 px = 1 dp" in out, (cmd, out)
    (d / "assets.json").write_text("{bad json")
    run("tokens", d, expect_exit="assets.json") and run("compare", d, d / "same.png", expect_exit="is not valid JSON")
    (d / "assets.json").unlink()
    (d / "design_spec.json").write_text(json.dumps({"palette": [{"name": "x", "hex": "#000"}], "typography": [{"role": "r"}]}))
    run("tokens", d, expect_exit="typography entry 1: missing 'size_px'")
    (d / "design_spec.json").write_text("[]")
    run("redline", d, expect_exit="is not a JSON object")
    (d / "design_spec.json").unlink()
    run("tokens", d, expect_exit="run `spec")
    out = run("compare", d, d / "same.png", "--fail-below", "50")  # compare works from the mockup alone
    assert "gate: score" in out and ": pass" in out
    (d / "mockup.png").unlink()
    run("compare", d, d / "same.png", expect_exit="mockup not found")
    shutil.rmtree(d)


# ---------------------------------------------------------------- G9-6: a usable M3 theme; measured spacing and insets

def test_theme_and_m3_typography():
    d, _ = _fonts_fixture()
    run("redline", d)
    out = run("tokens", d, "--prefix", "Acme", "--package", "com.acme.app.ui.theme", "--platforms", "compose,swiftui", "--verify")
    comp = d / "tokens" / "compose"
    ty, col, sh, th = [(comp / f).read_text() for f in ("Type.kt", "Color.kt", "Shape.kt", "Theme.kt")]
    assert "val AcmeM3Typography = Typography(" in ty and "headlineLarge = AcmeTypography.GreetingTitle," in ty
    assert "bodyLarge = AcmeTypography.CardBody," in ty and "labelLarge = AcmeTypography." in ty
    assert "val AcmeDarkColorScheme: ColorScheme = darkColorScheme(" in col and "val AcmeLightColorScheme: ColorScheme = lightColorScheme(" in col
    assert "TODO: verify against a light mockup" in col and "val AcmeColorScheme: ColorScheme = AcmeDarkColorScheme" in col
    assert "val AcmeShapes = Shapes(" in sh and "small = ChipShape," in sh and "medium = PlanCardShape," in sh
    assert "fun AcmeTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit)" in th
    assert "colorScheme = if (darkTheme) AcmeDarkColorScheme else AcmeLightColorScheme" in th and "shapes = AcmeShapes" in th
    assert not g._lint_tokens(sorted(comp.glob("*.kt"))) and "verify lint    : ok" in out
    assert "M3 colour roles left at baseline: primaryContainer" in out and "M3 type slots left at baseline: displayLarge" in out
    rep = json.loads((d / "tokens" / "tokens.json").read_text())
    assert rep["m3"]["m3_shapes"] == {"small": "Chip", "medium": "PlanCard"}
    sw = (d / "tokens" / "swiftui" / "AcmeTokens.swift").read_text()
    customs = [l for l in sw.splitlines() if ".custom(" in l]
    assert len(customs) == 3 and all("relativeTo: ." in l for l in customs), customs
    # HIG style is selected by fitted point size here; renderer metrics can move CardBody
    # between footnote and subheadline. The spec's bodyLarge is an M3 slot, not a HIG name.
    assert 'relativeTo: .title)' in sw and any(f'relativeTo: .{name})' in sw for name in ("footnote", "subheadline")) and 'relativeTo: .callout)' in sw
    if os.environ.get("GPTKIT_KOTLINC") or shutil.which("kotlinc"):
        assert rep["verify"]["compose"].startswith("ok (kotlinc"), rep["verify"]
    # no platform_style: the nearest slot by size is used and said so; a spec slot keeps the M3 line-height ratio
    spec = json.loads((d / "design_spec.json").read_text())
    spec["typography"][0].pop("platform_style")
    spec["typography"][1]["sample"] = "Your plan for today"  # single line: no measured pitch -> M3 bodyLarge ratio
    (d / "design_spec.json").write_text(json.dumps(spec))
    run("tokens", d, "--prefix", "Acme", "--platforms", "compose", "--out", d / "tok2")
    rep = {t["role"]: t for t in json.loads((d / "tok2" / "tokens.json").read_text())["typography"]}
    assert rep["Greeting title"]["m3_slot"] == "headlineMedium" and rep["Greeting title"]["m3_slot_source"] == "nearest by size"
    assert rep["Card body"]["line_height_source"].startswith("M3 bodyLarge ratio") and 20 <= rep["Card body"]["line_height"] <= 21.5
    assert g._m3_slot("Section title", 16)[0] == "titleMedium" and g._m3_slot("Body copy", 14)[0] == "bodyMedium"
    assert g._m3_slot("Tab label", 12)[0] == "labelMedium" and g._hig_style(17) == "headline" and g._hig_style(34) == "largeTitle"
    shutil.rmtree(d)


def test_spacing_sources_and_baseline():
    d, _ = _fonts_fixture()
    run("tokens", d, "--platforms", "compose", "--out", d / "tok_nored")
    sp = json.loads((d / "tok_nored" / "tokens.json").read_text())["spacing"]
    assert all(s_["source"] == "spec (GPT estimate)" for s_ in sp) and [s_["dp"] for s_ in sp] == [20, 20]
    assert "// GPT estimate" in (d / "tok_nored" / "compose" / "Dimens.kt").read_text()
    run("redline", d)
    rl = json.loads((d / "redline" / "redline.json").read_text())
    body = next(i for i in rl["items"] if i["name"] == "text: Your plan for today")
    asc = _font(28, 400).getmetrics()[0]
    truth_baseline_inset = (240 + asc - 200) * 0.5
    assert abs(body["baseline_inset_dp"] - truth_baseline_inset) <= 1, (body["baseline_inset_dp"], truth_baseline_inset)
    assert abs(body["layout_top_inset_dp"] - 20) <= 1 and abs(body["inset_dp"]["left"] - 20) <= 1, body
    assert body["inset_dp"]["top"] > body["layout_top_inset_dp"]  # ink sits below the layout box top
    out = run("tokens", d, "--platforms", "compose")
    sp = {s_["name"]: s_ for s_ in json.loads((d / "tokens" / "tokens.json").read_text())["spacing"]}
    # the card's content starts 16 dp in (the chip) and the text 20 dp in: the smallest measured inset is the padding
    assert sp["card padding"]["source"].startswith("redline: Plan card") and 15.5 <= sp["card padding"]["dp"] <= 21, sp
    assert sp["screen edge to card"]["source"].startswith("redline: Plan card left of screen edge") and sp["screen edge to card"]["dp"] == 20
    dims = (d / "tokens" / "compose" / "Dimens.kt").read_text()
    assert "// redline: Plan card" in dims and "GPT estimate" not in dims
    shutil.rmtree(d)


# ---------------------------------------------------------------- G9-9: portability extras

def test_dtcg_and_extras():
    d, out = _fonts_fixture()
    assert "WOFF2 skipped" in out or any((d / "fonts" / "ttf").glob("*.woff2"))
    run("redline", d)
    out = run("tokens", d, "--prefix", "Acme", "--dtcg", "--verify")
    tk = d / "tokens"
    doc = json.loads((tk / "tokens.tokens.json").read_text())
    assert doc["color"]["screen-background"] == {"$type": "color", "$value": TRUTH["bg"],
                                                 "$description": doc["color"]["screen-background"]["$description"]}
    assert "snapped to the mockup" in doc["color"]["screen-background"]["$description"]
    ty = doc["typography"]["card-body"]
    assert ty["$type"] == "typography" and ty["$value"]["fontFamily"] == "Inter" and ty["$value"]["fontWeight"] == 400
    assert ty["$value"]["fontSize"].endswith("px") and ty["$value"]["lineHeight"].endswith("px") and ty["$value"]["letterSpacing"].endswith("em")
    assert doc["dimension"]["radius"]["plan-card"]["$value"] == "16px" and "redline" in doc["dimension"]["radius"]["plan-card"]["$description"]
    assert doc["dimension"]["spacing"]["card-padding"]["$type"] == "dimension"
    theme = (tk / "web" / "tailwind.theme.css").read_text()
    assert theme.startswith("/* Tailwind v4") and "@import \"tailwindcss\"" in theme.splitlines()[0] and "@import" not in theme.split("*/", 1)[1]
    assert "@theme {" in theme and "--color-screen-background: #0E1428;" in theme
    assert not (tk / "web" / "tailwind.config.js").exists() and "verify tailwind: ok (node require)" in out
    prev = (tk / "compose" / "Preview.kt").read_text()
    assert "@Preview(name = \"Acme 1:1\", widthDp = 360, showBackground = true)" in prev and "AcmeTheme {" in prev
    assert not g._lint_tokens(sorted((tk / "compose").glob("*.kt")))
    cs = json.loads((tk / "swiftui" / "Colors.xcassets" / "screenBackground.colorset" / "Contents.json").read_text())
    comp = cs["colors"][0]["color"]["components"]
    assert (comp["red"], comp["green"], comp["blue"], comp["alpha"]) == ("0x0E", "0x14", "0x28", "1.000")
    assert (tk / "swiftui" / "Colors.xcassets" / "Contents.json").exists()
    draw_design().save(d / "same.png")
    out = run("compare", d, d / "same.png")
    assert "SSIM (7px box)" in out and json.loads((d / "compare" / "report.json").read_text())["overall"]["ssim_method"].startswith("SSIM (7px box")
    shutil.rmtree(d)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok {name} ({time.time() - t0:.1f}s)")
