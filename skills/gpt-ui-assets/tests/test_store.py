"""Store screenshots, offline: python3 tests/test_store.py

capture: simctl / adb are mocked at gptkit._sh; the status bar override is always cleared (also when the screenshot
fails), Android leaves demo mode, a folder import refuses image-model sizes.
plan / art / research: the API is mocked at gptkit.ask / generate / edit and the network at gptkit._fetch.
build + check: a real set is rendered from synthetic captures (skipped without Playwright + Chrome).
"""
import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def _run(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    K._store_parsers(sub)
    a = ap.parse_args(["store", *argv])
    a.fn(a)


def _screen(w=1320, h=2868, hue=(70, 140, 90), label="A"):
    """A synthetic app screen: a status bar, a header band, a grid of tiles and a button."""
    im = Image.new("RGB", (w, h), (246, 238, 226))
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, w, 190), fill=(250, 246, 240))
    dr.rounded_rectangle((60, 260, w - 60, 460), radius=40, fill=hue)
    for r in range(5):
        for c in range(5):
            x, y = 120 + c * 220, 700 + r * 220
            dr.rounded_rectangle((x, y, x + 190, y + 190), radius=24, fill=((c * 50) % 255, (r * 60) % 255, 150))
    dr.rounded_rectangle((200, h - 400, w - 200, h - 260), radius=70, fill=(40, 120, 60))
    dr.text((w // 2, 360), label, fill="white")
    return im


def _storyboard(d, slots):
    sb = {"app": "Tiles", "category": "Games / Puzzle", "voice": "calm",
          "brand": {"colors": ["#E9C9A0", "#D9A877"], "ink_dark": "#2E2018", "ink_light": "#FFF6EA"},
          "slots": slots, "feature": {"subtitle": "Calm tile puzzles", "plate_brief": "a soft dawn"},
          "privacy": [], "notes": [], "claims": ""}
    (Path(d) / "storyboard.json").write_text(json.dumps(sb))
    return sb


class _FakeDevices:
    """Records every simctl/adb call and answers like a booted simulator and one Android tablet."""

    def __init__(self, fail_screenshot=False):
        self.calls, self.fail = [], fail_screenshot

    def __call__(self, args, binary=False, timeout=60):
        self.calls.append(args)
        if args[:4] == ["xcrun", "simctl", "list", "devices"]:
            return json.dumps({"devices": {"iOS-26": [{"udid": "U1", "name": "iPhone 17 Pro Max", "state": "Booted"}]}})
        if args[:4] == ["xcrun", "simctl", "io", "U1"]:
            if self.fail:
                raise RuntimeError("store capture: xcrun simctl io U1 failed: boom")
            _screen().save(args[-1])
            return ""
        if args == ["adb", "devices"]:
            return "List of devices attached\nEMU1\tdevice\n"
        if args[-3:] == ["exec-out", "screencap", "-p"]:
            return _png(_screen(2560, 1600).convert("RGBA"))
        if args[-2:] == ["wm", "density"]:
            return "Physical density: 320\n"
        if args[-2:] == ["dumpsys", "window"]:
            return "  InsetsSource id=1 type=statusBars frame=[0,0][2560,64] visible=true\n"
        return ""


def test_store_names_shadow_no_other_module():
    """gptkit re-exports every module's names, so a store helper named like another module's would replace it."""
    own = {k for k in vars(K.store) if not k.startswith("__")}
    bad = [(m.__name__, k) for m in K.MODULES if m is not K.store
           for k in own & set(vars(m)) if vars(m)[k] is not vars(K.store)[k]]
    assert not bad, bad


def test_capture_ios_overrides_and_clears_the_status_bar():
    fake = _FakeDevices()
    K._sh = fake
    with tempfile.TemporaryDirectory() as d:
        _run(["capture", d, "--name", "home", "--platform", "ios"])
        cmds = [" ".join(c[:4]) for c in fake.calls]
        assert "xcrun simctl status_bar U1" in cmds
        over = next(c for c in fake.calls if "override" in c)
        assert over[over.index("--time") + 1] == "9:41" and over[over.index("--batteryState") + 1] == "discharging"
        assert fake.calls[-1][-1] == "clear"
        caps = json.loads((Path(d) / "captures.json").read_text())["captures"]
        assert caps[0]["platform"] == "ios" and caps[0]["device"] == "phone" and caps[0]["size"] == [1320, 2868]
        assert Image.open(Path(d) / caps[0]["file"]).mode == "RGB"


def test_capture_clears_the_status_bar_when_the_screenshot_fails():
    fake = _FakeDevices(fail_screenshot=True)
    K._sh = fake
    with tempfile.TemporaryDirectory() as d:
        try:
            _run(["capture", d, "--name", "home", "--platform", "ios"])
            raise AssertionError("a failed screenshot must raise")
        except RuntimeError as e:
            assert "boom" in str(e)
        assert fake.calls[-1][-1] == "clear"


def test_capture_android_uses_demo_mode_and_detects_a_tablet():
    fake = _FakeDevices()
    K._sh = fake
    with tempfile.TemporaryDirectory() as d:
        _run(["capture", d, "--name", "home", "--platform", "android"])
        demo = [c for c in fake.calls if "com.android.systemui.demo" in c]
        assert demo[0][-1] == "enter" and demo[-1][-1] == "exit"
        rec = json.loads((Path(d) / "captures.json").read_text())["captures"][0]
        assert rec["device"] == "tablet" and rec["orientation"] == "landscape" and rec["status_bar"] == 64


def test_import_refuses_image_model_sizes():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        _screen().save(src / "Home Screen.png")
        Image.new("RGB", (1024, 1536), "white").save(src / "mockup.png")
        _run(["capture", str(Path(d) / "p"), "--from", str(src)])
        caps = json.loads((Path(d) / "p" / "captures.json").read_text())["captures"]
        assert [c["name"] for c in caps] == ["home_screen"]


def test_lint_flags_prices_platforms_play_words_and_alt():
    sb = {"slots": [
        {"id": "a", "capture": "x", "caption": "Only $4.99 a month", "sub": "", "alt": "ok"},
        {"id": "b", "capture": "x", "caption": "Works on Android too", "sub": "", "alt": "Image of the board"},
        {"id": "c", "capture": "x", "caption": "The best word game", "caption_play": "The #1 new game, download now",
         "sub": "", "alt": "x" * 141},
        {"id": "d", "capture": "x", "caption": "Award-winning puzzles", "sub": "", "alt": "The board"}]}
    issues = K._lint(sb, claims="")
    by = {(lv, sid) for lv, sid, _ in issues}
    assert ("FAIL", "a") in by and ("FAIL", "b") in by and ("FAIL", "c") in by
    msgs = " ".join(m for _, sid, m in issues if sid == "c")
    assert "'#1'" in msgs and "'new'" in msgs and "download" in msgs and "141 chars" in msgs
    assert any(sid == "b" and "image of" in m for _, sid, m in issues)
    assert not any(sid == "d" and "Award" in m for _, sid, m in K._lint(sb, claims="Award-winning puzzles"))


def test_lint_warns_on_callouts_at_the_screen_ends():
    slot = lambda sid, box: {"id": sid, "capture": "x", "caption": "Fast", "sub": "", "alt": "ok",
                             "calls": [{"capture": "x", "box": box}]}
    sb = {"slots": [slot("top", [0, 0.1, 1, 0.3]), slot("bottom", [0, 0.87, 1, 0.96]), slot("tall", [0, 0.2, 1, 0.7]),
                    slot("mid", [0, 0.43, 1, 0.84])]}
    warned = {sid for lv, sid, m in K._lint(sb) if "callout box" in m}
    assert warned == {"top", "bottom", "tall"}


def test_plan_normalises_boxes_and_captures():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        _screen().save(src / "home.png")
        _screen(label="B").save(src / "level.png")
        _run(["capture", d, "--from", str(src)])
        seen = {}

        def fake_ask(parts, schema=None, name="result", model=None):
            seen["images"] = sum(1 for p in parts if not isinstance(p, str))
            seen["prompt"] = parts[0]
            return {"app": "Tiles", "category": "Games", "voice": "calm",
                    "brand": {"colors": ["#ffeedd"], "ink_dark": "#222222", "ink_light": "#ffffff", "head_font": "Nunito",
                              "body_font": "DM Sans", "art_style": "clay", "lettering_style": "clay"},
                    "slots": [{"id": "Hero Slot", "capture": "home", "caption": "Calm tile puzzles", "sub": "One a day.",
                               "caption_play": "", "sub_play": "", "arch": "callout", "ink": "dark",
                               "calls": [{"capture": "home", "box": [0.05, 0.09, 0.95, 0.16]}], "cards": [],
                               "plate_brief": "dawn", "alt": "The board", "why": "hook"},
                              {"id": "two", "capture": "nope", "caption": "Levels", "sub": "", "caption_play": "",
                               "sub_play": "", "arch": "fan", "ink": "light", "calls": [],
                               "cards": [{"capture": "home", "box": [0, 0, 1, 1]}], "plate_brief": "", "alt": "",
                               "why": ""}],
                    "feature": {"subtitle": "Calm", "plate_brief": "dawn"}, "privacy": [], "notes": []}

        K.ask = fake_ask
        _run(["plan", d, "--app", "Tiles: calm puzzles", "--claims", "Apple Design Award 2026"])
        sb = json.loads((Path(d) / "storyboard.json").read_text())
        assert seen["images"] == 2 and "Apple Design Award 2026" in seen["prompt"]
        s0, s1 = sb["slots"]
        assert s0["id"] == "hero_slot" and s0["calls"][0]["box"] == [0.05, 0.09, 0.95, 0.16]
        assert s1["capture"] == "home" and s1["arch"] == "framed"  # unknown capture fixed, a 1-card fan is framed
        assert (Path(d) / "storyboard.md").exists()


def test_art_rerolls_lettering_until_ocr_matches():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        _screen().save(src / "home.png")
        _run(["capture", d, "--from", str(src)])
        _storyboard(d, [{"id": "hero", "capture": "home", "caption": "Calm tile puzzles", "sub": "",
                         "plate_brief": "a soft dawn", "alt": "x"}])
        letter = Image.new("RGBA", (1536, 1024), (0, 0, 0, 0))
        ImageDraw.Draw(letter).rectangle((300, 400, 1200, 600), fill=(200, 120, 60, 255))
        K.generate = lambda prompt, **kw: (_png(letter), {})
        K.edit = lambda ref, prompt, **kw: (_png(Image.new("RGB", (851, 1848), (230, 200, 160))), {})
        answers = iter([{"has_text": False, "what": ""}, {"text": "CALM TILE PUZZLE"}, {"text": "calm tile\npuzzles"}])
        K.ask = lambda parts, schema=None, name="result", model=None: next(answers)
        _run(["art", d, "--lettering", "--jobs", "1"])
        art = json.loads((Path(d) / "art" / "art.json").read_text())
        assert art["letter_hero"]["ok"] and len(art["letter_hero"]["tries"]) == 2
        assert Image.open(Path(d) / "art" / "letter_hero.png").size == (901, 201)  # trimmed to the alpha bbox
        assert art["plate_hero"]["flags"] == [] and (Path(d) / "art" / "plate_hero.png").exists()


def test_idea_check_reads_the_shots_blind_then_compares():
    with tempfile.TemporaryDirectory() as d:
        slots = [{"id": f"s{i}", "capture": "home", "caption": cap, "sub": "", "alt": "x"}
                 for i, cap in enumerate(("Sleep sounds for babies", "Forty-two sounds", "Mix your own"), 1)]
        sb = _storyboard(d, slots)
        sb["promise"] = "Baby sleep sounds that help little ones settle"
        (Path(d) / "storyboard.json").write_text(json.dumps(sb))
        shots = []
        for i, s in enumerate(slots, 1):
            f = Path(d) / "build" / "ios-phone" / f"{i:02d}_{s['id']}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1206, 2622), (200, 210, 230)).save(f)
            (Path(d) / "build" / "bare" / "ios-phone").mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (420, 913), (200, 210, 230)).save(Path(d) / "build" / "bare" / "ios-phone" / f"{i:02d}_{s['id']}.jpg")
            shots.append({"file": str(f.relative_to(d)), "slot": s["id"], "size": [1206, 2622], "alt": "x"})
        (Path(d) / "build" / "report.json").write_text(json.dumps({"targets": {"ios-phone": {"shots": shots}}, "notes": []}))
        sent = []
        answers = iter([{"app": "Rain sounds to relax", "audience": "adults", "unclear": False,
                         "slots": [{"idea": "rain"}, {"idea": "a sound library"}, {"idea": "mixing"}]},
                        {"app_match": False, "app_note": "misses babies",
                         "slots": [{"match": False, "note": "no baby"}, {"match": True, "note": ""}, {"match": True, "note": ""}]}])
        K.ask = lambda parts, schema=None, name="result", model=None: sent.append(parts) or next(answers)
        _run(["check", d, "--idea"])
        blind = " ".join(p for p in sent[0] if isinstance(p, str))
        assert sum(not isinstance(p, str) for p in sent[0]) == 3            # the three captionless shots
        assert not any(c["caption"].lower() in blind.lower() for c in slots) and "babies" not in blind  # blind
        assert "Baby sleep sounds" in sent[1][0] and "Rain sounds to relax" in sent[1][0]
        res = json.loads((Path(d) / "build" / "check.json").read_text())
        ideas = [(x["level"], x["where"]) for x in res if x["where"].startswith("idea")]
        assert ideas == [("WARN", "idea"), ("WARN", "idea s1")], ideas


def test_research_downloads_and_sheets_without_ai():
    shot = _png(_screen(600, 1300).convert("RGB"))

    def fake_fetch(url, timeout=30):
        if "itunes.apple.com/search" in url:
            return json.dumps({"results": [{"trackId": 1, "trackName": "Words", "primaryGenreName": "Games",
                                            "userRatingCount": 10, "screenshotUrls": ["u1", "u2", "u3", "u4"]},
                                           {"trackId": 2, "trackName": "No shots", "screenshotUrls": []}]}).encode()
        return shot

    K._fetch = fake_fetch
    with tempfile.TemporaryDirectory() as d:
        _run(["research", d, "--term", "word game", "--no-ai"])
        apps = json.loads((Path(d) / "research" / "apps.json").read_text())
        assert [x["shots"] for x in apps] == [3]
        assert (Path(d) / "research" / "competitors_1.jpg").exists()


def test_build_refuses_a_device_frame_on_play():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        _screen().save(src / "home.png")
        _run(["capture", d, "--from", str(src)])
        _storyboard(d, [{"id": "hero", "capture": "home", "caption": "Calm", "alt": "x"}])
        try:
            _run(["build", d, "--frame", "device"])
            raise AssertionError("an iPhone frame on Play must be refused")
        except ValueError as e:
            assert "Play" in str(e)


def test_check_fails_wrong_size_and_alpha():
    with tempfile.TemporaryDirectory() as d:
        _storyboard(d, [{"id": "hero", "capture": "home", "caption": "Calm", "alt": "x"}])
        b = Path(d) / "build" / "ios-phone"
        b.mkdir(parents=True)
        Image.new("RGBA", (1320, 2868), (200, 0, 0, 255)).save(b / "01_hero.png")
        (Path(d) / "build" / "report.json").write_text(json.dumps({"targets": {"ios-phone": {"shots": [
            {"file": "build/ios-phone/01_hero.png", "slot": "hero", "size": [1320, 2868], "alt": "x",
             "contrast": {"h1": 2.1, "sub": 5}, "fidelity": {"mean": 4.0, "p99": 30}, "h1_px": 20}]}}}))
        try:
            _run(["check", d])
            raise AssertionError("check must exit 1 on a FAIL")
        except SystemExit as e:
            assert e.code == 1
        res = json.loads((Path(d) / "build" / "check.json").read_text())
        msgs = " ".join(x["msg"] for x in res if x["level"] == "FAIL")
        assert "needs 1206x2622" in msgs and "RGBA" in msgs and "drifted" in msgs and "headline contrast" in msgs


def test_headlines_share_the_largest_size_that_fits():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("  skip build: no playwright")
        return
    if not K._chrome():
        print("  skip build: no Chrome")
        return
    sizes = {}
    for name, caps in (("short", ["Calm tiles", "Daily puzzles"]),
                       ("long", ["Calm tiles", "Relax with a gentle puzzle every single evening"])):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            _screen().save(src / "home.png")
            _screen(hue=(160, 80, 60), label="B").save(src / "level.png")
            _run(["capture", d, "--from", str(src)])
            _storyboard(d, [{"id": f"s{i}", "capture": c, "caption": cap, "sub": "One small puzzle a day, every day.",
                             "ink": "dark", "alt": "Board"} for i, (c, cap) in enumerate(zip(("home", "level"), caps))])
            _run(["build", d, "--no-webfonts", "--no-pill"])
            rep = json.loads((Path(d) / "build" / "report.json").read_text())
            for key, t in rep["targets"].items():
                fonts = {round(s["h1_px"], 1) for s in t["shots"]}
                assert len(fonts) == 1, (name, key, fonts)  # one headline size across the set
                sizes[name, key] = fonts.pop()
                for s in t["shots"]:
                    rep_tag = s["tagline_share"]
                    assert key != "play-phone" or rep_tag <= K.TAGLINE_FIT + 0.005, (name, s)
    for key in ("ios-phone", "play-phone"):
        assert sizes["short", key] > sizes["long", key] * 1.15, sizes  # short copy earns a bigger headline


def test_build_and_check_render_a_real_set():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("  skip build: no playwright")
        return
    if not K._chrome():
        print("  skip build: no Chrome")
        return
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        _screen().save(src / "home.png")
        _screen(hue=(160, 80, 60), label="B").save(src / "level.png")
        _screen(hue=(60, 90, 170), label="C").save(src / "shop.png")
        _run(["capture", d, "--from", str(src)])
        _storyboard(d, [
            {"id": "hero", "capture": "home", "caption": "Calm tile puzzles", "sub": "One small puzzle a day.",
             "arch": "callout", "ink": "dark", "calls": [{"capture": "home", "box": [0.05, 0.09, 0.95, 0.16]}],
             "alt": "The daily board"},
            {"id": "modes", "capture": "level", "caption": "Three ways to play", "sub": "", "arch": "fan", "ink": "light",
             "cards": [{"capture": c, "box": [0.1, 0.2, 0.9, 0.6]} for c in ("level", "home", "shop")],
             "alt": "Three modes"},
            {"id": "shop", "capture": "shop", "caption": "Unlock themes", "sub": "Earned by playing.", "ink": "dark",
             "alt": "Themes"}])
        _run(["build", d, "--no-webfonts"])
        rep = json.loads((Path(d) / "build" / "report.json").read_text())
        assert set(rep["targets"]) == {"ios-phone", "play-phone"}
        for key, want in (("ios-phone", (1206, 2622)), ("play-phone", (1440, 2560))):
            shots = rep["targets"][key]["shots"]
            assert [Path(s["file"]).name for s in shots] == ["01_hero.png", "02_modes.png", "03_shop.png"]
            for s in shots:
                im = Image.open(Path(d) / s["file"])
                assert im.size == want and im.mode == "RGB"
                if s["fidelity"]:
                    assert s["fidelity"]["mean"] < 2.5, (key, s)
        light = rep["targets"]["ios-phone"]["shots"][1]["contrast"]
        assert light["h1"] >= 4.5 or light.get("pill"), light  # light ink on a light gradient is moved onto a pill
        assert (Path(d) / "build" / "ios-phone" / "search_row.jpg").exists()
        assert (Path(d) / "build" / "play-phone" / "alt.txt").read_text().startswith("01_hero.png\tThe daily board")
        Image.new("RGB", (1206, 2622)).save(Path(d) / "build" / "ios-phone" / "09_old.png")
        try:
            _run(["check", d])
        except SystemExit as e:
            raise AssertionError(f"check failed on a clean set: {(Path(d) / 'build' / 'check.json').read_text()}") from e
        res = json.loads((Path(d) / "build" / "check.json").read_text())
        assert any("09_old.png is not in this build" in x["msg"] for x in res), res
        _run(["feature", d, "--no-webfonts"])
        assert Image.open(Path(d) / "build" / "play-feature" / "feature.png").size == (1024, 500)


def test_floats_render_and_warn_when_they_crowd():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("  skip floats: no playwright")
        return
    if not K._chrome():
        print("  skip floats: no Chrome")
        return
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ext:
        src = Path(d) / "src"
        src.mkdir()
        _screen().save(src / "home.png")
        _run(["capture", d, "--from", str(src)])
        star = Path(ext) / "star.png"
        Image.new("RGBA", (256, 256), (255, 200, 40, 255)).save(star)
        _storyboard(d, [
            {"id": "edge", "capture": "home", "caption": "Calm tile puzzles", "ink": "dark", "alt": "Board",
             "floats": [{"src": str(star), "x": 0.9, "y": 0.7, "w": 0.12, "layer": "back"}]},
            {"id": "crowd", "capture": "home", "caption": "Calm tile puzzles", "ink": "dark", "alt": "Board",
             "floats": [{"src": str(star), "x": 0.5, "y": 0.06, "w": 0.15, "layer": "front"},
                        {"src": str(star), "x": 0.5, "y": 0.6, "w": 0.4, "layer": "front"}],
             "floats_play": [{"src": str(star), "x": 0.5, "y": 0.6, "w": 0.4, "layer": "front"}]}])
        _run(["build", d, "--no-webfonts"])
        assert (Path(d) / "floats" / "star.png").exists()
        rep = json.loads((Path(d) / "build" / "report.json").read_text())
        ios, play = rep["targets"]["ios-phone"]["shots"], rep["targets"]["play-phone"]["shots"]
        assert ios[0]["floats"] == 1 and not ios[0]["float_on_caption"] and ios[0]["float_cover"] == 0
        assert ios[1]["floats"] == 2 and ios[1]["float_on_caption"] and ios[1]["float_cover"] > 0.03
        assert play[1]["floats"] == 1 and not play[1]["float_on_caption"], "Play must read floats_play first"
        _run(["check", d])
        msgs = [x["msg"] for x in json.loads((Path(d) / "build" / "check.json").read_text())]
        assert any("touches the caption" in m for m in msgs) and any("of the app screen" in m for m in msgs), msgs
        _run(["feature", d, "--no-webfonts", "--title", "Tiles", "--icon", str(star)])
        assert (Path(d) / "build" / "feature_icon.png").exists()
        assert Image.open(Path(d) / "build" / "play-feature" / "feature.png").size == (1024, 500)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        saved = K._sh, K._fetch, K.ask, K.generate, K.edit
        try:
            t()
            print(f"ok   {t.__name__}")
        finally:
            K._sh, K._fetch, K.ask, K.generate, K.edit = saved
    print(f"{len(tests)} store tests passed")
