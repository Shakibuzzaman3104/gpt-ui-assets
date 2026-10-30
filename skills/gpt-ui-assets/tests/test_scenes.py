"""Painted store scenes, offline: python3 tests/test_scenes.py."""
import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")


def _run(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    K._store_parsers(sub)
    a = ap.parse_args(["store", *argv])
    a.fn(a)


def _png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _scene(w=1320, h=2868):
    q = np.float32([[310, 1000], [1020, 900], [1090, 2420], [215, 2520]])
    q *= [w / 1320, h / 2868]
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    radius = max(1, round(40 * w / 1320))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = (70, 90, 120)
    bg[..., 0] += np.linspace(0, 25, h, dtype=np.uint8)[:, None]
    bg[m > 0] = (0, 255, 0)
    cv2.circle(bg, tuple(q[2].astype(int)), round(85 * w / 1320), (240, 110, 20), -1)
    return Image.fromarray(bg), q


def test_quad_and_key():
    im, truth = _scene()
    alpha = K.green_alpha(np.asarray(im))
    q = K.quad_from_mask(alpha > 0.5)
    assert np.max(np.linalg.norm(q - truth, axis=1)) < 4, (q, truth)
    capture = Image.new("RGB", (300, 600), (100, 80, 160))
    result, _, _ = K.key_capture(im, [capture])
    a = np.asarray(result).astype(int)
    region = np.zeros(alpha.shape, np.uint8)
    cv2.fillConvexPoly(region, q.astype(np.int32), 1)
    assert np.max((a[..., 1] - np.maximum(a[..., 0], a[..., 2]))[region > 0]) <= 30
    x, y = truth[2].astype(int)
    assert tuple(a[y, x]) == (240, 110, 20)


def test_small_phone():
    big, _ = _scene()
    fit, error, flags, screen = K.scenes._check(big, 1, 0.22, [0.372])
    assert error is None and "SMALL-PHONE" not in flags and screen >= 0.42
    small = Image.new("RGB", big.size, (70, 90, 120))
    small.paste(big.resize((660, 1434)), (330, 700))
    fit, error, flags, screen = K.scenes._check(small, 1, 0.22, [0.372])
    assert error is None and "SMALL-PHONE" in flags and 0.25 < screen < 0.35


def test_occlusion():
    im, _ = _scene()
    assert "OCCLUDED" not in K.scenes._check(im, 1, 0.22, [0.372])[2]
    pixels = np.array(im)
    cv2.rectangle(pixels, (400, 2200), (900, 2400), (240, 110, 20), -1)
    assert "OCCLUDED" in K.scenes._check(Image.fromarray(pixels), 1, 0.22, [0.372])[2]


def test_mixed_compose():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        slots = [{"id": sid, "capture": "home", "caption": "A calm day", **({"scene_brief": "hills"} if sid != "flat" else {})}
                 for sid in ("hero", "flat", "last")]
        (d / "storyboard.json").write_text(json.dumps({"scenes": {"style": "painted"}, "slots": slots}))
        cap = d / "captures/ios-phone/home.png"
        cap.parent.mkdir(parents=True)
        Image.new("RGB", (372, 1000), (130, 80, 170)).save(cap)
        (d / "captures.json").write_text(json.dumps({"captures": [{"name": "home", "platform": "ios", "device": "phone",
            "orientation": "portrait", "file": str(cap.relative_to(d))}]}))
        scene, _ = _scene(853, 1843)
        K.generate = lambda prompt, **kw: (_png(scene), {})
        K._gf_ttf = lambda *args: None
        _run(["scene", temp, "--variants", "1"])
        targets = {}
        for key, size in (("ios-phone", (1206, 2622)), ("ios-phone-large", (1320, 2868))):
            path = d / "build" / key / "02_flat.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", size, (70, 90, 120)).save(path)
            targets[key] = {"shots": [{"slot": "flat", "file": str(path.relative_to(d)), "screen_box": {"x": 100, "y": 800, "w": 600, "h": 1200}}]}
        report = d / "build/report.json"
        report.write_text(json.dumps({"no_caption": True, "targets": targets}))
        _run(["compose", temp, "--flat", temp, "--targets", "ios-phone,ios-phone-large"])
        base = d / "scenes/main/build"
        result = json.loads((base / "report.json").read_text())
        for key in ("ios-phone", "ios-phone-large"):
            shots = result[key]
            assert list(shots) == ["hero", "flat", "last"]
            assert [shots[s["id"]]["kind"] for s in slots] == ["painted", "flat", "painted"]
            assert [Path(shots[s["id"]]["file"]).name for s in slots] == ["01_hero.png", "02_flat.png", "03_last.png"]
            assert all(Image.open(base / shots[s["id"]]["file"]).size == K.SCENE_TARGETS[key] for s in slots)
            assert (base / key / "sheet.jpg").exists()
        try:
            _run(["compose", temp, "--targets", "play-phone"])
            raise AssertionError("Play scene accepted")
        except ValueError as e:
            assert "App Store only" in str(e)
        _run(["compose", temp, "--flat", temp])
        assert set(json.loads((base / "report.json").read_text())) == {"ios-phone", "ios-phone-large"}
        report.write_text(json.dumps({"no_caption": False, "targets": targets}))
        try:
            _run(["compose", temp, "--flat", temp])
            raise AssertionError("captioned build accepted")
        except ValueError as e:
            assert "--no-caption" in str(e)


def test_perspective():
    _, quad = _scene()
    check = K.scenes.screen_perspective
    assert check(quad, 1320, 2868, 0.372)["ok"]
    rectangle = np.array([[330, 500], [990, 500], [990, 1934], [330, 1934]])
    result = check(rectangle, 1320, 2868, 1320 / 2868)
    assert result["ok"] and result["focal"] is None and result["skew"] == 0
    aspect = 1320 / 2868
    points = np.array([[0, 0, 0], [aspect, 0, 0], [aspect, 1, 0], [0, 1, 0]], dtype=float)
    camera = np.array([[2000., 0, 660], [0, 2000., 1434], [0, 0, 1]])
    projected, _ = cv2.projectPoints(points, np.array([0.15, 0.35, 0.08]),
                                     np.array([-0.2, -0.5, 2.]), camera, None)
    result = check(projected.reshape(4, 2), 1320, 2868, aspect)
    assert result["ok"] and abs(result["aspect"] - 1) < 1e-8
    assert abs(result["focal"] - 2000 / 2868) < 1e-8
    shear = np.array([[300, 900], [1000, 1100], [1000, 2500], [300, 2300]])
    assert not check(shear, 1320, 2868, 1320 / 2868)["ok"]
    pixels = np.zeros((2868, 1320, 3), np.uint8)
    cv2.fillConvexPoly(pixels, shear.astype(np.int32), (0, 255, 0))
    fit, error, flags, _ = K.scenes._check(Image.fromarray(pixels), 1, 0.22)
    assert fit > 0.95 and "perspective" in error and "BAD-PERSPECTIVE" in flags


def test_compose_bad_perspective():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        slot = {"id": "hero", "capture": "home", "caption": "A calm day"}
        (d / "storyboard.json").write_text(json.dumps({"slots": [slot]}))
        source = d / "scenes" / "main"
        source.mkdir(parents=True)
        shear = np.array([[300, 900], [1000, 1100], [1000, 2500], [300, 2300]])
        pixels = np.zeros((2868, 1320, 3), np.uint8)
        cv2.fillConvexPoly(pixels, shear.astype(np.int32), (0, 255, 0))
        Image.fromarray(pixels).save(source / "hero_v1.png")
        (source / "scenes.json").write_text(json.dumps({"slots": {"hero": {"pick": "hero_v1.png"}}}))
        captures = d / "captures"
        captures.mkdir()
        Image.new("RGB", (132, 286)).save(captures / "home.png")
        _run(["compose", temp, "--targets", "ios-phone-large", "--captures", str(captures)])
        build = source / "build"
        shot = json.loads((build / "report.json").read_text())["ios-phone-large"]["hero"]
        assert shot["file"] is None and "BAD-PERSPECTIVE" in shot["flags"]
        assert "perspective" in shot["notes"][0]
        assert not (build / "ios-phone-large" / "01_hero.png").exists()


def test_cut_edge():
    m = np.zeros((500, 300), np.uint8)
    cv2.rectangle(m, (0, 80), (240, 440), 1, -1)
    try:
        K.quad_from_mask(m)
        raise AssertionError("cut edge accepted")
    except ValueError as e:
        assert "cut" in str(e)


def _ref_project(d):
    slots = [{"id": sid, "capture": sid, "caption": "A calm day", "scene_brief": "soft hills"}
             for sid in ("hero", "next")]
    (d / "storyboard.json").write_text(json.dumps({"scenes": {"style": "painted", "style_ref": "ref.png"},
                                                  "slots": slots}))
    ref, _ = _scene(853, 1843)
    ref.save(d / "ref.png")
    return ref


def test_ref_prompt():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        ref = _ref_project(d)
        scene = Image.new("RGB", ref.size, (70, 90, 120))
        scene.paste(ref.resize((600, 1300)), (100, 300))
        prompts = []
        K.edit = lambda ref, prompt, **kw: (prompts.append(prompt), (_png(scene), {}))[1]
        _run(["scene", temp, "--variants", "2"])
        assert len(prompts) == 4
        assert all(p.startswith(K.scenes.REF_PROMPT) and "at most one corner" in p for p in prompts)
        manifest = json.loads((d / "scenes/main/scenes.json").read_text())
        assert all("COPIES-REF" not in v["flags"] for e in manifest["slots"].values() for v in e["variants"])


def test_copies_ref():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        ref = _ref_project(d)
        K.edit = lambda source, prompt, **kw: (_png(ref), {})
        _run(["scene", temp, "--variants", "2"])
        manifest = json.loads((d / "scenes/main/scenes.json").read_text())
        for entry in manifest["slots"].values():
            assert entry["pick"] is None
            assert all("copies the style reference" in v["error"] and "COPIES-REF" in v["flags"]
                       for v in entry["variants"])


def test_end_to_end():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        slots = [{"id": "hero", "capture": "home", "caption": "A calm day", "sub": "Play today", "scene_brief": "soft hills",
                  "scene_view": "front",
                  "scene_lines": [{"text": "A calm", "size": 0.8, "color": "color"}, {"text": "day", "color": "accent"}]},
                 {"id": "next", "capture": "next", "caption": "More to explore", "sub": "", "scene_brief": "soft sky"}]
        sb = {"brand": {}, "scenes": {"style": "painted"}, "slots": slots}
        (d / "storyboard.json").write_text(json.dumps(sb))
        caps = []
        for name in ("home", "next"):
            p = d / "captures" / "ios-phone" / f"{name}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (372, 1000), (130, 80, 170)).save(p)
            caps.append({"name": name, "platform": "ios", "device": "phone", "orientation": "portrait", "file": str(p.relative_to(d))})
        (d / "captures.json").write_text(json.dumps({"captures": caps}))
        scene, _ = _scene(853, 1843)
        prompts = []
        K.generate = lambda prompt, **kw: (prompts.append(prompt), (_png(scene), {}))[1]
        def darken(ref, prompt, **kw):
            im = Image.open(io.BytesIO(ref)).convert("RGB").resize(scene.size)
            a = np.array(im)
            green = K.green_alpha(a) > 0.5
            a[~green] = (a[~green].astype(np.float32) * 0.65).astype(np.uint8)
            a[green] = (0, 255, 0)
            return _png(Image.fromarray(a)), {}
        K.edit = darken
        K._gf_ttf = lambda *args: None
        _run(["scene", temp, "--variants", "2"])
        assert any("faces the camera straight on" in p for p in prompts)
        assert any("faces the camera straight on" not in p for p in prompts)
        manifest = json.loads((d / "scenes/main/scenes.json").read_text())
        assert all(manifest["slots"][s["id"]]["pick"] for s in slots)
        assert all("SMALL-PHONE" not in v["flags"] and v["screen"] >= 0.48
                   for entry in manifest["slots"].values() for v in entry["variants"])
        assert (d / "scenes/main/review.jpg").exists()
        _run(["scene", temp, "--set", "night", "--look", "moonlit"])
        assert json.loads((d / "storyboard.json").read_text())["scenes"]["sets"]["night"]["look"] == "moonlit"
        _run(["compose", temp])
        _run(["compose", temp, "--set", "night", "--targets", "ios-phone-65"])
        for setname, targets in (("main", ("ios-phone", "ios-phone-large")), ("night", ("ios-phone-65",))):
            base = d / "scenes" / setname / "build"
            report = json.loads((base / "report.json").read_text())
            for target in targets:
                for slot in slots:
                    shot = report[target][slot["id"]]
                    im = Image.open(base / shot["file"])
                    assert im.size == K.SCENE_TARGETS[target] and im.mode == "RGB"
                    assert shot["fit"] >= 0.85 and "OVERLAP" not in shot["flags"] and "SMALL-PHONE" not in shot["flags"], shot
                assert (base / target / "sheet.jpg").exists()


def test_bad_view():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        sb = {"brand": {}, "scenes": {"style": "painted"},
              "slots": [{"id": "hero", "capture": "home", "caption": "A calm day", "scene_brief": "soft hills",
                         "scene_view": "sideways"}]}
        (d / "storyboard.json").write_text(json.dumps(sb))
        K.generate = lambda prompt, **kw: (_png(Image.new("RGB", (853, 1843))), {})
        try:
            _run(["scene", temp])
            raise AssertionError("bad scene_view accepted")
        except ValueError as e:
            assert "hero" in str(e) and "front" in str(e) and "angled" in str(e)


def test_hardening():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        slots = [{"id": s, "capture": "home", "caption": "A calm day", "scene_brief": "hills"} for s in ("hero", "next")]
        sb = {"scenes": {"style": "painted"}, "slots": slots}
        (d / "storyboard.json").write_text(json.dumps(sb))
        caps = d / "caps"
        caps.mkdir()
        Image.new("RGB", (372, 1000), (130, 80, 170)).save(caps / "home.png")
        (d / "captures.json").write_text(json.dumps({"captures": [{"name": "home", "platform": "ios", "device": "phone",
            "orientation": "portrait", "file": "caps/home.png"}]}))
        scene, _ = _scene(853, 1843)
        K.generate = lambda prompt, **kw: (_png(scene), {})
        K._gf_ttf = lambda *args: None
        _run(["scene", temp, "--variants", "1"])
        source = d / "scenes/main"
        manifest_path = source / "scenes.json"
        before = manifest_path.read_bytes()
        review = (source / "review.jpg").read_bytes()
        _run(["scene", temp, "--pick", "hero=1", "--dry-run"])
        assert manifest_path.read_bytes() == before and (source / "review.jpg").read_bytes() == review
        _run(["scene", temp, "--variants", "1", "--slots", "hero", "--force"])
        assert (source / "_replaced/hero_v1.png").exists()
        _run(["scene", temp, "--variants", "1", "--slots", "hero", "--force"])
        assert (source / "_replaced/hero_v1.2.png").exists()
        args = ["compose", temp, "--targets", "ios-phone", "--captures", str(caps)]
        _run(args)
        build = source / "build"
        folder = build / "ios-phone"
        report = json.loads((build / "report.json").read_text())["ios-phone"]
        assert "SMALL-HEADLINE" not in report["hero"]["flags"] and (folder / "search_row.jpg").exists()
        (folder / "keep.txt").write_text("keep")
        manifest = json.loads(manifest_path.read_text())
        manifest["slots"]["next"]["pick"] = None
        manifest_path.write_text(json.dumps(manifest))
        sb["scenes"]["type"] = {"size": 0.05}
        (d / "storyboard.json").write_text(json.dumps(sb))
        _run(args)
        report = json.loads((build / "report.json").read_text())["ios-phone"]
        assert not (folder / "02_next.png").exists() and (folder / "keep.txt").exists()
        assert "SMALL-HEADLINE" in report["hero"]["flags"] and report["hero"]["search_pt"] < 9
        slots[0]["capture"] = "missing"
        (d / "storyboard.json").write_text(json.dumps(sb))
        _run(args)
        shot = json.loads((build / "report.json").read_text())["ios-phone"]["hero"]
        assert shot == {"file": None, "kind": "painted", "fit": None, "flags": [], "notes": ["missing capture 'missing'"]}
        try:
            _run(args + ["--flat", temp])
            raise AssertionError("missing report accepted")
        except ValueError as e:
            assert "--no-caption" in str(e)
        slots[0]["scene_captures"] = []
        (d / "storyboard.json").write_text(json.dumps(sb))
        try:
            _run(args)
            raise AssertionError("empty captures accepted")
        except ValueError as e:
            assert str(e) == "hero: scene_captures is empty"


def test_extra_screen_and_despill():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        slot = {"id": "hero", "capture": "home", "caption": "A calm day"}
        (d / "storyboard.json").write_text(json.dumps({"slots": [slot]}))
        source = d / "scenes/main"
        source.mkdir(parents=True)
        pixels = np.full((1843, 853, 3), (70, 90, 120), np.uint8)
        cv2.rectangle(pixels, (80, 500), (350, 1400), (0, 255, 0), -1)
        cv2.rectangle(pixels, (500, 500), (770, 1400), (0, 255, 0), -1)
        Image.fromarray(pixels).save(source / "hero_v1.png")
        (source / "scenes.json").write_text(json.dumps({"slots": {"hero": {"pick": "hero_v1.png"}}}))
        caps = d / "caps"
        caps.mkdir()
        Image.new("RGB", (372, 1000)).save(caps / "home.png")
        _run(["compose", temp, "--targets", "ios-phone", "--captures", str(caps)])
        shot = json.loads((source / "build/report.json").read_text())["ios-phone"]["hero"]
        assert shot["file"] is None and shot["notes"] == ["found 2 screens, slot needs 1"]
        scene, _ = _scene(853, 1843)
        pixels = np.array(scene)
        pixels[50:70, 20:40] = (40, 75, 40)
        result, _, _ = K.key_capture(Image.fromarray(pixels), [Image.new("RGB", (372, 1000))])
        assert tuple(np.asarray(result)[60, 30]) == (40, 75, 40)


def test_bad_reference():
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        _ref_project(d)
        Image.new("RGB", (853, 1843)).save(d / "ref.png")
        scene, _ = _scene(853, 1843)
        K.edit = lambda ref, prompt, **kw: (_png(scene), {})
        _run(["scene", temp, "--variants", "1"])
        assert (d / "scenes/main/scenes.json").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        saved = K.generate, K.edit, K._gf_ttf
        try:
            test()
            print(f"ok   {test.__name__}")
        finally:
            K.generate, K.edit, K._gf_ttf = saved
    print(f"{len(tests)} scene tests passed")
