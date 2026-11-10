"""Offline project status checks: python3 tests/test_status.py."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger
import gptkit as g  # noqa: E402


class TempDir:
    """Temporary fixture path, removed afterwards."""
    def __enter__(self):
        self.path = str(Path(tempfile.mkdtemp()).resolve())  # macOS: /var -> /private/var
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def call(*args):
    return run_cli("status", *args)


def run_cli(*args):
    old = sys.argv
    out = io.StringIO()
    sys.argv = ["gptkit.py", *map(str, args)]
    try:
        with contextlib.redirect_stdout(out):
            try:
                g.main()
            except SystemExit as e:
                assert e.code in (None, 0), out.getvalue()
    finally:
        sys.argv = old
    return out.getvalue()


def put(path, data="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    return path


def unit(doc, kind):
    return next(x for x in doc["units"] if x["type"] == kind)


def test_manifest_to_spec_next():
    with TempDir() as td:
        d = Path(td) / "build" / "home"
        put(d / "assets.json", '{"assets": []}')
        put(d / "manifest_overlay.png", "fake png")
        text = call(d)
        assert f"next: python3 $K spec {d}" in text


def test_stale_spec():
    with TempDir() as td:
        d = Path(td)
        spec = put(d / "design_spec.json")
        time.sleep(0.02)
        put(d / "assets.json", '{"assets": []}')
        doc = json.loads(call(d, "--json"))
        b = unit(doc, "build")
        assert any(c["name"] == "design_spec.json" and c["status"] == "STALE" for c in b["checks"])
        assert b["next"] == [f"python3 $K spec {d}"]


def test_forge_attention():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        put(d / "design_spec.json")
        # the shape forge writes: a flagged winner is kept as the final file and marked needs_attention
        put(d / "assets" / "a.png", "fake png")
        put(d / "assets" / "b.png", "fake png")
        put(d / "forge_report.json", json.dumps({
            "a": {"candidates": [{"path": "raw/a_c1.png", "score": {"flags": ["CUT-OFF"]}}], "complete": True,
                  "chosen": 1, "final": "assets/a.png", "needs_attention": True},
            "b": {"candidates": [{"path": "raw/b_c1.png", "score": {"flags": []}}], "complete": True,
                  "chosen": 1, "final": "assets/b.png", "needs_attention": False},
            "c": {"candidates": [], "complete": False}}))
        doc = json.loads(call(d, "--json"))
        b = unit(doc, "build")
        assert any("NEEDS ATTENTION: a" in p for p in b["problems"]), b["problems"]
        assert any(p == "no final file: c" for p in b["problems"]), b["problems"]
        assert not any(" b" in p or p.endswith(": b") for p in b["problems"]), b["problems"]
        assert "pick" in b["next"][0] and "--only a --force" in b["next"][0], b["next"]
        assert b["next"][1] == f"python3 $K forge {d}", b["next"]


def test_corrupt_forge_report():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        put(d / "forge_report.json", "{")
        doc = json.loads(call(d, "--json"))
        assert any("forge_report.json: invalid JSON" in p for p in unit(doc, "build")["problems"])


def test_sequence_missing_frames():
    with TempDir() as td:
        d = Path(td) / "sequence" / "rain"
        put(d / "sequence.json", '{"frames": 3, "guide_sha1": ["a", "b", "c"]}')
        doc = json.loads(call(Path(td), "--json"))
        seq = unit(doc, "sequence")
        assert "frames missing: [0, 1, 2]" in seq["problems"]
        assert seq["next"] == [f"python3 $K sequence {d} <name> render"]


def test_empty_dir():
    with TempDir() as td:
        out = call(td)
        assert f"no gptkit artifacts under {Path(td).resolve()}" in out
        assert "python3 $K mockup" in out


def test_json_shape():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        doc = json.loads(call(d, "--json"))
        assert set(("dir", "units", "next")) <= set(doc)
        b = unit(doc, "build")
        assert set(("checks", "problems", "next")) <= set(b)


def test_real_offline_kit_output():
    """Run the real offline SVG -> VectorDrawable command, then inspect that project tree."""
    with TempDir() as td:
        d = Path(td) / "build" / "home"
        put(d / "assets.json", '{"assets": []}')
        put(d / "design_spec.json")
        svg = put(d / "svg" / "mark.svg",
                  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
                  '<path fill="#000000" d="M3 3h18v18H3z"/></svg>')
        run_cli("vd", svg, "--out", d / "platform" / "drawable", "--no-check")
        assert list((d / "platform" / "drawable").glob("*.xml"))
        out = call(d)
        assert f"build {d}" in out and "design_spec.json" in out


def test_unforged_assets_and_design_link():
    from PIL import Image
    with TempDir() as td:
        root = Path(td)
        design, build = root / "design" / "home", root / "build" / "home"
        design.mkdir(parents=True); build.mkdir(parents=True)
        Image.new("RGB", (40, 60), (20, 26, 51)).save(design / "mockup_v1.png")
        put(design / "design.json", '{"platform": "android"}')
        Image.open(design / "mockup_v1.png").save(build / "mockup.png", optimize=True)  # re-encoded, same pixels
        put(build / "assets.json", json.dumps({"assets": [{"name": "hero", "format": "png"}, {"name": "gear", "format": "svg"}]}))
        put(build / "design_spec.json")
        doc = json.loads(call(root, "--json"))
        b, d = unit(doc, "build"), unit(doc, "design")
        assert b["next"][0] == "python3 $K forge " + str(build) and "svg" in b["next"][1], b["next"]
        assert not any(u["type"] == "design" and u["path"] == str(build) for u in doc["units"])  # mockup.png is not a design
        assert any(c["name"].startswith("manifest (") for c in d["checks"]) and d["next"] == [], d


def test_empty_output_dir_is_not_done():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        (d / "review").mkdir()
        (d / "assets").mkdir()
        checks = {c["name"]: c["status"] for c in unit(json.loads(call(d, "--json")), "build")["checks"]}
        assert "review/" not in checks and checks["assets/"] == "missing", checks
        put(d / "review" / "forge_x.png", "fake png")
        checks = {c["name"]: c["status"] for c in unit(json.loads(call(d, "--json")), "build")["checks"]}
        assert checks["review/"] == "done", checks


def store_unit(d):
    return unit(json.loads(call(d, "--json")), "store")


def test_store_project_steps_in_order():
    with TempDir() as td:
        d = Path(td) / "ui-assets" / "store"
        (d / "captures").mkdir(parents=True)
        put(d / "captures.json", json.dumps({"captures": [{"name": "home", "file": "captures/ios-phone/home.png"}]}))
        assert store_unit(d)["next"] == [f"python3 $K store plan {d} --app \"<name: what it does, for whom>\""]
        slots = [{"id": "home", "capture": "home", "caption": "Calm", "plate_brief": "soft sky"},
                 {"id": "list", "capture": "list", "caption": "Yours"}]
        put(d / "storyboard.json", json.dumps({"slots": slots, "feature": {"subtitle": "x"}}))
        u = store_unit(d)
        assert u["next"] == [f"python3 $K store art {d}"] and "plates not painted: plate_home" in u["problems"], u
        put(d / "art" / "plate_home.png", "fake png")
        put(d / "art" / "art.json", json.dumps({"plate_home": {"file": "art/plate_home.png", "flags": []}}))
        assert store_unit(d)["next"] == [f"python3 $K store build {d}"]
        report = {"targets": {"ios-phone": {"shots": []}, "play-phone": {"shots": [{"slot": "home"}]}}, "errors": []}
        put(d / "build" / "report.json", json.dumps(report))
        u = store_unit(d)
        assert u["next"] == [f"python3 $K store check {d}"] and "ios-phone: 0 shots built" in u["problems"], u
        put(d / "build" / "check.json", json.dumps([{"level": "FAIL", "where": "ios-phone", "msg": "0 screenshots"},
                                                     {"level": "WARN", "where": "build", "msg": "left out"}]))
        u = store_unit(d)
        assert "FAIL ios-phone: 0 screenshots" in u["problems"] and "1 WARN in build/check.json" in u["problems"], u
        assert u["next"][0].startswith("fix the FAIL lines"), u
        put(d / "build" / "check.json", json.dumps([{"level": "PASS", "where": "play-phone", "msg": "ok"}]))
        assert store_unit(d)["next"] == [f"python3 $K store feature {d} --logo <logo.png>"]
        put(d / "build" / "play-feature" / "feature.json")
        u = store_unit(d)
        assert u["next"] == [] and {c["status"] for c in u["checks"]} == {"done"}, u
        old = time.time() - 60
        os.utime(d / "build" / "report.json", (old, old))
        os.utime(d / "build" / "check.json", (old, old))
        u = store_unit(d)
        assert u["next"] == [f"python3 $K store build {d}"] and any(p.startswith("STALE: build/report.json") for p in u["problems"]), u
        assert "store " in call(d) and "no gptkit artifacts" not in call(d)


def test_store_painted_scenes():
    with TempDir() as td:
        d = Path(td) / "store"
        put(d / "captures.json", json.dumps({"captures": [{"name": "home"}]}))
        slots = [{"id": "home", "capture": "home", "caption": "Calm", "scene_brief": "a desk at dusk"},
                 {"id": "list", "capture": "list", "caption": "Yours", "scene_brief": "a shelf"}]
        put(d / "storyboard.json", json.dumps({"slots": slots}))
        u = store_unit(d)
        assert u["next"] == [f"python3 $K store scene {d}"] and "build/report.json" not in {c["name"] for c in u["checks"]}, u
        put(d / "scenes" / "main" / "scenes.json", json.dumps({"slots": {"home": {"pick": "home_v1.png"}, "list": {"pick": None}}}))
        u = store_unit(d)
        assert u["next"] == [f"python3 $K store scene {d} --set main --slots list"], u
        assert "scenes/main: no passing scene picked for list" in u["problems"], u
        put(d / "scenes" / "main" / "scenes.json", json.dumps({"slots": {"home": {"pick": "a.png"}, "list": {"pick": "b.png"}}}))
        assert store_unit(d)["next"] == [f"python3 $K store compose {d} --set main"]
        shots = {"home": {"file": "ios-phone/01_home.png", "flags": ["OCCLUDED"]}, "list": {"file": None, "flags": [], "notes": ["found 0 screens"]}}
        put(d / "scenes" / "main" / "build" / "report.json", json.dumps({"ios-phone": shots, "ios-phone-large": shots}))
        u = store_unit(d)
        assert u["next"] == [] and "scenes/main home: OCCLUDED (ios-phone, ios-phone-large)" in u["problems"], u
        assert "scenes/main list: found 0 screens (ios-phone, ios-phone-large)" in u["problems"], u


def test_store_damaged_files():
    with TempDir() as td:
        d = Path(td) / "store"
        put(d / "storyboard.json", "{not json")
        put(d / "build" / "report.json", "[]")
        put(d / "build" / "check.json", '{"level": "FAIL"}')
        u = store_unit(d)
        assert any("storyboard.json: invalid JSON" in p for p in u["problems"]), u
        assert u["next"] == [f"python3 $K store capture {d} --name <screen>"], u


def run():
    for fn in (test_empty_output_dir_is_not_done, test_manifest_to_spec_next, test_stale_spec, test_forge_attention,
               test_corrupt_forge_report, test_sequence_missing_frames, test_empty_dir, test_json_shape,
               test_real_offline_kit_output, test_unforged_assets_and_design_link, test_store_project_steps_in_order,
               test_store_painted_scenes, test_store_damaged_files):
        fn()
        print(f"ok {fn.__name__}")


if __name__ == "__main__":
    run()
