"""Offline checks of the ui-assets gallery (manifest, page, picks, history, server): python3 tests/test_gallery.py."""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger
os.environ.pop("GPTKIT_GALLERY", None)
from PIL import Image  # noqa: E402

import gptkit as K  # noqa: E402


class Project:
    """A temporary project with its own gptkit config dir, so gallery decisions never touch the user's."""
    def __enter__(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.old_cfg = os.environ.get("GPTKIT_CONFIG")
        os.environ["GPTKIT_CONFIG"] = str(self.root / ".cfg" / "env")
        self.ua = self.root / "ui-assets"
        return self

    def __exit__(self, *exc):
        if self.old_cfg is None:
            os.environ.pop("GPTKIT_CONFIG", None)
        else:
            os.environ["GPTKIT_CONFIG"] = self.old_cfg
        shutil.rmtree(self.root, ignore_errors=True)
        return False

    def manifest(self):
        return json.loads((self.ua / "manifest.json").read_text())["entries"]


def cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            K.main(argv=[str(x) for x in args])
        except SystemExit as e:
            assert e.code in (None, 0), (out.getvalue(), err.getvalue(), e.code)
    return out.getvalue(), err.getvalue()


def png(path, size=(64, 128), color=(200, 210, 220, 255)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)
    return path


@contextlib.contextmanager
def fake_images():
    calls = []

    def gen(prompt, size="1024x1024", **k):
        calls.append(prompt)
        w, h = map(int, size.split("x"))
        b = io.BytesIO()
        Image.new("RGB", (w, h), (len(calls) * 30 % 255, 120, 160)).save(b, "PNG")
        return b.getvalue(), {"quality": "high", "size": size}
    old = K.generate
    K.generate = gen
    try:
        yield calls
    finally:
        K.generate = old


def mockup(p, brief="habit tracker home", variants=3):
    return cli("mockup", brief, "--platform", "android", "--out", p.ua / "design" / "home", "--variants", variants,
               "--no-critique")


def test_mockup_run_is_indexed_with_its_prompt():
    with Project() as p, fake_images():
        _, err = mockup(p)
        assert "gallery: undecided for this project" in err, err
        m = p.manifest()
        paths = sorted(m)
        assert paths == [f"ui-assets/design/home/mockup_v{i}.png" for i in (1, 2, 3)], paths
        e = m[paths[0]]
        assert (e["kind"], e["unit"], e["group"], e["label"], e["state"], e["choice"]) == \
            ("mockup", "design/home", "mockup", "v1", "new", True), e
        assert e["cmd"] == "mockup" and e["prompt"] == "habit tracker home" and e["size"] == [1152, 2496], e
        assert (p.root / e["thumb"]).exists() and e["thumb"].startswith("ui-assets/.thumbs/")
        with Image.open(p.root / e["thumb"]) as t:
            assert max(t.size) <= 1280
        page = (p.ua / "index.html").read_text()
        data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1))
        assert len(data["entries"]) == 3 and not any(k.startswith("_") for k in data["entries"][0])


def test_outside_ui_assets_nothing_is_written():
    with Project() as p, fake_images():
        cli("mockup", "x", "--platform", "android", "--out", p.root / "design" / "home", "--no-critique")
        assert (p.root / "design" / "home" / "mockup_v1.png").exists()
        assert not p.ua.exists()


def test_rerun_keeps_overwritten_variants_in_history():
    with Project() as p, fake_images():
        mockup(p, "first brief", 2)
        before = (p.ua / "design/home/mockup_v1.png").read_bytes()
        mockup(p, "second brief", 2)
        m = p.manifest()
        hist = [e for e in m.values() if e.get("history")]
        assert len(hist) == 2 and all(e["replaced"] and e["prompt"] == "first brief" for e in hist), hist
        assert all(e["path"].startswith("ui-assets/.history/design/home/mockup_v") for e in hist)
        old = next(e for e in hist if "mockup_v1@" in e["path"])
        assert (p.root / old["path"]).read_bytes() == before
        assert m["ui-assets/design/home/mockup_v1.png"]["prompt"] == "second brief"


def test_pick_is_exclusive_and_drives_gitignore():
    with Project() as p, fake_images():
        mockup(p)
        cli("gallery", "on", p.root)
        cli("gallery", "set", p.ua / "design/home/mockup_v2.png", "picked", "--dir", p.root)
        cli("gallery", "set", p.ua / "design/home/mockup_v3.png", "picked", "--note", "bigger header", "--dir", p.root)
        cli("gallery", "set", "ui-assets/design/home/mockup_v1.png", "discarded", "--dir", p.root)
        m = p.manifest()
        states = {k.rsplit("/", 1)[1]: e["state"] for k, e in m.items()}
        assert states == {"mockup_v1.png": "discarded", "mockup_v2.png": "new", "mockup_v3.png": "picked"}, states
        assert m["ui-assets/design/home/mockup_v3.png"]["note"] == "bigger header"
        ignore = (p.ua / ".gitignore").read_text()
        assert "/design/home/mockup_v1.png" in ignore and "/design/home/mockup_v2.png" in ignore
        assert "/design/home/mockup_v3.png" not in ignore and ".thumbs/" in ignore
        out, _ = cli("gallery", "picks", p.root)
        assert "picked\n  ui-assets/design/home/mockup_v3.png" in out and "note: bigger header" in out, out
        assert "discarded\n  ui-assets/design/home/mockup_v1.png" in out, out
        _, err = mockup(p, "again", 1)
        assert "undecided" not in err, err


def test_gitignore_keeps_user_lines():
    with Project() as p:
        p.ua.mkdir(parents=True)
        (p.ua / ".gitignore").write_text("*.psd\n")
        png(p.ua / "design/home/mockup_v1.png"); (p.ua / "design/home/design.json").write_text("{}")
        cli("gallery", "index", p.root)
        cli("gallery", "index", p.root)
        text = (p.ua / ".gitignore").read_text()
        assert text.startswith("*.psd\n") and text.count("# end gptkit gallery") == 1, text


def test_off_stops_indexing_and_on_restores():
    with Project() as p, fake_images():
        mockup(p, "a", 1)
        cli("gallery", "off", p.root)
        assert not (p.ua / "index.html").exists() and not (p.ua / ".thumbs").exists()
        _, err = mockup(p, "b", 2)
        assert "gallery:" not in err, err
        assert len(p.manifest()) == 1
        cli("gallery", "on", p.root)
        assert len(p.manifest()) == 2 and (p.ua / "index.html").exists()


def test_classification():
    with Project() as p:
        d = p.ua / "design" / "home"
        (d / "design.json").parent.mkdir(parents=True); (d / "design.json").write_text("{}")
        for n in ("mockup_v1.png", "mockup_v1_critique.png", "mockup_v1_raw.png", "mockup_v1_qa1.png", "mockup_tablet.png"):
            png(d / n)
        (d / "mockup_v1_critique.json").write_text(json.dumps({"rounds": [{"issues": [{"severity": "high"}, {"severity": "low"}]}]}))
        b = p.ua / "build" / "home"
        (b / "assets.json").parent.mkdir(parents=True); (b / "assets.json").write_text('{"assets": []}')
        png(b / "mockup.png"); png(b / "manifest_overlay.png"); png(b / "review" / "forge_1.png")
        for k in (1, 2):
            png(b / "raw" / f"hero_c{k}.png")
        png(b / "raw" / "hero_cm.png"); png(b / "assets" / "hero.png"); png(b / "assets" / "solo.png")
        (b / "forge_report.json").write_text(json.dumps({"hero": {"chosen": {"path": "raw/hero_c2.png"}, "candidates": [
            {"path": "raw/hero_c2.png", "score": {"fid": 88, "flags": ["COPY"]}}]}}))
        (b / "svg").mkdir(); (b / "svg" / "bell.svg").write_text('<svg viewBox="0 0 24 24"/>'); png(b / "svg" / "preview" / "bell.png")
        a = p.ua / "icon" / "build"
        (a / "appicon.json").parent.mkdir(parents=True); (a / "appicon.json").write_text("{}")
        png(a / "android" / "play_store_512.png"); png(a / "android" / "res" / "mipmap-xxhdpi" / "ic_launcher.webp")
        png(a / "web" / "favicon-32.png")
        cli("gallery", "index", p.root)
        m = p.manifest()
        assert sorted(m) == ["ui-assets/build/home/assets/solo.png", "ui-assets/build/home/raw/hero_c1.png",
                             "ui-assets/build/home/raw/hero_c2.png", "ui-assets/build/home/svg/bell.svg",
                             "ui-assets/design/home/mockup_tablet.png", "ui-assets/design/home/mockup_v1.png",
                             "ui-assets/design/home/mockup_v1_qa1.png", "ui-assets/icon/build/android/play_store_512.png"], sorted(m)
        c2 = m["ui-assets/build/home/raw/hero_c2.png"]
        assert c2["state"] == "picked" and c2["kit"] and c2["info"] == {"fid": 88, "flags": ["COPY"]}, c2
        assert m["ui-assets/build/home/raw/hero_c1.png"]["state"] == "new"
        assert m["ui-assets/design/home/mockup_v1.png"]["info"]["issues"] == {"high": 1, "low": 1}
        assert not m["ui-assets/design/home/mockup_tablet.png"]["choice"]
        assert m["ui-assets/icon/build/android/play_store_512.png"]["final"]
        assert m["ui-assets/build/home/svg/bell.svg"]["size"] == [24, 24]


def test_forge_pick_goes_through_the_kit():
    with Project() as p:
        b = p.ua / "build" / "home"
        (b / "assets.json").parent.mkdir(parents=True); (b / "assets.json").write_text('{"assets": []}')
        for k in (1, 2):
            png(b / "raw" / f"hero_c{k}.png")
        report = {"hero": {"chosen": {"path": "raw/hero_c1.png"}, "candidates": []}}
        (b / "forge_report.json").write_text(json.dumps(report))
        cli("gallery", "index", p.root)
        seen = []

        def fake_pick(a):
            seen.append((a.dir, a.name, a.candidate))
            report["hero"]["chosen"]["path"] = "raw/hero_c2.png"
            (b / "forge_report.json").write_text(json.dumps(report))
        old = K.cmd_pick
        K.cmd_pick = fake_pick
        try:
            cli("gallery", "set", b / "raw" / "hero_c2.png", "picked", "--dir", p.root)
        finally:
            K.cmd_pick = old
        assert seen == [(str(b), "hero", str(b / "raw" / "hero_c2.png"))], seen
        m = p.manifest()
        assert m["ui-assets/build/home/raw/hero_c2.png"]["state"] == "picked"
        assert m["ui-assets/build/home/raw/hero_c1.png"]["state"] == "new"
        try:
            K.gallery_set(p.ua, b / "raw" / "hero_c2.png", "discarded")
            raise AssertionError("discarding the kit's chosen candidate must be refused")
        except ValueError as e:
            assert "pick another first" in str(e)


def test_page_survives_script_tags_in_prompts():
    with Project() as p, fake_images():
        mockup(p, "</script><script>alert(1)</script>", 1)
        page = (p.ua / "index.html").read_text()
        blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
        assert json.loads(blob)["entries"][0]["prompt"] == "</script><script>alert(1)</script>"


def test_server_guards_and_writes():
    from gptkit import gallery_server
    with Project() as p:
        png(p.ua / "design/home/mockup_v1.png"); png(p.ua / "design/home/mockup_v2.png")
        (p.ua / "design/home/design.json").write_text("{}")
        (p.root / "secret.txt").write_text("no")
        cli("gallery", "on", p.root)
        from functools import partial
        from http.server import ThreadingHTTPServer
        h = type("H", (gallery_server._Handler,), {"gdir": p.ua, "port": 0})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(h, directory=str(p.root)))
        h.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{h.port}"

        def req(path, data=None, headers=None):
            r = urllib.request.Request(base + path, data=data, headers=headers or {}, method="POST" if data else "GET")
            try:
                with urllib.request.urlopen(r) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
        try:
            assert req("/ui-assets/index.html")[0] == 200
            assert req("/secret.txt")[0] == 404
            assert req("/ui-assets/manifest.json")[0] == 200
            assert req("/ui-assets/../secret.txt")[0] == 404
            os.symlink(p.root / "secret.txt", p.ua / "leak.txt")
            assert req("/ui-assets/leak.txt")[0] == 404
            assert req("/ui-assets/index.html", headers={"Host": "evil.example"})[0] == 403
            body = json.dumps({"path": "ui-assets/design/home/mockup_v2.png", "state": "picked"}).encode()
            assert req("/api/state", body, {"Content-Type": "application/json"})[0] == 403
            code, out = req("/api/state", body, {"Content-Type": "application/json", "X-Gptkit": "1"})
            assert code == 200 and json.loads(out)["entry"]["state"] == "picked", out
            code, out = req("/api/manifest")
            assert code == 200 and any(e["state"] == "picked" for e in json.loads(out)["entries"])
        finally:
            srv.shutdown()
            srv.server_close()


def test_parallel_runs_keep_their_own_prompts():
    with Project() as p, fake_images():
        cli("mockup", "home brief", "--platform", "android", "--out", p.ua / "design" / "home", "--no-critique")
        png(p.ua / "design/profile/mockup_v1.png"); (p.ua / "design/profile/design.json").write_text("{}")
        os.utime(p.ua / "design/home/mockup_v1.png", (1e9, 1e9))  # made well before the next run
        # a profile file written while the home run was going must not take the home run's prompt
        run = K.before_command(K.build_parser()[0].parse_args(
            ["mockup", "home again", "--out", str(p.ua / "design" / "home")]), "mockup")
        png(p.ua / "design/profile/mockup_v2.png")
        K.after_command(run)
        m = p.manifest()
        assert m["ui-assets/design/home/mockup_v1.png"]["prompt"] == "home brief"
        assert "prompt" not in m["ui-assets/design/profile/mockup_v2.png"], m["ui-assets/design/profile/mockup_v2.png"]


def test_export_of_a_final_marks_the_picked_candidate_shipped():
    with Project() as p:
        b = p.ua / "build" / "home"
        (b / "assets.json").parent.mkdir(parents=True); (b / "assets.json").write_text('{"assets": []}')
        png(b / "raw" / "hero_c1.png"); png(b / "raw" / "hero_c2.png"); png(b / "assets" / "hero.png", (96, 64))
        (b / "forge_report.json").write_text(json.dumps({"hero": {"chosen": {"path": "raw/hero_c2.png"}}}))
        cli("gallery", "index", p.root)
        res = p.root / "app" / "src" / "main" / "res"
        cli("export", b / "assets" / "hero.png", "--dp", 48, "--res", res)
        m = p.manifest()
        assert m["ui-assets/build/home/raw/hero_c2.png"].get("shipped") == [str(res)], m["ui-assets/build/home/raw/hero_c2.png"]
        assert "shipped" not in m["ui-assets/build/home/raw/hero_c1.png"]


def test_animation_and_critique_revisions_are_indexed():
    with Project() as p:
        b = p.ua / "build" / "home"
        (b / "assets.json").parent.mkdir(parents=True); (b / "assets.json").write_text('{"assets": []}')
        f = b / "frames" / "hero_bounce"
        png(f / "frames_review.png"); png(f / "frame_01.png"); f.joinpath("preview.gif").write_bytes(b"GIF89a")
        png(f / "hero_bounce.webp")
        d = p.ua / "design" / "home"
        png(d / "mockup_v1.png"); (d / "design.json").write_text("{}")
        png(d / "mockup_v1_qa1.png"); png(d / "mockup_v1_qa2.png")
        (d / "mockup_v1_critique.json").write_text(json.dumps({"rounds": [
            {"issues": [{"severity": "high"}]},
            {"image": str(d / "mockup_v1_qa1.png"), "kept": True, "weight_before": 3, "weight_after": 1},
            {"image": str(d / "mockup_v1_qa2.png"), "kept": False, "weight_before": 1, "weight_after": 2}]}))
        cli("gallery", "index", p.root)
        m = p.manifest()
        anim = m["ui-assets/build/home/frames/hero_bounce/hero_bounce.webp"]
        assert (anim["kind"], anim["unit"], anim["group"]) == ("animation", "build/home", "hero_bounce"), anim
        assert not any(k.endswith(("preview.gif", "frame_01.png", "frames_review.png")) for k in m), sorted(m)
        q1, q2 = m["ui-assets/design/home/mockup_v1_qa1.png"], m["ui-assets/design/home/mockup_v1_qa2.png"]
        assert (q1["group"], q1["label"], q1["state"], q1["choice"]) == ("mockup", "v1 fix 1", "new", True), q1
        assert q2["state"] == "discarded" and q2["info"]["flags"] == ["NOT KEPT"], q2
        assert m["ui-assets/design/home/mockup_v1.png"]["info"]["issues"] == {"high": 1}


def test_page_data_has_no_raw_angle_brackets():
    with Project() as p, fake_images():
        mockup(p, "<!--<script> a < b", 1)
        page = (p.ua / "index.html").read_text()
        blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
        assert "<" not in blob and json.loads(blob)["entries"][0]["prompt"] == "<!--<script> a < b"


def test_retries_material_stage_and_store_layouts():
    with Project() as p:
        b = p.ua / "build" / "home"
        (b / "assets.json").parent.mkdir(parents=True); (b / "assets.json").write_text('{"assets": []}')
        (b / "svg" / "raw").mkdir(parents=True)
        (b / "svg" / "home.svg").write_text('<svg viewBox="0 0 24 24"/>')
        (b / "svg" / "raw" / "home_try1.svg").write_text('<svg viewBox="0 0 24 24"/>')
        h = p.ua / "hybrid" / "home"
        png(h / "raw" / "paper_c1.png"); png(h / "raw" / "paper_c2.png"); png(h / "paper.png"); png(h / "review" / "material_paper.png")
        (h / "material.json").write_text(json.dumps({"paper": {"best": "c2", "candidates": {
            "c1": {"file": "raw/paper_c1.png", "flags": ["EMPTY"]}, "c2": {"file": "raw/paper_c2.png", "flags": []}}}}))
        st = p.ua / "hybrid" / "shots"
        png(st / "idle.png"); png(st / "motion_sheet.png"); png(st / "motion.webp"); (st / "stage.json").write_text("{}")
        s = p.ua / "store"
        (s / "captures.json").parent.mkdir(parents=True); (s / "captures.json").write_text("{}")
        png(s / "captures" / "ios-phone" / "home.png")
        for t in ("ios-phone", "play-phone"):
            png(s / "build" / t / "01_hero.png"); png(s / "build" / t / "02_shop.png")
            (s / "build" / t / "sheet.jpg").write_bytes(b"x")
        png(s / "build" / ".previous" / "ios-phone-20261027-201936" / "01_hero.png")
        sc = s / "scenes" / "main"
        png(sc / "hero_v1.png"); png(sc / "hero_v2.png"); png(sc / "_replaced" / "hero_v1.png")
        (sc / "scenes.json").write_text(json.dumps({"slots": {"hero": {"pick": "hero_v2.png", "variants": [
            {"file": "hero_v1.png", "flags": ["OCCLUDED"]}, {"file": "hero_v2.png", "flags": []}]}}}))
        cli("gallery", "index", p.root)
        m = p.manifest()
        assert not any("_try" in k or "/captures/" in k or k.endswith(("paper.png", "motion_sheet.png", "sheet.jpg"))
                       for k in m), sorted(m)
        mat = {k.rsplit("/", 1)[1]: m[k] for k in m if "/raw/paper_" in k}
        assert {k: (e["kind"], e["unit"], e["state"]) for k, e in mat.items()} == {
            "paper_c1.png": ("hybrid", "hybrid/home", "new"), "paper_c2.png": ("hybrid", "hybrid/home", "picked")}, mat
        assert mat["paper_c1.png"]["info"]["flags"] == ["EMPTY"]
        assert (m["ui-assets/hybrid/shots/idle.png"]["kind"], m["ui-assets/hybrid/shots/motion.webp"]["kind"]) == ("hybrid", "animation")
        shots = {k: (m[k]["unit"], m[k]["group"]) for k in m if "/build/" in k and "/store/" in k}
        assert set(shots.values()) == {("store", "ios-phone"), ("store", "play-phone")}, shots
        v1, v2 = m["ui-assets/store/scenes/main/hero_v1.png"], m["ui-assets/store/scenes/main/hero_v2.png"]
        assert (v1["group"], v1["state"], v1["info"]["flags"], v2["state"]) == ("hero scene", "new", ["OCCLUDED"], "picked")
        old = m["ui-assets/store/scenes/main/_replaced/hero_v1.png"]
        assert old["replaced"] and old["group"] == "hero scene", old
        seen = []
        saved = K.cmd_store_scene
        K.cmd_store_scene = lambda a: seen.append((a.dir, a.set, a.pick))
        try:
            cli("gallery", "set", sc / "hero_v1.png", "picked", "--dir", p.root)
        finally:
            K.cmd_store_scene = saved
        assert seen == [(str(s), "main", "hero=1")], seen


def test_quick_rerun_keeps_untouched_prompts():
    with Project() as p, fake_images():
        mockup(p, "first", 3)
        mockup(p, "second", 1)  # within the same second: only v1 is rewritten
        m = p.manifest()
        assert m["ui-assets/design/home/mockup_v1.png"]["prompt"] == "second"
        assert m["ui-assets/design/home/mockup_v2.png"]["prompt"] == "first"


def test_damaged_manifest_is_kept_aside():
    with Project() as p:
        png(p.ua / "design/home/mockup_v1.png"); (p.ua / "design/home/design.json").write_text("{}")
        cli("gallery", "index", p.root)
        (p.ua / "manifest.json").write_text('{"entries": [')
        _, err = cli("gallery", "index", p.root)
        assert "was damaged" in err and len(p.manifest()) == 1
        assert len(list(p.ua.glob("manifest.damaged-*.json"))) == 1


def run():
    for fn in (test_mockup_run_is_indexed_with_its_prompt, test_outside_ui_assets_nothing_is_written,
               test_rerun_keeps_overwritten_variants_in_history, test_pick_is_exclusive_and_drives_gitignore,
               test_gitignore_keeps_user_lines, test_off_stops_indexing_and_on_restores, test_classification,
               test_forge_pick_goes_through_the_kit, test_page_survives_script_tags_in_prompts,
               test_server_guards_and_writes, test_parallel_runs_keep_their_own_prompts,
               test_export_of_a_final_marks_the_picked_candidate_shipped,
               test_animation_and_critique_revisions_are_indexed, test_page_data_has_no_raw_angle_brackets,
               test_retries_material_stage_and_store_layouts, test_quick_rerun_keeps_untouched_prompts,
               test_damaged_manifest_is_kept_aside):
        fn()
        print(f"ok {fn.__name__}")


if __name__ == "__main__":
    run()
