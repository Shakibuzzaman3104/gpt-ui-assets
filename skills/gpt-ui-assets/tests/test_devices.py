"""Offline form-factor coverage. Run: python3 tests/test_devices.py"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger


def _png(size):
    im = Image.new("RGB", size, "white")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


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
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fn(a)
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            print(e.code)
    return code, out.getvalue()


def args(td, **kw):
    base = argparse.Namespace(prompt="inbox", out=str(Path(td) / "out"), name="inbox", size=None, variants=1,
        edit=None, raw=False, style_ref=None, platform="android", model=None, no_critique=True,
        no_house_style=True, device=None, orientation=None, posture=None, page=None, set=None, anchor=None)
    vars(base).update(kw)
    return base


def test_defaults_and_validity():
    for platform, device in (("android", "phone"), ("ios", "phone"), ("web", "desktop"),
                             ("macos", "desktop"), ("windows", "desktop"), ("linux", "desktop")):
        assert K._device_record(platform)["canvas"] and K._device_record(platform)["logical_w"]
        assert K._device_record(platform)["canvas"] == K._device_record(platform, device)["canvas"]
    calls = []
    with tempfile.TemporaryDirectory() as td, patched(generate=lambda *a, **k: calls.append(1)):
        code, out = run(K.cmd_mockup, args(td, platform="ios", device="desktop"))
    assert code == 1 and "ios supports phone, tablet" in out and not calls, out
    assert K._device_record("android", "tablet", "landscape")["canvas"] == "2304x1440"
    assert K._device_record("ios", "tablet", "landscape")["canvas"] == "2320x1600"
    assert K._device_record("android", "foldable", "", "tabletop")["hinge"] == ("y", 0.5)


def test_device_canvas_prompt_and_metadata():
    calls = []

    def generate(prompt, size, **kw):
        calls.append((prompt, size))
        return _png(tuple(map(int, size.split("x")))), {"quality": "high", "size": size}

    with tempfile.TemporaryDirectory() as td, patched(generate=generate):
        code, _ = run(K.cmd_mockup, args(td, platform="web", size=None, device="desktop", page="landing"))
        assert code == 0
        prompt, size = calls[-1]
        assert size == "2304x1440" and "1440x900 desktop viewport" in prompt and "landing page" in prompt.lower()
        meta = json.loads((Path(td) / "out" / "design.json").read_text())
        assert meta["platform"] == "web" and meta["device"] == "desktop" and meta["screen_dp"] == 1440
        assert meta["logical_h"] == 900 and meta["canvas"] == "2304x1440"


def test_critique_device_rule_touch_and_hinge_note():
    seen = []
    issue = {"severity": "high", "category": "adaptivity", "problem": "control crosses fold", "edit": "move it",
             "bbox": {"x": 910, "y": 100, "w": 40, "h": 50}}
    with patched(ask=lambda parts, *a, **k: (seen.append(parts[0]) or {"issues": [issue]})):
        issues = K.critique(Image.new("RGB", (1840, 1904)), 852, "android",
                            device_record=K._device_record("android", "foldable", posture="flat"))
    assert issues and "48 logical-unit touch target" in seen[0] and "vertical hinge band" in seen[0]
    assert K._hinge_hits(issues, K._device_record("android", "foldable"), 1840, 1904) == issues


def test_foldable_fit_disabled_and_edit_canvas():
    calls = []
    def generate(prompt, size, **kw):
        calls.append(size)
        return _png((800, 900)), {"quality": "high", "size": size}
    def edit(ref, prompt, size, **kw):
        calls.append(size)
        return _png(tuple(map(int, size.split("x")))), {"quality": "high", "size": size}
    with tempfile.TemporaryDirectory() as td, patched(generate=generate,
            edit=edit, _fit_aspect=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fit a hinge"))):
        code, out = run(K.cmd_mockup, args(td, size=None, device="foldable", posture="flat"))
        assert code == 0 and calls == ["1840x1904", "1840x1904"] and "aspect=OFF" in out, (code, out, calls)
        png = Path(td) / "out" / "inbox_v1.png"
        png.write_bytes(_png((1440, 2304)))
        code, _ = run(K.cmd_mockup, args(td, device="tablet", orientation="portrait", edit=str(png)))
        assert code == 0 and calls[-1] == "1440x2304"


def test_set_outputs_and_skips_cross_aspect_copy_check():
    prompts = []
    anchor = None
    def edit(ref, prompt, size, **kw):
        prompts.append((prompt, size))
        return _png(tuple(map(int, size.split("x")))), {"quality": "high", "size": size}
    with tempfile.TemporaryDirectory() as td:
        anchor = Path(td) / "approved.png"
        anchor.write_bytes(_png((1152, 2496)))
        with patched(edit=edit):
            code, out = run(K.cmd_mockup, args(td, out=str(Path(td) / "out"), platform="android", set="tablet,desktop",
                anchor=str(anchor)))
        d = Path(td) / "out"
        assert code == 0 and (d / "inbox_tablet.png").exists() and (d / "inbox_desktop.png").exists(), (code, out)
        manifest = json.loads((d / "set.json").read_text())
        assert manifest["anchor"] == str(anchor) and set(manifest["devices"]) == {"tablet", "desktop"}
        assert len(prompts) == 2 and all("same information architecture and data, not a scaled copy" in x[0] for x in prompts)
        assert all("layout_diff=None" in out for _ in [0])


def test_manifest_excludes_desktop_window_chrome():
    with tempfile.TemporaryDirectory() as td:
        screen = Path(td) / "screen.png"
        screen.write_bytes(_png((2304, 1440)))
        prompts = []
        a = argparse.Namespace(mockup=str(screen), out=str(Path(td) / "manifest"), platform="macos", screen_dp=None, model=None)
        with patched(ask=lambda parts, *a, **k: (prompts.append(parts[0]) or {"assets": []})):
            code, _ = run(K.cmd_manifest, a)
        assert code == 0 and "traffic lights, caption buttons, title bar" in prompts[0]
        assert "taskbar, Dock and browser chrome" in prompts[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
