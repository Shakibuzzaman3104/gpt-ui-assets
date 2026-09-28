"""Hostile inputs through the CLI (verifier c2): bad flags, missing and damaged files, every JSON file the kit reads in
broken or older shapes, odd image modes, every mocked API failure shape of contract C2, partial results and Ctrl-C.
Each case asserts that no traceback reaches the user and that the one-line message names the file/flag and the fix.
Offline: requests.post/get are mocked at gptkit.requests, time.sleep is recorded. Run: python3 tests/test_cli_errors.py"""
import base64
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import gptkit as K  # noqa: E402

W, H = 1024, 1536


class R:
    """A requests.Response stand-in."""

    def __init__(self, status, body=None, text=None, headers=None):
        self.status_code, self.headers = status, headers or {}
        self.text = text if text is not None else json.dumps(body)
        self.content = self.text.encode()

    def json(self):
        return json.loads(self.text)


def _b64(fmt="PNG", size=(1024, 1024)):
    im = Image.new("RGBA" if fmt == "PNG" else "RGB", size, (0, 0, 0, 0) if fmt == "PNG" else (230, 120, 40))
    ImageDraw.Draw(im).ellipse([size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4], fill=(230, 120, 40, 255))
    b = io.BytesIO()
    im.save(b, fmt)
    return base64.b64encode(b.getvalue()).decode()


def _err(status, code, msg, etype="invalid_request_error", headers=None):
    return R(status, {"error": {"message": msg, "type": etype, "param": None, "code": code}}, headers=headers)


def _msg(content):
    return {"status": "completed", "output": [{"type": "message", "content": content}]}


REFUSAL = _msg([{"type": "refusal", "refusal": "I can't help with that."}])
# contract C2 failure shapes: (images endpoints, /responses) -> the text the printed line must carry
MODES = {
    "moderation": ((lambda: _err(400, "moderation_blocked", "Your request was rejected by the safety system."),) * 2,
                   ("[moderation_blocked]", "[moderation_blocked]")),
    "quota": ((lambda: _err(429, "insufficient_quota", "You exceeded your current quota.", "insufficient_quota"),) * 2,
              ("[insufficient_quota]", "[insufficient_quota]")),
    "retry_after": ((lambda: _err(429, "rate_limit_exceeded", "Rate limit reached.", "requests", {"Retry-After": "3"}),) * 2,
                    ("[exhausted]", "[exhausted]")),
    "500": ((lambda: R(500, {"error": {"message": "The server had an error.", "type": "server_error"}}),) * 2,
            ("[exhausted]", "[exhausted]")),
    "data_empty": ((lambda: R(200, {"data": []}),) * 2, ("no image (data: [])", "empty response")),
    "refusal": ((lambda: R(200, {"data": []}), lambda: R(200, REFUSAL)), ("no image (data: [])", "the model refused")),
    "incomplete": ((lambda: R(200, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}),) * 2,
                   ("no image (data: [])", "incomplete response (max_output_tokens: lower --effort")),
    "not_json": ((lambda: R(200, text="definitely not json"),
                  lambda: R(200, _msg([{"type": "output_text", "text": "Sure! Here is the answer you wanted."}]))),
                 ("[not_json]", "is not JSON")),
    "html": ((lambda: R(200, text="<html><body><h1>502 Bad Gateway</h1></body></html>"),) * 2, ("[not_json]", "[not_json]")),
    # a gateway answering JPEG: the image call succeeds (re-encoded PNG); the vision call is refused
    "jpeg": ((lambda: R(200, {"data": [{"b64_json": _b64("JPEG")}]}), lambda: R(200, REFUSAL)), (None, "the model refused")),
}


@contextlib.contextmanager
def env(**kv):
    keep = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    K._config.cache_clear()
    try:
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        K._config.cache_clear()


class Run:
    """Result of one in-process CLI run: code, out (stdout+stderr+exit message), tb, calls (posted URLs), sleeps."""

    def __init__(self, code, out, tb, calls, sleeps):
        self.code, self.out, self.tb, self.calls, self.sleeps = code, out, tb, calls, sleeps

    @property
    def last(self):
        lines = [x for x in self.out.splitlines() if x.strip()]
        return lines[-1] if lines else ""


def cli(*argv, api=None, interrupt=None, envs=None):
    """Run gptkit's main() in-process. api: a MODES key, or callable(url, json, data) -> R; interrupt(n) -> True raises
    KeyboardInterrupt inside the n-th API call (in whatever worker thread makes it)."""
    calls, sleeps, lock = [], [], threading.Lock()

    def post(url, headers=None, json=None, files=None, data=None, timeout=None):
        with lock:
            calls.append(url)
            n = len(calls)
        if interrupt and interrupt(n):
            raise KeyboardInterrupt
        assert api is not None, f"unexpected API call {url}"
        if callable(api):
            return api(url, json, data)
        img, resp = MODES[api][0]
        return (resp if url.endswith("/responses") else img)()

    def get(url, *a, **k):
        raise K.requests.ConnectionError(f"offline test: {url}")

    tbs = []
    saved = K.requests.post, K.requests.get, K.time.sleep, threading.excepthook, sys.argv
    K.requests.post, K.requests.get, K.time.sleep = post, get, sleeps.append
    threading.excepthook = lambda a: tbs.append("".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
    sys.argv = ["gptkit.py", *map(str, argv)]
    buf, code, tb = io.StringIO(), 0, None
    e = dict(GPTKIT_API_KEY="sk-test-key-123456", GPTKIT_CONFIG="/nonexistent/gptkit-env", GPTKIT_RETRIES="1",
             GPTKIT_MODERATION=None, GPTKIT_DEBUG=None, GPTKIT_BASE_URL=None, OPENAI_BASE_URL=None)
    e.update(envs or {})
    try:
        with env(**e), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                K.main()
            except SystemExit as x:
                code = x.code if isinstance(x.code, int) else (0 if x.code is None else 1)
                if isinstance(x.code, str):
                    buf.write(x.code + "\n")
            except BaseException:  # anything else would have been a traceback for the user
                tb, code = traceback.format_exc(), "TB"
    finally:
        K.requests.post, K.requests.get, K.time.sleep, threading.excepthook, sys.argv = saved
    out = buf.getvalue()
    if tbs or "Traceback (most recent call last)" in out:
        tb = (tb or "") + "\n".join(tbs) + out
    return Run(code, out, tb, calls, sleeps)


def clean_fail(r, *needles, code=None):
    """No traceback, a non-zero exit (or `code`), and every needle somewhere in the output."""
    assert r.tb is None, r.tb[-3000:]
    assert (r.code != 0) if code is None else (r.code == code), (r.code, r.out[-1500:])
    for n in needles:
        assert n in r.out, (n, r.out[-1500:])


# ---------------------------------------------------------------- fixtures

_FX = {}


def _flame(size=256):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([size * .1, size * .1, size * .9, size * .9], fill="#D1C4E9")
    d.polygon([(size * .5, size * .15), (size * .25, size * .8), (size * .75, size * .8)], fill="#FF7043")
    return im


def _spec():
    return {"palette": [{"name": "bg", "hex": "#EDE7F6", "usage": "background"}, {"name": "accent", "hex": "#5E35B1", "usage": "button"}],
            "typography": [{"role": "Button", "sample": "Continue", "family_guess": "Inter", "size_px": 20, "weight": 700,
                            "letter_spacing_em": 0, "color": "#FFFFFF", "platform_style": "labelLarge"}],
            "radii": [{"element": "button", "px": 12}], "spacing": [{"between": "edge", "px": 16}],
            "components": [{"name": "Button", "bbox": {"x": 300, "y": 1200, "w": 400, "h": 100}, "native": "Button", "notes": "Continue"}],
            "platform": "android", "px_to_dp": 0.4, "px_to_dp_design": 0.4}


def _build(root):
    """mockup + assets.json (png flame, svg drop, background sky) + design.json + design_spec.json + assets/flame.png +
    svg/drop.svg + svg_report.json + forge_report.json (one candidate) + a 4-frame guides layout + a rig layers.json."""
    P = root / "p"
    P.mkdir(parents=True)
    m = Image.new("RGB", (W, H), "#EDE7F6")
    d = ImageDraw.Draw(m)
    d.rectangle([0, 60, W, 500], fill=(40, 50, 120))
    d.ellipse([70, 600, 190, 720], fill="#D1C4E9")
    d.polygon([(130, 610), (100, 690), (160, 690)], fill="#FF7043")
    d.ellipse([400, 900, 480, 980], fill="#4FC3F7")
    d.ellipse([420, 920, 460, 960], fill="#0288D1")
    d.rectangle([300, 1200, 700, 1300], fill="#5E35B1")
    m.save(P / "mockup.png")
    assets = [
        {"name": "flame", "kind": "illustration", "format": "png", "bbox": {"x": 70, "y": 600, "w": 120, "h": 120},
         "transparent": True, "description": "orange flame on a lilac circle", "colors": ["#FF7043"], "animated": False},
        {"name": "drop", "kind": "icon", "format": "svg", "bbox": {"x": 395, "y": 895, "w": 90, "h": 90},
         "transparent": True, "description": "blue concentric drop", "colors": ["#4FC3F7"], "animated": False},
        {"name": "sky", "kind": "background", "format": "png", "bbox": {"x": 0, "y": 60, "w": 1024, "h": 440},
         "transparent": False, "description": "night sky", "colors": [], "animated": False}]
    (P / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": W, "height": H, "platform": "android",
                                               "screen_dp": 412, "assets": assets}, indent=1))
    (P / "design.json").write_text(json.dumps({"platform": "android", "screen_dp": 412}))
    (P / "design_spec.json").write_text(json.dumps(_spec()))
    (P / "assets").mkdir()
    _flame().save(P / "assets" / "flame.png")
    (P / "raw").mkdir()
    _flame().save(P / "raw" / "flame_c1.png")
    cand = {"cand": 1, "path": "raw/flame_c1.png", "score": {"fid": 20.0, "flags": []}, "meta": {}}
    (P / "forge_report.json").write_text(json.dumps({"flame": {"candidates": [cand], "chosen": cand, "complete": True}}))
    (P / "svg").mkdir()
    (P / "svg" / "drop.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">'
                                        '<circle cx="12" cy="12" r="9" fill="#4FC3F7"/><circle cx="12" cy="12" r="4" fill="#0288D1"/></svg>')
    (P / "svg_report.json").write_text(json.dumps({"drop": {"dp": 24}}))
    g = root / "guides"
    g.mkdir()
    for i in range(4):
        im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([20 + i * 10, 40, 60 + i * 10, 80], fill=(200, 200, 200, 255))
        im.save(g / f"{i}.png")
    for argv in (("sequence", P, "smoke", "layout", "--guides", g, "--fps", "4"),
                 ("rig", P, "flame", "--parts", "disc=the lilac disc; fire=the flame", "--assign", "disc=1; fire=2", "--allow-empty")):
        r = cli(*argv)
        assert r.code == 0 and r.tb is None, (argv, r.out[-800:], r.tb)
    assert (P / "sequence" / "smoke" / "sequence.json").exists() and (P / "rig" / "flame" / "layers.json").exists()


def fixture():
    """A fresh copy of the shared project (built once per process); returns (tmp root, project dir)."""
    if "src" not in _FX:
        _FX["src"] = Path(tempfile.mkdtemp(prefix="c2fx_"))
        _build(_FX["src"])
    t = Path(tempfile.mkdtemp(prefix="c2_"))
    shutil.copytree(_FX["src"], t, dirs_exist_ok=True)
    return t, t / "p"


# ---------------------------------------------------------------- flags and paths


def test_bad_flags_exit_naming_the_flag():
    t, P = fixture()
    flame = P / "assets" / "flame.png"
    L = ("sequence", P, "s", "layout", "--base", flame, "--sprites", flame, "--vent", "128,200")
    cases = [  # (argv, exit code, needles)
        (("research", "x", "--out", t / "r", "--jobs", "0"), 2, ["--jobs", "positive integer"]),
        (("mockup", "p", "--out", t / "m", "--size", "abc"), 1, ["--size must be WIDTHxHEIGHT"]),
        (("mockup", "p", "--out", t / "m", "--variants", "nan"), 2, ["--variants"]),
        (("critique", P / "mockup.png", "--screen-dp", "nan"), 2, ["--screen-dp"]),
        (("critique", P / "mockup.png", "--screen-dp", "-3"), 2, ["--screen-dp"]),
        (("forge", P, "--jobs", "1e9"), 2, ["--jobs"]),
        (("forge", P, "--only", "zzz"), 1, ["unknown asset(s): zzz", "known: flame, drop, sky"]),
        (("svg", P, "--grid", "nan"), 1, ["--grid nan"]),
        (("svg", P, "--grid", "-5"), 1, ["--grid -5"]),
        (("svg", P, "--pad", "-100"), 1, ["--pad -100"]),
        (("svg", P, "--pad", "12"), 1, ["less than half of --grid"]),
        (("icons", P, "bell", "--stroke", "nan"), 2, ["--stroke"]),
        (("vd", P / "svg" / "drop.svg", "--out", t / "vd", "--dp", "inf"), 2, ["--dp"]),
        (("frames", P, "flame", "--procedural", "float", "--count", "100000"), 1, ["--count 100000"]),
        (("frames", P, "flame", "--procedural", "float", "--fps", "1e-9"), 1, ["--fps 1e-09 must be between 1 and 120"]),
        (("frames", P, "flame", "--procedural", "float", "--amount", "1e308"), 1, ["--amount"]),
        (("frames", P, "flame", "--procedural", "float", "--preview-bg", "zz"), 2, ["--preview-bg", "colour"]),
        (("frames", P, "flame", "--procedural", "float", "--formats", "mp4"), 1, ["--formats"]),
        (L + ("--buoyancy", "nan"), 1, ["--buoyancy nan"]),
        (L + ("--buoyancy", "1e308"), 1, ["--buoyancy"]),
        (L + ("--grow", "0,0"), 1, ["--grow 0,0"]),
        (L + ("--fps", "1e5"), 1, ["300000 frames", "limit is 1000"]),
        (L + ("--life", "1e9"), 1, ["--life", "limit 2000"]),
        (L + ("--chuff", "1e-9"), 1, ["--chuff"]),
        (L + ("--size", "1e9"), 1, ["--size 1e+09"]),
        (L + ("--velocity", "nan,1"), 1, ["--velocity expects x,y numbers"]),
        (L + ("--loop", "1e6"), 1, ["sequence layout: --loop"]),
        (("sequence", P, "s", "layout", "--preview-bg", "zz"), 2, ["--preview-bg"]),
        (("vectorize", P, "flame", "--moving", "x", "--keyframes", "1000"), 1, ["--keyframes 1000"]),
        (("vectorize", P, "flame", "--moving", "x", "--preview-bg", "red;}body{x"), 2, ["--preview-bg"]),
        (("animate", P / "rig" / "flame" / "layers.json", "--part", "fire=flicker", "--keyframes", "100000"), 2, ["--keyframes"]),
        (("animate", P / "rig" / "flame" / "layers.json", "--part", "fire=flicker:nan"), 1, ["amp/phase out of range"]),
        (("animate", P / "rig" / "flame" / "layers.json", "--part", "nope=flicker"), 1, ["no static layer 'nope'"]),
        (("rig", P, "flame", "--parts", "a=b", "--keyframes", "100000"), 2, ["--keyframes"]),
        (("trace", flame, "--out", t / "t", "--bg", "notacolor"), 2, ["--bg"]),
        (("trace", flame, "--out", t / "t", "--tol", "nan"), 2, ["--tol"]),
        (("export", flame, "--dp", "1e6", "--res", t / "res"), 1, ["the limit is 8192 px", "not device pixels"]),
        (("export", flame, "--dp", "24", "--res", P / "mockup.png"), 1, ["--res", "is a file"]),
        (("export", flame, "--dp", "24", "--res", t / "res", "--quality", "1000"), 1, ["--quality 1000"]),
        (("appicon", "concepts", "b", "--out", t / "ac", "--crop", "0,0,0,0", "--mockup", P / "mockup.png"), 1, ["--crop 0,0,0,0"]),
        (("appicon", "build", flame, "--out", t / "ab", "--name", "Bad-Name"), 1, ["use --name bad_name"]),
        (("compare", P, P / "mockup.png", "--crop-top", "100000"), 1, ["--crop-top 100000"]),
        (("compare", P, P / "mockup.png", "--fail-below", "nan"), 1, ["--fail-below"]),
        (("tokens", P, "--platforms", "zzz"), 1, ["--platforms: unknown 'zzz'"]),
        (("tokens", P, "--package", "com.1bad..pkg"), 1, ["--package"]),
    ]
    for argv, code, needles in cases:
        r = cli(*argv)
        clean_fail(r, *needles, code=code)
        assert not r.calls, (argv, "no API call before the flags are validated")
    # a web --name with path separators is sanitised (printed) and never escapes --res
    r = cli("export", flame, "--dp", "24", "--res", t / "w", "--platform", "web", "--name", "../../evil")
    assert r.code == 0 and "renamed" in r.out and "evil" in r.out, r.out[-500:]
    assert sorted(q.name for q in (t / "w").rglob("*.webp")) == ["evil.webp", "evil@2x.webp"] and not (t / "evil.webp").exists()


def test_missing_and_misplaced_inputs():
    t, P = fixture()
    nope = t / "nope"
    cases = [
        (("critique", nope / "m.png"), ["mockup not found", "m.png"]),
        (("manifest", P / "assets.json", "--out", t / "mm"), ["cannot read mockup", "assets.json"]),
        (("spec", nope), ["project dir not found", "run manifest first"]),
        (("forge", nope), ["project dir not found"]),
        (("fix", P, "zzz"), ["unknown asset: zzz"]),
        (("pick", P, "flame", "7"), ["candidate 7 not found", "have: c1"]),
        (("score", P, "flame", P / "assets.json"), ["cannot read file", "assets.json"]),
        (("vectorize", nope, "flame", "--moving", "x"), ["assets.json not found", "--svg FILE"]),
        (("vectorize", P, "flame", "--rebuild"), ["layers.json not found", "without --rebuild"]),
        (("vectorize", P, "flame", "--moving", "x", "--svg", P / "mockup.png"), ["mockup.png is not a text SVG file"]),
        (("sequence", P, "smoke2", "render"), ["sequence render: no layout in", "sequence.json", "layout` first"]),
        (("sequence", P, "smoke2", "build"), ["sequence build: no layout in"]),
        (("sequence", P, "s", "layout", "--guides", nope), ["guide folder not found", "nope"]),
        (("animate", nope / "layers.json", "--part", "a=sway"), ["layers.json not found"]),
        (("tokens", nope), ["design_spec.json", "run `spec"]),
        (("compare", P, nope / "s.png"), ["screenshot not found"]),
        (("appicon", "build", nope / "c.png", "--out", t / "ab"), ["concept not found"]),
        (("mockup", "x", "--out", P / "mockup.png"), ["File exists", "a file is where the kit needs a directory"]),
        (("research", "x", "--out", P / "mockup.png"), ["a file is where the kit needs a directory"]),
        (("tokens", P, "--out", P / "mockup.png"), ["check --out/--res"]),
        (("vd", P / "svg" / "drop.svg", "--out", P / "mockup.png"), ["check --out/--res"]),
    ]
    for argv, needles in cases:
        clean_fail(cli(*argv), *needles)
    assert not (P / "sequence" / "smoke2").exists(), "a stage before layout leaves no empty folders behind"


# ---------------------------------------------------------------- damaged JSON files

GENERIC = ["", "{not json", "[]", "null", "42", '"str"', '{"a": 1']


def _json_cases(P):
    man = json.loads((P / "assets.json").read_text())

    def m(**kw):
        return json.dumps({**man, **kw})

    def a0(**kw):
        x = json.loads(json.dumps(man))
        x["assets"][0].update(kw)
        return json.dumps(x)
    sp = lambda **kw: json.dumps({**_spec(), **kw})
    seq = P / "sequence" / "smoke" / "sequence.json"
    lay = P / "rig" / "flame" / "layers.json"
    return [  # (file, contents, commands, needle naming the file)
        (P / "assets.json", GENERIC, [("forge", P, "--only", "flame"), ("tokens", P), ("redline", P)], "assets.json"),
        (P / "assets.json", [m(assets={}), m(assets=["x"]), m(mockup=5), a0(bbox="x"), a0(name=5)], [("forge", P, "--only", "flame")],
         "assets.json"),
        (P / "assets.json", [m(width="abc"), m(screen_dp="abc"), a0(name=5), a0(description=None), a0(bbox={"x": "a"})],
         [("tokens", P), ("redline", P)], None),  # sanitised: the command may succeed, but never a traceback
        (P / "design_spec.json", GENERIC + [sp(palette=[1, 2]), sp(typography=[{"role": "x", "size_px": "big"}]),
                                            sp(radii=[{"element": "r", "px": "x"}]), sp(components=["x"])],
         [("tokens", P), ("redline", P)], "design_spec.json"),
        (P / "design_spec.json", [sp(px_to_dp="abc"), sp(px_to_dp=-1), sp(palette="x")], [("tokens", P), ("redline", P)], None),
        (P / "forge_report.json", ['{"flame": "x"}', '{"flame": {"candidates": "x"}}', '{"flame": {"candidates": [{"cand": 1}]}}'],
         [("pick", P, "flame", "1"), ("fix", P, "flame")], "forge_report.json"),
        (P / "forge_report.json", ['{"flame": {"candidates": [], "chosen": 5}}'], [("pick", P, "flame", "1"), ("fix", P, "flame")],
         "damaged"),
        (P / "forge_report.json", GENERIC[1:2], [("pick", P, "flame", "1")], "forge report"),
        (seq, GENERIC + ['{"frames": "x"}', '{"frames": -1}', '{"frames": 4}', '{"frames": 4, "fps": "x", "canvas": 128}'],
         [("sequence", P, "smoke", s) for s in ("render", "build", "plan", "propagate", "reroll")], "sequence.json"),
        (lay, GENERIC + ['{"viewBox": "x", "static_layers": "x"}', '{"viewBox": "0 0 100 100", "static_layers": [5]}'],
         [("animate", lay, "--part", "fire=flicker")], "layers.json"),
        (P / "icon_spec.json", GENERIC, [("icons", P, "bell"), ("svg", P, "--check")], "icon_spec.json"),
        (P / "svg_report.json", ['{"drop": {"dp": "abc"}}', '{"drop": {"dp": -5}}', '{"drop": "x"}'],
         [("vd", P / "svg" / "drop.svg", "--out", P / "vdout", "--no-check")], None),
        (P / "fonts" / "fonts.json", ['{"families": "x", "styles": "x"}', '{"families": [], "styles": [5]}', "[]"],
         [("tokens", P)], "fonts.json"),
        (P / "redline" / "redline.json", ['{"items": "x", "radii": "x"}', '{"items": [5], "radii": []}'], [("tokens", P)], "redline.json"),
    ]


def test_damaged_json_files_name_themselves():
    t, P = fixture()
    for path, contents, commands, needle in _json_cases(P):
        orig = path.read_text() if path.exists() else None
        for text in contents:
            for argv in commands:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                r = cli(*argv, api="refusal")
                assert r.tb is None, (path.name, text[:60], argv[0], r.tb[-2500:])
                if needle:
                    assert r.code != 0 and needle in r.out, (path.name, text[:60], argv[:4], r.code, r.out[-600:])
        if orig is None:
            path.unlink()
        else:
            path.write_text(orig)
    # the stage that replaces a damaged sequence.json is never blocked by it
    seq = P / "sequence" / "smoke" / "sequence.json"
    seq.write_text("{not json")
    r = cli("sequence", P, "smoke", "layout", "--guides", t / "guides", "--fps", "4")
    assert r.code == 0 and r.tb is None and "starting a new one" in r.out, r.out[-800:]
    assert json.loads(seq.read_text())["frames"] == 4


def test_unexpected_error_is_one_line_unless_debug():
    t, P = fixture()
    (P / "icon_spec.json").write_text('{"icons": [5]}')  # passes the object check, then breaks deeper in svg
    r = cli("svg", P, api="refusal")
    clean_fail(r, "gptkit svg: unexpected", "GPTKIT_DEBUG=1")
    r = cli("svg", P, api="refusal", envs={"GPTKIT_DEBUG": "1"})
    assert r.code == "TB" and "TypeError" in r.tb


# ---------------------------------------------------------------- odd images


def _odd_images(t):
    base = _flame()
    out = {}

    def save(name, im, fmt="PNG", **kw):
        p = t / f"{name}.{'jpg' if fmt == 'JPEG' else 'png'}"
        im.save(p, fmt, **kw)
        out[name] = p
    save("P", base.quantize(64))
    save("L", base.convert("L"))
    save("LA", base.convert("LA"))
    save("I16", Image.fromarray(np.asarray(base.convert("L")).astype(np.uint16) * 257).convert("I;16"))
    ex = Image.Exif()
    ex[0x0112] = 6  # rotate 90
    save("CMYK_exif", base.convert("RGB").resize((300, 200)).convert("CMYK"), "JPEG", exif=ex.tobytes())
    save("1x1", Image.new("RGBA", (1, 1), (255, 0, 0, 255)))
    save("transparent", Image.new("RGBA", (256, 256), (0, 0, 0, 0)))
    save("nonsquare", base.resize((256, 64)))
    (t / "trunc.png").write_bytes(out["P"].read_bytes()[:200])
    out["trunc"] = t / "trunc.png"
    (t / "empty.png").write_bytes(b"")
    out["empty"] = t / "empty.png"
    return out


def test_odd_image_modes():
    t, P = fixture()
    ims = _odd_images(t)
    slow = {"trace": ("CMYK_exif", "transparent", "trunc"), "pick": ("LA", "transparent", "empty")}  # trace runs Chrome: ~3 s each
    for name, f in ims.items():
        for argv in (("export", f, "--dp", "24", "--res", t / "res"), ("export", f, "--dp", "24", "--res", t / "A.xcassets", "--platform", "ios"),
                     ("trace", f, "--out", t / "tr"), ("score", P, "flame", f), ("pick", P, "flame", f)):
            if argv[0] in slow and name not in slow[argv[0]]:
                continue
            r = cli(*argv)
            assert r.tb is None, (name, argv[0], r.tb[-2000:])
            if name in ("trunc", "empty"):
                assert r.code != 0 and (str(f) in r.out or f.name in r.out), (name, argv[0], r.out[-500:])
            elif name == "transparent" and argv[0] in ("trace", "pick"):
                assert r.code != 0, (name, argv[0])
            else:
                assert r.code == 0, (name, argv[0], r.out[-800:])
    # CMYK JPEG with EXIF orientation 6: exported upright (portrait 2:3), as RGB(A) PNG for iOS
    im = Image.open(t / "A.xcassets" / "CMYK_exif.imageset" / "CMYK_exif@1x.png")
    assert im.height > im.width and im.mode in ("RGB", "RGBA"), (im.size, im.mode)
    # a huge source is animated from a 2048 px copy instead of running out of memory
    big = Image.new("RGBA", (2560, 2560), (0, 0, 0, 0))
    ImageDraw.Draw(big).ellipse([400, 400, 2100, 2100], fill=(255, 100, 0, 255))
    big.save(P / "assets" / "big.png", compress_level=1)
    r = cli("frames", P, "big", "--procedural", "float", "--count", "2", "--formats", "gif", "--px", "64")
    assert r.code == 0 and r.tb is None and "animating a 2048x2048 copy" in r.out, r.out[-800:]


# ---------------------------------------------------------------- API failure shapes (contract C2)


def _api_commands(t, P):
    _flame(512).save(t / "concept.png")
    return {  # command -> (argv, which endpoint fails first: "img" or "txt")
        "research": (("research", "an idea", "--out", t / "r"), "txt"),
        "blueprint": (("blueprint", t / "b", "--idea", "x"), "txt"),
        "mockup": (("mockup", "a screen", "--out", t / "m", "--variants", "2"), "img"),
        "mockup_edit": (("mockup", "a screen", "--out", t / "m2", "--edit", P / "mockup.png"), "img"),
        "critique": (("critique", P / "mockup.png"), "txt"),
        "manifest": (("manifest", P / "mockup.png", "--out", t / "mm"), "txt"),
        "spec": (("spec", P), "txt"),
        "forge": (("forge", P, "--only", "sky", "--candidates", "2"), "img"),
        "fix": (("fix", P, "flame"), "txt"),
        "svg": (("svg", P), "txt"),
        "iconspec": (("iconspec", P), "txt"),
        "icons": (("icons", P, "bell, moon"), "txt"),
        "frames": (("frames", P, "flame", "--moving", "the flame", "--count", "4"), "img"),
        "sequence": (("sequence", P, "smoke", "render", "--jobs", "2"), "img"),
        "vectorize": (("vectorize", P, "flame", "--moving", "the flame"), "txt"),
        "concepts": (("appicon", "concepts", "brief", "--out", t / "ac", "--variants", "2"), "img"),
        "rig_gpt": (("rig", P, "flame", "--parts", "disc=the lilac disc; fire=the flame", "--route", "gpt", "--namer", "none"), "img"),
        "rig_namer": (("rig", P, "flame", "--parts", "disc=the lilac disc; fire=the flame"), "txt"),
        "tokens": (("tokens", P, "--vision"), "txt"),
    }


def test_api_failure_shapes_through_every_command():
    heavy = {"rig_gpt", "rig_namer", "tokens", "icons"}  # slower commands: the HTTP-status shapes are shared code, run them on a subset
    for mode, (_, (img_needle, txt_needle)) in MODES.items():
        for name in _api_commands(*fixture()):
            if name in heavy and mode not in ("moderation", "data_empty", "not_json"):
                continue
            argv, first = _api_commands(*fixture())[name]
            r = cli(*argv, api=mode)
            assert r.tb is None, (mode, name, r.tb[-2500:])
            needle = img_needle if first == "img" else txt_needle
            if mode == "jpeg" and first == "img":
                continue  # the image call succeeded: only "no traceback" is asserted
            assert needle in r.out, (mode, name, needle, r.out[-1200:])
            if name != "tokens":  # tokens --vision falls back to local text location and succeeds
                assert r.code != 0, (mode, name, r.out[-800:])
            assert "prompt: '" in r.out or 'prompt: "' in r.out, (mode, name, "the failing prompt is quoted", r.out[-1200:])
            if mode == "quota":
                assert len(r.calls) == len(set(r.calls)) or name in ("mockup", "forge", "sequence", "concepts", "rig_gpt"), \
                    (name, "insufficient_quota is never retried", r.calls)
                assert not r.sleeps, (name, r.sleeps)
            if mode == "retry_after":
                assert 3.0 in r.sleeps, (name, "Retry-After honoured", r.sleeps)


def test_bearer_key_never_printed():
    t, P = fixture()
    key = "sk-live-SECRET-0123456789"
    r = cli("spec", P, api=lambda u, j, d: _err(401, "invalid_api_key", f"Incorrect API key provided: Bearer {key}"),
            envs={"GPTKIT_API_KEY": key})
    clean_fail(r, "[invalid_api_key]", "Bearer [redacted]")
    assert key not in r.out and "SECRET" not in r.out


# ---------------------------------------------------------------- partial results


def _ok_image(url, j, d):
    return R(200, {"data": [{"b64_json": _b64("PNG")}]})


def _nth_fails(n, ok=_ok_image, fail=lambda: _err(400, "moderation_blocked", "Your request was rejected by the safety system.")):
    """An API whose n-th call (1-based, in call order) fails with moderation_blocked; the others use `ok`."""
    seen, lock = [0], threading.Lock()

    def api(url, j, d):
        with lock:
            seen[0] += 1
            k = seen[0]
        return fail() if k == n else ok(url, j, d)
    return api


def test_partial_results_are_kept():
    # forge: one of two assets fails -> the other is forged and in the report; the failed one is named with its re-run line
    t, P = fixture()
    r = cli("forge", P, "--only", "flame,sky", "--candidates", "1", "--jobs", "1", "--force", api=_nth_fails(2))
    assert r.tb is None and "moderation_blocked" in r.out and "prompt:" in r.out, r.out[-1500:]
    rep = json.loads((P / "forge_report.json").read_text())
    assert sum(1 for n in ("flame", "sky") if rep.get(n, {}).get("chosen")) >= 1, rep.keys()
    assert "re-run: forge" in r.out or "failed:" in r.out, r.out[-600:]
    # sequence render: the frame whose call fails stays missing, the others are written and a re-run renders only it
    r = cli("sequence", P, "smoke", "render", "--jobs", "1", api=_nth_fails(1, fail=lambda: _err(400, "invalid_request_error", "bad")))
    raw = sorted(p.name for p in (P / "sequence" / "smoke" / "raw").glob("[0-9][0-9][0-9].png"))
    assert r.tb is None and len(raw) == 3 and "1 of 4 frames still missing" in r.out and "finished frames are kept" in r.out, (raw, r.out[-800:])
    r = cli("sequence", P, "smoke", "render", "--jobs", "1", api=_ok_image)
    assert r.code == 0 and len(r.calls) == 1, (r.calls, r.out[-500:])
    # appicon concepts: variant 2 of 3 fails -> 1 and 3 are written and listed in concepts.json
    r = cli("appicon", "concepts", "moon app", "--out", t / "ac", "--variants", "3", "--jobs", "1", api=_nth_fails(2))
    assert r.tb is None and "moderation_blocked" in r.out, r.out[-800:]
    pngs = sorted(p.name for p in (t / "ac").glob("concept_v*.png"))
    assert pngs == ["concept_v1.png", "concept_v3.png"], pngs
    assert json.loads((t / "ac" / "concepts.json").read_text()), "concepts.json lists the good variants"
    # rig --route gpt: one part isolation fails -> the other is cached in parts_gpt/ and a re-run makes one call
    r = cli("rig", P, "flame", "--parts", "disc=the lilac disc; fire=the flame", "--route", "gpt", "--namer", "none",
            api=_nth_fails(1))
    assert r.tb is None and r.code != 0 and "1 succeeded and are cached" in r.out, r.out[-800:]
    assert len(list((P / "rig" / "flame" / "parts_gpt").glob("*_raw.png"))) == 1
    r = cli("rig", P, "flame", "--parts", "disc=the lilac disc; fire=the flame", "--route", "gpt", "--namer", "none",
            api=_nth_fails(1))  # the re-run asks only for the missing part (failing it again keeps the test fast)
    assert r.tb is None and len(r.calls) == 1 and "isolating 1 part(s)" in r.out and "next to 1 from an earlier run" in r.out, \
        (r.calls, r.out[-800:])


def test_svg_batches_keep_the_good_ones():
    t, P = fixture()
    man = json.loads((P / "assets.json").read_text())
    icon = man["assets"][1]
    man["assets"] += [dict(icon, name=f"ic{i}", bbox={"x": 395, "y": 895, "w": 90, "h": 90}) for i in range(7)]
    (P / "assets.json").write_text(json.dumps(man))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="#4FC3F7"/></svg>'

    def ok(url, j, d):
        names = j["text"]["format"]["schema"]["properties"]["files"]["items"]["properties"]["name"].get("enum", [])
        return R(200, _msg([{"type": "output_text", "text": json.dumps({"files": [{"name": n, "svg": svg} for n in names]})}]))
    r = cli("svg", P, "--no-retry", api=_nth_fails(1, ok=ok))  # 8 icons = 2 batches of 6 + 2; the first request fails
    assert r.tb is None and "moderation_blocked" in r.out, r.out[-1500:]
    written = {p.stem for p in (P / "svg").glob("*.svg")}
    assert len(written - {"drop"}) >= 2, written  # the batch that answered is on disk


# ---------------------------------------------------------------- Ctrl-C


def test_ctrl_c_in_a_worker_leaves_a_resumable_state():
    for argv_fn, again in (
        (lambda t, P: ("forge", P, "--only", "flame,sky", "--candidates", "2", "--jobs", "2", "--force"), True),
        (lambda t, P: ("sequence", P, "smoke", "render", "--jobs", "2"), True),
        (lambda t, P: ("appicon", "concepts", "brief", "--out", t / "ac", "--variants", "3", "--jobs", "1"), False),
        (lambda t, P: ("rig", P, "flame", "--parts", "disc=a; fire=b", "--route", "gpt", "--namer", "none"), False),
        (lambda t, P: ("mockup", "a screen", "--out", t / "m", "--variants", "2"), False),
    ):
        t, P = fixture()
        argv = argv_fn(t, P)
        r = cli(*argv, api=_ok_image, interrupt=lambda n: n == 2)
        assert r.code == 130 and r.tb is None and f"gptkit {argv[0]}: interrupted" in r.out, (argv[:3], r.code, r.out[-800:])
        for j in t.rglob("*.json"):
            json.loads(j.read_text())  # every state file is still valid JSON
        assert not list(t.rglob("*.json.tmp")) and not list(t.rglob("*.png.tmp")), "no half-written state"
        if again:
            r = cli(*argv, api=_ok_image)
            assert r.tb is None and r.code in (0, 1), (argv[:3], r.out[-800:])


def test_real_sigint_exits_at_once():
    """SIGINT while two pool workers sit in a (mocked) 60 s API call: the CLI exits 130 within seconds, not after the
    calls return (the executor would otherwise join its threads at interpreter exit)."""
    with tempfile.TemporaryDirectory() as td:
        code = (f"import sys, time, runpy, requests\nsys.path.insert(0, {str(SCRIPTS)!r})\n"
                "def post(*a, **k):\n    print('posted', flush=True); time.sleep(60)\n"
                "requests.post = post\n"
                f"sys.argv = ['gptkit.py', 'mockup', 'a screen', '--out', {str(Path(td) / 'm')!r}, '--variants', '2']\n"
                f"runpy.run_path({str(SCRIPTS / 'gptkit.py')!r}, run_name='__main__')\n")
        e = {**os.environ, "GPTKIT_API_KEY": "sk-test-1234567890", "GPTKIT_CONFIG": "/nonexistent/env", "GPTKIT_RETRIES": "0"}
        e.pop("GPTKIT_DEBUG", None)
        p = subprocess.Popen([sys.executable, "-c", code], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t0 = time.time()
        while time.time() - t0 < 20:
            line = p.stdout.readline()
            if "posted" in line:
                break
        time.sleep(0.3)
        p.send_signal(signal.SIGINT)
        t1 = time.time()
        try:
            out, err = p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            raise AssertionError("Ctrl-C did not stop the run while API calls were in flight")
        assert p.returncode == 130 and time.time() - t1 < 10, (p.returncode, time.time() - t1)
        assert "Traceback" not in err and "gptkit mockup: interrupted" in err, err


if __name__ == "__main__":
    t_all = time.time()
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            t = time.time()
            fn()
            print(f"ok {k} ({time.time() - t:.1f}s)")
    print(f"total {time.time() - t_all:.1f}s")
