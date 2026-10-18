"""forge / fix / pick / score, perceptual metrics, _load / _rgba. Offline: the API is mocked.
Run: python3 tests/test_forge.py (or pytest)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K  # noqa: E402
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger

W, H = 1024, 1536


def png(im, fmt="PNG"):
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def project(root, assets=None, extra=None):
    """A synthetic 1024x1536 design with a flame on a lilac circle, a drop icon, a sky background and a tiny square."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    m = Image.new("RGB", (W, H), "#EDE7F6")
    d = ImageDraw.Draw(m)
    sky = np.zeros((440, W, 3), np.uint8)
    for y in range(440):
        sky[y] = (40 + y // 6, 50 + y // 8, 120 + y // 5)
    m.paste(Image.fromarray(sky), (0, 60))
    d.ellipse([700, 120, 820, 240], fill="#FFD9B3")
    d.text((40, 100), "Good evening", fill="white")
    d.ellipse([70, 600, 190, 720], fill="#D1C4E9")
    d.polygon([(130, 610), (100, 690), (160, 690)], fill="#FF7043")
    d.ellipse([110, 650, 150, 700], fill="#FFB74D")
    d.ellipse([400, 900, 480, 980], fill="#4FC3F7")
    d.ellipse([420, 920, 460, 960], fill="#0288D1")
    d.rectangle([600, 900, 609, 909], fill="#E91E63")
    d.ellipse([990, 1100, 1060, 1170], fill="#8BC34A")
    m.save(root / "mockup.png")
    if assets is None:
        assets = [
            {"name": "flame", "kind": "illustration", "format": "png", "bbox": {"x": 70, "y": 600, "w": 120, "h": 120},
             "transparent": True, "description": "orange flame on a lilac circle", "colors": ["#FF7043"], "animated": False},
            {"name": "drop", "kind": "icon", "format": "png", "bbox": {"x": 395, "y": 895, "w": 90, "h": 90},
             "transparent": True, "description": "blue concentric drop", "colors": ["#4FC3F7"], "animated": False},
            {"name": "sky", "kind": "background", "format": "png", "bbox": {"x": 0, "y": 60, "w": 1024, "h": 440},
             "transparent": False, "description": "night sky with moon", "colors": [], "animated": False},
        ]
    man = {"mockup": "mockup.png", "width": W, "height": H, "platform": "android", "screen_dp": 412, "assets": assets}
    man.update(extra or {})
    (root / "assets.json").write_text(json.dumps(man, indent=1))
    return root


def art(ref, transparent, keep_backdrop=False, opaque_bg=None, S=640):
    """Fake redraw: segment the reference crop by distance from its border colour, then (like a real redraw) let the
    element fill 80% of an S px canvas with alpha. keep_backdrop/opaque return the whole crop instead."""
    r = ref.convert("RGB")
    a = np.asarray(r).astype(int)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)
    mask = np.abs(a - bg).sum(-1) > 40
    im = Image.fromarray(np.dstack([a, mask * 255]).astype("uint8"), "RGBA")
    if keep_backdrop or not transparent:
        im = r.convert("RGBA")
    else:
        im = im.crop(im.getchannel("A").getbbox() or (0, 0, *im.size))
    k = 0.8 * S / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    out = Image.new("RGBA", (S, S), opaque_bg or (0, 0, 0, 0))
    out.alpha_composite(im, ((S - im.width) // 2, (S - im.height) // 2))
    return out


class Mock:
    """gptkit.edit / gptkit.ask replacement: `plan` maps the call index (1-based) to bytes, an exception or 'ok'."""

    def __init__(self, plan=None, ask=None):
        self.plan, self.calls, self.prompts, self.asks = plan or {}, 0, [], []
        self.ask_result = ask
        import threading
        self.lock = threading.Lock()

    def edit(self, ref_png, prompt, size="1024x1024", background="auto", model=None, fidelity="high"):
        with self.lock:
            self.calls += 1
            n = self.calls
            self.prompts.append({"prompt": prompt, "size": size, "background": background})
        ref = Image.open(io.BytesIO(ref_png))
        meta = {"quality": "medium", "size": size, "background": background, "usage": None}
        what = self.plan.get(n, "ok")
        if isinstance(what, Exception):
            raise what
        if callable(what):
            what = what(ref, background)
        if what == "ok":
            return png(art(ref, background == "transparent")), meta
        if what == "jpeg":
            return png(art(ref, background == "transparent").convert("RGB"), "JPEG"), meta
        if what == "blank":
            return png(Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))), meta
        if what == "opaque":
            return png(art(ref, False, opaque_bg=(255, 255, 255, 255))), meta
        return what, meta

    def ask(self, parts, schema=None, name="result", model=None):
        self.asks.append({"parts": parts, "schema": schema})
        r = self.ask_result
        return r(parts) if callable(r) else r


@contextlib.contextmanager
def patched(mock):
    old = K.edit, K.ask
    K.edit, K.ask = mock.edit, mock.ask
    try:
        yield mock
    finally:
        K.edit, K.ask = old


def run(*argv):
    """Run the CLI in-process; returns (exit code, stdout+stderr)."""
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["gptkit.py", *map(str, argv)]
    code = 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                K.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                if isinstance(e.code, str):
                    buf.write(e.code + "\n")
    finally:
        sys.argv = old
    return code, buf.getvalue()


def load_quiet(d):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            r = K._load(d)
        except SystemExit as e:
            return None, str(e.code), buf.getvalue()
    return r, None, buf.getvalue()


# ---------------------------------------------------------------- G3-1: _load / _select / _rgba


def test_load_validation_and_defaults():
    with tempfile.TemporaryDirectory() as td:
        # missing dir / missing assets.json / corrupt JSON / missing keys: one line each, no traceback
        _, err, _ = load_quiet(Path(td) / "nope")
        assert err and "not found" in err and "nope" in err
        d = project(Path(td) / "p")
        (d / "assets.json").write_text("{not json")
        _, err, _ = load_quiet(d)
        assert err and "not valid JSON" in err
        (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": W, "height": H}))
        _, err, _ = load_quiet(d)
        assert err and "missing assets" in err
        # missing screen_dp -> filled from the platform, with a note; defaults applied and printed; spec's read works
        base = {"name": "flame", "kind": "illustration", "format": "png", "bbox": {"x": 70, "y": 600, "w": 120, "h": 120}}
        old = {"name": "sky", "kind": "background", "bbox": {"x": 0, "y": 60, "w": 1024, "h": 440}}
        man = {"mockup": "mockup.png", "width": W, "height": H, "platform": "android", "assets": [base, old]}
        (d / "assets.json").write_text(json.dumps(man))
        (_, m, mock), err, out = load_quiet(d)
        assert err is None and m["screen_dp"] == 412 and "screen_dp" in out
        assert m["assets"][0]["transparent"] is True and m["assets"][0]["description"] == "" and m["assets"][0]["colors"] == []
        assert m["assets"][1]["transparent"] is False and m["assets"][1]["format"] == "png"
        assert "defaults applied" in out and "flame" in out and "transparent=False" in out
        # bbox {x:985, w:80} on a 1024 mockup -> clamped and printed; floats/strings normalised to ints
        man["assets"] = [dict(base, bbox={"x": 985, "y": 1095, "w": 80, "h": 80}),
                         dict(base, name="f2", bbox={"x": 10.6, "y": "20", "w": 30.2, "h": "40"})]
        (d / "assets.json").write_text(json.dumps(man))
        (_, m, _), err, out = load_quiet(d)
        assert err is None and m["assets"][0]["bbox"] == {"x": 985, "y": 1095, "w": 39, "h": 80}
        assert "flame bbox clamped" in out and "was x=985" in out
        assert m["assets"][1]["bbox"] == {"x": 11, "y": 20, "w": 30, "h": 40}
        assert all(isinstance(v, int) for v in m["assets"][1]["bbox"].values())
        # {w:0}, garbage strings, far outside, no name, no bbox -> exit naming the asset
        for bad in ({"x": 1, "y": 1, "w": 0, "h": 5}, {"x": "a", "y": 1, "w": 5, "h": 5}, {"x": 5000, "y": 5000, "w": 10, "h": 10}):
            man["assets"] = [dict(base, name="zed", bbox=bad)]
            (d / "assets.json").write_text(json.dumps(man))
            _, err, _ = load_quiet(d)
            assert err and "zed" in err and "bbox" in err, (bad, err)
        man["assets"] = [{"kind": "icon", "bbox": {"x": 1, "y": 1, "w": 5, "h": 5}}]
        (d / "assets.json").write_text(json.dumps(man))
        _, err, _ = load_quiet(d)
        assert err and "needs 'name' and 'bbox'" in err
        man["assets"] = [{"name": "nobox", "kind": "icon"}]
        (d / "assets.json").write_text(json.dumps(man))
        _, err, _ = load_quiet(d)
        assert err and "nobox" in err
        # missing mockup file -> one line naming it
        man["assets"] = [base]
        (d / "assets.json").write_text(json.dumps(man))
        (d / "mockup.png").unlink()
        _, err, _ = load_quiet(d)
        assert err and "mockup not found" in err


def test_select_unknown():
    m = {"assets": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    assert [x["name"] for x in K._select(m, None)] == ["a", "b", "c"]
    assert [x["name"] for x in K._select(m, "c, a")] == ["a", "c"]
    for bad, fn in (("x", lambda: K._select(m, "a,x")), ("nosuch", lambda: K._asset(m, "nosuch"))):
        try:
            fn()
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert "unknown asset" in str(e.code) and bad in str(e.code) and "known: a, b, c" in str(e.code)
    assert K._asset(m, "b") is m["assets"][1]


def test_rgba_exif_and_16bit():
    with tempfile.TemporaryDirectory() as td:
        # a landscape JPEG with EXIF orientation 6 opens portrait
        im = Image.new("RGB", (80, 40), "red")
        ex = Image.Exif()
        ex[0x0112] = 6
        p = Path(td) / "rot.jpg"
        im.save(p, exif=ex.tobytes())
        r = K._rgba(p)
        assert r.mode == "RGBA" and r.size == (40, 80)
        # a 16-bit greyscale gradient PNG opens 8-bit with mean about 128
        g = np.linspace(0, 65535, 256, dtype=np.uint16)[None, :].repeat(16, 0)
        p16 = Path(td) / "g16.png"
        Image.fromarray(g, "I;16").save(p16)
        r = K._rgba(p16)
        arr = np.asarray(r)
        assert r.mode == "RGBA" and abs(arr[..., 0].mean() - 127.5) < 1.5 and arr[..., 3].min() == 255
        # a 32-bit int image with 8-bit values passes through unscaled; odd modes still open
        Image.fromarray(np.full((8, 8), 200, np.int32), "I").save(Path(td) / "i32.png")
        assert np.asarray(K._rgba(Path(td) / "i32.png"))[..., 0].mean() == 200
        for mode in ("P", "L", "LA", "1", "RGB"):
            q = Path(td) / f"m_{mode.lower()}.png"
            Image.new(mode, (8, 8)).save(q)
            assert K._rgba(q).mode == "RGBA"


# ---------------------------------------------------------------- metrics: Lab, fid ordering, flags


def test_lab_reference_and_roundtrip():
    lab = K._lab(np.array([[255, 255, 255], [255, 0, 0], [0, 0, 0]], np.uint8))
    assert np.allclose(lab[0], [100, 0, 0], atol=0.05) and np.allclose(lab[2], [0, 0, 0], atol=0.05)
    assert np.allclose(lab[1], [53.24, 80.09, 67.20], atol=0.1)
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (2000, 3), np.uint8)
    assert np.array_equal(K._lab_to_rgb(K._lab(rgb)), rgb)


def _flame_mock():
    m = Image.new("RGBA", (400, 400), "white")
    d = ImageDraw.Draw(m)
    d.ellipse([100, 100, 300, 300], fill="#D1C4E9")
    d.polygon([(200, 150), (160, 250), (240, 250)], fill="#FF7043")
    return m, {"x": 158, "y": 148, "w": 84, "h": 104}


def _tri(color="#FF7043", pts=((512, 100), (150, 924), (874, 924)), size=1024):
    a = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(a).polygon(list(pts), fill=color)
    return a


def test_fid_ordering():
    m, bb = _flame_mock()
    perfect = _tri()
    good = _tri(pts=((512, 130), (170, 900), (854, 900)))
    wrong_col = _tri("#42A5F5")
    wrong_shape = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    ImageDraw.Draw(wrong_shape).rectangle([150, 100, 874, 924], fill="#FF7043")
    f = [K.score(m, bb, a)["fid"] for a in (perfect, good, wrong_col, wrong_shape)]
    assert f[0] < f[1] < f[2] and f[0] < f[3], f
    assert K.score(m, bb, perfect)["flags"] == [] and K.score(m, bb, perfect)["iou"] > 0.9


def test_backdrop_flag_table():
    # the five cases of the g3 P0-3 prototype table: only kept cards/circles are flagged
    from PIL import ImageFont
    mock = Image.new("RGBA", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(mock)
    d.text((60, 100), "Berlin", fill="#1C1B1F", font=ImageFont.load_default(64))
    d.rounded_rectangle([48, 1380, 976, 1470], 45, fill="#1E88E5")

    def cloud(dr, o, k, w=3):
        for e in ([10, 30, 50, 60], [30, 15, 70, 55], [45, 30, 80, 60]):
            dr.ellipse([o[0] + e[0] * k, o[1] + e[1] * k, o[0] + e[2] * k, o[1] + e[3] * k], fill="white",
                       outline="#90A4AE", width=max(1, int(w * k)))
    cloud(d, (212, 400), 7)
    bbox = {"x": 212 + 70 - 4, "y": 400 + 105 - 4, "w": 7 * 70 + 8, "h": 7 * 45 + 8}
    a = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    cloud(ImageDraw.Draw(a), (100, 260), 10.5)
    sc = K.score(mock, bbox, a)
    assert sc["flags"] == [] and sc["fid"] < 15, sc  # white cloud on white page, correct
    b = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    ImageDraw.Draw(b).rounded_rectangle([40, 120, 984, 900], 80, fill="white")
    b.alpha_composite(a)
    assert "BACKDROP" in K.score(mock, bbox, b)["flags"]  # kept the white card
    m, bb = _flame_mock()
    assert K.score(m, bb, _tri())["flags"] == []  # tight bbox, correct flame
    kc = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    ImageDraw.Draw(kc).ellipse([20, 20, 1004, 1004], fill="#D1C4E9")
    ImageDraw.Draw(kc).polygon([(512, 250), (330, 700), (694, 700)], fill="#FF7043")
    assert "BACKDROP" in K.score(m, bb, kc)["flags"]  # kept the lilac circle
    mm = Image.new("RGBA", (100, 100), "#EDE7F6")
    ImageDraw.Draw(mm).rectangle([45, 45, 54, 54], fill="#E91E63")
    t = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    ImageDraw.Draw(t).rectangle([100, 100, 924, 924], fill="#E91E63")
    assert K.score(mm, {"x": 45, "y": 45, "w": 10, "h": 10}, t)["flags"] == []  # tiny square filling its bbox
    # the outside ring is the backdrop even for a bbox flush with the mockup edge
    assert K._backdrop(mm, {"x": 45, "y": 45, "w": 10, "h": 10}) == (237, 231, 246)
    assert K._backdrop(mm, {"x": 0, "y": 0, "w": 100, "h": 100}) == (237, 231, 246)


def test_flags_positive_negative_and_empty():
    m, bb = _flame_mock()
    blank = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    sc = K.score(m, bb, blank)
    assert sc["flags"] == ["EMPTY"], sc  # no CUT-OFF for an empty canvas
    speck = blank.copy()
    ImageDraw.Draw(speck).rectangle([500, 500, 510, 510], fill="#FF7043")  # 0.01% of the canvas
    assert "EMPTY" in K.score(m, bb, speck)["flags"]
    opaque = Image.new("RGBA", (1024, 1024), "white")
    ImageDraw.Draw(opaque).polygon([(512, 100), (150, 924), (874, 924)], fill="#FF7043")
    fl = K.score(m, bb, opaque)["flags"]
    assert "OPAQUE" in fl and "CUT-OFF" in fl and "EMPTY" not in fl
    cut = _tri(pts=((512, 0), (0, 1023), (1023, 1023)))
    assert "CUT-OFF" in K.score(m, bb, cut)["flags"]
    # COPY: the mockup itself and the reference crop
    ref = K.ref_crop(m, bb)
    assert "COPY:returned-mockup" in K.score(m, bb, m.copy())["flags"]
    assert "COPY:returned-reference" in K.score(m, bb, ref.convert("RGBA"), ref)["flags"]
    assert not any(f.startswith("COPY") for f in K.score(m, bb, _tri(), ref)["flags"])
    # opaque background asset: no OPAQUE/CUT-OFF/BACKDROP flags apply
    assert K.score(m, {"x": 0, "y": 0, "w": 400, "h": 200}, m.crop((0, 0, 400, 200)).resize((1536, 768)), None, False)["flags"] == []


def test_rank_weights():
    mk = lambda *flags, fid=10.0: {"fid": fid, "diff": fid, "flags": list(flags)}
    assert K._flag_weight(["EMPTY"]) == 1000 and K._flag_weight(["COPY:returned-mockup"]) == 100
    assert K._flag_weight(["OPAQUE", "CUT-OFF", "BACKDROP"]) == 23
    # an empty image with one flag no longer beats a real one with three
    assert K._rank(mk("OPAQUE", "CUT-OFF", "BACKDROP", fid=60), None) < K._rank(mk("EMPTY", fid=5), None)
    assert K._rank(mk("CUT-OFF", fid=30), None) < K._rank(mk("OPAQUE", fid=10), None)
    assert K._rank(mk(fid=30), None) < K._rank(mk("CUT-OFF", fid=10), None)
    assert K._rank(mk(fid=10), {"quality": "low"}) > K._rank(mk(fid=12), {"quality": "medium"})
    assert K._unusable(mk("EMPTY")) == "EMPTY" and K._unusable(mk("COPY:returned-reference")).startswith("COPY")
    assert K._unusable(mk("OPAQUE", "CUT-OFF")) is None


# ---------------------------------------------------------------- forge / fix / pick / score commands


def test_forge_partial_failure():
    # one undecodable body, one JPEG, one moderation error, one transport error: the run finishes, names the failures
    with tempfile.TemporaryDirectory() as td:
        d = project(td)
        plan = {1: b"<html>not an image</html>", 4: "jpeg",
                3: K.ApiError(400, 'HTTP 400 /images/edits: {"error":{"code":"moderation_blocked"}}'),
                5: RuntimeError("/images/edits failed after retries: HTTP 500")}
        with patched(Mock(plan)) as mk:  # --jobs 1: calls run in submission order flame c1,c2, drop c1,c2, sky c1,c2
            code, out = run("forge", d, "--candidates", "2", "--jobs", "1", "--no-color-match")
        assert mk.calls == 6 and "Traceback" not in out, out
        assert code == 0, out
        rep = json.loads((d / "forge_report.json").read_text())
        assert set(rep) == {"flame", "drop", "sky"} and all(rep[n]["complete"] for n in rep)
        assert [c["cand"] for c in rep["flame"]["candidates"]] == [2] and [c["cand"] for c in rep["sky"]["candidates"]] == [2]
        assert rep["drop"]["candidates"][0]["meta"]["source_format"] == "JPEG" and rep["drop"]["needs_attention"]
        assert "flame c1 failed" in out and "drop c1 failed" in out and "sky c1 failed" in out
        assert sorted(Path(d, "assets").glob("*.png")) == [d / "assets" / n for n in ("drop.png", "flame.png", "sky.png")]
        assert " failed: " in out and "no decodable image" in out and "moderation_blocked" in out and "HTTP 500" in out
        assert "provider returned JPEG" in out
        assert all(Image.open(p).format == "PNG" for p in Path(d, "raw").glob("*_c*.png"))
        # every stored path is relative to the project dir and carries its candidate number
        for e in rep.values():
            for c in e["candidates"]:
                assert not Path(c["path"]).is_absolute() and c["path"].startswith("raw/") and isinstance(c["cand"], int)
            assert e["final"].startswith("assets/")
        assert "forged 3/3 assets" in out and list(Path(d, "review").glob("forge_????????-??????_1.png"))


def test_forge_all_fail_exits_one():
    with tempfile.TemporaryDirectory() as td:
        d = project(td)
        err = K.ApiError(429, 'HTTP 429 /images/edits: {"error":{"code":"rate_limit_exceeded"}}')
        with patched(Mock({i: err for i in range(1, 7)})) as mk:
            code, out = run("forge", d, "--jobs", "2")
        assert code == 1 and mk.calls == 6 and "Traceback" not in out
        assert "forged 0/3 assets; failed: " in out and "--only" in out and "flame" in out and "sky" in out
        rep = json.loads((d / "forge_report.json").read_text())
        assert all(rep[n]["needs_attention"] and rep[n]["chosen"] is None for n in rep)
        assert not list(Path(d, "assets").glob("*.png"))
        # a rerun re-queues them (nothing is treated as done)
        with patched(Mock()) as mk:
            code, out = run("forge", d, "--jobs", "2", "--no-color-match")
        assert code == 0 and mk.calls == 6 and "forged 3/3" in out


def test_empty_candidate_never_shipped():
    flame = [{"name": "flame", "kind": "illustration", "format": "png", "bbox": {"x": 70, "y": 600, "w": 120, "h": 120},
              "transparent": True, "description": "flame", "colors": [], "animated": False}]
    with tempfile.TemporaryDirectory() as td:
        d = project(td, flame)
        # blank + opaque + good -> the good one wins
        with patched(Mock({1: "blank", 2: "opaque", 3: "ok"})):
            code, out = run("forge", d, "--candidates", "3", "--jobs", "1", "--no-color-match")
        rep = json.loads((d / "forge_report.json").read_text())["flame"]
        assert code == 0 and rep["chosen"]["cand"] == 3 and rep["chosen"]["score"]["flags"] == [], out
        assert [c["cand"] for c in rep["candidates"]] == [3, 2, 1]  # rank order: good, opaque, empty
        assert Image.open(d / "assets" / "flame.png").getchannel("A").getextrema()[1] == 255
        # blank + opaque only, forced: no final file written, one line says why, exit 1
        (d / "assets" / "flame.png").unlink()
        with patched(Mock({1: "blank", 2: "blank"})):
            code, out = run("forge", d, "--candidates", "2", "--jobs", "1", "--force")
        assert code == 1 and not (d / "assets" / "flame.png").exists()
        assert "no usable candidate" in out and "EMPTY" in out and "--only flame --force --candidates 3" in out
        rep = json.loads((d / "forge_report.json").read_text())["flame"]
        assert rep["needs_attention"] and rep["unusable"] == "EMPTY" and rep["final"] is None
        # a rerun without --force re-queues it (needs_attention), and a flagged-only asset is re-queued too
        with patched(Mock({1: "opaque", 2: "opaque"})) as mk:
            code, out = run("forge", d, "--candidates", "2", "--jobs", "1", "--no-color-match")
        assert mk.calls == 2 and code == 0 and (d / "assets" / "flame.png").exists() and "NEEDS ATTENTION" in out and "pick" in out
        with patched(Mock()) as mk:
            code, out = run("forge", d, "--candidates", "1", "--no-color-match")
        assert mk.calls == 1 and code == 0
        with patched(Mock()) as mk:
            code, out = run("forge", d)
        assert mk.calls == 0 and "nothing to forge" in out


def test_report_relative_and_merge():
    with tempfile.TemporaryDirectory() as td:
        d = project(Path(td) / "a")
        with patched(Mock()):
            code, out = run("forge", d, "--only", "flame", "--candidates", "1", "--no-color-match")
        assert code == 0
        with patched(Mock()):
            code, out = run("forge", d, "--only", "drop", "--candidates", "1", "--no-color-match")
        rep = json.loads((d / "forge_report.json").read_text())
        assert set(rep) == {"flame", "drop"}  # the second run kept the first run's entry
        # fix/pick from another cwd, with the project moved: relative paths still resolve
        moved = Path(td) / "elsewhere" / "proj"
        moved.parent.mkdir()
        os.rename(d, moved)
        old = os.getcwd()
        os.chdir(Path(td) / "elsewhere")
        try:
            with patched(Mock()):
                code, out = run("pick", "proj", "flame", "1", "--no-color-match")
            assert code == 0 and "flame <- " in out, out
            with patched(Mock(ask={"differences": ["too tall"], "edit_instruction": "make it shorter"})) as mk:
                code, out = run("fix", "proj", "drop", "--no-color-match")
            assert code == 0 and mk.calls == 1 and "round 1: fid" in out, out
        finally:
            os.chdir(old)
        rep = json.loads((moved / "forge_report.json").read_text())
        assert rep["flame"]["picked_manually"] and rep["flame"]["final"] == "assets/flame.png"
        assert rep["drop"]["candidates"][-1]["cand"] == "fix1" and (moved / "raw" / "drop_fix1.png").exists()
        # an older report with cwd-relative paths is read via the raw/<basename> fallback
        rep["flame"]["chosen"]["path"] = "somewhere/else/raw/flame_c1.png"
        rep["flame"]["color_match"] = None
        (moved / "forge_report.json").write_text(json.dumps(rep))
        assert K._current(moved, "flame", rep["flame"]) == moved / "raw" / "flame_c1.png"
        # concurrent-safe write: entries added by someone else between read and write survive
        K._update_report(moved, {"zzz": {"complete": True}})
        assert set(json.loads((moved / "forge_report.json").read_text())) == {"flame", "drop", "zzz"}


def test_ctrl_c_cancels():
    import concurrent.futures as cf
    import time
    real = cf.as_completed

    def interrupting(futs, *a, **k):
        it = real(futs, *a, **k)
        yield next(it)
        raise KeyboardInterrupt

    def slow(n):  # staggered: the second worker is still busy when the interrupt lands after the first result
        def go(ref, background):
            time.sleep(0.4 + 0.8 * n)
            return "ok"
        return go
    with tempfile.TemporaryDirectory() as td:
        d = project(td)
        cf.as_completed = interrupting
        try:
            with patched(Mock({i: slow(i) for i in range(1, 7)})) as mk:
                code, out = run("forge", d, "--candidates", "2", "--jobs", "2", "--no-color-match")
        finally:
            cf.as_completed = real
        assert code == 130 and "interrupted" in out and mk.calls <= 3, (mk.calls, out)
        rep = json.loads((d / "forge_report.json").read_text())
        assert rep and any(not e["complete"] and e["candidates"] for e in rep.values())
        # the resume draws only the missing candidates and reuses the scored ones
        with patched(Mock()) as mk2:
            code, out = run("forge", d, "--candidates", "2", "--jobs", "2", "--no-color-match")
        assert code == 0 and mk2.calls == 6 - sum(len(e["candidates"]) for e in rep.values()) and "reused" in out
        rep = json.loads((d / "forge_report.json").read_text())
        assert all(e["complete"] and len(e["candidates"]) == 2 for e in rep.values())


def test_pick_matches_labels():
    with tempfile.TemporaryDirectory() as td:
        d = project(td, extra={})
        with patched(Mock({1: "opaque", 2: "ok", 3: "ok"})):
            code, out = run("forge", d, "--only", "flame", "--candidates", "3", "--jobs", "1", "--no-color-match")
        rep = json.loads((d / "forge_report.json").read_text())["flame"]
        assert rep["chosen"]["cand"] != 1 and rep["candidates"][-1]["cand"] == 1  # opaque one ranks last
        labels = [K._cand_label(c, rep["candidates"][0]) for c in rep["candidates"]]
        assert labels[0].startswith("BEST c") and labels[-1].startswith("c1 fid=") and "OPAQUE" in labels[-1]
        # picking the number on the label resolves to that raw file
        with patched(Mock()):
            code, out = run("pick", d, "flame", "1", "--no-color-match")
        rep = json.loads((d / "forge_report.json").read_text())["flame"]
        assert code == 0 and rep["chosen"]["path"] == "raw/flame_c1.png" and rep["needs_attention"] is True
        assert rep["picked_manually"] and rep["final_size"]
        # a review sheet tile label is wrapped, never blindly cut
        tile = K._tile(Image.new("RGBA", (50, 50), "red"), 220, "BEST c2 fid=23.5 OPAQUE CUT-OFF BACKDROP extra words")
        assert tile.size == (220, 220)
        assert K._mid("a_very_long_asset_name_indeed", 12) == "a_ver~indeed"


def test_pick_and_score_errors():
    with tempfile.TemporaryDirectory() as td:
        d = project(td)
        code, out = run("pick", d, "nosuch", "1")
        assert code == 1 and "unknown asset" in out and "known: flame, drop, sky" in out and "Traceback" not in out
        code, out = run("pick", d, "flame", "1")
        assert code == 1 and "run forge" in out and "--only flame first" in out
        code, out = run("fix", d, "flame")
        assert code == 1 and "run forge" in out
        with patched(Mock()):
            run("forge", d, "--only", "flame", "--candidates", "2", "--jobs", "1", "--no-color-match")
        code, out = run("pick", d, "flame", "9")
        assert code == 1 and "candidate 9 not found" in out and "(have: c1, c2)" in out
        code, out = run("pick", d, "flame", "raw/nope.png")
        assert code == 1 and "candidate must be" in out
        blank = d / "blank.png"
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(blank)
        code, out = run("pick", d, "flame", blank)
        assert code == 1 and "EMPTY" in out
        # score: JSON per file, COPY:returned-reference detectable, missing file one line
        code, out = run("score", d, "flame", d / "raw" / "flame_c1.png")
        assert code == 0 and json.loads(out.split(" ", 1)[1])["fid"] > 0
        ref = d / "ref.png"
        K.ref_crop(K._rgba(d / "mockup.png"), {"x": 70, "y": 600, "w": 120, "h": 120}).save(ref)
        code, out = run("score", d, "flame", ref)
        assert code == 0 and "COPY:returned-reference" in out
        code, out = run("score", d, "flame", d / "missing.png")
        assert code == 1 and "not found" in out and "missing.png" in out
        code, out = run("forge", "/nonexistent/dir")
        assert code == 1 and "not found" in out and "Traceback" not in out
        code, out = run("forge", d, "--only", "nosuch")
        assert code == 1 and "unknown asset" in out
        code, out = run("forge", d, "--only", "sky", "--include-svg", "--force", "--candidates", "0")
        assert code == 2 and "--candidates" in out and "positive integer" in out


def test_forge_argparse():
    for argv, flag in ((["forge", "x", "--jobs", "0"], "--jobs"), (["forge", "x", "--candidates", "nan"], "--candidates"),
                       (["fix", "x", "n", "--rounds", "-1"], "--rounds"), (["forge", "x", "--color-match", "3"], "--color-match"),
                       (["pick", "x", "n", "1", "--color-match", "-0.5"], "--color-match")):
        code, out = run(*argv)
        assert code == 2 and flag in out, (argv, out)
    code, out = run("forge", "--help")
    assert code == 0 and "--include-svg" in out and "--candidates" in out
    for cmd in ("fix", "pick", "score"):
        assert run(cmd, "-h")[0] == 0


def test_fix_prompts_and_empty_diff():
    with tempfile.TemporaryDirectory() as td:
        d = project(td)
        with patched(Mock()):
            run("forge", d, "--only", "flame,sky", "--candidates", "1", "--jobs", "2", "--no-color-match")
        # empty differences -> no edit call, message
        with patched(Mock(ask={"differences": [], "edit_instruction": ""})) as mk:
            code, out = run("fix", d, "flame", "--rounds", "2")
        assert code == 0 and mk.calls == 0 and "no visible differences" in out and len(mk.asks) == 1
        assert mk.asks[0]["schema"] == K.FIX_SCHEMA  # strict schema, not json_object
        # transparent asset: prompt says transparent + no backdrop; vision prompt mentions the backdrop
        with patched(Mock(ask={"differences": ["too tall"], "edit_instruction": "shorter"})) as mk:
            code, out = run("fix", d, "flame", "--no-color-match")
        assert code == 0 and mk.calls == 1
        assert "no backdrop" in mk.prompts[0]["prompt"] and "transparent background" in mk.prompts[0]["prompt"]
        assert mk.prompts[0]["background"] == "transparent" and mk.prompts[0]["prompt"].endswith("Corrections: shorter")
        # background asset: vision prompt ignores UI, edit prompt keeps it full-bleed, never says "no backdrop"
        with patched(Mock(ask={"differences": ["moon too big"], "edit_instruction": "smaller moon"})) as mk:
            code, out = run("fix", d, "sky", "--accept")
        assert code == 0 and mk.calls == 1
        vis = mk.asks[0]["parts"][0]
        assert "ignore all UI" in vis and "compare only the artwork" in vis
        ep = mk.prompts[0]["prompt"]
        assert "full-bleed" in ep and "keep it full-bleed, no UI, no text" in ep and "no backdrop" not in ep
        assert mk.prompts[0]["background"] == "opaque"
        rep = json.loads((d / "forge_report.json").read_text())
        assert rep["sky"]["chosen"]["cand"] == "fix1" and rep["sky"]["candidates"][-1]["cand"] == "fix1"
        # --accept keeps a worse result, but never an EMPTY one
        with patched(Mock({1: "blank"}, ask={"differences": ["x"], "edit_instruction": "y"})) as mk:
            code, out = run("fix", d, "flame", "--accept", "--no-color-match")
        assert code == 0 and "discarded (EMPTY)" in out
        assert json.loads((d / "forge_report.json").read_text())["flame"]["chosen"]["cand"] != "fix2"
        # opaque non-background asset: edge-to-edge wording, no contradiction
        opq = [{"name": "tile", "kind": "illustration", "format": "png", "bbox": {"x": 40, "y": 560, "w": 300, "h": 200},
                "transparent": False, "description": "card art", "colors": [], "animated": False}]
        d2 = project(Path(td) / "p2", opq)
        with patched(Mock()) as mk:
            run("forge", d2, "--candidates", "1")
        assert "edge to edge" in mk.prompts[0]["prompt"] and "no letterbox" in mk.prompts[0]["prompt"]
        assert "Opaque background matching" not in mk.prompts[0]["prompt"]
        with patched(Mock(ask={"differences": ["x"], "edit_instruction": "y"})) as mk:
            run("fix", d2, "tile")
        assert "no letterbox" in mk.prompts[0]["prompt"] and "no backdrop" not in mk.prompts[0]["prompt"]


def test_forge_prompt_and_ref_centering():
    m = Image.new("RGBA", (400, 400), "#EDE7F6")
    ImageDraw.Draw(m).ellipse([0, 100, 60, 160], fill="#8BC34A")  # touches x=0
    bb = {"x": 0, "y": 100, "w": 60, "h": 60}
    c = K.ref_crop(m, bb, center=True)
    box = K.content_box(Image.fromarray(np.dstack([np.asarray(c), (np.abs(np.asarray(c).astype(int) - [237, 231, 246]).sum(-1) > 30) * 255]).astype(np.uint8), "RGBA"))
    cx, cy = (box[0] + box[2]) / 2 / c.width, (box[1] + box[3]) / 2 / c.height
    assert abs(cx - 0.5) < 0.02 and abs(cy - 0.5) < 0.02 and c.width == c.height
    plain = K.ref_crop(m, bb)
    assert plain.width != plain.height  # the default crop is still clamped, as svg/icons expect
    with tempfile.TemporaryDirectory() as td:
        d = project(td)
        with patched(Mock()) as mk:
            run("forge", d, "--only", "flame", "--candidates", "3", "--jobs", "1", "--no-color-match")
        ps = [x["prompt"] for x in mk.prompts]
        assert all("in the CENTRE" in p and "No watermark" in p for p in ps)
        assert "PRIORITY" not in ps[0] and "colour values" in ps[1] and "silhouette" in ps[2]


def test_color_match_unique_colours():
    m, bb = _flame_mock()
    a = _tri("#FF9060")
    out = K.color_match(m, bb, a, 0.75)
    assert out.size == a.size and np.array_equal(np.asarray(out)[..., 3], np.asarray(a)[..., 3])
    assert np.all(np.asarray(out)[np.asarray(a)[..., 3] == 0, :3] == 0)
    assert K.score(m, bb, out)["fid"] <= K.score(m, bb, a)["fid"]
    assert K.color_match(m, bb, Image.new("RGBA", (64, 64), (0, 0, 0, 0))) is not None  # too few pixels: unchanged


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
