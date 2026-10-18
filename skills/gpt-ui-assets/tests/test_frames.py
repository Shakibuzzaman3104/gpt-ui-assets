"""Offline checks for the frames pipeline: python3 tests/test_frames.py (or pytest)."""
import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
os.environ.setdefault("GPTKIT_LEDGER", "off")  # nor append mocked calls to the user's cost ledger
import gptkit as g  # noqa: E402


def flame(i, n=8, size=160, tip=0):
    """A teardrop whose tip sways with i (and rises by `tip` px per frame, making every frame unique)."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    sway = 18 * math.sin(2 * math.pi * i / n)
    d.polygon([(size / 2 + sway, 8 + tip * i), (size - 30, size - 50), (30, size - 50)], fill=(255, 150, 40, 255))
    d.ellipse([30, size - 100, size - 30, size - 10], fill=(255, 150, 40, 255))
    d.ellipse([60, size - 70, size - 60, size - 20], fill=(120, 50, 30, 255))
    return g.trim(im)


def ident(crops, truth):
    thumbs = [g._premul_thumb(g._square(t)[0], 48) for t in truth]
    return [int(np.argmin([np.abs(g._premul_thumb(g._square(c)[0], 48) - t).mean() for t in thumbs])) for c in crops]


def sheet_of(frames, centres, size):
    sh = Image.new("RGBA", size, (0, 0, 0, 0))
    for f, (x, y) in zip(frames, centres):
        sh.alpha_composite(f, (int(x - f.width / 2), int(y - f.height / 2)))
    return sh


def test_slice_irregular_layout():
    random.seed(1)
    fr = [flame(i, tip=4) for i in range(8)]
    centres = [(170 + 330 * c + random.uniform(-30, 30), 120 + 250 * r) for r, n in enumerate((3, 3, 2)) for c in range(n)]
    crops = g._slice_sheet(sheet_of(fr, centres, (1000, 700)), 8, 4)
    assert ident(crops, fr) == list(range(8))


def test_slice_twelve_with_grid_lines():
    fr = [flame(i, 12, tip=3) for i in range(12)]
    sh = sheet_of(fr, [(125 + 250 * (i % 4), 110 + 220 * (i // 4)) for i in range(12)], (1000, 660))
    d = ImageDraw.Draw(sh)
    for x in (250, 500, 750):
        d.line([(x, 0), (x, 660)], fill=(80, 80, 80, 220), width=2)
    crops = g._slice_sheet(g._remove_dividers(sh), 12, 4)
    assert ident(crops, fr) == list(range(12))
    assert all(c.width < 200 for c in crops)


def test_qa_flags_bad_frame_and_seam():
    fr = g._align_frames([flame(i) for i in range(8)], "bottom")
    ref = fr[0]
    assert not any(r["flags"] for r in g._frame_qa(fr, ref)["frames"])
    bad = list(fr)
    bad[3] = ImageEnhance.Color(bad[3]).enhance(0.2)
    assert "colour" in g._frame_qa(bad, ref)["frames"][3]["flags"]
    sway = g._align_frames([flame(i, 16) for i in range(16)], "bottom")
    assert not g._frame_qa(sway, ref)["sequence"]["flags"]
    assert "loop-seam" in g._frame_qa(sway[:5], ref)["sequence"]["flags"]


def test_procedural_loops_and_lottie():
    src = g._square(flame(0))[0]
    for kind in g.MOTIONS:
        frames, params, lo, art = g._procedural(src, kind, 16, 1.0)
        assert len({f.size for f in frames}) == 1
        assert g._frame_qa(frames, src)["sequence"]["loop_ratio"] < 2
        lot = g.lottie_motion(art, kind, 16, 1.0, lo, frames[0].size, 24, 240)
        assert lot["op"] == 16 and len(lot["assets"]) == 1
    seq = g.lottie_image_sequence(frames[:4], 12)
    assert [l["ip"] for l in seq["layers"]] == [0, 1, 2, 3] and len(seq["assets"]) == 4


def test_match_colors_moves_toward_reference():
    fr = [flame(i) for i in range(4)]
    tinted = [ImageEnhance.Color(f).enhance(1.8) for f in fr]
    before = g._frame_qa(tinted, fr[0])["sequence"]["dE_ref"]
    after = g._frame_qa(g._match_colors(tinted, fr[0]), fr[0])["sequence"]["dE_ref"]
    assert after < before / 2


# -- outputs (G5-1)

ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


def _anim_args(**kw):
    ns = dict(name="Rain-Drop", procedural="pulse", fps=None, count=None, amount=1.0, formats="gif,webp,apng,lottie",
              px=120, preview_bg="#1b1d33")
    ns.update(kw)
    return argparse.Namespace(**ns)


def _webp_chunks(path):
    """(loop, [ANMF durations]) straight from the RIFF container, independent of Pillow's reader."""
    import struct
    b = path.read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WEBP"
    i, loop, durs = 12, None, []
    while i < len(b):
        tag, n = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        data = b[i + 8:i + 8 + n]
        if tag == b"ANIM":
            loop = struct.unpack("<H", data[4:6])[0]
        elif tag == b"ANMF":
            durs.append(int.from_bytes(data[12:15], "little"))
        i += 8 + n + (n & 1)
    return loop, durs


def _decoded(path):
    im = Image.open(path)
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(np.asarray(im.convert("RGBA")).copy())
    return out


def test_write_anim_android_names_and_durations():
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "frames"
    out.mkdir()
    src = g._square(flame(0))[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g._frames_procedural(_anim_args(), src, out)        # frames build Rain-Drop --procedural pulse: 24 @ 24 fps
    xml = out / "android" / "drawable" / "rain_drop_pulse_anim.xml"
    assert xml.exists(), sorted(p.name for p in out.rglob("*"))
    items = list(ET.parse(xml).getroot())
    assert len(items) == 24
    for it in items:
        name = it.get(ANDROID_NS + "drawable").split("/")[1]
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name
        assert (out / "android" / "drawable-nodpi" / f"{name}.png").exists(), name
    assert sum(int(it.get(ANDROID_NS + "duration")) for it in items) == 1000
    assert Image.open(out / "android" / "drawable-nodpi" / "rain_drop_pulse_frame_01.png").width <= 120
    assert "android resource name: rain_drop_pulse" in buf.getvalue()
    loop, durs = _webp_chunks(out / "Rain-Drop_pulse.webp")
    assert loop == 0 and len(durs) == 24 and abs(sum(durs) - 1000) <= 1, (loop, sum(durs))
    ap = Image.open(out / "Rain-Drop_pulse.apng")
    total = 0
    for i in range(ap.n_frames):
        ap.seek(i)
        total += ap.info["duration"]
    assert ap.n_frames == 24 and abs(total - 1000) <= 1 and ap.info.get("loop") == 0, (total, ap.info)
    gi = Image.open(out / "preview.gif")
    total = 0
    for i in range(gi.n_frames):
        gi.seek(i)
        total += gi.info["duration"]
    assert gi.n_frames == 24 and total == 1000, total
    if g._android_sdk():
        assert "aapt2 compile" in buf.getvalue() and "FAILED" not in buf.getvalue(), buf.getvalue()
        assert "valid" not in buf.getvalue().split("aapt2")[1]
    else:
        assert "aapt2 not found" in buf.getvalue()
    # a formats subset writes only those, and 11 frames at 12 fps keep the exact 917 ms loop
    out2 = tmp / "sub"
    out2.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        g._write_anim(out2, "flame", [flame(i, 11) for i in range(11)], 12, _anim_args(formats="webp"))
    assert sorted(p.name for p in out2.iterdir()) == ["android", "flame.webp", "frames_review.png"]
    assert sum(_webp_chunks(out2 / "flame.webp")[1]) == round(11 * 1000 / 12)
    assert g._durations(24, 24) == [42, 41, 42] * 8 and sum(g._durations(7, 24, 10)) == 290
    shutil.rmtree(tmp)


def test_apng_loop_seam():
    """Every APNG frame decodes to exactly the frame written (sub-rectangle disposal/blend can only ghost mid-loop
    or at the wrap, and the spec clears the buffer at each play, so an exact per-frame decode is the whole check)."""
    tmp = Path(tempfile.mkdtemp())
    fr = g._align_frames([flame(i, 8, tip=6) for i in range(8)], "bottom")
    with contextlib.redirect_stdout(io.StringIO()):
        g._write_anim(tmp, "flame", fr, 12, _anim_args(formats="apng", px=400))
    got = _decoded(tmp / "flame.apng")
    assert len(got) == 8
    for i, (d, f) in enumerate(zip(got + got[:1], fr + fr[:1])):
        want = np.asarray(f)
        assert d.shape == want.shape
        assert np.array_equal(d[..., 3], want[..., 3]), f"alpha of frame {i}"
        vis = want[..., 3] > 0
        assert np.array_equal(d[..., :3][vis], want[..., :3][vis]), f"rgb of frame {i}"
    shutil.rmtree(tmp)


# -- cmd_frames end to end with edit() mocked (G5-3, G5-4)

def _frames_args(tmp, **kw):
    ns = dict(dir=str(tmp), name="flame", moving="the flame", static=None, static_order="front", motion=None, count=None,
              cols=4, anchor="bottom", fps=None, procedural=None, amount=1.0, reuse=None, reroll=False, match_colors=False,
              formats="webp", px=120, preview_bg="#1b1d33", model=None)
    ns.update(kw)
    return argparse.Namespace(**ns)


def _campfire(with_flame=True, size=200):
    """A flame (the moving part) over a log (the static part) on one square canvas, or the log alone."""
    im = Image.new("RGBA", (size, size))
    if with_flame:
        f = flame(0)
        f.thumbnail((120, 120))
        im.alpha_composite(f, ((size - f.width) // 2, 24))
    ImageDraw.Draw(im).rounded_rectangle([30, size - 56, size - 30, size - 24], 10, fill=(120, 80, 40, 255))
    return im


def _asset_dir(tmp, img=None):
    (tmp / "assets").mkdir(parents=True, exist_ok=True)
    (img or _campfire()).save(tmp / "assets" / "flame.png")
    return tmp


class _Sheet:
    """edit() drawing the sprite sheet the prompt asks for (k of the requested frames), or failing in a named way."""
    def __init__(self, mode="ok", k=None):
        self.mode, self.k, self.calls = mode, k, []

    def __enter__(self):
        self.real = g.edit

        def fake(ref, prompt, size="1024x1024", background="auto", model=None, fidelity="high"):
            self.calls.append((size, prompt))
            w, h = map(int, size.split("x"))
            if "Keep ONLY" in prompt:                                       # the static layer: the log alone
                return g._png_bytes(_campfire(False).resize((w, h))), {}
            if "frame" in prompt and "sprite sheet" not in prompt:         # a single-frame re-roll
                j = int(re.search(r"frame (\d+) of", prompt).group(1))
                return g._png_bytes(g._square(flame(j - 1))[0].resize((w, h))), {}
            if self.mode == "raise":
                raise g.ApiError(400, 'HTTP 400 /images/edits: {"error":{"code":"moderation_blocked","message":"rejected"}}')
            if self.mode == "opaque":
                return g._png_bytes(Image.new("RGBA", (w, h), (255, 255, 255, 255))), {}
            if self.mode == "html":
                return b"<html>502 Bad Gateway</html>", {}
            m = re.search(r"as (\d+) animation frames.*?(\d+) columns? x (\d+) rows?", prompt)
            n, cols, rows = map(int, m.groups())
            sh = Image.new("RGBA", (w, h))
            cw, ch = w / cols, h / rows
            for i in range(self.k if self.k is not None else n):
                f = flame(i, n) if not (self.mode == "bad3" and i == 3) else ImageEnhance.Color(flame(i, n)).enhance(0.1)
                f.thumbnail((int(cw * 0.7), int(ch * 0.7)))
                sh.alpha_composite(f, (int((i % cols) * cw + (cw - f.width) / 2), int((i // cols) * ch + (ch - f.height) / 2)))
            return g._png_bytes(sh), {}
        g.edit = fake
        return self

    def __exit__(self, *exc):
        g.edit = self.real


def _run(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def _exits(fn, *args):
    try:
        _run(fn, *args)
    except SystemExit as e:
        return str(e)
    raise AssertionError("must exit")


def test_frames_e2e_reuse_and_reroll():
    tmp = _asset_dir(Path(tempfile.mkdtemp()))
    out = tmp / "frames" / "flame"
    with _Sheet() as sh:
        text = _run(g.cmd_frames, _frames_args(tmp, static="the base", count=8))
    assert [c[0] for c in sh.calls] == ["1536x1024", "1024x1024"] or sorted(c[0] for c in sh.calls) == ["1024x1024", "1536x1024"]
    assert "8 columns" not in sh.calls[0][1] and "4 columns x 2 rows" in sh.calls[0][1] and "with NO the base at all" in sh.calls[0][1]
    assert "sliced 8 frames" in text and "flagged frames: none" in text, text
    qa = json.loads((out / "qa.json").read_text())
    assert qa["sequence"]["frames"] == 8 and "registration_iou" in qa["sequence"] and not qa["sequence"]["flags"]
    assert len(list(out.glob("frame_*.png"))) == 8 and (out / "static.png").exists() and (out / "flame.webp").exists()
    assert (out / "android" / "drawable" / "flame_anim.xml").exists()
    with _Sheet() as sh:                                                     # --reuse all: no calls at all
        _run(g.cmd_frames, _frames_args(tmp, static="the base", count=8, reuse="all"))
    assert not sh.calls
    with _Sheet() as sh:                                                     # --reuse static: the sheet only
        _run(g.cmd_frames, _frames_args(tmp, static="the base", count=8, reuse="static"))
    assert [c[0] for c in sh.calls] == ["1536x1024"]
    shutil.rmtree(out)
    assert "--reuse needs" in _exits(g.cmd_frames, _frames_args(tmp, count=8, reuse="all"))
    # a washed-out frame 4 is flagged and --reroll redraws it from a neighbour (1 call), kept because it scores better
    (tmp / "frames" / "flame" / "reroll_f04.png").parent.mkdir(parents=True, exist_ok=True)
    (tmp / "frames" / "flame" / "reroll_f04.png").write_bytes(b"old")
    with _Sheet("bad3") as sh:
        text = _run(g.cmd_frames, _frames_args(tmp, count=8, reroll=True))
    assert "f4(colour" in text and "re-roll f4" in text and "KEPT" in text, text
    assert len(sh.calls) == 2 and "frame 4:" in sh.calls[1][1]
    assert (out / "reroll_f04.png").read_bytes() != b"old"                   # stale re-rolls are cleared, new one written
    assert "colour" not in json.loads((out / "qa.json").read_text())["frames"][3]["flags"]
    # the prompt reads correctly without --static and clamps --cols to --count
    with _Sheet() as sh:
        text = _run(g.cmd_frames, _frames_args(tmp, count=2, cols=4))
    assert "— the flame — as 2 animation frames" in sh.calls[0][1] and "2 columns x 1 row " in sh.calls[0][1]
    assert "using 2 columns" in text
    shutil.rmtree(tmp)


def test_opaque_render_detected():
    tmp = _asset_dir(Path(tempfile.mkdtemp()))
    out = tmp / "frames" / "flame"
    with _Sheet("opaque") as sh:
        msg = _exits(g.cmd_frames, _frames_args(tmp, count=8))
    assert "sheet call failed" in msg and "\n" not in msg, msg
    assert (out / "sheet_raw.opaque.png").exists() and not (out / "sheet_raw.png").exists()
    assert not list(out.glob("frame_*.png")) and not (out / "flame.webp").exists()
    share, corners = g._opaque_share(Image.new("RGBA", (64, 64), (255, 255, 255, 255)))
    assert share == 1.0 and corners
    assert g._opaque_error(g._square(flame(0))[0], "x") is None
    err = g._opaque_error(Image.new("RGBA", (64, 64), (0, 0, 0, 255)), "raw/003.png", "m")
    assert err.startswith("raw/003.png: the image came back with an opaque background") and "ignored by m at" in err
    with _Sheet("html"):
        msg = _exits(g.cmd_frames, _frames_args(tmp, count=8))
    assert "not a decodable image" in msg or "sheet call failed" in msg, msg
    shutil.rmtree(tmp)


def test_frame_count_flag():
    tmp = _asset_dir(Path(tempfile.mkdtemp()))
    with _Sheet(k=6):
        text = _run(g.cmd_frames, _frames_args(tmp, count=8))
    assert "only 6 of 8 frames" in text, text
    qa = json.loads((tmp / "frames" / "flame" / "qa.json").read_text())
    assert "frame-count" in qa["sequence"]["flags"] and qa["sequence"]["frames"] == 6
    shutil.rmtree(tmp)


def test_asset_usability():
    tmp = _asset_dir(Path(tempfile.mkdtemp()), Image.new("RGBA", (100, 100)))
    assert "fully transparent" in _exits(g.cmd_frames, _frames_args(tmp, procedural="pulse"))
    tmp = _asset_dir(Path(tempfile.mkdtemp()), g._square(flame(0))[0].convert("RGB").convert("RGBA"))
    text = _run(g.cmd_frames, _frames_args(tmp, procedural="pulse", count=4))
    assert "no transparent background" in text and (tmp / "frames" / "flame_pulse" / "flame_pulse.webp").exists(), text
    assert "asset not found" in _exits(g.cmd_frames, _frames_args(tmp, name="nosuch", procedural="pulse"))
    assert "--formats" in _exits(g.cmd_frames, _frames_args(tmp, procedural="pulse", formats="mp4"))
    assert "--count must be at least 2" in _exits(g.cmd_frames, _frames_args(tmp, count=1))
    assert "--moving is required" in _exits(g.cmd_frames, _frames_args(tmp, moving=None))
    shutil.rmtree(tmp)


def test_frames_partial_failure():
    tmp = _asset_dir(Path(tempfile.mkdtemp()))
    out = tmp / "frames" / "flame"
    with _Sheet("raise") as sh:
        msg = _exits(g.cmd_frames, _frames_args(tmp, static="the base", count=8))
    assert len(sh.calls) == 2 and (out / "static_raw.png").exists() and not (out / "sheet_raw.png").exists()
    assert msg == "frames: the sheet call failed (see above); static_raw.png was kept: re-run with --reuse static", msg
    with _Sheet() as sh:
        _run(g.cmd_frames, _frames_args(tmp, static="the base", count=8, reuse="static"))
    assert [c[0] for c in sh.calls] == ["1536x1024"] and len(list(out.glob("frame_*.png"))) == 8
    shutil.rmtree(tmp)


# -- argv validation (G5-7)

def _main_exit(argv):
    """(exit code, stderr) of gptkit's main() for argv; any exception other than SystemExit is a failure."""
    real, err = sys.argv, io.StringIO()
    sys.argv = ["gptkit"] + argv
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            g.main()
    except SystemExit as e:
        return (e.code if isinstance(e.code, int) else 1), (err.getvalue() + (str(e.code) if isinstance(e.code, str) else ""))
    finally:
        sys.argv = real
    return 0, err.getvalue()


def test_argv_table():
    tmp = _asset_dir(Path(tempfile.mkdtemp()))
    base = ["frames", str(tmp), "flame"]
    table = [
        (["--count", "0", "--moving", "x"], 2, "positive integer"),
        (["--count", "-3", "--moving", "x"], 2, "positive integer"),
        (["--count", "1", "--moving", "x"], 1, "--count must be at least 2"),
        (["--cols", "0", "--moving", "x"], 2, "positive integer"),
        (["--fps", "0", "--moving", "x"], 2, "positive number"),
        (["--fps", "nan", "--moving", "x"], 2, "positive number"),
        (["--fps", "inf", "--moving", "x"], 2, "positive number"),
        (["--px", "0", "--procedural", "pulse"], 2, "positive integer"),
        (["--amount", "0", "--procedural", "pulse"], 2, "positive number"),
        (["--procedural", "hop"], 2, "invalid choice"),
        (["--formats", "mp4", "--procedural", "pulse"], 1, "--formats"),
        (["--formats", "", "--procedural", "pulse"], 1, "--formats"),
        ([], 1, "--moving is required"),
        (["--reuse", "all"], 1, "--moving is required"),
    ]
    for extra, code, text in table:
        got, err = _main_exit(base + extra)
        assert got == code and text in err and "Traceback" not in err, (extra, got, err)
    got, err = _main_exit(["frames", str(tmp), "nosuch", "--procedural", "pulse"])
    assert got == 1 and "asset not found" in err, err
    got, err = _main_exit(["frames", str(tmp / "nodir"), "flame", "--procedural", "pulse"])
    assert got == 1 and "asset not found" in err, err
    assert _main_exit(base + ["--procedural", "pulse", "--count", "4", "--formats", "webp"])[0] == 0
    shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
