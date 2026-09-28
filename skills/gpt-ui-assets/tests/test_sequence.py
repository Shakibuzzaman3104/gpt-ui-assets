"""Offline checks for the layout-driven sequence route: python3 tests/test_sequence.py (or pytest).

No API calls: the guides themselves stand in for GPT's renders, which is what a perfect render of the layout is.
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gptkit as g  # noqa: E402

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def _fixture(tmp):
    """A dark 'engine' with a chimney on a 256 canvas, and a soft round puff sprite."""
    base = Image.new("RGBA", (256, 256))
    d = ImageDraw.Draw(base)
    d.rounded_rectangle([40, 150, 220, 230], 20, fill=(60, 60, 70, 255))
    d.rectangle([70, 115, 100, 150], fill=(50, 50, 60, 255))
    base.save(tmp / "base.png")
    puff = Image.new("RGBA", (80, 64))
    ImageDraw.Draw(puff).ellipse([2, 2, 78, 62], fill=(235, 232, 240, 255))
    puff.save(tmp / "puff.png")
    return tmp / "base.png", tmp / "puff.png"


def _args(tmp, stage, **kw):
    ns = dict(dir=str(tmp), name="smoke", stage=stage, guides=None, base=None, sprites=None, vent=None, loop=2.0,
              fps=12, chuff=0.5, life=2.5, velocity="0,-60", spread="6,8", drag=1.5, wind="40,-3", buoyancy=-10,
              size=40, grow="0.5,1.4", front=False, moving="The smoke", locked="The engine", look=None, jobs=2,
              force=False, ref=None, below=None, quality=80, frames=None, candidates=1, preview_bg="#a0aabe", model=None,
              every=1, ebsynth=None, guide_weight=2.0, position_weight=1.5, engine=None, ebsynth_timeout=None,
              keep_work=False)
    ns.update(kw)
    return argparse.Namespace(**ns)


def _plume(tmp):
    base, puff = _fixture(tmp)
    return dict(base=str(base), sprites=str(puff), vent="85,112")


def test_plume_layout_is_periodic():
    p = dict(loop=2.0, chuff=0.5, life=2.5, vent=[85, 112], velocity=[0, -60], spread=[6, 8], drag=1.5,
             wind=[40, -3], buoyancy=-10, size=40, grow=[0.5, 1.4], sprites=["a", "b"], front=False)
    per_loop = 4
    for t in (0.0, 0.3, 1.1):
        now, later = g.plume_layout(t, p), g.plume_layout(t + 2.0, p)
        assert {k + per_loop for k in now} == set(later)
        for k, st in now.items():
            assert np.allclose(st[:4], later[k + per_loop][:4])
    # terminal velocity: an old puff drifts with the wind and rises by buoyancy / drag
    x1, y1 = g.plume_state(2.0, 0, p)[:2]
    x2, y2 = g.plume_state(2.1, 0, p)[:2]
    assert abs((x2 - x1) / 0.1 - 40) < 3 and (y2 - y1) / 0.1 < 0


def test_layout_rejects_a_loop_that_cannot_repeat():
    tmp = Path(tempfile.mkdtemp())
    try:
        g.cmd_sequence_layout(_args(tmp, "layout", chuff=0.7, **_plume(tmp)))
    except SystemExit as e:
        assert "whole number" in str(e)
    else:
        raise AssertionError("a 2 s loop with 0.7 s emissions must be rejected")
    shutil.rmtree(tmp)


def _perfect_renders(tmp):
    """Build a 12-frame sequence (128 canvas) whose 'renders' are the guides themselves."""
    g.cmd_sequence_layout(_args(tmp, "layout", **_small(tmp)))
    d = tmp / "sequence" / "smoke"
    for guide in sorted((d / "guides").glob("*.png")):
        shutil.copyfile(guide, d / "raw" / guide.name)
    return d


def test_build_cuts_the_locked_art_and_writes_a_loop():
    tmp = Path(tempfile.mkdtemp())
    d = _perfect_renders(tmp)
    meta = json.loads((d / "sequence.json").read_text())
    assert meta["frames"] == 12 and len(list((d / "guides").glob("*.png"))) == 12
    g.cmd_sequence_build(_args(tmp, "build", ref=str(tmp / "puff_s.png")))
    webp = Image.open(d / "smoke.webp")
    assert not json.loads((d / "report.json").read_text())["outliers"]   # no false alarms on a perfect render
    assert webp.n_frames == 12
    total = 0
    for i in range(webp.n_frames):
        webp.seek(i)
        webp.load()
        total += webp.info["duration"]
    assert total == 1000                                # 12 frames at 12 fps: exactly one second
    frame = np.asarray(Image.open(d / "frames" / "004.png"))
    assert frame[95, 65, 3] == 0                        # the engine body is cut away
    assert frame[..., 3].max() > 200                    # the smoke is kept
    report = json.loads((d / "report.json").read_text())
    assert len(report["rect"]) == 4 and report["frames"] == 12
    if HAVE_CV2:
        assert report["backwards"] == 0                 # a perfect render of the layout only ever moves forward
        assert report["seam_step"] > 0
    shutil.rmtree(tmp)


def test_qa_names_a_render_that_ignored_the_layout():
    if not HAVE_CV2:
        return
    tmp = Path(tempfile.mkdtemp())
    d = _perfect_renders(tmp)
    # frame 9 comes back with the engine where it belongs but its smoke drawn somewhere else, as a render that
    # ignored the layout would (a whole-frame shift is not a failure: registration puts it back)
    meta = json.loads((d / "sequence.json").read_text())
    smoke = g.plume_guide(9 / meta["fps"], meta["plume"], None, g._seq_sprites(d, meta), canvas=meta["canvas"])
    bad = Image.new("RGBA", smoke.size)
    bad.alpha_composite(smoke, (-9, 5))
    bad.alpha_composite(g._rgba(g._seq_path(d, meta["base"])))
    bad.resize((1024, 1024), Image.LANCZOS).save(d / "raw" / "009.png")
    g.cmd_sequence_build(_args(tmp, "build"))
    report = json.loads((d / "report.json").read_text())
    assert 9 in report["outliers"], report["outlier_scores"]
    shutil.rmtree(tmp)


def _have_ebsynth():
    try:
        g._ebsynth_bin(argparse.Namespace(ebsynth=None))
        return True
    except SystemExit:
        return False


def _small(tmp):
    """A 12-frame plume on a 128 canvas (12 fps, keys every 3): the same shapes as _fixture at half size."""
    base = Image.new("RGBA", (128, 128))
    d = ImageDraw.Draw(base)
    d.rounded_rectangle([20, 75, 110, 115], 10, fill=(60, 60, 70, 255))
    d.rectangle([35, 57, 50, 75], fill=(50, 50, 60, 255))
    base.save(tmp / "base_s.png")
    puff = Image.new("RGBA", (40, 32))
    ImageDraw.Draw(puff).ellipse([1, 1, 39, 31], fill=(235, 232, 240, 255))
    puff.save(tmp / "puff_s.png")
    return dict(base=str(tmp / "base_s.png"), sprites=str(tmp / "puff_s.png"), vent="42,56", loop=1.0, chuff=0.5,
                life=1.5, velocity="0,-30", spread="3,4", wind="20,-2", buoyancy=-5, size=20, grow="0.5,1.4", fps=12)


def _keys_from_guides(d, every):
    for guide in sorted((d / "guides").glob("*.png")):
        i = int(guide.stem)
        if i % every == 0:
            shutil.copyfile(guide, d / "raw" / guide.name)


def test_propagate_carries_keys_to_every_frame():
    if not _have_ebsynth():
        print("  (skipped: no ebsynth binary)")
        return
    tmp = Path(tempfile.mkdtemp())
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
    d = tmp / "sequence" / "smoke"
    _keys_from_guides(d, 3)                             # only the keys are "rendered"
    msg = _exits(g.cmd_sequence_build, _args(tmp, "build", engine="gpt"))
    assert "propagate" in msg                           # build points at propagate when only keys exist
    out = _quiet(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, jobs=4))
    assert "20 EbSynth runs, 0 reused" in out, out
    assert Image.open(d / "smoke.webp").n_frames == 12
    frame = np.asarray(Image.open(d / "frames" / "004.png"))
    assert frame[..., 3].max() > 200 and frame[95, 65, 3] == 0
    report = json.loads((d / "report.json").read_text())
    if HAVE_CV2:
        assert report["backwards"] <= 1, report["speed_along"]
        assert report["follows"] > 5, report["follows"]          # the texture travels with the plume
    assert not list((d / "propagate").glob("syn_*.png"))         # work files are gone after a clean run
    assert "divide" in _exits(g.cmd_sequence_propagate, _args(tmp, "propagate", every=5))
    shutil.rmtree(tmp)


class _Ebsynth:
    """Pretend the ebsynth binary is (or is not) installed, for the plan only."""
    def __init__(self, present):
        self.present = present

    def __enter__(self):
        self.real = g._find_ebsynth
        g._find_ebsynth = lambda a: "/fake/ebsynth" if self.present else None

    def __exit__(self, *exc):
        g._find_ebsynth = self.real


def _guides(tmp, n, draw, size=256):
    """A folder of n guide frames on a square canvas, drawn by draw(ImageDraw, i)."""
    src = Path(tempfile.mkdtemp(dir=tmp, prefix="guides_"))
    for i in range(n):
        im = Image.new("RGBA", (size, size))
        draw(ImageDraw.Draw(im), i)
        im.save(src / f"{i:03d}.png")
    return src


def _dot(x, y, fill=(200, 210, 230, 255), r=14):
    return lambda dr, i: dr.ellipse([x(i) - r, y(i) - r, x(i) + r, y(i) + r], fill=fill)


def _plan(tmp, present=True, **kw):
    with _Ebsynth(present):
        g.cmd_sequence_layout(_args(tmp, "layout", **kw))
    return json.loads((tmp / "sequence" / "smoke" / "sequence.json").read_text())["route"]


def test_plan_uses_ebsynth_with_quarter_second_keys():
    tmp = Path(tempfile.mkdtemp())
    route = _plan(tmp, **_small(tmp))                   # 12 frames at 12 fps
    assert route["engine"] == "ebsynth" and route["every"] == 3, route
    assert route["image_calls"] == 4 and route["ebsynth_runs"] == 20
    shutil.rmtree(tmp)


def test_plan_skips_ebsynth_when_it_is_not_installed():
    tmp = Path(tempfile.mkdtemp())
    route = _plan(tmp, present=False, **_small(tmp))
    assert route["engine"] == "gpt" and route["every"] == 1 and "not installed" in route["reasons"][0]
    shutil.rmtree(tmp)


def test_plan_skips_ebsynth_when_keys_save_nothing():
    tmp = Path(tempfile.mkdtemp())
    route = _plan(tmp, **{**_small(tmp), "fps": 4})     # a quarter second is a single frame at 4 fps
    assert route["engine"] == "gpt" and "every frame" in route["reasons"][0], route
    shutil.rmtree(tmp)
    tmp = Path(tempfile.mkdtemp())                      # 23 frames: no key interval repeats with the loop
    route = _plan(tmp, fps=24, guides=str(_guides(tmp, 23, _dot(lambda i: 60 + 6 * i, lambda i: 128), 128)))
    assert route["engine"] == "gpt" and "divides" in route["reasons"][0], route
    shutil.rmtree(tmp)


def test_plan_carries_travel_and_renders_what_no_key_shows():
    tmp = Path(tempfile.mkdtemp())                      # a drop falling steadily: EbSynth can carry it
    route = _plan(tmp, fps=24, guides=str(_guides(tmp, 12, _dot(lambda i: 128, lambda i: 30 + 8 * i))))
    assert route["engine"] == "ebsynth" and route["every"] == 6, route
    shutil.rmtree(tmp)

    def splash(dr, i):                                  # a drop, and on every 6th frame (from 3) a splash ring
        _dot(lambda i: 128, lambda i: 60)(dr, i)
        if i % 6 == 3:
            dr.ellipse([80, 150, 176, 246], outline=(200, 210, 230, 255), width=6)
    tmp = Path(tempfile.mkdtemp())
    route = _plan(tmp, fps=24, guides=str(_guides(tmp, 12, splash)))
    assert route["engine"] == "ebsynth" and route["every"] == 3, route   # keys every 6 would never show the splash
    shutil.rmtree(tmp)
    tmp = Path(tempfile.mkdtemp())                      # a new colour on every frame: nothing for a key to carry
    colours = [(230, 60, 50, 255), (60, 200, 80, 255), (70, 90, 230, 255), (240, 220, 60, 255)]
    route = _plan(tmp, fps=24, guides=str(_guides(tmp, 12, lambda dr, i: _dot(lambda i: 128, lambda i: 128,
                                                                             colours[i % 4])(dr, i))))
    assert route["engine"] == "gpt" and "no key shows" in route["reasons"][0], route
    shutil.rmtree(tmp)


def test_plan_warns_about_a_dense_plume():
    tmp = Path(tempfile.mkdtemp())
    route = _plan(tmp, **{**_small(tmp), "chuff": 0.1, "life": 2.5})
    assert any("generic plume" in w for w in route["warnings"]), route
    shutil.rmtree(tmp)
    tmp = Path(tempfile.mkdtemp())
    assert not _plan(tmp, **_small(tmp))["warnings"]
    shutil.rmtree(tmp)


def test_build_hands_keys_to_propagate_as_planned():
    tmp = Path(tempfile.mkdtemp())
    _plan(tmp, **_small(tmp))
    d = tmp / "sequence" / "smoke"
    for i in range(0, 12, 3):
        shutil.copyfile(d / "guides" / f"{i:03d}.png", d / "raw" / f"{i:03d}.png")
    seen = {}
    real = g.cmd_sequence_propagate
    g.cmd_sequence_propagate = lambda a: seen.setdefault("every", a.every)
    try:
        with _Ebsynth(True):
            g.cmd_sequence_build(_args(tmp, "build", every=None))
    finally:
        g.cmd_sequence_propagate = real
    assert seen == {"every": 3}
    shutil.rmtree(tmp)


def test_render_every_counts_only_the_keys():
    import contextlib, io
    tmp = Path(tempfile.mkdtemp())
    g.cmd_sequence_layout(_args(tmp, "layout", **_small(tmp)))
    d = tmp / "sequence" / "smoke"
    real = g._seq_render_one
    g._seq_render_one = lambda d_, meta, i, out, model: shutil.copyfile(d_ / "guides" / f"{i:03d}.png", out) and out
    shutil.copyfile(d / "guides" / "000.png", d / "raw" / "003_cand0.png")   # a reroll candidate is not a frame
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g.cmd_sequence_render(_args(tmp, "render", every=6))
    finally:
        g._seq_render_one = real
    assert sorted(p.name for p in (d / "raw").glob("0??.png")) == ["000.png", "006.png"]
    assert "still missing" not in buf.getvalue(), buf.getvalue()
    shutil.rmtree(tmp)


def test_brief_names_the_frame_and_binds_the_layout():
    meta = {"frames": 72, "seconds": 3.0, "fps": 24, "moving": "The smoke", "locked": "The toy steam train"}
    brief = g._seq_prompt(meta, 9)
    assert "frame 10 of 72" in brief and "3-second" in brief and "24 frames per second" in brief
    assert "The toy steam train is final artwork and is locked" in brief
    assert "Render the smoke as the finished" in brief and "the layout is binding" in brief


# -- layout freshness, guide validation, empty layouts (G5-2)

class _Edit:
    """edit() returns the reference image itself (a perfect render of the guide) and counts the calls."""
    def __init__(self, fn=None):
        self.calls, self.fn = [], fn

    def __enter__(self):
        self.real = g.edit

        def fake(ref, prompt, size="1024x1024", background="auto", model=None, fidelity="high"):
            self.calls.append(prompt)
            return (self.fn(ref, prompt) if self.fn else ref), {}
        g.edit = fake
        return self

    def __exit__(self, *exc):
        g.edit = self.real


def _quiet(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kw)
    return buf.getvalue()


def _exits(fn, *args, **kw):
    try:
        _quiet(fn, *args, **kw)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} must exit")


def test_layout_fresh_meta_and_stale_raws():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", moving="The smoke", **_small(tmp)))
    meta = json.loads((d / "sequence.json").read_text())
    assert meta["plume"] and len(meta["guide_sha1"]) == 12 and meta["base"] == "inputs/base.png"
    assert all(str(s).startswith("inputs/") for s in meta["plume"]["sprites"]) and (d / "inputs" / "sprite_0.png").exists()
    with _Edit() as ed:
        _quiet(g.cmd_sequence_render, _args(tmp, "render", every=1))
    assert len(ed.calls) == 12 and json.loads((d / "raw" / "005.json").read_text())["guide_sha1"] == meta["guide_sha1"][5]
    # a guide-folder layout replaces the plume (no stale plume, route, qa or rect survives)
    meta = json.loads((d / "sequence.json").read_text())
    (d / "sequence.json").write_text(json.dumps({**meta, "qa": {"outliers": [3]}, "rect": [1, 2, 3, 4], "propagated": {}}))
    guides = _guides(tmp, 8, _dot(lambda i: 128, lambda i: 40 + 20 * (i % 4)))
    with _Ebsynth(False):
        out = _quiet(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(guides), fps=8))
    meta2 = json.loads((d / "sequence.json").read_text())
    assert "plume" not in meta2 and meta2["base"] is None and meta2["frames"] == 8
    assert not any(k in meta2 for k in ("qa", "rect", "propagated")) and meta2["moving"] == "The smoke"
    assert "guides in order: 000.png, 001.png, 002.png ... 005.png, 006.png, 007.png" in out
    # the old renders are stale: build refuses them by name, render redoes every frame
    msg = _exits(g.cmd_sequence_build, _args(tmp, "build"))
    assert "raw/000.png" in msg and "older layout" in msg, msg
    with _Edit() as ed:
        _quiet(g.cmd_sequence_render, _args(tmp, "render", every=1))
    assert len(ed.calls) == 8
    # the same plume again: its 4 surviving raws (008..011) hash-match the identical guides and are kept
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
    with _Edit() as ed:
        _quiet(g.cmd_sequence_render, _args(tmp, "render", every=1))
    assert len(ed.calls) == 8, len(ed.calls)
    # a different wind: every frame is rendered again, none reused
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **{**_small(tmp), "wind": "-20,-2"}))
    with _Edit() as ed:
        _quiet(g.cmd_sequence_render, _args(tmp, "render", every=1))
    assert len(ed.calls) == 12, len(ed.calls)
    with _Edit() as ed:                                   # and nothing is re-rendered when the layout stands
        _quiet(g.cmd_sequence_render, _args(tmp, "render", every=1))
    assert not ed.calls
    shutil.rmtree(tmp)


def test_guide_order_and_validation():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    src = tmp / "g"
    src.mkdir()
    for i in range(12):                                    # 0.png .. 11.png, a dot 15 px lower per file
        im = Image.new("RGBA", (256, 256))
        ImageDraw.Draw(im).ellipse([118, 20 + 15 * i, 138, 40 + 15 * i], fill=(200, 210, 230, 255))
        im.save(src / f"{i}.png")
    with _Ebsynth(False):
        out = _quiet(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(src), fps=12))
    assert "guides in order: 0.png, 1.png, 2.png ... 9.png, 10.png, 11.png" in out, out
    assert "do not loop" in out, out
    ys = np.nonzero(np.asarray(Image.open(d / "guides" / "002.png"))[..., 3] > 128)[0]
    assert abs(ys.mean() / 4 - 60) < 3, ys.mean() / 4                     # guide 002 is 2.png (dot at y=60)
    # non-square, mixed sizes: one-line exits naming the sizes
    bad = tmp / "ns"
    bad.mkdir()
    for i in range(4):
        Image.new("RGBA", (256, 128), (0, 0, 0, 0)).save(bad / f"{i}.png")
    ImageDraw.Draw(Image.open(bad / "0.png")).ellipse([0, 0, 10, 10])
    msg = _exits(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(bad), fps=12))
    assert "square" in msg and "256x128" in msg, msg
    Image.new("RGBA", (512, 512)).save(bad / "1.png")
    Image.new("RGBA", (256, 256)).save(bad / "0.png")
    msg = _exits(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(bad), fps=12))
    assert "one size" in msg and "512x512" in msg and "256x256" in msg, msg
    # an opaque backdrop without --base warns; the same guides with --base do not
    baked = _guides(tmp, 8, lambda dr, i: (dr.rectangle([0, 0, 256, 256], fill=(150, 190, 230, 255)),
                                           dr.ellipse([120, 20 + 25 * (i % 4), 136, 36 + 25 * (i % 4)], fill=(230, 60, 60, 255))))
    with _Ebsynth(False):
        out = _quiet(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(baked), fps=8))
        assert "opaque backdrop" in out, out
        sky = Image.new("RGBA", (256, 256), (150, 190, 230, 255))
        ImageDraw.Draw(sky).rectangle([100, 0, 156, 130], fill=(0, 0, 0, 0))
        sky.save(tmp / "sky.png")
        out = _quiet(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(baked), base=str(tmp / "sky.png"), fps=8))
        assert "opaque backdrop" not in out and (d / "inputs" / "base.png").exists(), out
        # a guide set that does not loop is called out (the 0..11 dots above); a looping one is not
        loop = _guides(tmp, 12, _dot(lambda i: 128, lambda i: 40 + 20 * (i % 6)))
        out = _quiet(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(loop), fps=12))
        assert "do not loop" not in out, out
    # missing folder / no PNGs / missing sprite: one line each
    assert "not found" in _exits(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(tmp / "nope"), fps=12))
    (tmp / "empty").mkdir()
    assert "no PNG guides" in _exits(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(tmp / "empty"), fps=12))
    msg = _exits(g.cmd_sequence_layout, _args(tmp, "layout", **{**_plume(tmp), "sprites": str(tmp / "nope.png")}))
    assert "sprite not found" in msg, msg
    assert "x,y" in _exits(g.cmd_sequence_layout, _args(tmp, "layout", **{**_plume(tmp), "vent": "abc"}))
    shutil.rmtree(tmp)


def test_empty_union_exits():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    empty = _guides(tmp, 8, lambda dr, i: None)
    msg = _exits(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(empty), fps=8))
    assert "no moving content" in msg, msg
    # blank renders: build exits with one line instead of a numpy traceback
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
    with _Edit(lambda ref, prompt: g._png_bytes(Image.new("RGBA", (1024, 1024)))):
        _quiet(g.cmd_sequence_render, _args(tmp, "render", every=1))
    msg = _exits(g.cmd_sequence_build, _args(tmp, "build"))
    assert "every frame is empty" in msg and "\n" not in msg, msg
    # a relative --base, then plan and build from another cwd, work (inputs are copied into the sequence dir)
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        _quiet(g.cmd_sequence_layout, _args(".", "layout", **{**_small(tmp), "base": "base_s.png", "sprites": "puff_s.png"}))
        for guide in sorted((d / "guides").glob("*.png")):
            shutil.copyfile(guide, d / "raw" / guide.name)
        os.chdir("/")
        _quiet(g.cmd_sequence_plan, _args(str(tmp), "plan"))
        _quiet(g.cmd_sequence_build, _args(str(tmp), "build"))
        assert (d / "smoke.webp").exists()
    finally:
        os.chdir(cwd)
    # propagate / build before any layout
    fresh = Path(tempfile.mkdtemp())
    assert "layout" in _exits(g.cmd_sequence_propagate, _args(fresh, "propagate"))
    assert "layout" in _exits(g.cmd_sequence_build, _args(fresh, "build"))
    shutil.rmtree(tmp)
    shutil.rmtree(fresh)


# -- render failures (G5-3, G5-4)

def test_opaque_render_detected():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))

    def opaque(ref, prompt):
        bg = Image.new("RGBA", (1024, 1024), (240, 240, 240, 255))
        bg.alpha_composite(Image.open(io.BytesIO(ref)).convert("RGBA"))
        return g._png_bytes(bg)
    with _Edit(opaque) as ed:
        try:
            out = _quiet(g.cmd_sequence_render, _args(tmp, "render", every=6))
        except SystemExit as e:
            msg, out = str(e), ""
        else:
            raise AssertionError(out)
    assert len(ed.calls) == 2, len(ed.calls)                                  # no retry: it would come back the same
    assert "2 of 2 frames still missing" in msg, msg
    assert (d / "raw" / "000.opaque.png").exists() and not (d / "raw" / "000.png").exists()
    assert "keys missing" in _exits(g.cmd_sequence_propagate, _args(tmp, "propagate", every=6, ebsynth="/bin/true"))
    shutil.rmtree(tmp)


def test_render_no_blind_retry():
    tmp = Path(tempfile.mkdtemp())
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
    body = '{"error":{"message":"Your request was rejected as a result of our safety system.","code":"moderation_blocked"}}'

    def blocked(ref, prompt):
        raise g.ApiError(400, "HTTP 400 /images/edits: " + body)
    buf = io.StringIO()
    with _Edit(blocked) as ed, contextlib.redirect_stdout(buf):
        try:
            g.cmd_sequence_render(_args(tmp, "render", every=3))
        except SystemExit as e:
            msg = str(e)
        else:
            raise AssertionError("render must exit when frames are missing")
    assert len(ed.calls) == 4, len(ed.calls)                                  # one call per frame, no retry on 400
    text = buf.getvalue()
    assert "moderation_blocked" in text and "rephrase --moving/--locked/--look" in text and "prompt:" in text, text
    assert text.count("safety system") == 1 and "same error" in text          # the full message once, then a reference
    assert "4 of 4 frames still missing" in msg, msg
    # a 503 is retried (3 attempts with backoff), a 400 is not; the sleeps are stubbed
    sleeps = []
    real_sleep = g.time.sleep
    g.time.sleep = lambda s: sleeps.append(s)

    def down(ref, prompt):
        raise g.ApiError(503, "HTTP 503 /images/edits: upstream")
    try:
        with _Edit(down) as ed, contextlib.redirect_stdout(io.StringIO()):
            try:
                g.cmd_sequence_render(_args(tmp, "render", every=6, jobs=1))
            except SystemExit:
                pass
    finally:
        g.time.sleep = real_sleep
    assert len(ed.calls) == 6 and sleeps == [4, 8] * 2, (len(ed.calls), sleeps)
    # a corrupt body (HTML from a gateway) is not written as a raw
    with _Edit(lambda ref, prompt: b"<html>502</html>") as ed, contextlib.redirect_stdout(io.StringIO()):
        try:
            g.cmd_sequence_render(_args(tmp, "render", every=6))
        except SystemExit as e:
            assert "still missing" in str(e)
    assert len(ed.calls) == 2 and not list((tmp / "sequence" / "smoke" / "raw").glob("0??.png"))
    shutil.rmtree(tmp)


# -- EbSynth failures and resume (G5-4), persisted decisions (G5-5), QA against the layout and plan cache (G5-6)

def _fake_ebsynth(tmp, name, body):
    p = tmp / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return str(p)


EB_ARGS = 'while [ $# -gt 0 ]; do case "$1" in -style) S="$2";; -output) O="$2";; esac; shift; done\n'


def test_ebsynth_failures():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
    _keys_from_guides(d, 3)
    cnt = tmp / "count"
    ok = _fake_ebsynth(tmp, "eb_ok", f'echo x >> "{cnt}"\n' + EB_ARGS + 'cp "$S" "$O"\n')
    fail = _fake_ebsynth(tmp, "eb_fail", 'echo "error: guide sizes differ"\nexit 3\n')
    noout = _fake_ebsynth(tmp, "eb_noout", 'exit 0\n')
    hang = _fake_ebsynth(tmp, "eb_hang", 'sleep 30\n')
    fail_at5 = _fake_ebsynth(tmp, "eb_fail5", f'echo x >> "{cnt}"\n[ $(wc -l < "{cnt}") -ge 5 ] && exit 7\n' + EB_ARGS + 'cp "$S" "$O"\n')
    for exe, want in ((fail, "exit 3"), (noout, "wrote no output"), (hang, "timed out after 1 s")):
        msg = _exits(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=exe, ebsynth_timeout=1, jobs=1))
        first, _, rest = msg.partition("\n")
        assert "EbSynth" in first and "frame" in first and "from key" in first and want in first, msg
        assert rest.startswith("  command: ") and exe in rest, msg
    assert "guide sizes differ" in _exits(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=fail))
    # a run that dies part-way leaves its finished syntheses; the next run does only the rest
    cnt.write_text("")
    msg = _exits(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=fail_at5, jobs=1))
    assert "exit 7" in msg
    cnt.write_text("")
    out = _quiet(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=ok, keep_work=True))
    runs = len(cnt.read_text().split())
    assert runs == 16 and "16 EbSynth runs, 4 reused" in out and "reused)" in out, (runs, out)
    assert (d / "smoke.webp").exists() and list((d / "propagate").glob("syn_*.png"))
    cnt.write_text("")                                    # with the work kept, a second run makes 0 EbSynth runs
    out = _quiet(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=ok))
    assert cnt.read_text() == "" and "0 EbSynth runs, 20 reused" in out, out
    assert not list((d / "propagate").glob("syn_*.png"))
    cnt.write_text("")                                    # --force redoes them all
    _quiet(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=ok, force=True))
    assert len(cnt.read_text().split()) == 20
    assert "--ebsynth" in _exits(g.cmd_sequence_propagate, _args(tmp, "propagate", every=3, ebsynth=str(tmp / "nope")))
    shutil.rmtree(tmp)


def test_engine_and_build_opts_persist():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    with _Ebsynth(True):
        _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
        assert json.loads((d / "sequence.json").read_text())["route"]["engine"] == "ebsynth"
        _quiet(g.cmd_sequence_plan, _args(tmp, "plan", engine="gpt"))
        meta = json.loads((d / "sequence.json").read_text())
        assert meta["engine"] == "gpt" and meta["route"]["engine"] == "gpt"
        with _Edit() as ed:                               # render follows the recorded engine: every frame
            _quiet(g.cmd_sequence_render, _args(tmp, "render", every=None))
        assert len(ed.calls) == 12, len(ed.calls)
        _quiet(g.cmd_sequence_plan, _args(tmp, "plan", engine="auto"))
        assert json.loads((d / "sequence.json").read_text())["route"]["engine"] == "ebsynth"
    # build options are remembered: --below on the first build still applies to the second
    _quiet(g.cmd_sequence_build, _args(tmp, "build", below=60, ref=str(tmp / "puff_s.png"), quality=None, preview_bg=None))
    meta = json.loads((d / "sequence.json").read_text())
    assert meta["build_opts"] == {"ref": "inputs/ref.png", "below": 60, "quality": 80, "preview_bg": "#a0aabe"}, meta["build_opts"]
    out = _quiet(g.cmd_sequence_build, _args(tmp, "build", quality=None, preview_bg=None))
    assert "using --ref inputs/ref.png --below 60 from the last build" in out, out
    frame = np.asarray(Image.open(d / "frames" / "005.png"))
    assert frame[60:, :, 3].max() == 0 and frame[:60, :, 3].max() > 0
    assert "--below 999" in _exits(g.cmd_sequence_build, _args(tmp, "build", below=999))
    assert "--every 5 must divide" in _exits(g.cmd_sequence_render, _args(tmp, "render", every=5))
    shutil.rmtree(tmp)


def test_reroll_after_propagate():
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **_small(tmp)))
    for guide in sorted((d / "guides").glob("*.png")):
        shutil.copyfile(guide, d / "raw" / guide.name)
    g._seq_meta(d, propagated={"every": 3, "keys": [0, 3, 6, 9]})
    with _Edit() as ed:
        msg = _exits(g.cmd_sequence_reroll, _args(tmp, "reroll", frames="4"))
    assert "frame 4 was synthesised from keys 3 and 6" in msg and "raw/003.png" in msg and not ed.calls, msg
    assert "comma list" in _exits(g.cmd_sequence_reroll, _args(tmp, "reroll", frames="a,b"))
    assert "outside 0-11" in _exits(g.cmd_sequence_reroll, _args(tmp, "reroll", frames="30"))
    g._seq_meta(d, propagated=None)
    assert "no outliers recorded" in _exits(g.cmd_sequence_reroll, _args(tmp, "reroll"))
    # a real reroll: frame 4 drawn off its layout is replaced by a candidate that follows it; the old raw is kept
    if HAVE_CV2:
        meta = json.loads((d / "sequence.json").read_text())
        smoke = g.plume_guide(4 / meta["fps"], meta["plume"], None, g._seq_sprites(d, meta), canvas=meta["canvas"])
        bad = Image.new("RGBA", smoke.size)
        bad.alpha_composite(smoke, (-9, 5))
        bad.alpha_composite(g._rgba(g._seq_path(d, meta["base"])))
        bad.resize((1024, 1024), Image.LANCZOS).save(d / "raw" / "004.png")
        _quiet(g.cmd_sequence_build, _args(tmp, "build"))
        outliers = json.loads((d / "report.json").read_text())["outliers"]
        assert 4 in outliers, outliers
        with _Edit() as ed:
            out = _quiet(g.cmd_sequence_reroll, _args(tmp, "reroll", candidates=1))
        assert len(ed.calls) == len(outliers) and "frames replaced" in out, out
        assert (d / "raw" / "004.prev.png").exists() and (d / "raw" / "004.json").exists()
        _quiet(g.cmd_sequence_build, _args(tmp, "build"))
        assert not json.loads((d / "report.json").read_text())["outliers"]
    shutil.rmtree(tmp)


def test_qa_against_layout():
    """Perfect renders of layouts that the neighbour-only QA used to flag: puff births 9 frames apart, a spray fan."""
    if not HAVE_CV2:
        return
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", **{**_small(tmp), "loop": 1.5, "chuff": 0.75, "life": 2.0, "fps": 8}))
    for guide in sorted((d / "guides").glob("*.png")):
        shutil.copyfile(guide, d / "raw" / guide.name)
    _quiet(g.cmd_sequence_build, _args(tmp, "build"))
    report = json.loads((d / "report.json").read_text())
    assert report["frames"] == 12 and report["vs_layout"] and not report["outliers"], report["outlier_scores"]
    assert report["backwards"] == 0
    import math
    spr = [(0.3 * math.sin(7 * k), (k * 0.37) % 1) for k in range(30)]

    def spray(dr, i):
        for ang, ph in spr:
            u = (i / 12 * 2 + ph) % 1
            x = 20 + 100 * u * math.cos(ang)
            y = 100 - 100 * u * math.sin(ang + 0.6) + 60 * u * u
            r = 2 + u
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(120, 190, 255, int(255 * (1 - u))))
    _quiet(g.cmd_sequence_layout, _args(tmp, "layout", guides=str(_guides(tmp, 12, spray, 128)), fps=12))
    for guide in sorted((d / "guides").glob("*.png")):
        shutil.copyfile(guide, d / "raw" / guide.name)
    _quiet(g.cmd_sequence_build, _args(tmp, "build"))
    report = json.loads((d / "report.json").read_text())
    assert not report["outliers"] and report["backwards"] == 0, (report["outliers"], report["speed_along"])
    shutil.rmtree(tmp)


def test_plan_cached():
    import time
    tmp = Path(tempfile.mkdtemp())
    d = tmp / "sequence" / "smoke"
    with _Ebsynth(True):
        _quiet(g.cmd_sequence_layout, _args(tmp, "layout", fps=24,
                                            guides=str(_guides(tmp, 12, _dot(lambda i: 128, lambda i: (30 + 8 * i) % 256)))))
        meta = json.loads((d / "sequence.json").read_text())
        t = time.time()
        route = g._seq_route(d, meta, _args(tmp, "plan"))
        assert time.time() - t < 1.0 and route["every"] == 6 and route["key"], route
        buf = _quiet(g.cmd_sequence_plan, _args(tmp, "plan"))
        assert "plan:" in buf                                # asked for: printed again, from the cache
        (d / "guides" / "000.png").write_bytes((d / "guides" / "001.png").read_bytes())
        meta["guide_sha1"][0] = "changed"
        assert g._seq_route(d, meta, _args(tmp, "plan"))["key"] != route["key"]   # a changed guide replans
    shutil.rmtree(tmp)


# -- argv validation (G5-7)

def _main_exit(argv):
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
    tmp = Path(tempfile.mkdtemp())
    small = _small(tmp)
    plume = ["--base", small["base"], "--sprites", small["sprites"], "--vent", small["vent"]]
    seq = ["sequence", str(tmp), "smoke"]
    table = [
        (["layout", "--chuff", "0"] + plume, 2, "positive number"),
        (["layout", "--drag", "0"] + plume, 2, "positive number"),
        (["layout", "--fps", "0"] + plume, 2, "positive number"),
        (["layout", "--loop", "0"] + plume, 2, "positive number"),
        (["layout", "--life", "0"] + plume, 2, "positive number"),
        (["layout", "--size", "0"] + plume, 2, "positive number"),
        (["layout", "--wind", "-40,-3"] + plume, 2, "expected one argument"),
        (["layout"] + plume + ["--vent", "abc"], 1, "x,y numbers"),
        (["layout"] + plume + ["--velocity", "1,2,3"], 1, "x,y numbers"),
        (["layout"] + plume + ["--sprites", str(tmp / "nope.png")], 1, "sprite not found"),
        (["layout"] + plume + ["--base", str(tmp / "nob.png")], 1, "--base not found"),
        (["layout", "--guides", str(tmp / "nodir")], 1, "guide folder not found"),
        (["layout"], 1, "needs --base, --sprites and --vent"),
        (["layout", "--loop", "1", "--chuff", "0.3"] + plume, 1, "whole number"),
        (["render", "--jobs", "0"], 2, "positive integer"),
        (["render", "--every", "0"], 2, "positive integer"),
        (["render"], 1, "layout"),
        (["build"], 1, "layout"),
        (["propagate"], 1, "layout"),
        (["reroll", "--candidates", "0"], 2, "positive integer"),
        (["reroll"], 1, "layout"),
        (["build", "--below", "0"], 2, "positive number"),
        (["build", "--quality", "0"], 2, "positive integer"),
        (["plan", "--engine", "foo"], 2, "invalid choice"),
        (["propagate", "--ebsynth-timeout", "-1"], 2, "positive number"),
        (["fly"], 2, "invalid choice"),
    ]
    for argv, code, text in table:
        got, err = _main_exit(seq + argv)
        assert got == code and text in err and "Traceback" not in err, (argv, got, err)
    with _Ebsynth(False):
        got, err = _main_exit(seq + ["layout", "--wind=-40,-3", "--loop", "1", "--chuff", "0.5", "--fps", "12"] + plume)
    assert got == 0, err                                  # the = form is how a negative x is passed
    for argv, text in ((["render", "--every", "5"], "must divide"), (["reroll", "--frames", "a,b"], "comma list"),
                       (["reroll", "--frames", "40"], "outside 0-11"),
                       (["propagate", "--every", "5", "--ebsynth", "/bin/true"], "divide")):
        got, err = _main_exit(seq + argv)
        assert got == 1 and text in err and "Traceback" not in err, (argv, got, err)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
