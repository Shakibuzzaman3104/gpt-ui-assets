"""Offline checks for the layout-driven sequence route: python3 tests/test_sequence.py (or pytest).

No API calls: the guides themselves stand in for GPT's renders, which is what a perfect render of the layout is.
"""
import argparse
import json
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
              force=False, ref=None, below=None, quality=80, frames=None, candidates=1, preview_bg="#a0aabe", model=None)
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
    """Build a sequence whose 'renders' are the guides themselves."""
    g.cmd_sequence_layout(_args(tmp, "layout", **_plume(tmp)))
    d = tmp / "sequence" / "smoke"
    for guide in sorted((d / "guides").glob("*.png")):
        shutil.copyfile(guide, d / "raw" / guide.name)
    return d


def test_build_cuts_the_locked_art_and_writes_a_loop():
    tmp = Path(tempfile.mkdtemp())
    d = _perfect_renders(tmp)
    meta = json.loads((d / "sequence.json").read_text())
    assert meta["frames"] == 24 and len(list((d / "guides").glob("*.png"))) == 24
    g.cmd_sequence_build(_args(tmp, "build", ref=str(tmp / "puff.png")))
    webp = Image.open(d / "smoke.webp")
    assert not json.loads((d / "report.json").read_text())["outliers"]   # no false alarms on a perfect render
    assert webp.n_frames == 24
    frame = np.asarray(Image.open(d / "frames" / "005.png"))
    assert frame[190, 130, 3] == 0                      # the engine body is cut away
    assert frame[..., 3].max() > 200                    # the smoke is kept
    report = json.loads((d / "report.json").read_text())
    assert len(report["rect"]) == 4 and report["frames"] == 24
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
    smoke = g.plume_guide(9 / meta["fps"], meta["plume"], None, canvas=meta["canvas"])
    bad = Image.new("RGBA", smoke.size)
    bad.alpha_composite(smoke, (-18, 10))
    bad.alpha_composite(g._rgba(meta["base"]))
    bad.resize((1024, 1024), Image.LANCZOS).save(d / "raw" / "009.png")
    g.cmd_sequence_build(_args(tmp, "build"))
    report = json.loads((d / "report.json").read_text())
    assert 9 in report["outliers"], report["outlier_scores"]
    shutil.rmtree(tmp)


def test_brief_names_the_frame_and_binds_the_layout():
    meta = {"frames": 72, "seconds": 3.0, "fps": 24, "moving": "The smoke", "locked": "The toy steam train"}
    brief = g._seq_prompt(meta, 9)
    assert "frame 10 of 72" in brief and "3-second" in brief and "24 frames per second" in brief
    assert "The toy steam train is final artwork and is locked" in brief
    assert "Render the smoke as the finished" in brief and "the layout is binding" in brief


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
