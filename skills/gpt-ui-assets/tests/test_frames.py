"""Offline checks for the frames pipeline: python3 tests/test_frames.py (or pytest)."""
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
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
    fr = g._align([flame(i) for i in range(8)], "bottom")
    ref = fr[0]
    assert not any(r["flags"] for r in g._frame_qa(fr, ref)["frames"])
    bad = list(fr)
    bad[3] = ImageEnhance.Color(bad[3]).enhance(0.2)
    assert "colour" in g._frame_qa(bad, ref)["frames"][3]["flags"]
    sway = g._align([flame(i, 16) for i in range(16)], "bottom")
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
    seq = g.lottie_frames(frames[:4], 12)
    assert [l["ip"] for l in seq["layers"]] == [0, 1, 2, 3] and len(seq["assets"]) == 4


def test_match_colors_moves_toward_reference():
    fr = [flame(i) for i in range(4)]
    tinted = [ImageEnhance.Color(f).enhance(1.8) for f in fr]
    before = g._frame_qa(tinted, fr[0])["sequence"]["dE_ref"]
    after = g._frame_qa(g._match_colors(tinted, fr[0]), fr[0])["sequence"]["dE_ref"]
    assert after < before / 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
