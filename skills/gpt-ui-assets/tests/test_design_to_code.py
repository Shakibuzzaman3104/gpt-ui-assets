"""Offline checks for the design-to-code measurements (run: python3 -m pytest skills/gpt-ui-assets/tests, or as a script)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gptkit as g  # noqa: E402


def test_ciede2000_reference_pairs():
    pairs = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425), ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0),
             ((50, 2.5, 0), (73, 25, -18), 27.1492), ((2.0776, 0.0795, -1.135), (0.9033, -0.0636, -0.5514), 0.9082)]
    for a, b, want in pairs:
        assert abs(float(g._de00(np.array(a), np.array(b))) - want) < 1e-3


def test_surface_geometry_recovers_rounded_rect():
    ss, (x, y, w, h, r) = 4, (60, 50, 300, 140, 24)
    big = Image.new("RGB", (420 * ss, 260 * ss), "#0E1428")
    ImageDraw.Draw(big).rounded_rectangle([x * ss, y * ss, (x + w) * ss - 1, (y + h) * ss - 1], r * ss, fill="#171E36",
                                          outline="#26304F", width=ss)
    rgb = np.asarray(big.resize((420, 260), Image.BOX))
    geo = g._surface_geometry(rgb, {"x": x + 3, "y": y - 2, "w": w - 5, "h": h + 3})
    assert geo and geo["shape"] == "rect"
    assert max(abs(a - b) for a, b in zip(geo["box"], (x, y, w, h))) <= 1.5
    assert abs(geo["radius"] - r) <= 1.5


def test_locate_text_finds_the_ink_box():
    font = ImageFont.load_default(size=40)
    im = Image.new("RGB", (500, 120), "#0E1428")
    ImageDraw.Draw(im).text((40, 30), "Quiet Lake", font=font, fill="#F8F7FB")
    layer = Image.new("L", im.size, 0)
    ImageDraw.Draw(layer).text((40, 30), "Quiet Lake", font=font, fill=255)
    truth = layer.point(lambda v: 255 if v > 90 else 0).getbbox()
    got = g._locate_text(np.asarray(im), [20, 15, 330, 90], "Quiet Lake", 40, 0, "#F8F7FB")
    x, y, w, h = got["box"]
    assert max(abs(a - b) for a, b in zip((x, y, x + w, y + h), truth)) <= 2
    assert g._de_hex(got["ink"], "#F8F7FB") < 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
