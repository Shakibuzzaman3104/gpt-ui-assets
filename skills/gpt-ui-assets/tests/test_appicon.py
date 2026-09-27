"""Offline checks for `gptkit.py appicon build` (no API calls): python3 -m unittest discover skills/gpt-ui-assets/tests"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

K = Path(__file__).resolve().parents[1] / "scripts" / "gptkit.py"


def concept(path, r):
    S = 1024
    yy, xx = np.mgrid[0:S, 0:S] / S
    d = np.sqrt((xx - 0.5) ** 2 + (yy - 0.45) ** 2)
    bg = np.stack([40 - 30 * d, 45 - 30 * d, 110 - 70 * d], -1).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(bg)
    m = Image.new("L", (S, S), 0)
    dm = ImageDraw.Draw(m)
    dm.ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], fill=255)
    dm.ellipse([S / 2 - r * 0.55, S / 2 - r * 1.25, S / 2 + r * 1.45, S / 2 + r * 0.75], fill=0)
    im.paste(Image.new("RGB", (S, S), (255, 219, 205)), (0, 0), m)
    im.save(path)


def opened(p):
    with Image.open(p) as im:
        return im.size, im.mode, dict(im.info)


def build(tmp, r, *extra):
    src = Path(tmp) / f"c{r}.png"
    concept(src, r)
    out = Path(tmp) / f"b{r}"
    subprocess.run([sys.executable, str(K), "appicon", "build", str(src), "--out", str(out), "--mono", "silhouette",
                    "--no-validate", *extra], check=True, capture_output=True, env={"GPTKIT_CONFIG": "/nonexistent"})
    return out, json.loads((out / "appicon.json").read_text())


class AppIconBuild(unittest.TestCase):
    def test_small_subject_is_kept_and_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, rep = build(tmp, 260)
            self.assertEqual(rep["background"]["kind"], "gradient")
            self.assertEqual(rep["safe_zone"]["shrink"], 1.0)
            self.assertTrue(rep["safe_zone"]["ok"])
            self.assertLess(rep["composite_diff_vs_concept"], 3)
            self.assertTrue(rep["ios"]["opaque"])
            res = out / "android" / "res"
            for dens, px in (("mdpi", 108), ("xxxhdpi", 432)):
                self.assertEqual(opened(res / f"mipmap-{dens}" / "ic_launcher_foreground.webp")[0], (px, px))
            self.assertEqual(opened(res / "mipmap-xxhdpi" / "ic_launcher_round.webp")[0], (144, 144))
            xml = (res / "mipmap-anydpi-v26" / "ic_launcher.xml").read_text()
            self.assertIn("@drawable/ic_launcher_background", xml)
            self.assertIn("@mipmap/ic_launcher_monochrome", xml)
            self.assertEqual(opened(out / "ios" / "AppIcon.appiconset" / "AppIcon-1024.png")[1], "RGB")
            self.assertEqual(opened(out / "web" / "favicon.ico")[2]["sizes"], {(16, 16), (32, 32), (48, 48)})

    def test_oversized_subject_is_shrunk_into_safe_zone(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rep = build(tmp, 470, "--bg", "color")
            self.assertLess(rep["safe_zone"]["shrink"], 1.0)
            self.assertLessEqual(rep["safe_zone"]["subject_radius_dp"], 33.0)
            self.assertTrue(rep["web_maskable"]["ok"])


if __name__ == "__main__":
    unittest.main()
