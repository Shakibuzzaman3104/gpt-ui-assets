"""Offline checks for in-between frames: python3 tests/test_tween.py (or pytest)."""
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")
import gptkit as g  # noqa: E402


def disk(x, size=96, r=14):
    """A shaded disk centred at (x, size/2) on a transparent canvas."""
    im = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(im)
    for k in range(r, 0, -1):
        c = 120 + 135 * (r - k) // r
        d.ellipse([x - k, size / 2 - k, x + k, size / 2 + k], fill=(c, 90, 40, 255))
    return im


def centroid_x(img):
    a = np.asarray(img)[..., 3].astype(float)
    return (a.sum(0) * np.arange(a.shape[1])).sum() / a.sum()


LOOP = [30, 40, 50, 60, 50, 40]                       # there and back: the wrap pair 40 -> 30 closes the loop


def test_flow_midpoints_follow_the_motion():
    keys = [disk(x) for x in LOOP]
    out, engine = g.tween(keys, 2, "flow")
    assert engine == "flow" and len(out) == 12 and all(out[2 * i] is keys[i] for i in range(6))
    for i in range(6):
        want = (LOOP[i] + LOOP[(i + 1) % 6]) / 2
        mid = out[2 * i + 1]
        assert abs(centroid_x(mid) - want) < 1.0, (i, centroid_x(mid), want)
        a = np.asarray(mid)[..., 3]
        assert (a > 250).sum() > 0.85 * (np.asarray(keys[i])[..., 3] > 250).sum()  # one solid disk, not two faint ones
    thirds, _ = g.tween(keys, 3, "flow")
    assert len(thirds) == 18 and abs(centroid_x(thirds[1]) - 33.33) < 1.0 and abs(centroid_x(thirds[2]) - 36.67) < 1.0


def test_blend_and_missing_opencv():
    keys = [disk(x) for x in LOOP]
    out, engine = g.tween(keys, 2, "blend")
    solid = lambda im: (np.asarray(im)[..., 3] > 250).sum()                # noqa: E731
    assert engine == "blend" and solid(out[1]) < 0.85 * solid(keys[0])   # a crossfade: two half-strength disks
    saved = sys.modules.get("cv2")
    sys.modules["cv2"] = None                                             # import cv2 now raises ImportError
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, engine = g.tween(keys, 2, "flow")
    finally:
        if saved is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = saved
    assert engine == "blend" and "OpenCV is not installed" in buf.getvalue()
    assert g.tween(keys, 1)[0] == keys and g.tween(keys[:1], 2)[1] is None


def test_rife_engine():
    rife = g.find_rife()
    if not rife:
        print("  (rife-ncnn-vulkan not installed: RIFE engine test skipped)")
        return
    keys = [disk(x) for x in LOOP]
    out, engine = g.tween(keys, 2, "rife")
    assert engine == "rife" and len(out) == 12
    for i in range(6):
        want = (LOOP[i] + LOOP[(i + 1) % 6]) / 2
        a = np.asarray(out[2 * i + 1])
        assert abs(centroid_x(out[2 * i + 1]) - want) < 1.5 and a[0, 0, 3] == 0   # alpha recovered, background clear


def test_rife_paths():
    os.environ["GPTKIT_RIFE"] = "/nonexistent/rife-ncnn-vulkan"
    try:
        g.find_rife()
        raise AssertionError("must exit")
    except SystemExit as e:
        assert "GPTKIT_RIFE /nonexistent/rife-ncnn-vulkan not found" in str(e)
    finally:
        del os.environ["GPTKIT_RIFE"]
    assert "rife-v4.6" in g.RIFE_INSTALL and "rife-ncnn-vulkan-20221029-" in g.RIFE_INSTALL


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
