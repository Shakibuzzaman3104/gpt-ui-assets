"""Offline checks for platform profiles, native-scale calibration and moving-part registration: python3 tests/test_platform.py"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")  # the offline tests never reach a real Codex CLI
import gptkit as g  # noqa: E402


def test_platform_rule_names_native_components():
    assert "NavigationBar" in g._platform_rule("android") and "SF Symbols" in g._platform_rule("ios")
    assert g._platform_rule(None) == ""
    assert g._screen_dp("android") == 412 and g._screen_dp(None) == 390 and g._screen_dp("ios", 400) == 400


def test_calibration_from_native_anchors():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        m = {"mockup": "mockup.png", "width": 1024, "height": 1536, "platform": "android", "screen_dp": 412,
             "assets": [{"name": "home", "bbox": {"x": 0, "y": 0, "w": 36, "h": 36}}]}
        spec_ = {"components": [
            {"name": "Bottom navigation", "native": "NavigationBar (4 destinations)", "bbox": {"x": 0, "y": 1390, "w": 1024, "h": 139}},
            {"name": "Add", "native": "FloatingActionButton", "bbox": {"x": 880, "y": 1250, "w": 88, "h": 88}},
            {"name": "Summary", "native": "Card (filled)", "bbox": {"x": 40, "y": 200, "w": 940, "h": 300}}]}
        g._calibrate(d, spec_, m, 1024)
        k = spec_["px_to_dp"]
        assert abs(k - np.median([80 / 139, 56 / 88])) < 1e-4 and spec_["native_scale"]["applied"]
        assert spec_["effective_width_dp"] == round(1024 * k) and spec_["px_to_dp_design"] == round(412 / 1024, 5)
        assert json.loads((d / "assets.json").read_text())["assets"][0]["dp"]["w"] == round(36 * k, 1)
        spec_design = {"components": spec_["components"]}
        g._calibrate(d, spec_design, dict(m), 1024, "design")
        assert spec_design["px_to_dp"] == round(412 / 1024, 5) and not spec_design["native_scale"]["applied"]


def test_calibration_from_ios_anchors():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        m = {"mockup": "mockup.png", "width": 1024, "height": 1536, "platform": "ios", "screen_dp": 393, "assets": []}
        spec_ = {"components": [
            {"name": "Tabs", "native": "TabBar (4 tabs)", "bbox": {"x": 60, "y": 1420, "w": 900, "h": 85}},
            {"name": "Header", "native": "NavigationStack bar with large title", "bbox": {"x": 0, "y": 100, "w": 1024, "h": 200}},
            {"name": "List", "native": "List (inset grouped)", "bbox": {"x": 30, "y": 320, "w": 960, "h": 600}}]}
        g._calibrate(d, spec_, m, 1024)
        ns = spec_["native_scale"]
        assert ns["applied"] and [a["component"] for a in ns["anchors"]] == ["Tabs"], ns
        assert abs(ns["px_to_dp_native"] - round(49 / 85, 5)) < 1e-9 and spec_["px_to_dp"] == round(49 / 85, 5)
        assert spec_["effective_width_dp"] == round(1024 * 49 / 85)
        pats = g.NATIVE_ANCHORS["ios"]

        def anchor(native):
            return next((dp for pat, dp, _ in pats if re.search(pat, native, re.I)), None)
        assert anchor("Navigation bar (inline title)") == 44 and anchor("NavigationStack bar, inline") == 44
        assert anchor("Nav bar") == 44 and anchor("NavigationStack bar with large title") is None
        assert anchor("floating Liquid Glass TabBar (up to 5 tabs)") == 49 and anchor("Tab bar") == 49
        assert anchor("Toggle") == 31 and anchor("Picker (segmented)") == 32 and anchor("List (inset grouped)") is None


def test_registration_ignores_static_slivers():
    size = (400, 400)
    src = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(src)
    d.ellipse((60, 150, 340, 380), fill=(240, 150, 140, 255))      # body (static)
    d.ellipse((250, 30, 320, 100), fill=(240, 190, 40, 255))       # coin (moving)
    static = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(static).ellipse((80, 165, 320, 365), fill=(240, 150, 140, 255))  # redrawn slightly smaller
    frames = []
    for i in range(4):
        f = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        ImageDraw.Draw(f).ellipse((10 + i, 10, 110 - i, 110), fill=(240, 190, 40, 255))
        frames.append(f)
    s, org, v = g._placement(src, static, frames)
    assert v > 0.5, v
    cx = org[0] + 60 * s
    assert 250 < cx < 320, (org, s)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
