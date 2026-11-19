"""Offline checks for structured art directions and native calibration."""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")
import gptkit as g
from gptkit import design
from gptkit.system import contrast_ratio


def direction():
    return {"name": "Test", "mood": "calm", "theme": "light", "layout": "rows", "palette": "teal",
            "type": "sans", "shape": "soft", "spacing": "airy", "signature": "ring", "avoid": "noise",
            "system": {"preset": "material3", "roles": [
                {"role": "primary", "name": "deep teal", "hex": "#0F766E"},
                {"role": "onPrimary", "name": "teal", "hex": "#10766E"}],
                "type": [{"role": "body", "size": 16, "line_height": 24, "weight": 400}],
                "spacing_base": 4, "spacing": [4, 8, 16], "radius": [{"use": "cards", "value": 12}]}}


def validate(schema, value):
    kind = schema["type"]
    if kind == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]) == set(value)
        for key, child in schema["properties"].items():
            validate(child, value[key])
    elif kind == "array":
        assert isinstance(value, list)
        for item in value:
            validate(schema["items"], item)
    else:
        assert isinstance(value, {"string": str, "integer": int, "number": (int, float)}[kind])
        if "enum" in schema:
            assert value in schema["enum"]


def test_schema_and_prompt():
    d = direction()
    validate(g.DIRECTION_SCHEMA, d)
    assert "deep teal #0F766E" in g._direction_text(d)
    assert "body 16/24 regular" in g._direction_text(d)
    del d["system"]
    assert g._direction_text(d) == ("Art direction (light theme; calm). Layout: rows Palette: teal Type: sans "
                                    "Shape: soft Spacing: airy Signature: ring Avoid: noise")


def test_contrast_fix():
    d = direction()
    design._validate_direction_system(d)
    roles = {r["role"]: r["hex"] for r in d["system"]["roles"]}
    assert contrast_ratio(roles["onPrimary"], roles["primary"]) >= 4.5
    assert any(x["type"] == "fix" and x["role"] == "onPrimary" for x in d["system_issues"])


def test_mockup_role_contrast():
    d = direction()
    img = Image.new("RGB", (100, 100), "#0F766E")
    ImageDraw.Draw(img).rectangle((20, 20, 80, 80), fill="#10766E")
    issues = design._system_contrast_issues(img, d)
    assert any(x["category"] == "contrast" and x["measured_contrast"] < 4.5 for x in issues)
    issue = {"problem": "below 4.5:1", "severity": "medium", "measured_contrast": 4.9}
    g._weigh_contrast(issue)
    assert issue["severity"] == "medium"
    issue["measured_contrast"] = 5.0
    g._weigh_contrast(issue)
    assert issue["severity"] == "low"


def test_calibration_outlier_and_single():
    components = [{"name": name, "native": "FloatingActionButton", "bbox": {"w": 56 / k, "h": 56 / k}}
                  for name, k in [("first", 1.0), ("second", 1.02), ("outlier", 1.6)]]
    with tempfile.TemporaryDirectory() as td:
        m = {"platform": "android", "screen_dp": 400, "assets": []}
        spec = {"components": components}
        g._calibrate(Path(td), spec, m, 1000)
        assert [a["component"] for a in spec["native_scale"]["anchors"]] == ["first", "second"]
        assert abs(spec["px_to_dp"] - 1.01) < 0.0001
        single = {"components": components[:1]}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            g._calibrate(Path(td), single, dict(m), 1000)
        assert single["px_to_dp"] == 0.4 and not single["native_scale"]["applied"] and "first" in out.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
