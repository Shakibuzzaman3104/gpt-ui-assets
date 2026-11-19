"""Offline token geometry checks: python3 tests/test_tokens_geometry.py."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")
import gptkit as g


def test_measured_edges_and_above():
    items = [{"name": "card", "kind": "surface", "px": [24, 20, 300, 100], "gap_below_dp": ["footer", 42]}]
    assert g._measured_spacing("card right margin", items, 1, (360, 200))[0] == 36
    assert g._measured_spacing("card bottom margin", items, 1, (360, 200))[0] == 80
    assert g._measured_spacing("gap above card", items, 1, (360, 200)) is None
    items.append({"name": "header", "kind": "surface", "px": [24, 0, 300, 10], "gap_below_dp": ["card", 10]})
    assert g._measured_spacing("gap above card", items, 1, (360, 200))[0] == 10


def test_presets_and_audit():
    for platform, expected in (("compose", "material3"), ("swiftui", "ios"), ("web", "web"), ("flutter", "material3")):
        assert g._token_preset([platform]) == expected
    assert g._token_preset(["swiftui"], "macOS") == "macos"
    spec = {"spacing": [{"between": "odd gap", "px": 18.5}],
            "radii": [{"element": n, "px": v} for n, v in
                      (("outer", 28), ("inner", 28), ("a", 4), ("b", 8), ("c", 12), ("d", 16), ("e", 20))],
            "typography": [{"role": "caption", "size_px": 9}]}
    redline = {"radii": [{"element": n, "component": n, "measured_px": 28} for n in ("outer", "inner")],
               "items": [{"name": "outer", "kind": "surface"},
                         {"name": "inner", "kind": "surface", "parent": "outer",
                          "inset_dp": {"left": 16, "right": 16, "top": 16, "bottom": 16}}]}
    tm = g._token_model(spec, [], {}, None, redline, preset="web")
    audit = g._shape_audit(tm)
    assert any("spacing grid" in x for x in audit["issues"])
    assert any("distinct non-zero radii" in x for x in audit["issues"])
    assert any("concentricity" in x for x in audit["issues"])
    assert any("caption" in x and "minimum" in x for x in audit["type"])


def test_shapes_and_type_emitters():
    spec = {"radii": [{"element": "sheet", "px": 24}],
            "typography": [{"role": "body", "size_px": 17}]}
    redline = {"radii": [{"element": "sheet", "component": "sheet", "measured_px": 24,
                         "corner_radii_px": [24, 24, 0, 0]}]}
    tm = g._token_model(spec, [], {}, None, redline, preset="ios")
    assert tm["radii"][0]["corners"] == {"topStart": 24, "topEnd": 24, "bottomStart": 0, "bottomEnd": 0}
    args = SimpleNamespace(package="com.example.ui.theme", r_package=None, prefix="Demo", author=None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        g._emit_compose(tm, args, root / "compose")
        g._emit_swift(tm, args, root / "swift")
        g._emit_web(tm, args, root / "web")
        g._emit_flutter(tm, args, root / "flutter")
        assert "topStart = 24.dp, topEnd = 24.dp, bottomEnd = 0.dp, bottomStart = 0.dp" in (root / "compose/Shape.kt").read_text()
        swift = (root / "swift/DemoTokens.swift").read_text()
        assert "UnevenRoundedRectangle(topLeadingRadius: 24" in swift and ".system(.body, design: .default" in swift
        css = (root / "web/tokens.css").read_text()
        assert "--radius-sheet: 24px 24px 0px 0px" in css and "font-size: 1.0625rem; /* 17px */" in css
        assert "BorderRadius.only(topLeft: Radius.circular(24)" in (root / "flutter/demo_tokens.dart").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
