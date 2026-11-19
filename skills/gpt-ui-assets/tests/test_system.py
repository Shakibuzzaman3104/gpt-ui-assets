"""Offline checks for design-system presets: python3 tests/test_system.py (or pytest)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")
from gptkit import system as m  # noqa: E402


def test_contrast_ratio():
    assert m.contrast_ratio("#000000", "#FFFFFF") == 21
    assert m.contrast_ratio("#00000000", "#FFFFFF") == 1.0          # alpha 0 composites to the background
    assert 4.4 < m.contrast_ratio("#777777", "#FFFFFF") < 4.5


def test_check_pairs_fails_grey_on_white():
    roles = {"onSurface": "#777777", "surface": "#FFFFFF"}
    issues = m.check_pairs(roles, "web")
    assert len(issues) == 1
    assert issues[0]["fg"] == "onSurface" and issues[0]["bg"] == "surface" and issues[0]["kind"] == "text"
    assert issues[0]["ratio"] == 4.48 and issues[0]["need"] == 4.5
    assert m.check_pairs(roles, "web", size=24) == []                # large text only needs 3:1


def test_check_pairs_ui_kind():
    roles = {"outline": "#999999", "surface": "#FFFFFF"}
    issues = m.check_pairs(roles, "ios")
    assert len(issues) == 1 and issues[0]["kind"] == "ui" and issues[0]["need"] == 3.0


def test_check_pairs_skips_absent_roles():
    roles = {"onPrimary": "#FFFFFF", "primary": "#6750A4"}
    assert m.check_pairs(roles, "material3", size=16) == []
    assert m.check_pairs({"surface": "#FFFFFF"}, "web") == []


def test_large_text_thresholds():
    assert m.is_large_text(24, 400, "web")
    assert m.is_large_text(18.66, 700, "web")
    assert not m.is_large_text(18, 400, "web")
    assert not m.is_large_text(23.9, 400, "material3")
    assert m.is_large_text(18, 400, "ios")
    assert m.is_large_text(14, 700, "ios")
    assert m.text_threshold(24, 400, "web") == 3.0
    assert m.text_threshold(14, 400, "web") == 4.5
    assert m.text_threshold(14, 700, "macos") == 3.0


def test_lint_type_flags():
    tiny = m.lint_type([{"name": "tiny", "size": 9}], "web")
    assert any("tiny" in s and "minimum" in s for s in tiny)
    body = m.lint_type([{"name": "bodyNote", "size": 16, "line_height": 16}], "web")
    assert any("bodyNote" in s and "1.2" in s for s in body)
    ten = [{"name": "s%d" % i, "size": 12 + 2 * i} for i in range(10)]
    assert any("10 distinct sizes" in s for s in m.lint_type(ten, "web"))
    wide = m.lint_type([{"name": "wide", "size": 14, "tracking_em": 0.2}], "web")
    assert any("wide" in s and "tracking" in s for s in wide)
    assert not m.lint_type([{"name": "ok", "size": 16, "line_height": 24, "tracking_em": 0.0}], "web")


def test_snap_spacing():
    out = m.snap_spacing([13, 15, 1], "material3")
    assert [o["snapped"] for o in out] == [12, 16, 1]
    assert out[0]["error"] == 1.0 and out[1]["error"] == -1.0 and out[2]["error"] == 0.0
    assert m.snap(13, [12, 16]) == (12, 1)
    assert m.snap(15, [12, 16]) == (16, -1)


def test_snap_radius():
    assert m.snap_radius(26, "material3")[:2] == ("extraLarge", 28)
    assert m.snap_radius(15, "material3")[:2] == ("large", 16)
    assert m.snap_radius(14, "material3")[:2] == ("medium", 12)      # a tie snaps to the smaller step
    assert m.snap_radius(40, "material3", height=80) == ("full", 9999, 0.0)
    assert m.snap_radius(12, "material3", height=80)[0] == "medium"
    assert "continuous" in m.PRESETS["ios"]["radius"]


def test_nested_radius():
    assert m.check_nested(28, 16, 12) is None
    msg = m.check_nested(28, 16, 28)
    assert msg and "12" in msg
    assert m.check_nested(12, 20, 0) is None


def test_audits():
    spacing = m.audit_spacing([13, 15, 16, 17, 18, 24, 25, 32, 40, 48], "material3")
    assert any(s.startswith("18 ") for s in spacing)
    assert any("distinct" in s for s in spacing)
    assert any("distinct" in s for s in m.audit_radius([4, 8, 12, 16, 20, 28], "material3"))
    assert any("99" in s for s in m.audit_radius([99], "material3"))
    assert m.audit_radius([4, 8, 12], "ios") == []


def test_nearest_style():
    assert m.nearest_style(22, "material3")["name"] == "titleLarge"
    assert m.nearest_style(17, "ios")["name"] in ("body", "headline")
    assert m.nearest_style(17, "ios", line_height=22, weight=400)["name"] == "body"
    hit = m.nearest_style(29, "fluent2")
    assert hit["name"] == "hero700" and hit["distance"] == 1


def test_presets():
    assert set(m.PRESETS) == {"material3", "ios", "macos", "fluent2", "web"}
    for key, p in m.PRESETS.items():
        assert p["source"] and all(str(u).startswith("http") for u in p["source"].values()), key
        assert p["type"], key
        sizes = [s["size"] for s in p["type"]]
        assert sizes == sorted(sizes, reverse=True), key
        assert p["grid"] == 4 and p["unit"] in ("dp", "pt", "px"), key
        assert p["margins"]["verified"] is False, key
        assert p["min_text"] <= p["default_body"], key
    m3 = {s["name"]: s for s in m.PRESETS["material3"]["type"]}
    assert len(m3) == 15
    assert (m3["bodyLarge"]["size"], m3["bodyLarge"]["line_height"]) == (16, 24)
    assert (m3["displayLarge"]["size"], m3["displayLarge"]["line_height"]) == (57, 64)
    assert m3["titleMedium"]["weight"] == 500 and m3["titleMedium"]["tracking_em"] > 0
    assert m.PRESETS["material3"]["dark_base"].startswith("tone 6")
    assert m.PRESETS["ios"]["dark_base"].startswith("pure black")
    assert m.PRESETS["material3"]["min_target"] == 48 and m.PRESETS["web"]["min_target"] == 24


def test_refuted_edge_cases():
    assert m.check_pairs({"onSurface": "#125B8EEF", "surface": "#C7E09B"}, "web")
    assert any("not a positive number" in i for i in m.lint_type([{"name": "body", "size": 0, "line_height": 12}], "ios"))
    assert m.snap_radius(0, "ios", height=2)[1] == 0
    assert not m.audit_radius([5, 9, 14], "ios")
    assert all(t["size"] >= m.PRESETS["web"]["min_text"] for t in m.PRESETS["web"]["type"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
