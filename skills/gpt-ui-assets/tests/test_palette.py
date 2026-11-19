"""Offline checks for tonal palettes and colour schemes."""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")
from gptkit import palette as m


SEEDS = ["#6750A4", "#FF0000", "#00C853", "#FFD600", "#0066CC", "#FFE066"]


def test_tones_and_gamut():
    for seed in SEEDS[:4]:
        assert m.tone_hex(seed, 0) == "#000000"
        assert m.tone_hex(seed, 100) == "#FFFFFF"
        hue = m.lch(seed)[2]
        for tone in range(10, 96):
            colour = m.tone_hex(seed, tone)
            assert re.fullmatch(r"#[0-9A-F]{6}", colour)
            lightness, chroma, actual_hue = m.lch(colour)
            assert abs(lightness - tone) < 0.6, (seed, tone, colour, lightness)
            if chroma > 10:
                assert abs((actual_hue - hue + 180) % 360 - 180) < 3
        assert list(m.tonal_palette(seed)) == m.TONES


def test_role_tones_and_modes():
    light = m.scheme("#6750A4", "light")
    dark = m.scheme("#6750A4", "dark")
    assert abs(m.lch(light["primary"])[0] - 40) < 1
    assert abs(m.lch(dark["primary"])[0] - 80) < 1
    assert abs(m.lch(light["surface"])[0] - 98) < 1
    assert abs(m.lch(dark["surface"])[0] - 6) < 1
    assert dark["surface"] != "#000000"
    assert all(light[role] != dark[role] for role in light if role not in ("scrim", "shadow"))
    assert light["scrim"] == dark["scrim"] == "#000000"


def test_measured_and_contrast():
    result = m.schemes("#6750A4", "light", measured={"primary": "#E91E63"})
    assert result["light"]["primary"] == "#E91E63"
    assert result["dark"]["primary"] != "#E91E63"
    assert any("dark" in note for note in result["notes"])
    assert m.contrast("#000000", "#FFFFFF") == 21
    for seed in SEEDS:
        result = m.schemes(seed, "light", measured={"primary": seed})
        for mode in ("light", "dark"):
            colours = result[mode]
            for fg, bg in (("onPrimary", "primary"), ("onSurface", "surface"),
                           ("onSurfaceVariant", "surface"), ("onPrimaryContainer", "primaryContainer")):
                assert m.contrast(colours[fg], colours[bg]) >= 4.5, (seed, mode, fg, bg)
    assert any("onPrimary" in note for note in m.schemes("#FFE066", "light", measured={"primary": "#FFE066"})["notes"])


def test_measured_surface_moves_its_family_and_repairs_text():
    from gptkit import system
    result = m.schemes("#6750A4", "light", measured={"surface": "#F4F1EA"})
    assert result["light"]["surface"] == "#F4F1EA"
    assert abs(m.lch(result["light"]["surfaceContainer"])[0] - (m.lch("#F4F1EA")[0] - 4)) < 1
    assert not system.check_pairs(result["light"], "material3")
    odd = m.schemes("#6750A4", "light", measured={"surfaceContainer": "#000000"})
    assert any("no onSurface tone" in note for note in odd["notes"])
    assert m.contrast("#00000000", "#FFFFFF") == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
