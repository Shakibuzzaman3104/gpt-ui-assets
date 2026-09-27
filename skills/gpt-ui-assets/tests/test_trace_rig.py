"""Offline checks for trace / rig / animate (no API). Run: python3 skills/gpt-ui-assets/tests/test_trace_rig.py"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gptkit as g  # noqa: E402


def synthetic(size=128):
    """Radial-gradient disc with a soft glow and a highlight: small stand-in for a forged illustration."""
    yy, xx = np.mgrid[:size, :size] + 0.5
    r = np.hypot(xx - size / 2, yy - size / 2)
    R = size * 0.3
    t = np.clip(r / R, 0, 1)[..., None]
    rgb = (1 - t) * np.array([255, 220, 120]) + t * np.array([230, 110, 40])
    alpha = np.where(r <= R, 1.0, np.clip(0.45 * np.exp(-(r - R) / (size * 0.06)), 0, 1))
    hl = np.hypot(xx - size * 0.42, yy - size * 0.4) < size * 0.06
    rgb[hl] = [255, 250, 235]
    return Image.fromarray(np.dstack([rgb, alpha * 255]).astype(np.uint8), "RGBA")


def test_ring_winding():
    m = np.zeros((64, 64), bool)
    yy, xx = np.mgrid[:64, :64]
    d = np.hypot(xx - 32, yy - 32)
    m[(d < 24) & (d > 12)] = True
    subs = g._subpaths(g.mask_to_path(m))
    assert len(subs) == 2
    t = np.linspace(0, 1, 12)
    areas = []
    for s in subs:
        pts, cur = [], np.array(s["pts"][0])
        for seg in s["segs"]:
            b = np.array([cur, seg[1], seg[1], seg[1]] if seg[0] == "L" else [cur, *seg[1:]], float)
            pts.append(g._bez(b, t))
            cur = b[3]
        areas.append(g._area(np.vstack(pts)))
    assert areas[0] * areas[1] < 0, "hole must wind opposite to the outline (nonzero fill)"


def test_trace_and_vd():
    src = synthetic()
    res = g.trace_image(src, "illustration", work=128)
    roles = {L["role"] for L in res["layers"]}
    assert roles == {"glow", "region"}, roles
    assert any(L["fill"]["type"] in ("radial", "linear") for L in res["layers"] if L["role"] == "region")
    svg = g.layers_svg(res["layers"], res["size"], src.size)
    ET.fromstring(svg)
    vd = g.svg_to_vd(svg, 48)
    assert "<vector" in vd and "transform" not in vd
    ET.fromstring(g._tr_vd_to_svg(vd))
    for L in res["layers"]:
        g._subpaths(L["d"])  # only absolute M/L/C/Z
        assert "e-" not in L["d"] and " ." not in L["d"], "lint InvalidVectorPath: no exponents / bare decimals"


def test_bake_translate():
    assert g._bake_translate("M0 0 C1 2 3 4 5 6 Z", 10, 20) == "M 10 20 C 11 22 13 24 15 26 Z"


def test_procedural_keys():
    d = "M10 90 C20 50 40 20 50 5 C60 20 80 50 90 90 L50 95Z M40 80 L60 80 L50 60Z"
    rest = g.map_path(d, lambda x, y: (x, y))
    for kind in g.TR_MOTIONS:
        keys = g.procedural_keyframes([d], kind, 6, phase=0.3)[0]
        assert keys[0] == rest, f"{kind}: key 0 must be the rest pose"
        assert len({"".join(t for t in g._TOK.findall(k) if t.isalpha()) for k in keys}) == 1, kind
        assert len(set(keys)) > 1, f"{kind}: must move"


def test_animate_to_lottie():
    res = g.trace_image(synthetic(), "illustration", work=128)
    vec = g.animate_layers(g.layers_vec(res), g._parse_motions("art=flicker; glow=flicker@art"), 6)
    assert not g._validate_vec(vec)
    assert not vec["static_layers"] and vec["animated_layers"]
    lot = g._tidy_lottie(g.build_lottie(vec, 256, 1.2, 30))
    json.dumps(lot)
    assert lot["layers"] and all(L["ty"] == 4 for L in lot["layers"])


def test_animate_traced_file_without_chrome():
    import argparse
    import os
    import tempfile
    res = g.trace_image(synthetic(), "illustration", work=96)
    old = os.environ.get("GPTKIT_CHROME")
    os.environ["GPTKIT_CHROME"] = "none"
    g._chrome.cache_clear()
    try:
        assert g.chrome_shot("<p>x</p>", 16, 16) is None
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "drop.layers.json"
            src.write_text(json.dumps(g.layers_vec(res)))
            g.cmd_animate(argparse.Namespace(layers=str(src), part="art=bob", keyframes=4, seconds=1.0, fps=24, size=128,
                                             ease="linear", out=None, name=None, preview_bg="#000000"))
            assert (Path(td) / "anim" / "drop" / "drop.json").exists()
    finally:
        if old is None:
            os.environ.pop("GPTKIT_CHROME")
        else:
            os.environ["GPTKIT_CHROME"] = old
        g._chrome.cache_clear()


def test_blank_filmstrip_is_not_rendered():
    import tempfile
    vec = g.animate_layers(g.layers_vec(g.trace_image(synthetic(), "illustration", work=64)), g._parse_motions("art=bob"), 4)
    saved = g.lottie_filmstrip, g._tr_render_svgs
    g.lottie_filmstrip = lambda lot, frames, size=256, bg="#1b1d33": Image.new("RGB", (size * len(frames), size), bg)
    g._tr_render_svgs = lambda svgs, sizes: None
    try:
        with tempfile.TemporaryDirectory() as td:
            assert g.write_animation(vec, td, "t", 1.0, 24, 128)["lottie_rendered"] is False
    finally:
        g.lottie_filmstrip, g._tr_render_svgs = saved


def test_render_fidelity():
    if not g._chrome():
        print("  (skip: no Chrome)")
        return
    src = synthetic()
    res = g.trace_image(src, "illustration", work=128)
    ren = g._tr_render_svgs([g.layers_svg(res["layers"], res["size"], src.size)], [src.size])[0]
    m = g.trace_metrics(src, ren)
    assert m["iou"] > 0.97 and m["diff"] < 4, m


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
