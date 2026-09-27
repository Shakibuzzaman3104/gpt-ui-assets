"""Offline checks for the vector animation builders: python3 tests/test_vector.py (no API calls)."""
import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

spec = importlib.util.spec_from_file_location("gptkit", Path(__file__).parent.parent / "scripts" / "gptkit.py")
gk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gk)

HEART = ("M22 40.5 8.2 27.1C3.9 22.8 1.75 19.2 1.75 14.1 1.75 7.6 6.1 3.5 11.7 3.5 16.1 3.5 19.4 6.2 22 9.7 "
         "24.6 6.2 27.9 3.5 32.3 3.5 37.9 3.5 42.25 7.6 42.25 14.1 42.25 19.2 40.1 22.8 35.8 27.1Z")
MOTION = {"anchor_x": 22, "anchor_y": 22, "keys": [
    {"t": 0, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "ease_in"},
    {"t": 0.25, "dx": 0, "dy": -2, "sx": 1.25, "sy": 1.25, "rotate": 8, "opacity": 0.6, "ease": "overshoot"},
    {"t": 0.6, "dx": 0, "dy": 0, "sx": 1, "sy": 1, "rotate": 0, "opacity": 1, "ease": "linear"}]}


def vec():
    return {"viewBox": "0 0 44 44", "defs": "", "layer_order": ["halo", "heart", "blob"],
            "static_layers": [{"name": "heart", "motion": MOTION, "paths": [
                {"d": HEART, "fill": "none", "opacity": 1, "stroke": "#8D8DB4", "stroke_width": 3.5}]}],
            "animated_layers": [{"name": "blob", "fill": "#FFB15E", "opacity": 1, "stroke": "none", "stroke_width": 0,
                                 "motion": {"anchor_x": 0, "anchor_y": 0, "keys": []},
                                 "keyframes": ["M10 10 L20 10 L15 20 Z", "M10 12 C12 8 18 8 20 12 L15 22 Z",
                                               "M9 10 L21 10 L15 18 Z", "M10 10 L20 10 L15 20 Z"]}],
            "glows": [{"name": "halo", "cx": 22, "cy": 22, "rx": 20, "ry": 14, "color": "#B9B6FF", "opacity": 0.5,
                       "motion": {"anchor_x": 22, "anchor_y": 22, "keys": []}}]}


def test_paths():
    subs = gk._subpaths("m10 10 h10 v10 a5 5 0 0 1 -10 0 q0 -5 5 -5 t5 0 s5 5 0 5 z")
    assert all(s[0] in "LC" for sp in subs for s in sp["segs"]) and subs[0]["closed"]
    arc = gk._subpaths("M0 0 A10 10 0 0 1 20 0")[0]["segs"]
    assert len(arc) == 4 and all(abs(math.dist(s[3], (10, 0)) - 10) < 1e-6 for s in arc)
    assert gk._subpaths(gk._d(subs)) == gk._subpaths(gk._d(gk._subpaths(gk._d(subs))))


def test_build():
    v = vec()
    assert not gk._validate_vec(v), gk._validate_vec(v)
    gk._normalize_vec(v)
    assert len(v["animated_layers"][0]["keyframes"]) == 3, "closing duplicate keyframe should be dropped"
    rep = gk.motion_report(v)
    assert rep["layers"]["heart"]["move"] > 0.05 and rep["layers"]["heart"]["fade"] >= 0.4
    lot = gk.build_lottie(v, 512, 1.2, 30)
    assert [lyr["nm"] for lyr in lot["layers"]] == ["blob", "heart", "halo"]
    heart = lot["layers"][1]["ks"]
    assert heart["s"]["a"] == 1 and heart["r"]["a"] == 1 and heart["a"]["k"][:2] == [22, 22]
    assert all(k["o"]["y"][0] <= 1 for k in heart["o"]["k"][:-1]), "opacity must not overshoot"
    fit = gk._fit_viewbox(v)
    assert fit and gk._vec_vb(dict(v, viewBox=fit))[2] > 44
    root = ET.fromstring(gk.build_avd(v, 1.2, 24))
    names = [t.get("{http://schemas.android.com/apk/res/android}name") for t in root.findall("target")]
    assert "heart" in names and {"blob_0", "blob_1", "blob_2"} <= set(names), names


if __name__ == "__main__":
    test_paths()
    test_build()
    print("ok")
