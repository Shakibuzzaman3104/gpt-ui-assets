"""Offline evaluator for the kit's <animated-vector> XML: node attribute values at loop fraction t, following the
framework's semantics for what build_avd emits (PVH keyframes with per-keyframe pathInterpolator, windowed
pathInterpolator android:pathData, repeatMode=reverse, pathData number interpolation). Test helper, no Android needed."""
import re
import xml.etree.ElementTree as ET

A = "{http://schemas.android.com/apk/res/android}"
AAPT_ATTR = "{http://schemas.android.com/aapt}attr"
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def bez(x1, y1, x2, y2, x):
    """y of the cubic-bezier easing (x1, y1, x2, y2) at progress x."""
    f = lambda u, a, b: 3 * (1 - u) ** 2 * u * a + 3 * (1 - u) * u * u * b + u ** 3
    lo, hi = 0.0, 1.0
    for _ in range(50):
        u = (lo + hi) / 2
        lo, hi = (u, hi) if f(u, x1, x2) < x else (lo, u)
    return f((lo + hi) / 2, y1, y2)


def path_interp(gk, d, x):
    """y of a pathInterpolator android:pathData (M/L/C from (0,0) to (1,1)) at progress x."""
    subs = gk._subpaths(d)
    p0 = subs[0]["pts"][0]
    for s in subs[0]["segs"]:
        p1 = s[-1]
        if p0[0] <= x <= p1[0] + 1e-9:
            if s[0] == "L":
                return p0[1] if p1[0] == p0[0] else p0[1] + (p1[1] - p0[1]) * (x - p0[0]) / (p1[0] - p0[0])
            c1, c2 = s[1], s[2]
            f = lambda u, k: (1 - u) ** 3 * p0[k] + 3 * (1 - u) ** 2 * u * c1[k] + 3 * (1 - u) * u * u * c2[k] + u ** 3 * p1[k]
            lo, hi = 0.0, 1.0
            for _ in range(50):
                u = (lo + hi) / 2
                lo, hi = (u, hi) if f(u, 0) < x else (lo, u)
            return f((lo + hi) / 2, 1)
        p0 = p1
    return 1.0


def interp_of(gk, el):
    at = next((c for c in el if c.tag == AAPT_ATTR and c.get("name") == "android:interpolator"), None)
    if at is None:
        return lambda x: x
    pi = at[0]
    if pi.get(A + "pathData"):
        return lambda x, d=pi.get(A + "pathData"): path_interp(gk, d, x)
    c = [float(pi.get(A + k)) for k in ("controlX1", "controlY1", "controlX2", "controlY2")]
    return lambda x: bez(*c, x)


def pvh_value(gk, pvh, t):
    kfs = [(float(k.get(A + "fraction")), float(k.get(A + "value")), interp_of(gk, k)) for k in pvh]
    for (f0, v0, _), (f1, v1, ip) in zip(kfs, kfs[1:]):
        if f0 <= t <= f1:
            x = 0 if f1 == f0 else (t - f0) / (f1 - f0)
            return v0 + (v1 - v0) * ip(x)
    return kfs[-1][1]


def morph(a, b, p):
    """pathData interpolation: same command structure, numbers mixed linearly."""
    na, nb = NUM.findall(a), NUM.findall(b)
    assert len(na) == len(nb) and NUM.sub("#", a) == NUM.sub("#", b), "incompatible pathData"
    it = iter(float(x) + (float(y) - float(x)) * p for x, y in zip(na, nb))
    return NUM.sub(lambda m: f"{next(it):.4f}", a)


def at(gk, avd_xml, t, loop_ms):
    """{android:name: {attribute: value}} of every named node at loop fraction t (numbers as floats, pathData as str)."""
    root = ET.fromstring(avd_xml)
    vec = next(c for c in root if c.tag == AAPT_ATTR)[0]
    nodes = {e.get(A + "name"): e for e in vec.iter() if e.get(A + "name")}
    for tgt in root.findall("target"):
        node = nodes[tgt.get(A + "name")]
        anim = tgt.find(AAPT_ATTR)[0]
        for oa in (list(anim) if anim.tag == "set" else [anim]):
            dur = float(oa.get(A + "duration"))
            tt = t
            if oa.get(A + "repeatMode") == "reverse":
                ph = (t * loop_ms / dur) % 2
                tt = ph if ph <= 1 else 2 - ph
            if oa.get(A + "propertyName") == "pathData":
                node.set(A + "pathData", morph(oa.get(A + "valueFrom"), oa.get(A + "valueTo"), interp_of(gk, oa)(tt)))
            for pvh in oa.findall("propertyValuesHolder"):
                node.set(A + pvh.get(A + "propertyName"), f"{pvh_value(gk, pvh, tt):.5f}")
    out = {}
    for name, e in nodes.items():
        attrs = {}
        for k, v in e.attrib.items():
            k = k.replace(A, "")
            try:
                attrs[k] = float(v) if k != "pathData" else v
            except ValueError:
                attrs[k] = v
        out[name] = attrs
    return out
