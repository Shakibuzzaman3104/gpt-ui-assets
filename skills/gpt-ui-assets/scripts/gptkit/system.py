"""Design-system presets from public guidelines and the checks a measured design must pass."""
from .measure import _hexi, _norm_hex

# ---- system: design-system presets and audits (BRIEF.md, verified 19/11/26)

_M3_SPACING = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
_IOS_SPACING = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48]
_MACOS_SPACING = [0, 4, 8, 12, 16, 20, 24, 32]
_FLUENT_SPACING = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
_CARBON_SPACING = [2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 160]

_M3_RADIUS = {"none": 0, "extraSmall": 4, "small": 8, "medium": 12, "large": 16, "largeIncreased": 20,
              "extraLarge": 28, "extraLargeIncreased": 32, "extraExtraLarge": 48, "full": 9999}
_FLUENT_RADIUS = {"r0": 0, "r1": 2, "r2": 4, "r3": 6, "r4": 8, "r5": 12, "r6": 16, "r7": 24, "r8": 32,
                  "r9": 40, "full": 9999}
_APPLE_RADIUS_VALUES = [0, 4, 6, 8, 10, 12, 16, 20, 24, 28, 34]

_M3_TRACKING = {"displayLarge": -0.25, "displayMedium": -0.25, "displaySmall": -0.25, "headlineLarge": 0,
                "headlineMedium": 0, "headlineSmall": 0, "titleLarge": 0, "titleMedium": 0.15, "titleSmall": 0.1,
                "bodyLarge": 0.5, "bodyMedium": 0.25, "bodySmall": 0.4, "labelLarge": 0.1, "labelMedium": 0.5,
                "labelSmall": 0.5}

_SF_TRACKING = {6: 41, 7: 34, 8: 26, 9: 19, 10: 12, 11: 6, 12: 0, 13: -6, 14: -11, 15: -16, 16: -20, 17: -26,
                18: -25, 19: -24, 20: -23, 21: -18, 22: -12, 23: -4, 24: 3, 25: 6, 26: 8, 27: 11, 28: 14, 29: 14,
                30: 14, 31: 13, 32: 13, 33: 12, 34: 12, 37: 10, 38: 10, 39: 10, 40: 10, 43: 9}

_FLUENT_RAMP = [("base100", 10, 14), ("base200", 12, 16), ("base300", 14, 20), ("base400", 16, 22),
                ("base500", 20, 28), ("base600", 24, 32), ("hero700", 28, 36), ("hero800", 32, 40),
                ("hero900", 40, 52), ("hero1000", 68, 92)]

_APPLE_SOURCE = {
    "type": "https://developer.apple.com/design/human-interface-guidelines/typography",
    "color": "https://developer.apple.com/design/human-interface-guidelines/color",
    "layout": "https://developer.apple.com/design/human-interface-guidelines/layout",
    "accessibility": "https://developer.apple.com/design/human-interface-guidelines/accessibility",
}


def _style(name, size, line_height, weight=400, tracking_em=0.0):
    """One type style record."""
    return {"name": name, "size": size, "line_height": line_height, "weight": weight, "tracking_em": tracking_em}


def _m3(name, size, line_height, weight=400):
    """An M3 role; tracking is stored in sp in the brief and converted to em here."""
    return _style(name, size, line_height, weight, _M3_TRACKING[name] / size)


def _sf(size):
    """SF tracking for a point size, in em (the table is 1/1000 em)."""
    return _SF_TRACKING.get(size, 0) / 1000.0


def _fluent_type():
    """The Fluent 2 ramp, regular weight, largest first."""
    return [_style(n, s, l) for n, s, l in reversed(_FLUENT_RAMP)]


def _apple_radius():
    """Apple radii r0-r10 plus full, marked continuous."""
    d = {"r%d" % i: v for i, v in enumerate(_APPLE_RADIUS_VALUES)}
    d["full"] = 9999
    d["continuous"] = True
    return d


_MATERIAL3_TYPE = [
    _m3("displayLarge", 57, 64),
    _m3("displayMedium", 45, 52),
    _m3("displaySmall", 36, 44),
    _m3("headlineLarge", 32, 40),
    _m3("headlineMedium", 28, 36),
    _m3("headlineSmall", 24, 32),
    _m3("titleLarge", 22, 28),
    _m3("titleMedium", 16, 24, 500),
    _m3("bodyLarge", 16, 24),
    _m3("titleSmall", 14, 20, 500),
    _m3("bodyMedium", 14, 20),
    _m3("labelLarge", 14, 20, 500),
    _m3("bodySmall", 12, 16),
    _m3("labelMedium", 12, 16, 500),
    _m3("labelSmall", 11, 16, 500),
]

_IOS_TYPE = [
    _style("largeTitle", 34, 41, 400, _sf(34)),
    _style("title1", 28, 34, 400, _sf(28)),
    _style("title2", 22, 28, 400, _sf(22)),
    _style("title3", 20, 25, 400, _sf(20)),
    _style("headline", 17, 22, 600, _sf(17)),
    _style("body", 17, 22, 400, _sf(17)),
    _style("callout", 16, 21, 400, _sf(16)),
    _style("subhead", 15, 20, 400, _sf(15)),
    _style("footnote", 13, 18, 400, _sf(13)),
    _style("caption1", 12, 16, 400, _sf(12)),
    _style("caption2", 11, 13, 400, _sf(11)),
]

_MACOS_TYPE = [
    _style("largeTitle", 26, 32, 400, _sf(26)),
    _style("title1", 22, 26, 400, _sf(22)),
    _style("title2", 17, 22, 400, _sf(17)),
    _style("title3", 15, 20, 400, _sf(15)),
    _style("headline", 13, 16, 700, _sf(13)),
    _style("body", 13, 16, 400, _sf(13)),
    _style("callout", 12, 15, 400, _sf(12)),
    _style("subheadline", 11, 14, 400, _sf(11)),
    _style("footnote", 10, 13, 400, _sf(10)),
    _style("caption1", 10, 13, 400, _sf(10)),
    _style("caption2", 10, 13, 500, _sf(10)),
]

PRESETS = {
    "material3": {
        "source": {"type": "https://m3.material.io/styles/typography/type-scale-tokens",
                   "color": "https://m3.material.io/styles/color/roles",
                   "shape": "https://m3.material.io/styles/shape/shape-scale-tokens",
                   "layout": "https://developer.android.com/develop/ui/compose/layouts/adaptive/use-window-size-classes"},
        "unit": "dp", "grid": 4, "spacing": _M3_SPACING, "radius": _M3_RADIUS, "type": _MATERIAL3_TYPE,
        "min_text": 11, "default_body": 16, "min_target": 48,
        "margins": {"compact": 16, "medium": 24, "expanded": 24, "verified": False},
        "breakpoints": [600, 840, 1200, 1600],
        "dark_base": "tone 6, never pure black",
    },
    "ios": {
        "source": dict(_APPLE_SOURCE),
        "unit": "pt", "grid": 4, "rhythm": 8, "spacing": _IOS_SPACING, "radius": _apple_radius(), "fixed_scale": False, "type": _IOS_TYPE,
        "min_text": 11, "default_body": 17, "min_target": 44,
        "margins": {"compact": 16, "regular": 20, "verified": False},
        "breakpoints": [],
        "dark_base": "pure black base, nearly black elevated",
    },
    "macos": {
        "source": dict(_APPLE_SOURCE),
        "unit": "pt", "grid": 4, "spacing": _MACOS_SPACING, "radius": _apple_radius(), "fixed_scale": False, "type": _MACOS_TYPE,
        "min_text": 10, "default_body": 13, "min_target": 28,
        "margins": {"default": 16, "verified": False},
        "breakpoints": [],
        "dark_base": None,
    },
    "fluent2": {
        "source": {"tokens": "https://github.com/microsoft/fluentui/tree/master/packages/tokens/src/global",
                   "type": "https://fluent2.microsoft.design/typography"},
        "unit": "px", "grid": 4, "spacing": _FLUENT_SPACING, "radius": dict(_FLUENT_RADIUS),
        "type": _fluent_type(), "min_text": 10, "default_body": 14, "min_target": 24,
        "margins": {"default": 16, "verified": False},
        "breakpoints": [],
        "dark_base": None,
    },
    "web": {
        "source": {"spacing": "https://carbondesignsystem.com/elements/spacing/overview/",
                   "type": "https://github.com/microsoft/fluentui/blob/master/packages/tokens/src/global/fonts.ts",
                   "contrast": "https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html",
                   "target": "https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html"},
        "unit": "px", "grid": 4, "rem_base": 16, "spacing": _CARBON_SPACING, "radius": dict(_FLUENT_RADIUS),
        "type": [t for t in _fluent_type() if t["size"] >= 12], "min_text": 12, "default_body": 16, "min_target": 24,
        "margins": {"default": 16, "verified": False},
        "breakpoints": [],
        "dark_base": None,
    },
}

PAIRS = [
    ("onPrimary", "primary", "text"),
    ("onPrimaryContainer", "primaryContainer", "text"),
    ("onSecondary", "secondary", "text"),
    ("onSecondaryContainer", "secondaryContainer", "text"),
    ("onTertiary", "tertiary", "text"),
    ("onTertiaryContainer", "tertiaryContainer", "text"),
    ("onError", "error", "text"),
    ("onErrorContainer", "errorContainer", "text"),
    ("onSurface", "surface", "text"),
    ("onSurfaceVariant", "surface", "text"),
    ("onSurface", "surfaceContainer", "text"),
    ("onSurface", "surfaceContainerHighest", "text"),
    ("onBackground", "background", "text"),
    ("primary", "surface", "text"),
    ("outline", "surface", "ui"),
]

_WCAG = 0.2126 + 0.7152 + 0.0722
_APPLE_PLATFORMS = ("ios", "macos")
_BODY_WORDS = ("body", "label", "callout", "footnote", "caption", "subhead")


def _lum(rgb):
    """WCAG relative luminance of an sRGB triple (coefficients normalised so white is exactly 1.0)."""
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in (c / 255.0 for c in rgb)]
    return (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]) / _WCAG


def contrast_ratio(fg, bg):
    """WCAG 2.x contrast ratio of fg on bg; a translucent fg is composited over bg first."""
    fhex, alpha = _norm_hex(fg)
    bhex, _ = _norm_hex(bg)
    f, b = _hexi(fhex), _hexi(bhex)
    if alpha < 1.0:
        f = [alpha * x + (1 - alpha) * y for x, y in zip(f, b)]
    hi, lo = sorted((_lum(f), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def is_large_text(size, weight, platform):
    """WCAG large text: 24px or 18.66px bold on web-family platforms; 18pt or any bold on Apple platforms."""
    if platform in _APPLE_PLATFORMS:
        return size >= 18 or weight >= 700
    return size >= 24 or (weight >= 700 and size >= 18.66)


def text_threshold(size, weight, platform):
    """Contrast needed for this text: 3.0 when it counts as large, else 4.5."""
    return 3.0 if is_large_text(size, weight, platform) else 4.5


def check_pairs(roles, platform, mode="light", size=14, weight=400):
    """Contrast failures for every role pair that has both sides present in roles."""
    out = []
    for fg, bg, kind in PAIRS:
        if fg not in roles or bg not in roles:
            continue
        ratio = contrast_ratio(roles[fg], roles[bg])
        need = 3.0 if kind == "ui" else text_threshold(size, weight, platform)
        if ratio < need:
            out.append({"fg": fg, "bg": bg, "mode": mode, "ratio": round(ratio, 2), "need": need, "kind": kind})
    return out


def lint_type(styles, platform):
    """Type-scale problems against a preset's minimum size, leading, tracking and distinct-size budget."""
    p = PRESETS[platform]
    out = []
    sizes = sorted({s["size"] for s in styles})
    for s in styles:
        name, size = s.get("name", "?"), s["size"]
        if size < p["min_text"]:
            out.append(f"{name}: {size:g} is below the {platform} minimum text size {p['min_text']}")
        if not size or size <= 0:
            out.append(f"{name}: size {size!r} is not a positive number")
            continue
        lh = s.get("line_height")
        if lh is not None:
            ratio = lh / size
            if not 1.0 <= ratio <= 1.8:
                out.append(f"{name}: line height {lh:g} on {size:g} is a {ratio:.2f} ratio, outside 1.0-1.8")
            elif any(w in name.lower() for w in _BODY_WORDS) and ratio < 1.2:
                out.append(f"{name}: body/label leading {ratio:.2f} is below 1.2")
        tracking = s.get("tracking_em")
        if tracking is not None:
            if not -0.05 <= tracking <= 0.12:
                out.append(f"{name}: tracking {tracking:+.3f}em is outside -0.05 to 0.12")
            if size >= 32 and tracking > 0.02:
                out.append(f"{name}: display tracking {tracking:+.3f}em is loose at {size:g}; keep it under 0.02")
    if len(sizes) > 8:
        out.append(f"{len(sizes)} distinct sizes ({', '.join('%g' % s for s in sizes)}); keep the scale to 8 or fewer")
    for i, a in enumerate(styles):
        for b in styles[i + 1:]:
            if (a.get("name") != b.get("name") and abs(a["size"] - b["size"]) < 1
                    and a.get("weight", 400) == b.get("weight", 400)):
                out.append(f"{a['name']} and {b['name']}: sizes {a['size']:g} and {b['size']:g} differ by under 1 "
                           f"at the same weight")
    return out


def snap(value, steps):
    """Nearest step and the signed error (value - step)."""
    if not steps:
        return value, 0.0
    step = min(steps, key=lambda s: (abs(value - s), s))
    return step, round(value - step, 6)


def snap_spacing(values, platform):
    """Snap spacing values to a preset's scale; values under 2 are left alone."""
    steps = PRESETS[platform]["spacing"]
    out = []
    for v in values:
        if v < 2:
            out.append({"value": v, "snapped": v, "error": 0.0})
        else:
            snapped, error = snap(v, steps)
            out.append({"value": v, "snapped": snapped, "error": error})
    return out


def _radius_items(platform):
    """Numeric radius tokens of a preset, ordered (the continuous flag is skipped)."""
    return [(k, v) for k, v in PRESETS[platform]["radius"].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def snap_radius(value, platform, height=None):
    """Nearest radius token name, value and error; a full pill is only found when the element height is given."""
    items = _radius_items(platform)
    if height is not None and value > 0 and value >= height / 2 - 1:
        return "full", 9999, 0.0
    finite = [(k, v) for k, v in items if k != "full"]
    name, step = min(finite, key=lambda kv: (abs(value - kv[1]), kv[1]))
    return name, step, round(value - step, 6)


def audit_spacing(values, platform, tol=1):
    """Spacing values off the preset grid by more than tol, plus a warning past 8 distinct values."""
    steps = PRESETS[platform]["spacing"]
    out = []
    for v in values:
        if v < 2:
            continue
        snapped, _ = snap(v, steps)
        if abs(v - snapped) > tol:
            out.append(f"{v:g} is {abs(v - snapped):g} off the {platform} spacing grid (nearest {snapped:g})")
    distinct = sorted(set(values))
    if len(distinct) > 8:
        out.append(f"{len(distinct)} distinct spacing values; keep to 8 or fewer")
    return out


def audit_radius(radii, platform):
    """Radius problems: more than 5 distinct non-zero values, or a value far from every preset step."""
    steps = [v for k, v in _radius_items(platform) if k != "full"]
    out = []
    non_zero = sorted({r for r in radii if r > 0})
    if len(non_zero) > 5:
        out.append(f"{len(non_zero)} distinct non-zero radii ({', '.join('%g' % r for r in non_zero)}); keep to 5")
    for r in radii:
        if r <= 0 or r >= 9999 or not PRESETS[platform].get("fixed_scale", True):
            continue
        gap = min(abs(r - s) for s in steps)
        if gap > 2:
            out.append(f"radius {r:g} is {gap:g} from every {platform} radius step")
    return out


def check_nested(outer_radius, padding, inner_radius, tol=2):
    """Concentric corners: the inner radius should be outer minus padding."""
    want = max(0.0, outer_radius - padding)
    if abs(want - inner_radius) <= tol:
        return None
    return (f"inner radius {inner_radius:g} breaks concentricity: expected {want:g} "
            f"(outer {outer_radius:g} - padding {padding:g}, tolerance {tol:g})")


def nearest_style(size, platform, line_height=None, weight=None):
    """Preset type style closest to a size (and optional leading and weight), with its distance."""
    best = None
    for s in PRESETS[platform]["type"]:
        d = abs(s["size"] - size)
        if line_height is not None:
            d += 0.25 * abs(s["line_height"] - line_height)
        if weight is not None:
            d += 0.5 * abs(s["weight"] - weight) / 100.0
        if best is None or d < best[0]:
            best = (d, s)
    out = dict(best[1])
    out["distance"] = round(best[0], 6)
    return out
