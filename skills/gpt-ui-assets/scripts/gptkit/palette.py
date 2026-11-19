"""CIELAB tonal palettes and Material-style colour schemes."""
import math

from .measure import _hexi, _lab_arr, _norm_hex, _to_hex


TONES = [0, 4, 6, 10, 12, 17, 20, 22, 24, 30, 40, 50, 60, 70, 80, 87, 90, 92, 94, 95, 96, 98, 99, 100]

# https://android.googlesource.com/platform/frameworks/support/+/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/
ROLE_TONES = {
    "primary": ("primary", 40, 80), "onPrimary": ("primary", 100, 20),
    "primaryContainer": ("primary", 90, 30), "onPrimaryContainer": ("primary", 10, 90),
    "secondary": ("secondary", 40, 80), "onSecondary": ("secondary", 100, 20),
    "secondaryContainer": ("secondary", 90, 30), "onSecondaryContainer": ("secondary", 10, 90),
    "tertiary": ("tertiary", 40, 80), "onTertiary": ("tertiary", 100, 20),
    "tertiaryContainer": ("tertiary", 90, 30), "onTertiaryContainer": ("tertiary", 10, 90),
    "error": ("error", 40, 80), "onError": ("error", 100, 20),
    "errorContainer": ("error", 90, 30), "onErrorContainer": ("error", 10, 90),
    "background": ("neutral", 98, 6), "onBackground": ("neutral", 10, 90),
    "surface": ("neutral", 98, 6), "onSurface": ("neutral", 10, 90),
    "surfaceVariant": ("neutralVariant", 90, 30), "onSurfaceVariant": ("neutralVariant", 30, 80),
    "surfaceDim": ("neutral", 87, 6), "surfaceBright": ("neutral", 98, 24),
    "surfaceContainerLowest": ("neutral", 100, 4), "surfaceContainerLow": ("neutral", 96, 10),
    "surfaceContainer": ("neutral", 94, 12), "surfaceContainerHigh": ("neutral", 92, 17),
    "surfaceContainerHighest": ("neutral", 90, 22), "outline": ("neutralVariant", 50, 60),
    "outlineVariant": ("neutralVariant", 80, 30), "inverseSurface": ("neutral", 20, 90),
    "inverseOnSurface": ("neutral", 95, 20), "inversePrimary": ("primary", 80, 40),
    "scrim": ("neutral", 0, 0), "shadow": ("neutral", 0, 0),
}


def lch(hex):
    """Return CIELAB lightness, chroma and hue in degrees for a colour."""
    lightness, a, b = (float(v) for v in _lab_arr(_hexi(hex)))
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _rgb(lightness, chroma, hue):
    angle = math.radians(hue)
    fy = (lightness + 16) / 116
    fx, fz = fy + chroma * math.cos(angle) / 500, fy - chroma * math.sin(angle) / 200
    def inverse(t):
        return t ** 3 if t ** 3 > 216 / 24389 else (116 * t - 16) * 27 / 24389
    x, y, z = 0.95047 * inverse(fx), inverse(fy), 1.08883 * inverse(fz)
    linear = (3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
              -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
              0.0556434 * x - 0.2040259 * y + 1.0572252 * z)
    return tuple(12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055 for v in linear)


def _tone(lightness, chroma, hue):
    if not 0 <= lightness <= 100 or not math.isfinite(chroma) or chroma < 0:
        raise ValueError("tone must be in 0..100 and chroma must be nonnegative")
    if lightness == 0:
        return "#000000"
    if lightness == 100:
        return "#FFFFFF"
    rgb = _rgb(lightness, chroma, hue)
    if not all(0 <= v <= 1 for v in rgb):
        low, high = 0.0, chroma
        for _ in range(12):
            mid = (low + high) / 2
            if all(0 <= v <= 1 for v in _rgb(lightness, mid, hue)):
                low = mid
            else:
                high = mid
            if high - low <= 0.5:
                break
        rgb = _rgb(lightness, low, hue)
    return _to_hex([v * 255 for v in rgb])


def tone_hex(hex, tone, chroma=None):
    """Set CIELAB lightness while preserving hue and fitting chroma to sRGB."""
    _, original, hue = lch(hex)
    return _tone(tone, original if chroma is None else chroma, hue)


def tonal_palette(hex, chroma=None):
    """Return the standard tonal stops for a seed colour."""
    _, original, hue = lch(hex)
    return {t: _tone(t, original if chroma is None else chroma, hue) for t in TONES}


def seed_palettes(primary, secondary=None, tertiary=None, error="#B3261E"):
    """Create primary, supporting, error and neutral tonal palettes."""
    _, chroma, hue = lch(primary)
    make = lambda c, h: {t: _tone(t, c, h) for t in TONES}
    return {"primary": tonal_palette(primary),
            "secondary": tonal_palette(secondary) if secondary is not None else make(min(chroma, 16), hue),
            "tertiary": tonal_palette(tertiary) if tertiary is not None else make(min(chroma, 24), hue + 60),
            "error": tonal_palette(error), "neutral": make(4, hue), "neutralVariant": make(8, hue)}


def contrast(a, b):
    """Unrounded WCAG contrast ratio; a translucent foreground is composited over the background first."""
    from .system import contrast_ratio  # deferred: keeps palette importable on its own
    return contrast_ratio(a, b)


def _scheme(primary, mode, secondary, tertiary, error, measured):
    if mode not in ("light", "dark"):
        raise ValueError("mode must be 'light' or 'dark'")
    palettes = seed_palettes(primary, secondary, tertiary, error)
    index = 1 if mode == "light" else 2
    result = {role: palettes[spec[0]][spec[index]] for role, spec in ROLE_TONES.items()}
    measured = measured or {}
    for role, colour in measured.items():
        if role not in ROLE_TONES:
            raise ValueError("unknown measured role: " + role)
        _norm_hex(colour)
        result[role] = colour
    notes = []
    anchor = next((r for r in ("surface", "background") if r in measured), None)
    touched = set(measured)
    if anchor:
        base = ROLE_TONES["surface"][index]
        shift = lch(measured[anchor])[0] - base
        _, _, hue = lch(primary)
        for role in ("surface", "background", "surfaceDim", "surfaceBright", "surfaceContainerLowest", "surfaceContainerLow",
                     "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest"):
            if role not in measured:
                result[role] = _tone(min(100, max(0, ROLE_TONES[role][index] + shift)), 4, hue)
                touched.add(role)
        if abs(shift) > 2:
            notes.append(f"{mode}: surface family follows the measured {anchor} (tone shift {shift:+.0f})")
    surfaces = ["surface", "surfaceDim", "surfaceBright", "surfaceContainerLowest", "surfaceContainerLow",
                "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest"]
    backgrounds = {"onPrimary": ["primary"], "onPrimaryContainer": ["primaryContainer"],
                   "onSecondary": ["secondary"], "onSecondaryContainer": ["secondaryContainer"],
                   "onTertiary": ["tertiary"], "onTertiaryContainer": ["tertiaryContainer"],
                   "onError": ["error"], "onErrorContainer": ["errorContainer"], "onBackground": ["background"],
                   "onSurface": surfaces, "onSurfaceVariant": surfaces + ["surfaceVariant"],
                   "inverseOnSurface": ["inverseSurface"]}
    for foreground, bgs in backgrounds.items():
        hit = [b for b in bgs if b in touched]
        if not hit or foreground in measured:
            continue
        palette, light, dark = ROLE_TONES[foreground]
        start = light if mode == "light" else dark
        direction = -1 if start < lch(result[hit[0]])[0] else 1
        seed = {"primary": primary, "secondary": secondary, "tertiary": tertiary, "error": error}.get(palette)
        if seed is None:
            _, c, h = lch(primary)
            c, h = (min(c, 16), h) if palette == "secondary" else ((min(c, 24), h + 60) if palette == "tertiary" else (4 if palette == "neutral" else 8, h))
        else:
            _, c, h = lch(seed)
        candidates = list(range(start, 101 if direction > 0 else -1, direction))
        candidates += list(range(0 if direction > 0 else 100, start, direction))
        for tone in candidates:
            candidate = _tone(tone, c, h)
            if min(contrast(candidate, result[b]) for b in bgs) >= 4.5:
                result[foreground] = candidate
                if tone != start:
                    notes.append(f"{mode}: adjusted {foreground} from tone {start} to {tone} for {', '.join(hit)}")
                break
        else:
            notes.append(f"{mode}: no {foreground} tone reaches 4.5:1 on every surface it sits on ({', '.join(bgs)})")
    return result, notes


def scheme(primary, mode, secondary=None, tertiary=None, error="#B3261E", measured=None):
    """Build one colour scheme, preserving measured roles for that mode."""
    return _scheme(primary, mode, secondary, tertiary, error, measured)[0]


def schemes(primary, measured_mode, measured=None, **kw):
    """Build both modes and report derived mode and contrast adjustments."""
    if measured_mode not in ("light", "dark"):
        raise ValueError("measured_mode must be 'light' or 'dark'")
    if set(kw) - {"secondary", "tertiary", "error"}:
        raise TypeError("unexpected scheme arguments: " + ", ".join(sorted(set(kw) - {"secondary", "tertiary", "error"})))
    result, notes = {}, []
    for mode in ("light", "dark"):
        result[mode], adjustments = _scheme(primary, mode, kw.get("secondary"), kw.get("tertiary"), kw.get("error", "#B3261E"), measured if mode == measured_mode else None)
        notes.extend(adjustments)
    derived = "dark" if measured_mode == "light" else "light"
    result["notes"] = [f"{derived} mode derived from the seed palettes"] + notes
    return result
