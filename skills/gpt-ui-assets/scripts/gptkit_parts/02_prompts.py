# gptkit part 02_prompts: prompts and JSON schemas.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---------------------------------------------------------------- prompts

MOCKUP_WRAP = ("High-fidelity production UI design render of a single app screen on a {aspect} canvas. {prompt} "
               "Flat, pixel-crisp screen render at 1:1, no device frame, no hands, no perspective, no watermark. "
               "Real, legible UI text. Consistent spacing grid and one cohesive icon style.")

# House style: the owner's standing art direction, added to every new design (mockup and --style-ref, not --edit).
# GPTKIT_HOUSE_STYLE (env or config file) replaces it; "off" drops it; --no-house-style skips it for one run.
HOUSE_STYLE = ("House style (unless the brief says otherwise): a clean, decluttered, modern UI. Generous whitespace, one "
               "clear focal point and hierarchy, only the elements this screen needs, one accent colour, no ornamental "
               "noise. Add illustration or artwork only where it carries meaning or warmth (a hero, an empty state, "
               "onboarding), in one consistent style. Show real product content only: no ads, ad banners, promos or "
               "placeholder marketing unless the brief asks for them.")


def _house_style(skip=False):
    text = None if skip else (_config().get("house_style") or HOUSE_STYLE)
    return "" if not text or text.strip().lower() in ("off", "none", "0", "false") else " " + text.strip()


PLATFORMS = {
    "android": {
        "label": "native Android app (Material 3 / Material 3 Expressive, Jetpack Compose)",
        "system_ui": "Android status bar at the top (time, signal, battery) and the gesture navigation handle at the bottom; edge-to-edge layout",
        "components": ("TopAppBar (small/center-aligned/medium/large), NavigationBar (3-5 destinations, pill active indicator), "
                       "NavigationRail (tablets), FloatingActionButton / ExtendedFAB, Cards (elevated/filled/outlined), "
                       "Buttons (filled/tonal/outlined/text), SegmentedButton, Chips (assist/filter/input/suggestion), "
                       "Switch, Checkbox, RadioButton, Slider, SearchBar, Tabs, ListItem, Badge, ModalBottomSheet, "
                       "Dialog, Snackbar, progress indicators"),
        "metrics": "4 dp baseline / 8 dp spacing grid, 48 dp minimum touch targets, 24 dp icons, "
                   "corner radii 4/8/12/16/28 dp (extra-small..extra-large)",
        "type": "Material 3 type scale: Display/Headline/Title/Body/Label in Large/Medium/Small (e.g. Headline Small 24 sp, "
                "Title Large 22 sp, Body Large 16 sp, Label Large 14 sp)",
        "icons": "Material Symbols (Outlined or Rounded, consistent weight), filled variant for the selected tab",
        "screen_dp": 412,
    },
    "ios": {
        "label": "native iOS app (Human Interface Guidelines, iOS 26 Liquid Glass, SwiftUI)",
        "system_ui": "iOS status bar with Dynamic Island at the top and the home indicator at the bottom; respect safe areas",
        "components": ("NavigationStack bar with large title, floating Liquid Glass TabBar (up to 5 tabs), Toolbar, "
                       "List (inset grouped), Section, sheet with detents, Menu, context menu, Alert, "
                       "confirmation dialog, Picker / segmented control, Toggle, Slider, Stepper, searchable field, "
                       "DatePicker, ProgressView, swipe actions"),
        "metrics": "44 pt minimum touch targets, 16-20 pt margins, 8 pt spacing rhythm, continuous-corner rounded rectangles",
        "type": "SF Pro with Dynamic Type styles: Large Title 34, Title 1 28, Title 2 22, Title 3 20, Headline 17 semibold, "
                "Body 17, Callout 16, Subheadline 15, Footnote 13, Caption 12/11",
        "icons": "SF Symbols (matching weight to adjacent text), filled variant for the selected tab",
        "screen_dp": 393,
    },
    "web": {
        "label": "responsive web app (semantic HTML/CSS, accessible)",
        "system_ui": "no device chrome; a browser viewport",
        "components": ("header with primary navigation, sidebar or top nav, breadcrumbs, buttons, links, form fields with "
                       "labels, selects, checkboxes, toggles, cards, tables, tabs, modals/dialogs, toasts, pagination"),
        "metrics": "8 px spacing grid, 44 px minimum touch targets on mobile, visible focus states, "
                   "breakpoints 390 / 768 / 1280 / 1440 px",
        "type": "a web font pairing with a clear rem-based type scale (e.g. 14/16/20/24/32/40 px)",
        "icons": "one consistent icon set (e.g. Material Symbols or Lucide)",
        "screen_dp": 390,
    },
}

PLATFORM_RULE = ("Target platform: {label}. Build the screen from this platform's native components wherever one fits: "
                 "{components}. Show the {system_ui}. Metrics: {metrics}. Typography: {type}. Icons: {icons}. "
                 "Anything custom must still look and behave native to this platform.")


def _platform_rule(platform):
    return " " + PLATFORM_RULE.format(**PLATFORMS[platform]) if platform else ""


def _design_meta(path):
    """design.json written by `mockup` next to a mockup (platform, screen width), if any."""
    for d in (Path(path).parent, Path(path).parent.parent):
        f = d / "design.json"
        if f.exists():
            try:
                meta = json.loads(f.read_text())
            except (OSError, ValueError) as e:
                print(f"  ! ignoring unreadable {f}: {e}")
                return {}
            meta = meta if isinstance(meta, dict) else {}
            if meta.get("platform") is not None and meta["platform"] not in PLATFORMS:
                print(f"  ! {f}: unknown platform {meta.pop('platform')!r} ignored (use {', '.join(PLATFORMS)})")
            return meta
    return {}


def _screen_dp(platform, given=None):
    return given or PLATFORMS.get(platform or "", {}).get("screen_dp", 390)


STYLE_REF_PROMPT = ("The attached image is an APPROVED screen of this app. Design a DIFFERENT screen of the same app on a "
                    "{aspect} canvas: {prompt} Use the attached screen ONLY as the style guide: the same design language "
                    "(palette, typefaces and weights, text colours, eyebrow/label treatment, corner radii, surfaces, "
                    "borders, glow, icon style, illustration style, spacing rhythm) and the same shared chrome (app name, "
                    "navigation bar with the same items and icons) where this screen has it. Do NOT copy its layout, "
                    "sections, cards or copy: compose a new layout for this screen's purpose, with new content. "
                    "Flat, pixel-crisp screen render at 1:1, no device frame. Real, legible UI text.")

CRITIQUE_PROMPT = """Do design QA on this app UI mockup ({W}x{H} px; the screen is {dp}dp wide, so 1dp = {k:.2f}px;
target platform: {platform}). It will be built in code. Report only real defects a developer or user would hit, most
severe first:
- text: every string that is garbled, misspelled, nonsense, truncated or oddly cased (quote it exactly).
- alignment/spacing: misaligned edges or baselines, uneven gaps between repeated items, inconsistent margins.
- icons: mixed icon styles (outline vs filled, stroke weight, corners), ambiguous or malformed glyphs.
- contrast/accessibility: text or icons below WCAG AA (4.5:1 body, 3:1 large text and icons), text illegibly small
  RELATIVE to the rest of the screen (not absolute sizes: the drawing scale is inexact and is calibrated later),
  controls that cannot get a {touch}dp touch target (a smaller visible pill or icon with room around it is fine).
- overlap: a floating element (FAB, snackbar, sheet, badge, mini player) covering text, numbers, list rows or
  artwork; elements colliding, or hidden under the navigation bar or system bars. High when it hides content.
- platform: {platform} guideline problems (navigation pattern, tab count, safe areas, standard controls), and system
  UI that belongs to another platform (e.g. an iOS "9:41" status bar or Dynamic Island on Android, Android status
  icons or a gesture pill on iOS).
- implementability: what code can't build cleanly (text baked into artwork, impossible overlaps, inconsistent radii,
  content cut off).{ref_rule}
Report each root cause ONCE: group repeated instances (e.g. several labels with the same size problem) into one
issue. Per issue: severity (high = broken, illegible or blocks the build; medium = visible defect; low = nit),
category, a bbox around the problem (or its first instance) in this image's pixels, the problem in one sentence, and
ONE edit instruction (<= 30 words) that fixes only that issue. No taste preferences. A clean design gets few or no
issues."""

CRITIQUE_REF_RULE = """
- consistency: the SECOND image is an approved screen of the same app. Flag differences in app name, navigation
  items/icons, typefaces, palette and component styling (bbox refers to the first image)."""

ISSUE_ITEM = {"type": "object", "additionalProperties": False,
              "required": ["severity", "category", "bbox", "problem", "edit"],
              "properties": {
                  "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                  "category": {"type": "string", "enum": ["text", "alignment", "icons", "contrast", "accessibility",
                                                          "overlap", "platform", "implementability", "consistency"]},
                  "bbox": {"type": "object", "additionalProperties": False, "required": ["x", "y", "w", "h"],
                           "properties": {k: {"type": "integer"} for k in "xywh"}},
                  "problem": {"type": "string"}, "edit": {"type": "string"}}}

CRITIQUE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["issues"],
                   "properties": {"issues": {"type": "array", "items": ISSUE_ITEM}}}

VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["checks", "new_issues"],
                 "properties": {
                     "checks": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                           "required": ["id", "fixed"],
                                                           "properties": {"id": {"type": "integer"},
                                                                          "fixed": {"type": "boolean"}}}},
                     "new_issues": {"type": "array", "items": ISSUE_ITEM}}}

SEVERITY = {"high": 3, "medium": 2, "low": 1}


ASSET_PROMPT = ("The attached image is a zoomed, low-resolution screenshot region from an APPROVED app design. "
                "Recreate ONLY the {kind} \"{name}\" ({description}) from scratch as a clean, high-resolution "
                "standalone production asset that matches the design 1:1: identical silhouette, proportions, "
                "number of layers/parts, colour values{colors}, gradient directions, highlights and glow. "
                "The target is the element in the CENTRE of the attached image (about the middle 45% of its width and "
                "height); everything around it is context only. Do NOT upscale, trace or copy the screenshot pixels. "
                "Do NOT include the backdrop, container shape, text, or any neighbouring element. No watermark, "
                "signature or drop shadow unless the design has one. {bg}")

BACKGROUND_PROMPT = ("The attached image is a cropped region of an APPROVED app UI design. Recreate ONLY the {kind} "
                     "\"{name}\" ({description}) as a standalone full-bleed {aspect}, from scratch at full resolution, "
                     "with exactly the framing of the attached crop. REMOVE every UI element drawn on top of it: all text, "
                     "buttons, chips, icons, badges, rounded corners and borders. Where UI covered it, continue the "
                     "artwork naturally. Keep the exact composition, colours and lighting of the design.")

# per-candidate emphasis (c1, c2, c3, then repeating) so best-of-N samples different failure modes
CANDIDATE_FOCUS = ["",
                   " PRIORITY: the exact colour values, saturation, lightness, gradient directions and glow strength as "
                   "in the reference; do not brighten, saturate, warm, cool or shift any hue.",
                   " PRIORITY: the exact silhouette, proportions and relative sizes of every part as in the reference "
                   "(how big each part is relative to the others, stroke/ring thickness, spacing); do not enlarge, "
                   "simplify or add parts."]


def _aspect_words(w, h):
    r = w / h
    if r > 1.2:
        return f"wide landscape image (about {r:.1f}:1)"
    if r < 1 / 1.2:
        return f"tall portrait image (about 1:{1 / r:.1f})"
    return "square image"

MANIFEST_PROMPT = """This is an APPROVED mobile/web UI design ({W}x{H} px). Inventory every visual ASSET a developer
needs as a separate file to build it in code: icons, illustrations, images/photos, logos, decorative graphics, and
artwork that sits behind UI (kind "background"). Rules:
- Exclude plain text, solid/gradient fills, simple shapes, dividers, shadows and standard containers code can draw.
- Exclude system UI the OS draws: status bar (time, wifi, signal, battery, notification icons), Dynamic Island,
  home indicator, gesture handle, system navigation buttons.
- One entry per distinct graphic. Identical repeats (same icon used twice) appear once.
- bbox is TIGHT around the graphic itself in this image's pixels (exclude any circle/card behind an icon unless that
  backdrop is part of the artwork).
- format "svg" for flat/line/monochrome/simple-gradient glyphs (tab bar, toolbar, chevrons, UI icons); "png" for
  painterly, textured, glowing, 3D, photographic or detailed illustrations.
- transparent=false only for full-bleed artwork/photos (kind background/photo).
- animated=true if it is a natural candidate for motion (flame, loader, mascot, weather, pulsing element).
- name: short snake_case noun for the graphic itself (no file extension, no format/kind suffix).
- description: precise enough to redraw it from scratch — shapes, parts, layer order, colours, gradients, stroke weight.
- colors: main hex colours sampled from the design.
- material_symbol / sf_symbol: for an icon that is a standard glyph, the closest Google Material Symbols name
  (snake_case, e.g. home, settings, search, favorite) and Apple SF Symbols name (e.g. house.fill, gearshape,
  magnifyingglass, heart); "" for custom glyphs and for every non-icon."""

SVG_RULES = """Write each asset as a standalone SVG that matches the design 1:1 (shape, proportions, stroke weight,
corner rounding, colours). Hard rules so it converts to Android VectorDrawable and normalises onto a 24 grid:
- Root: <svg xmlns="http://www.w3.org/2000/svg" viewBox="..."> (the xmlns is required).
- Elements allowed: path, circle, ellipse, rect, line, polyline, polygon, g, defs, linearGradient, radialGradient,
  stop. No text, image, filter, mask, clipPath, use, symbol, style, CSS classes, marker, pattern, nested svg.
- No transform attribute on ANY element (bake positions into the coordinates), no gradientTransform.
- Plain numbers only: no % and no units (px, pt, em) in coordinates, sizes, stroke widths or opacities.
- Colours as hex. Gradients use gradientUnits="userSpaceOnUse" with explicit numeric coordinates: every
  linearGradient has x1 y1 x2 y2, every radialGradient has cx cy r (no fx/fy), and each has at least two stops.
- No stroke-dasharray, vector-effect, paint-order or marker-start/mid/end (VectorDrawable cannot express them).
- Square viewBox (e.g. "0 0 24 24") for icons unless the graphic is clearly non-square.
- Keep stroke-based icons stroke-based (stroke, stroke-width, stroke-linecap, stroke-linejoin); fill="none" on them."""

VECTOR_PROMPT = """Vectorise this asset from an APPROVED design for a looping vector animation (Lottie and Android
AnimatedVectorDrawable). The MOVING part is: {moving}. The STATIC part is: {static}. Motion: {motion}, one loop is
{seconds}s. {strength} Rules:
- Every path "d" uses ONLY absolute M, L, C and Z commands (no H/V/S/Q/A, no relative commands).
- Shape change (flame, liquid, cloth, smoke): an animated layer with {k} keyframes that all have the IDENTICAL command
  sequence (same count and order of M/L/C/Z) and morph into each other. The loop runs back from the last keyframe to
  the first, so never repeat the first keyframe at the end.
- Rigid motion (bounce, sway, tilt, spin, pulse, pop, fade, expanding rings): the layer's "motion". anchor_x/anchor_y
  is the pivot in viewBox units. keys: t (0..1 of the loop, first key t=0), dx/dy offset, sx/sy scale factor (1 = rest),
  rotate (degrees, clockwise), opacity (multiplier 0..1), ease (curve towards the next key). If the last key is before
  t=1 the loop eases back to the first key. keys: [] means the layer does not move. A layer can morph AND move.
- glows: soft radial halos (centre, radii, colour, peak opacity) drawn as a gradient fading to transparent. Add one for
  every glow or light in the design (behind a flame, around a lamp) and animate its opacity/scale; otherwise [].
- Several copies of a moving part (ripples, sparks, notes) are separate layers with time-shifted keys.
- fill is a colour, url(#id) of a gradient in "defs" (gradientUnits="userSpaceOnUse", explicit coordinates in viewBox
  units, at least 2 stops, no gradientTransform) or "none"; stroke is a colour or "none", stroke_width in viewBox units
  (0 without stroke). Colours are #RRGGBB only; transparency goes in opacity (0..1), never in the colour.
- Names: lowercase snake_case, unique across layers and glows. Coordinates: at most 1 decimal.
- Draw only the asset itself: no background, backdrop, card, plate or shadow behind it.
- Match the design's silhouette, proportions and colours. layer_order lists every layer and glow, bottom to top."""

MOTION_STRENGTH = {
    "subtle": "Keep the motion calm: moving parts shift or deform by about 2-4% of their size.",
    "normal": "The motion must read clearly at icon size: moving parts shift, deform or scale by about 6-12% of their size.",
    "strong": "Make the motion pronounced and lively: moving parts shift, deform or scale by 12-25% of their size "
              "(flame tips sway widely, pulses scale by 20%).",
}
MOTION_MIN = {"subtle": 0.015, "normal": 0.04, "strong": 0.08}

EASES = {"linear": (0, 0, 1, 1), "ease_in_out": (0.42, 0, 0.58, 1), "ease_out": (0, 0, 0.58, 1),
         "ease_in": (0.42, 0, 1, 1), "overshoot": (0.34, 1.56, 0.64, 1)}

_NUMBER_T = {"type": "number"}
_UNIT_T = {"type": "number", "minimum": 0, "maximum": 1}
_POS_T = {"type": "number", "minimum": 0}
# pattern / minimum / maximum / minItems are part of OpenAI strict structured outputs; a gateway that rejects them
# answers 400 on text.format, which ask() (g1) retries without strict mode, and _canon_vec/_validate_vec re-check here.
_NAME_T = {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}
_COLOR_T = {"type": "string", "pattern": "^(#[0-9A-Fa-f]{6}|url\\(#[A-Za-z_][\\w-]*\\)|none)$"}
_HEX_T = {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"}
_MOTION = {"type": "object", "additionalProperties": False, "required": ["anchor_x", "anchor_y", "keys"],
           "properties": {"anchor_x": _NUMBER_T, "anchor_y": _NUMBER_T, "keys": {"type": "array", "items": {
               "type": "object", "additionalProperties": False,
               "required": ["t", "dx", "dy", "sx", "sy", "rotate", "opacity", "ease"],
               "properties": {**{k: _NUMBER_T for k in ("dx", "dy", "sx", "sy", "rotate")}, "t": _UNIT_T, "opacity": _UNIT_T,
                              "ease": {"type": "string", "enum": list(EASES)}}}}}}
_PAINT = {"fill": _COLOR_T, "opacity": _UNIT_T, "stroke": _COLOR_T, "stroke_width": _POS_T,
          "fill_rule": {"type": "string", "enum": ["nonzero", "evenodd"]},
          "cap": {"type": "string", "enum": ["butt", "round", "square"]}, "join": {"type": "string", "enum": ["miter", "round", "bevel"]}}

VECTOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["viewBox", "defs", "static_layers", "animated_layers", "glows", "layer_order"],
    "properties": {
        "viewBox": {"type": "string", "pattern": "^-?[0-9.]+ -?[0-9.]+ [0-9.]+ [0-9.]+$"}, "defs": {"type": "string"},
        "layer_order": {"type": "array", "items": _NAME_T},
        "static_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "paths", "motion"],
            "properties": {"name": _NAME_T, "motion": _MOTION, "paths": {"type": "array", "minItems": 1, "items": {
                "type": "object", "additionalProperties": False, "required": ["d", *_PAINT],
                "properties": {"d": {"type": "string"}, **_PAINT}}}}}},
        "animated_layers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", *_PAINT, "keyframes", "motion"],
            "properties": {"name": _NAME_T, **_PAINT, "motion": _MOTION,
                           "keyframes": {"type": "array", "minItems": 2, "items": {"type": "string"}}}}},
        "glows": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "cx", "cy", "rx", "ry", "color", "opacity", "motion"],
            "properties": {"name": _NAME_T, "cx": _NUMBER_T, "cy": _NUMBER_T, "rx": {"type": "number", "exclusiveMinimum": 0},
                           "ry": {"type": "number", "exclusiveMinimum": 0}, "opacity": _UNIT_T, "color": _HEX_T, "motion": _MOTION}}},
    },
}

MANIFEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["assets"],
    "properties": {"assets": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["name", "kind", "bbox", "format", "transparent", "animated", "description", "colors",
                     "material_symbol", "sf_symbol"],
        "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["icon", "illustration", "photo", "logo", "decoration", "background"]},
            "bbox": {"type": "object", "additionalProperties": False, "required": ["x", "y", "w", "h"],
                     "properties": {k: {"type": "integer"} for k in "xywh"}},
            "format": {"type": "string", "enum": ["svg", "png"]},
            "transparent": {"type": "boolean"}, "animated": {"type": "boolean"},
            "description": {"type": "string"}, "colors": {"type": "array", "items": {"type": "string"}},
            "material_symbol": {"type": "string"}, "sf_symbol": {"type": "string"}}}}},
}

SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["palette", "typography", "radii", "spacing", "components"],
    "properties": {
        "palette": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                               "required": ["name", "hex", "usage"],
                                               "properties": {"name": {"type": "string"}, "hex": {"type": "string"},
                                                              "usage": {"type": "string"}}}},
        "typography": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["role", "sample", "family_guess", "size_px", "weight", "letter_spacing_em", "color", "platform_style"],
                                                  "properties": {"role": {"type": "string"}, "sample": {"type": "string"},
                                                                 "platform_style": {"type": "string"},
                                                                 "family_guess": {"type": "string"}, "size_px": {"type": "number"},
                                                                 "weight": {"type": "integer"}, "letter_spacing_em": {"type": "number"},
                                                                 "color": {"type": "string"}}}},
        "radii": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["element", "px"],
                                             "properties": {"element": {"type": "string"}, "px": {"type": "number"}}}},
        "spacing": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["between", "px"],
                                               "properties": {"between": {"type": "string"}, "px": {"type": "number"}}}},
        "components": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["name", "bbox", "native", "notes"],
                                                  "properties": {"name": {"type": "string"}, "notes": {"type": "string"},
                                                                 "native": {"type": "string"},
                                                                 "bbox": {"type": "object", "additionalProperties": False,
                                                                          "required": ["x", "y", "w", "h"],
                                                                          "properties": {k: {"type": "integer"} for k in "xywh"}}}}},
    },
}
