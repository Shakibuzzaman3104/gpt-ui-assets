"""Prompts and JSON schemas."""
import json
from pathlib import Path

from .core import _config

# ---------------------------------------------------------------- prompts

MOCKUP_WRAP = ("High-fidelity production UI design render of a single {surface} on a {aspect} canvas. {prompt} "
               "Flat, pixel-crisp screen render at 1:1, no device frame, no hands, no perspective, no watermark. "
               "Real, legible UI text. Consistent spacing grid and one cohesive icon style.")

# House style: the owner's standing art direction, added to every new design (mockup and --style-ref, not --edit).
# GPTKIT_HOUSE_STYLE (env or config file) replaces it; "off" drops it; --no-house-style skips it for one run.
HOUSE_STYLE = ("House style (unless the brief says otherwise): a clean, decluttered, modern UI. Generous whitespace, one "
               "clear focal point and hierarchy, only the elements this screen needs, one accent colour, no ornamental "
               "noise. Add illustration or artwork only where it carries meaning or warmth (a hero, an empty state, "
               "onboarding), in one consistent style. Show real product content only: no ads, ad banners, promos or "
               "placeholder marketing unless the brief asks for them. If the brief names no product, invent an original "
               "product name; never use a real company's or app's name, logo or brand mark.")


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
    "macos": {"label": "native macOS 26 app (SwiftUI / AppKit)", "system_ui": "macOS unified app toolbar and sidebar",
              "components": "NavigationSplitView, sidebar, toolbar, inspector, popover, native controls",
              "metrics": "8 pt spacing rhythm, 28 pt minimum controls", "type": "SF Pro 13 pt body and native text styles",
              "icons": "SF Symbols", "screen_dp": 1280},
    "windows": {"label": "native Windows 11 Fluent 2 app (WinUI 3)", "system_ui": "Windows title bar and caption buttons",
                "components": "NavigationView, CommandBar, InfoBar, teaching tip, native controls",
                "metrics": "4 px base spacing, 32 epx minimum controls", "type": "Segoe UI Variable 14 px body",
                "icons": "Fluent System Icons", "screen_dp": 1280},
    "linux": {"label": "native GNOME/libadwaita desktop app", "system_ui": "GNOME header bar and window controls",
              "components": "AdwHeaderBar, AdwNavigationView, sidebar, dialogs, native controls",
              "metrics": "8 px spacing rhythm, 32 px minimum controls", "type": "Cantarell or Inter 14 px body",
              "icons": "GNOME symbolic icons", "screen_dp": 1280},
}

def _device(canvas, logical_w, logical_h, touch, hinge, aspect, system_ui, components, metrics, rule, checks):
    return {"canvas": canvas, "logical_w": logical_w, "logical_h": logical_h, "touch": touch,
            "hinge": hinge, "aspect": aspect, "system_ui": system_ui, "components": components,
            "metrics": metrics, "rule": rule, "checks": checks}


_TABLET_CHECKS = "not a stretched phone or a single full-width column over about 840 dp; avoid lines over 75 characters and giant edge-to-edge buttons; navigation fits (rail/sidebar/iPad top tab bar rather than a phone bottom bar at expanded width); panes are sensible; dialogs centre or float; device chrome is correct"
_DESKTOP_CHECKS = "correct OS chrome (macOS traffic lights top-left; Windows/Linux caption buttons top-right); no mobile status/home/tab chrome; pointer-sized controls; hover and focus states; sidebar, toolbar and content structure"
_WEB_CHECKS = "above-the-fold hierarchy; centred max-width container and readable line length; breakpoint-appropriate navigation; no device/browser chrome; visible focus/hover; labelled forms"
DEVICES = {
    ("android", "phone", "default"): _device("1152x2496", 412, 915, 48, None,
        "tall 9:19.5 phone canvas", "Android status bar and gesture navigation handle", "NavigationBar, TopAppBar, cards, sheets",
        "4/8 dp grid, 48 dp touch targets", "Form factor: Android phone, compact width; use a single-column phone layout and bottom navigation.", "phone layout and Android system UI"),
    ("ios", "phone", "default"): _device("1152x2496", 393, 852, 44, None, "tall 9:19.5 phone canvas",
        "iOS status bar with Dynamic Island and home indicator", "NavigationStack, floating Liquid Glass TabBar, sheets",
        "8 pt rhythm, 44 pt touch targets", "Form factor: iPhone full-screen app with safe areas and native bottom tab bar.", "phone layout, safe areas and iPhone system UI"),
    ("web", "phone", "app"): _device("1152x2496", 390, 844, 44, None, "tall 9:19.5 phone canvas",
        "no device or browser chrome", "responsive web navigation, cards, labelled forms", "44 px touch targets, visible focus, 8 px grid",
        "Form factor: responsive web app at a 390x844 mobile breakpoint; hamburger or bottom navigation and 44 px touch targets.", _WEB_CHECKS),
    ("web", "phone", "landing"): _device("1152x2496", 390, 844, 44, None, "tall 9:19.5 phone canvas",
        "no device or browser chrome", "responsive web navigation, hero, CTA", "44 px touch targets, visible focus, 8 px grid",
        "Form factor: responsive landing page at a 390x844 mobile breakpoint; compact nav, stacked hero and visible primary CTA.", _WEB_CHECKS),
    ("android", "tablet", "landscape"): _device("2304x1440", 1280, 800, 48, None, "wide landscape canvas about 1.6:1",
        "Android status bar and edge-to-edge app UI", "NavigationSuiteScaffold, WideNavigationRail (collapsed 96 dp / expanded 220-360 dp), ListDetailPaneScaffold, SupportingPaneScaffold (70/30), LazyVerticalGrid(Adaptive)",
        "4/8 dp grid, 48 dp touch targets", "Form factor: Android tablet in landscape, a 1280x800 dp window (Expanded width), wide landscape canvas about 1.6:1. Use a NavigationRail on the left (96 dp, icons with labels) and a two-pane list-detail layout (ListDetailPaneScaffold): a 360 dp list pane and a flexible detail pane, 24 dp gap, 24 dp outer margins. Do NOT draw an enlarged phone layout or one stretched column.", _TABLET_CHECKS),
    ("android", "tablet", "portrait"): _device("1440x2304", 800, 1280, 48, None, "tall portrait tablet canvas",
        "Android status bar and edge-to-edge app UI", "NavigationSuiteScaffold, NavigationRail or NavigationBar, ListDetailPaneScaffold, LazyVerticalGrid(Adaptive)",
        "4/8 dp grid, 48 dp touch targets", "Form factor: Android tablet portrait, 800x1280 dp (Medium width). Use a rail or bottom bar and a single pane or 50/50 split; retain tablet margins and do not stretch a phone column.", _TABLET_CHECKS),
    ("ios", "tablet", "landscape"): _device("2320x1600", 1210, 834, 44, None, "wide iPad landscape canvas about 1.45:1",
        "iPad status bar with time/date top left and battery top right, home indicator bottom centre; no Dynamic Island or notch",
        "TabView .sidebarAdaptable or NavigationSplitView, split sidebar/list/detail, popovers, pointer hover",
        "44 pt touch targets, 16-20 pt margins", "Form factor: iPad (iPadOS 26) full-screen app in landscape, 1210x834 pt. A sidebar on the left (TabView .sidebarAdaptable / NavigationSplitView) or the floating tab bar at top centre, and a split view sidebar | list | detail. iPad status bar with time and date at top left and battery at top right, NO Dynamic Island or notch; home indicator at bottom centre. Popovers instead of full-screen sheets, pointer hover effects.", _TABLET_CHECKS + "; no Dynamic Island"),
    ("ios", "tablet", "portrait"): _device("1600x2320", 834, 1210, 44, None, "tall iPad portrait canvas",
        "iPad status bar with time/date top left and battery top right, home indicator bottom centre; no Dynamic Island or notch",
        "TabView .sidebarAdaptable or NavigationSplitView, split view, popovers", "44 pt touch targets, 16-20 pt margins",
        "Form factor: iPad (iPadOS 26) full-screen app in portrait, 834x1210 pt. Use a sidebar or top floating tab bar with a useful single or split pane. Status bar time/date top left, battery top right; NO Dynamic Island or notch; home indicator bottom centre. Prefer popovers and show pointer hover affordances.", _TABLET_CHECKS + "; no Dynamic Island"),
    ("android", "foldable", "flat"): _device("1840x1904", 852, 883, 48, ("x", 0.5), "nearly square, slightly taller than wide",
        "Android status bar and gesture handle; no device frame", "NavigationSuiteScaffold, WideNavigationRail, ListDetailPaneScaffold, LazyVerticalGrid(Adaptive)",
        "48 dp touch targets", "Form factor: book-style foldable (Pixel 9 Pro Fold class) fully unfolded, a nearly square 852x883 dp inner display, slightly taller than wide. A vertical fold runs exactly down the centre (x = 50%): treat it as a gutter, two panes, one on each side, no text, button, number or face crossing the centre line. Do not draw the hinge, crease or device.", "report any critical element crossing the vertical hinge band (+-16 dp) as high severity with a bbox; panes split at hinge or avoid it; use near-square shape"),
    ("android", "foldable", "tabletop"): _device("1904x1840", 883, 852, 48, ("y", 0.5), "nearly square tabletop canvas",
        "Android status bar; no device frame", "ShortNavigationBarMedium, SupportingPaneScaffold",
        "48 dp touch targets", "Form factor: book-style foldable half-opened in tabletop posture, 883x852 dp. The horizontal fold is exactly at y = 50%: content or media in the top half, controls in the bottom half, nothing across the fold. Do not draw the device or crease.", "report any critical element crossing the horizontal hinge band (+-16 dp) as high severity with a bbox; content/media top and controls bottom; use near-square shape"),
    ("android", "foldable", "folded"): _device("1104x2480", 443, 994, 48, None, "tall cover-screen canvas",
        "Android status bar and gesture handle", "phone TopAppBar, NavigationBar, compact single-column layout", "48 dp touch targets",
        "Form factor: folded book-style phone using its 443x994 dp cover screen; compose a phone-like single-column layout.", "cover screen uses phone layout and Android chrome"),
}

_DESKTOP_RULES = {
 "macos": "Form factor: A macOS 26 app window, 1280x800 pt, filling the canvas edge to edge: no wallpaper, Dock, menu bar or drop shadow. Three traffic-light buttons at top left inside a unified toolbar with the title and SF Symbol buttons, a translucent source-list sidebar about 220 pt wide, a content area, an optional inspector. SF Pro 13 pt body, 28 pt controls, pointer density, one hover row and one keyboard focus ring visible.",
 "windows": "Form factor: A Windows 11 Fluent 2 (WinUI 3) app window, 1280x800 epx, filling the canvas. A 48 px Mica title bar with a 16 px app icon and title at the left, a centred search box, minimize/maximize/close at the top right. A NavigationView left pane (expanded, about 320 px, icons with labels, Settings at the bottom). Segoe UI Variable 14 px body, 4 px control corners, 8 px window/dialog corners.",
 "linux": "Form factor: A GNOME/libadwaita app window, 1280x800 px, filling the canvas. Header bar with main actions at the start, title or view switcher centred, menu at the end, window controls at the right; Adwaita sidebar. Cantarell/Inter 14 px body and native pointer controls.",
}
for _os in ("macos", "windows", "linux"):
    DEVICES[(_os, "desktop", "default")] = _device("2304x1440", 1280, 800,
        28 if _os == "macos" else 32, None, "wide landscape canvas about 1.6:1",
        "OS-native window title bar, toolbar and sidebar", PLATFORMS[_os]["components"], PLATFORMS[_os]["metrics"],
        _DESKTOP_RULES[_os], _DESKTOP_CHECKS)
DEVICES[("android", "desktop", "default")] = _device("2304x1440", 1280, 800, 48, None,
    "wide landscape canvas about 1.6:1", "Android desktop window chrome without phone status/navigation bars", "NavigationRail, list-detail panes, toolbar",
    "48 dp touch targets", "Form factor: Android app in a 1280x800 dp desktop window; omit phone status/navigation bars and use a rail, toolbar and multi-pane content.", _DESKTOP_CHECKS)
for _dev, _page in (("desktop", "app"), ("desktop", "landing"), ("tablet", "app"), ("tablet", "landing"), ("phone", "app"), ("phone", "landing")):
    _canvas, _lw, _lh, _rule = {
      ("desktop", "app"): ("2304x1440", 1440, 900, "A web app with a 240-280 px sidebar, 56-64 px top bar with search and account, page header with actions, KPI cards or table."),
      ("desktop", "landing"): ("2304x1440", 1440, 900, "A landing page with header nav (logo, links, CTA), hero headline, one-line value proposition and primary CTA visible, then the start of social proof/features."),
      ("tablet", "app"): ("1536x2048", 768, 1024, "Responsive tablet app with 44 px touch targets and navigation suited to this breakpoint."),
      ("tablet", "landing"): ("1536x2048", 768, 1024, "Responsive tablet landing page with navigation and 44 px touch targets."),
      ("phone", "app"): ("1152x2496", 390, 844, "Responsive mobile web app with hamburger or bottom navigation and 44 px touch targets."),
      ("phone", "landing"): ("1152x2496", 390, 844, "Responsive mobile landing page with stacked hero, visible CTA and 44 px touch targets."),
    }[(_dev, _page)]
    if _dev == "desktop":
        _rule = "Form factor: Web " + _rule + " At a 1440x900 desktop viewport, first screenful only, flat, no browser window, address bar or device. Content in a centred 1200-1280 px container on a 12-column grid."
    DEVICES[("web", _dev, _page)] = _device(_canvas, _lw, _lh, 24 if _dev == "desktop" else 44, None,
        "wide desktop viewport about 1.6:1" if _dev == "desktop" else ("tablet portrait viewport" if _dev == "tablet" else "tall mobile viewport"),
        "no browser chrome or device frame", "semantic responsive web navigation, forms and content", "WCAG target size; visible focus and hover",
        _rule, _WEB_CHECKS)

DEVICE_VALID = {"android": {"phone", "tablet", "foldable", "desktop"}, "ios": {"phone", "tablet"},
                "web": {"phone", "tablet", "desktop"}, "macos": {"desktop"}, "windows": {"desktop"}, "linux": {"desktop"}}


def _device_record(platform, device=None, orientation=None, posture=None, page=None):
    if not platform:
        return None
    default = "desktop" if platform in ("web", "macos", "windows", "linux") else "phone"
    device = device or default
    allowed = DEVICE_VALID.get(platform, {"phone"})
    if device not in allowed:
        raise ValueError(f"{platform} supports {', '.join(sorted(allowed))}")
    if orientation and device != "tablet":
        raise ValueError("--orientation applies to --device tablet (portrait|landscape)")
    if posture and device != "foldable":
        raise ValueError("--posture applies to --device foldable (flat|tabletop|folded)")
    if page and platform != "web":
        raise ValueError("--page applies to --platform web (app|landing)")
    if device == "tablet":
        variant = orientation or ("landscape" if platform == "android" else "landscape")
    elif device == "foldable":
        variant = posture or "flat"
    elif platform == "web":
        variant = page or "app"
    else:
        variant = "default"
    key = (platform, device, variant)
    if key not in DEVICES:
        valid = sorted(k[2] for k in DEVICES if k[:2] == key[:2])
        raise ValueError(f"{platform} {device} supports {', '.join(valid)}")
    return DEVICES[key]

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
            per_screen = meta.get("screens", {}).get(Path(path).name) if isinstance(meta.get("screens"), dict) else None
            if isinstance(per_screen, dict):
                meta = dict(meta, **per_screen)
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

CRITIQUE_PROMPT = """Do design QA on this app UI mockup ({W}x{H} px; the screen is {dp} logical units wide, so 1 unit = {k:.2f}px;
target platform: {platform}). It will be built in code. Report only real defects a developer or user would hit, most
severe first:
- text: every string that is garbled, misspelled, nonsense, truncated or oddly cased (quote it exactly).
- alignment/spacing: misaligned edges or baselines, uneven gaps between repeated items, inconsistent margins.
- icons: mixed icon styles (outline vs filled, stroke weight, corners), ambiguous or malformed glyphs.
- contrast/accessibility: text or icons below WCAG AA (4.5:1 body, 3:1 large text and icons), text illegibly small
  RELATIVE to the rest of the screen (not absolute sizes: the drawing scale is inexact and is calibrated later),
  controls that cannot get a {touch} logical-unit touch target (a smaller visible pill or icon with room around it is fine).
- overlap: a floating element (FAB, snackbar, sheet, badge, mini player) covering text, numbers, list rows or
  artwork; elements colliding, or hidden under the navigation bar or system bars. High when it hides content.
- platform: {platform} guideline problems (navigation pattern, tab count, safe areas, standard controls), and system
  UI that belongs to another platform (e.g. an iOS "9:41" status bar or Dynamic Island on Android, Android status
  icons or a gesture pill on iOS).
- adaptivity: {device_rule}
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

ADAPT_PROMPT = ("The attached image is the approved screen of this app and style reference. Create the SAME screen for a "
                "different form factor: {device_rule} Keep the same app name, navigation destinations, icons, palette, "
                "type and dataset. Recompose for this form factor (phone list -> tablet list-detail; bottom bar -> rail/sidebar; "
                "feed -> adaptive grid; detail -> 70/30 supporting pane; web hero -> stacked mobile hero). Preserve the same "
                "information architecture and data, not a scaled copy. {prompt} Render a distinct responsive composition.")

ISSUE_ITEM = {"type": "object", "additionalProperties": False,
              "required": ["severity", "category", "bbox", "problem", "edit"],
                  "properties": {
                  "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                  "category": {"type": "string", "enum": ["text", "alignment", "icons", "contrast", "accessibility",
                                                          "overlap", "platform", "adaptivity", "implementability", "consistency"]},
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
- Exclude window chrome (traffic lights, caption buttons, title bar), taskbar, Dock and browser chrome.
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
