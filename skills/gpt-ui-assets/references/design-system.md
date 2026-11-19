# Design system: type, colour, spacing, shape

Read before writing art directions, when reviewing a mockup's type, colour or geometry, and before `tokens`. The numbers
live in `scripts/gptkit/system.py` (presets and checks) and `scripts/gptkit/palette.py` (tonal palettes and schemes), each
with its source. Values marked *unverified* come from guidance whose page could not be read as text; treat them as defaults,
not rules. Sources were read on 19/11/26.

## The order of authority

1. The platform's own guideline for anything people touch or read: touch targets, minimum text size, contrast, safe areas.
2. The approved design for everything else: if it uses a 20 dp radius or a 15 sp label on purpose, keep it.
3. The preset scale, to snap values the image model drew loosely (13 → 12, 26 → 28).

When 2 and 3 disagree, keep the design's value if it repeats across screens or the user chose it; snap it if it occurs
once. A value that breaks rule 1 is always fixed, and the fix is reported.

## Typography

| | Material 3 (sp) | iOS Dynamic Type, Large (pt) | macOS (pt) | Fluent 2 (px) |
|---|---|---|---|---|
| Body | Body Large 16/24, Medium 14/20, Small 12/16 | Body 17/22 | Body 13/16 | 14/20, 16/22 |
| Titles | Title L/M/S 22/28, 16/24, 14/20 | Title 1-3 28/34, 22/28, 20/25 | Title 1-3 22/26, 17/22, 15/20 | 20/28, 24/32 |
| Display | Display L/M/S 57/64, 45/52, 36/44; Headline 32/40, 28/36, 24/32 | Large Title 34/41 | Large Title 26/32 | 28/36 … 68/92 |
| Small | Label L/M/S 14/20, 12/16, 11/16 | Footnote 13/18, Caption 12/16, 11/13 | Footnote and caption 10/13 | 10/14, 12/16 |
| Minimum | 11 (Label Small) | 11 | 10 | 10 |

Sources: [Compose TypeScaleTokens](https://github.com/androidx/androidx/blob/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/TypeScaleTokens.kt),
[HIG Typography](https://developer.apple.com/design/human-interface-guidelines/typography),
[Fluent 2 tokens](https://github.com/microsoft/fluentui/blob/master/packages/tokens/src/global/fonts.ts).

Rules:
- Use the platform's text roles and their sizes for text people read: body 16 sp on Android, 17 pt on iOS, 13 pt on
  macOS, 16 px on the web. Express the hierarchy with 4-6 roles; more than 8 sizes on one screen is noise.
- Line height: about 1.4-1.5 × size for body text, 1.1-1.3 for headings; never below 1.0. M3 body runs at 1.5, iOS body at 1.29.
- Tracking tightens as size grows. M3: +0.5 sp on 16 sp body, -0.25 sp on 57 sp display. SF Pro's system font adjusts
  itself (+12/1000 em at 10 pt, -26 at 17 pt, +12 at 34 pt), so **custom fonts on iOS need their own tracking, system fonts
  need none**. A matched tracking outside -0.05…+0.12 em, or above +0.02 em at 32 px and over, is a measuring error.
- Bind sizes to the platform's scaling: `sp` and the M3 `Typography` on Android, `Font.TextStyle` (or
  `.custom(_:size:relativeTo:)`) on iOS so Dynamic Type works, `rem` on the web.
- Two weights carry most interfaces (regular and medium or semibold); bold for titles only. Thin weights need larger sizes.
- On the web, keep running text to 45-75 characters a line.

## Colour

M3 builds every role from tonal palettes, where tone is CIELAB L* (0 black, 100 white).
[ColorLightTokens](https://github.com/androidx/androidx/blob/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/ColorLightTokens.kt) /
[ColorDarkTokens](https://github.com/androidx/androidx/blob/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/ColorDarkTokens.kt):

| Role | Light tone | Dark tone |
|---|---|---|
| primary / onPrimary | 40 / 100 | 80 / 20 |
| primaryContainer / onPrimaryContainer | 90 / 10 | 30 / 90 |
| surface / onSurface | N98 / N10 | N6 / N90 |
| surfaceContainerLowest … Highest | N100, 96, 94, 92, 90 | N4, 10, 12, 17, 22 |
| onSurfaceVariant / outline / outlineVariant | NV30 / NV50 / NV80 | NV80 / NV60 / NV30 |
| error | 40 | 80 |

State layers over content: hover 8%, focus 10%, pressed 10%, dragged 16%
([StateTokens](https://github.com/androidx/androidx/blob/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/StateTokens.kt));
disabled content 38%, disabled container 12% (*unverified*).

Contrast (WCAG 2.2 AA, [1.4.3](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html),
[1.4.11](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)):
- Text 4.5:1. Large text 3:1, where large is 24 CSS px, or 18.66 px bold. HIG uses points: 18 pt and up, or any bold
  weight, may use 3:1 ([HIG Accessibility](https://developer.apple.com/design/human-interface-guidelines/accessibility)).
- Icons, borders, focus rings and other controls: 3:1 against what touches them.
- HIG asks for 7:1 on small text in custom colours where possible, and for checking both appearances and Increase Contrast.
- APCA is advisory only: no adopted standard requires it, and its reference code has a restricted licence.

Rules:
- One accent marks the primary action, the selected navigation item and the single most important value: about 10%
  of the screen.
- Name colours by role (primary, surface, onSurface …), never by hue, so light and dark can differ.
- **A dark scheme is derived, never copied.** `palette.schemes()` keeps the measured colours exactly in the appearance
  they were measured in and derives the other appearance from the tone table.
- Platform difference: M3 dark surfaces sit at tone 6, never pure black. iOS's dark base background is pure black, and
  elevated surfaces are nearly black ([HIG Dark Mode](https://developer.apple.com/design/human-interface-guidelines/dark-mode)).
- Image models follow colour names better than hex codes. Give both ("deep teal #0F766E for primary actions"), and
  snap the measured colours afterwards.

## Spacing and layout

| | Grid | Steps | Touch target | Side margin |
|---|---|---|---|---|
| Material 3 | 4 dp (8 dp rhythm) | 0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64 | 48 dp | 16 dp compact, 24 dp medium and up (*unverified*) |
| iOS | 4 pt (8 pt rhythm) | 4 … 48 | 44 × 44 pt (28 minimum) | 16-20 pt (*unverified*) |
| macOS | 4 pt | 4 … 32 | 28 × 28 pt (20 minimum) | |
| Fluent 2 | 2/4 px | 0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32 | 24 px (WCAG 2.5.8) | |
| Web (Carbon) | 2/4 px | 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 160 | 24 × 24 CSS px minimum | |

M3 window classes: compact under 600 dp, medium 600-840, expanded 840-1200, large 1200-1600, extra-large 1600 and up
([window size classes](https://developer.android.com/develop/ui/compose/layouts/adaptive/use-window-size-classes)).
Sources: [Fluent spacings](https://github.com/microsoft/fluentui/blob/master/packages/tokens/src/global/spacings.ts),
[@carbon/layout](https://www.npmjs.com/package/@carbon/layout),
[WCAG 2.5.8](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html),
[HIG Layout](https://developer.apple.com/design/human-interface-guidelines/layout).

Rules:
- Every gap, padding and margin is a step of the platform's scale. Use more space between groups than inside one, and
  give the same relation the same value on every screen.
- Respect safe areas and system bars. Content never sits under the status bar, the Dynamic Island or the home indicator.
- Inset content at least as much as the platform's side margin. Wider windows add margin or panes, not longer lines.

## Shape

| | Steps |
|---|---|
| Material 3 | none 0, extraSmall 4, small 8, medium 12, large 16, largeIncreased 20, extraLarge 28, extraLargeIncreased 32, extraExtraLarge 48, full |
| Fluent 2 | 0, 2, 4, 6, 8, 12, 16, 24, 32, 40, circular |
| Apple | No fixed scale; continuous ("squircle") corners |

Sources: [Compose ShapeTokens](https://github.com/androidx/androidx/blob/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/ShapeTokens.kt),
[Fluent borderRadius](https://github.com/microsoft/fluentui/blob/master/packages/tokens/src/global/borderRadius.ts).

Rules:
- Three to five radii per app, each tied to a component family: controls, cards, sheets and dialogs, chips and pills.
- Nested corners are concentric: inner radius = outer radius − padding (28 dp card with 16 dp padding → 12 dp inside).
  A nested shape with the same radius as its parent looks pinched.
- Sheets round only their top corners, and a pill is `full`, not a large number.
- On Apple platforms use continuous corners (`RoundedRectangle(cornerRadius:style: .continuous)`).

## Tokens

Write tokens in three tiers, following [DTCG 2025.10](https://www.designtokens.org/tr/2025.10/format/) and M3's
`md.ref` / `md.sys` / `md.comp`:
- **Primitive:** tonal palettes, the spacing steps, the radius steps, the type sizes.
- **Semantic:** colour roles in light and dark, type roles, `space.*`, `shape.*`. Screens use only these.
- **Component** (optional): `button.container = {color.primary}`.

Aliases point down one tier only, and a value is never copied. `tokens` writes `tokens.json` in this format.
