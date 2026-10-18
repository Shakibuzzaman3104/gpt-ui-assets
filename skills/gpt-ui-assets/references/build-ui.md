# Phase 6: build the UI 1:1 (measure, don't guess)

Read before writing UI code from a design: `fonts`, `redline`, `tokens`, `compare`. `$K` is `scripts/gptkit.py` in this skill's directory.

GPT's spec is a list of estimates. Replace them with measurements from the pixels before writing UI code:
```bash
python3 $K fonts   build/home     # Google Fonts match by rendering + measuring (network; cached under ~/.cache/gptkit)
python3 $K redline build/home     # real edges, corner radii, insets and gaps in dp
python3 $K tokens  build/home --package com.acme.app.ui.theme --prefix Acme [--author "Name"] --verify
```
- **Network.** `fonts` and `compare`'s font verdict talk to fonts.google.com (family list, refetched after 7 days; with
  `GOOGLE_FONTS_API_KEY` set the Developer API at www.googleapis.com is the fallback), fonts.googleapis.com + fonts.gstatic.com
  (one TTF per family/weight tried, cached) and raw.githubusercontent.com (licence texts). Offline with a warm cache everything
  runs; offline without one, `fonts` exits with one line naming the host and `compare` prints `(font verdict skipped: ...)`
  and still writes its report. A corrupt cache is deleted and refetched.
- **`fonts`** writes `ttf/` (PostScript file names), Android `res/font/*.ttf` + a font-family XML with both `android:` and `app:`
  attributes (checked with `aapt2 compile` when the SDK is found; link needs `platforms/android-*/android.jar`), a real
  `ios/UIAppFonts.plist`, `web/fonts.css` (self-hosted `@font-face`, WOFF2 first when `fonttools` + `brotli` are installed, else
  TTF and a printed note; `font-display: swap`; a fallback stack per category) plus
  `web/fonts-link.html` (preconnect + css2 `<link>` with `display=swap`), `licenses/` and `fonts.json` (the fitted size, weight,
  tracking and `ascent_em` per style; file paths relative to the fonts dir).
  - **Read `fonts_review.png`.**
  - iOS: add `ttf/*.ttf` to the target, then `/usr/libexec/PlistBuddy -c "Merge build/home/fonts/ios/UIAppFonts.plist" App/Info.plist`
    and use `Font.custom("<PostScript name>", size:, relativeTo:)` (the tokens file already does).
  - On GPT-drawn text, sans families often tie within 0.01; the tool says so, keeps the spec's family when it's close, and
    `--family` (resolved case-insensitively, close matches suggested, `decided_by: forced`) overrides it; `--extra` adds families
    to every shortlist. Weights that tie within one step keep the spec's weight (`weight_tie` in fonts.json); stem darkening is
    fitted per candidate, so Chrome/Skia and Pillow/Linux renders both work.
  - `! Style: fits X poorly; Y fits much better` means that style may really use a second family. Latin scripts only.
  - Flags: `--candidates N` similar families per group (0 = named only), `--weights N` weights tried per family, `--tie` margin
    for keeping the named family, `--show N` candidates per review row, `--out DIR`, `--vision`.
- **`redline`** writes surface boxes (full-bleed sheets and bars included: sides on the image border are the border, radii per
  corner in `corner_radii_px`, `shadow: true` under drop shadows), corner radii (rect, pill or circle), and for every item its
  parent, insets and the gap below in dp. Text `px`/`dp` are **ink** boxes: pad from `baseline_inset_dp`
  (`Modifier.paddingFromBaseline` / `.alignmentGuide(.firstTextBaseline)`) or, with fonts.json, `layout_top_inset_dp` (the text
  layout box, what `Modifier.padding` sees), not from the ink inset. Spec radii are linked to surfaces by exact name first
  (`match_score`), each surface at most once; items it can't measure stay listed as `unmeasured` with a reason. `--vision`
  locates text the components don't cover.
- **`tokens`** writes Compose `Color.kt` (`AcmeColors` object, measured `AcmeDarkColorScheme`/`AcmeLightColorScheme` with the other
  appearance scaffolded and marked TODO), `Type.kt` (`AcmeTypography` styles plus `AcmeM3Typography = Typography(headlineLarge = …)`
  from the spec's `platform_style`, nearest slot by size otherwise), `Shape.kt` (`AcmeShapes` from chip/card/fab/sheet radii),
  `Dimens.kt` (spacing with `// redline: …` or `// GPT estimate` comments), `Theme.kt` (`AcmeTheme { }`), a SwiftUI tokens file
  (`Font.custom(…, relativeTo: .<HIG style>)` for Dynamic Type) with `Colors.xcassets/*.colorset` (one appearance; add the
  dark values when you have a dark mockup), `web/tokens.css` (imports `fonts.css` when present), `tailwind.tokens.cjs` (a Tailwind
  v3 preset: `presets: [require('./tailwind.tokens.cjs')]`) and `tailwind.theme.css` (an `@theme` block for Tailwind v4:
  `@import "tailwindcss"; @import "./tailwind.theme.css";`), plus `compose/Preview.kt`, a `@Preview(widthDp = <effective width>)`
  stub for Roborazzi or the Compose Preview Screenshot Testing plugin. `--dtcg` adds `tokens.tokens.json` (W3C Design Tokens,
  the Style Dictionary v4 subset: `color`, `typography`, `dimension.radius`, `dimension.spacing`, provenance in `$description`)
  for Style Dictionary, Tokens Studio and friends.
  It warns about spacing off the 4 dp grid and more than four distinct non-pill corner radii.
  `--snap-grid` rounds spacing to the grid: a deliberate departure from the mockup that `compare` will show.
  - Every palette hex (`#rgb`, `#rrggbbaa`, `rgb()`/`rgba()` accepted; alpha kept) is first snapped to the mockup's pixels;
    `palette_check.png` shows the guess, the snapped value and the ΔE00. Identifiers are unique across colours, styles, radii,
    spacing and font families; Swift keywords are backticked and SwiftUI colour names get a `Token` suffix.
  - `--verify` lints the emitted code (unique identifiers, no recursive colour references, valid `FontWeight`/`Color` literals),
    type-checks the Kotlin with `kotlinc` against Compose API stubs when it is on PATH (or `GPTKIT_KOTLINC`), `swiftc` only on
    macOS, loads the Tailwind config in node and parses the CSS in headless Chrome. It never reports a check that did not run as ok.
  - It prints which M3 colour roles and type slots stay at the baseline; native components (Switch, FAB, NavigationBar) use those.
  - Flags: `--platforms compose,swiftui,web` (aliases android, ios), `--r-package` (the module namespace holding `R`, AGP 8),
    `--fonts FILE`, `--no-snap` (keep the spec's hex), `--dtcg`, `--out DIR`, `--vision`.
  - If `tokens`/`fonts` print "missing: … (add --vision)", add `--vision` (one cached call to locate the text); "vision couldn't
    find it" means the sample text in design_spec.json does not match the mockup.

Build the screen with the platform's **native components**, using `spec`'s `native` widget per component and `platform_style` per
text style, plus these files and the exported assets. Follow the project's conventions. Then close the loop:
```bash
# screenshot the built screen (emulator/simulator/device, or headless Chrome at the target width), then:
python3 $K compare build/home shot.png [--vision] [--crop-top PX] [--crop-bottom PX] [--shot-width-dp N] [--fail-below 95]
```
- `compare` writes `compare/report.json`, `heatmap.png` (design | aligned implementation | ΔE heatmap with the worst regions boxed)
  and a ranked list of concrete fixes, for example:
  - `'Sleep Stories' 7% larger: renders like ~30.0sp (design 28sp, token AcmeTypography.SleepStories; ink height 56 -> 60px)`
    (the measured size; "use 26.2sp" is added only when the implemented font is the specified one)
  - `'Breathe' looks like Nunito 400 rather than Lora 400`; on Android `looks like Roboto 400 … (fontFamily not set?)`
  - `fill colour off by dE 4.8 (#A69BF7 -> #9A8BE6)`
  - `shifted +12dp vertically`
  - `element … missing in the implementation` (no shift/size hints follow a missing or replaced element)
- It detects a status bar and a taller screen automatically. A shorter screenshot prints `compared X% of the design` and marks
  regions below the fold `outside screenshot`; when every text is off by the same factor it prints one `all text is ~N% smaller`
  line instead of per-text hints (wrong dp width or font scale). "Differs inside the artwork" regions come from the forged assets,
  not your code. `--vision` adds qualitative descriptions; trust the measured numbers for sizes and positions.
- **Screenshot at the design's effective width** (`effective_width_dp`, printed by `spec` and `compare`; pass `--shot-width-dp N`
  or `--density D` so a mismatch > 5% is called out). Use a Compose `@Preview(widthDp = N)` rendered with Roborazzi (or the Compose
  Preview Screenshot Testing Gradle plugin; Paparazzi works too), an emulator display override, or a Chrome window N CSS px wide.
  A narrower real device reflows the layout, and `compare` will report that as differences.
- `--fail-below SCORE` exits 1 when `overall.score` is lower (a CI gate; no gate by default). Other flags: `--mockup`, `--top N`,
  `--out DIR`, `--fonts FILE`.
- Fix, re-screenshot and re-run until the only remaining items are ones you accept.
- **Custom lettering** (a wordmark or handwritten caption) that `fonts` reports as having no reliable match: forge it as a PNG asset and
  `trace` it, or `svg` it, and ship it as a VectorDrawable rather than forcing a font.
- `redline` misses very low-contrast surfaces (white cards on cream, about ΔE 1.5). Take those from `spec`.
