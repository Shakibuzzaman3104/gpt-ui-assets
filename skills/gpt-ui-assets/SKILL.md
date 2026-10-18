---
name: gpt-ui-assets
description: Design app and web screens with OpenAI's GPT image and vision models, then turn an approved design into production assets redrawn 1:1 (never cropped from the screenshot) — SVG/VectorDrawable icons, PNG/WebP illustrations and backgrounds, Android/iOS/web launcher icons, frame animations, Lottie and AnimatedVectorDrawable, smoke/steam/rain/spray loops in the art's own style — plus theme code and fonts measured from the pixels and a screenshot diff of the built UI. Also writes GPT-researched feature blueprints that feed the mockups. Use whenever the user wants GPT/ChatGPT to design screens, icons, illustrations or an app icon; approves a GPT design and says "go ahead", "build it" or "give me the assets"; wants the assets "used in the UI"; wants an animated drawable, Lottie or AVD made from existing art; or complains that GPT cropped assets out of a mockup. Not for go/no-go idea validation (product-idea-validator) or design direction without GPT (design-muse).
license: MIT
metadata:
  version: 4.0.0
---

# GPT UI assets

**You are the driver.** GPT is one instrument among several. Route every job to whoever does it best, and never
accept "GPT can't" as the end of the road:

| Job | Best done by |
|---|---|
| Market research (web-searched, cited) and the feature blueprint | GPT reasoning with web search (`research`, `blueprint`) |
| Screen mockups, painterly art, illustrations, photos, textures, raster frames | GPT image (`mockup`, `forge`, `frames`) |
| Concept screens that must look finished, or move | **You** build the screen in HTML/CSS/JS; GPT paints only the materials (`material`, `state`); `stage` renders it |
| Asset inventory, design QA, icon-set style, vector keyframes | GPT vision (`manifest`, `critique`, `iconspec`, `vectorize`) |
| Flat/line icons as vectors | GPT vision writes SVG (`svg`, `icons`). If it stays weak after its retry, **you** fix the SVG by hand, then `svg --check` |
| Ranking, colour correction, safe zones, registration, QA | Local measurement in the kit. Trust numbers over the model's claims |
| Raster art → vectors, animatable parts | Local tracing and rigging (`trace`, `rig`, `animate`) |
| Exact colours, fonts, geometry, theme code, 1:1 check of the built screen | Measurement from pixels (`tokens`, `fonts`, `redline`, `compare`) plus you, the engineer |
| Anything a human must judge | You: Read every review sheet before handing anything over |

`K` is `scripts/gptkit.py` inside this skill's directory. Every command documents itself with `python3 $K <cmd> -h`.
```bash
K="<this skill's directory>/scripts/gptkit.py"
```

## Before you start

- **New machine or a failing call:** run `python3 $K doctor` and read `references/setup.md`. It covers the API key,
  gateways and optional tools (Chrome, Playwright, aapt2, EbSynth).
- **Resuming a project** (after a break, a context reset or in a new session): run `python3 $K status <dir>`. It lists
  each unit's finished, missing and stale steps, the problems the kit's reports recorded, and the exact next command.
  No API calls.
- **Cost.** Image calls take 35–80 s each and cost money or quota. Paid commands print a `plan: ~N image calls … ≈ $X`
  line before the first call. `forge`, `sequence render` and `research` take `--dry-run`. `GPTKIT_BUDGET_USD` stops any
  command whose estimate exceeds it. `python3 $K cost --by cmd` shows what has been spent. Run slow commands in the
  background, and run independent ones in parallel (e.g. `forge` and `svg`).
- **Models** are routed per job automatically. Read `references/models.md` before pinning one, when a transparent
  render comes back opaque, or when `doctor` warns that a model is retiring.

## The workflow

Each phase has a reference file. **Read it before running that phase's commands for the first time in a session.** It
holds the flags, the QA lines to look for and the fixes for each flag. The commands below are the shape, not the whole
story.

| Phase | When | Core commands | Read |
|---|---|---|---|
| 0. Research | An app idea needs a feature list, competitors or screens to design | `research "<idea>" --platform P --out research/x`, then `blueprint research/x` | `references/research.md` |
| 1. Design | The user wants screens designed | `mockup "<brief>" --platform P --out design/home`, `critique`, `--edit`, `--style-ref` | `references/design.md` |
| 1b. Hybrid | The concept must look portfolio-grade or move | `material`, `state`, `stage --init`, `stage`, `stage --compose` | `references/hybrid.md` |
| 2–4. Assets | The user approves a design ("go ahead") | `manifest`, `spec`, `iconspec`, `forge`, `svg`, `icons`, `pick`, `fix`, `trace`, `vd`, `export` | `references/assets.md` |
| App icon | A launcher icon is needed | `appicon concepts`, `appicon build` | `references/appicon.md` |
| 5. Animation | Art should move in place, pop, flicker or loop | `frames`, `vectorize`, `rig`, `animate` | `references/animation.md` |
| 5. Sequences | Something travels: smoke, steam, drops, spray, embers | `sequence layout → plan → render → build → reroll` | `references/sequence.md` |
| 6. Build 1:1 | Writing the UI code from the design | `fonts`, `redline`, `tokens`, then `compare` against a screenshot | `references/build-ui.md` |

A typical "design a screen and give me the assets" run:
```bash
python3 $K mockup "habit tracker home: date, progress, 5 habits with check + streak, 4-tab nav; mint + warm white" \
  --platform android --out design/home --variants 2
python3 $K critique design/home/mockup_v1.png          # Read the critique sheet, then show the user the variants
# ...iterate with --edit until the user approves; screen 2 onward uses --style-ref <approved screen>
python3 $K manifest design/home/mockup_v1.png --out build/home   # Read manifest_overlay.png; fix assets.json by hand
python3 $K spec build/home                              # calibrated scale, palette, type, native widget per component
python3 $K forge build/home --candidates 3 &  python3 $K svg build/home &  wait   # Read review/forge_*.png, review/svg_*.png
python3 $K vd build/home/svg/*.svg --out app/src/main/res/drawable
python3 $K export build/home/assets/hero.png --dp 240 --res app/src/main/res --name hero
python3 $K fonts build/home && python3 $K redline build/home && python3 $K tokens build/home --package com.acme.ui.theme --verify
# build the screen with spec's native widgets, screenshot it at the design's effective width, then:
python3 $K compare build/home shot.png
```

## Rules that hold in every phase

1. **Know the platform first** (`android`, `ios` or `web`). Ask if it isn't clear. `--platform` drives native components,
   system UI, metrics and icon family, and every later command inherits it from `design.json`.
2. **Read every review sheet yourself before the user sees the result.** These include `*_critique.png`,
   `manifest_overlay.png`, `review/forge_*.png`, `review/svg_*.png`, `icons/preview.png`, the `appicon` sheets,
   `frames_review.png`, `keys_review.png`, `layout_review.png`, `fonts_review.png` and `compare/heatmap.png`. The
   metrics rank candidates, but your eyes decide.
3. **Never crop an asset out of the mockup, and never text-only prompt an asset of an approved design.** A text-only
   prompt gives a generic piece in a different style. `forge`, `fix`, `frames`, `sequence` and `material` pass the
   design as the reference.
4. **Design screen 2 onward with `--style-ref <approved screen>`.** Describing the style in words drifts.
5. **Measure, don't guess.** Vision hex guesses drift by up to ΔE 10, and GPT draws native components at ~0.7× scale.
   Take colours, sizes, radii and fonts from `spec`, `tokens`, `redline` and `fonts`, not from the model's prose.
6. **Flags outrank scores.** `EMPTY` and `COPY` are never shipped. After two unsuccessful `fix` rounds, tell the user
   what still differs instead of looping.
7. **Read the error code before rewording a prompt.** `[moderation_blocked]` needs a rephrase; `exhausted` and 5xx need
   a retry later; a model error needs `doctor`.
8. **Never claim a check that did not run.** `link skipped`, `not measured: <reason>` and `render check skipped` mean
   unverified. Say so.
9. **An image edit re-renders the whole image.** For a state change, use `state`, which composites only the changed
   region back. Never ship a raw edit as a state frame.
10. **GPT renders; it does not animate.** Code or the kit decides motion (`--procedural`, `sequence` layouts, `rig`),
    and GPT paints only the moving layer.

## Working with the user

- Show design variants and let the user choose. Walk them through `blueprint.md`, especially the MVP cut, before
  designing every screen.
- App-icon concepts, animation routes and whether to ship platform icons (Material Symbols / SF Symbols) instead of the
  generated SVGs are the user's call. Present the options with the review sheet.
- Hand over files with their destinations (`res/drawable`, `Assets.xcassets`, `public/`) and say what was verified and
  what was not.

## Other skills that fit around this one

- **design-muse**, for art direction and real-app references before a `mockup`. Put the chosen direction (palette,
  type, shape, style family) into the mockup brief, and log the result there.
- **product-idea-validator**, for a go/no-go verdict before `research`. **product-blueprint**, when the user needs a full
  PRD and technical design rather than a feature blueprint.
- **play-store-listings**, after `appicon build`: `android/play_store_512.png` is the listing icon, and the approved
  mockups are the source for the feature graphic and screenshots.

## Reference files

| File | Read when |
|---|---|
| `references/setup.md` | Installing, `doctor` problems, gateway quirks, optional tools |
| `references/models.md` | Choosing or pinning a model, prices, retirement dates |
| `references/research.md` | `research`, `blueprint` |
| `references/design.md` | `mockup`, `critique`, `--edit`, `--style-ref` |
| `references/hybrid.md` | `material`, `state`, `stage`, carrying a hybrid screen into Compose |
| `references/assets.md` | `manifest`, `spec`, `iconspec`, `forge`, `fix`, `pick`, `score`, `svg`, `icons`, `trace`, `vd`, `export` |
| `references/appicon.md` | `appicon concepts`, `appicon build`, installing the icon |
| `references/animation.md` | `frames`, `vectorize`, `rig`, `animate` |
| `references/sequence.md` | `sequence` (layout-driven loops, EbSynth) |
| `references/build-ui.md` | `fonts`, `redline`, `tokens`, `compare` |
| `references/api-findings.md` | The raw measurements behind every rule; read only to settle a doubt |
