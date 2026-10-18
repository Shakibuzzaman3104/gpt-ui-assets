# Phase 1b: hybrid screens (code builds the UI, GPT paints the materials)

Read before `material`, `state` or `stage`, or when a concept must look portfolio-grade or move. `$K` is `scripts/gptkit.py` in this skill's directory.

A GPT mockup is a plausible screen, not a designed one. It comes back with chunky type, no hairlines, big saturated
fills and dense text, and it can't show motion. When the concept has to hold up next to portfolio-grade references,
or has to move, split the work: **code** owns layout, type, hairlines, states, controls and choreography, and **GPT**
paints only what code can't fake: real materials and objects such as foil, paper, glass, food or fabric. Keep
`mockup` for fast exploration.
```bash
python3 $K material "<the object: material, light, camera; no UI>" --out hybrid/assets --name strip --candidates 2   # background
python3 $K stage hybrid/screen.html --init                  # starter page with the page contract; build the screen meanwhile
python3 $K state hybrid/assets/strip.png "<what changes>" --circle CX,CY,R --out hybrid/assets/strip_after.png
python3 $K stage hybrid/screen.html --out hybrid/shots --still before=still --still after=state=after --motion 6.5
python3 $K stage hybrid/shot.html --out hybrid/shots --size 1600x1200 --dpr 2    # the presentation shot
```
1. **Run in parallel.** Start `material` in the background (each image call takes 35–80 s) and write the page while it
   renders. Read `review/material_<name>.png`; `CUT-OFF`, `OPAQUE` and `EMPTY` rank the candidates, but you pick.
2. **Measure the asset's geometry, never eyeball it.** Find where its parts sit (cavity centres, radii, aspect) locally
   with PIL/numpy, check them with an overlay, and write them to a `geo.js` the page reads. Rows, rings and masks then
   land on the material exactly.
3. **State variants come from `state`, not from a raw edit.** An edit re-renders the whole image even when told to
   change one region (measured on a foil strip: mean drift of 8.5–13.5 grey levels everywhere, with no shift).
   `state` registers each candidate, composites only the changed region back with a feathered mask, and reports
   `max difference outside the mask: 0`. `NO-CHANGE` means the edit ignored the request.
4. **Motion is code.**
   - Cut sprites of only the moving part (the pill, not the dome rim around it). A sprite that includes its container
     reads as zooming out when it scales.
   - Reveal the after-state underneath with an animatable mask radius: `@property --hole { syntax: '<length>' }`.
     Chrome does not interpolate gradient masks.
   - Make every motion a Web Animation. Default to `fill: "forwards"`; only the entrance uses `"both"`. With the
     timeline paused for capture, a later `both` animation applies its first keyframe before its delay and leaks into
     the idle frames.
   - The page exposes `demo()` and `seek(ms)` under `?capture`, so `stage` scrubs exact frames with no wall-clock
     jitter. `--init` writes this contract.
5. **Check the key moment frame by frame.** Read `motion_sheet.png`, then crop the key moment (the tear, the pop) at
   30 fps. A dissolve that should have been a snap only shows up close.
6. **Craft rules that made the difference.**
   - One hero, with generous empty space.
   - Two type weights, and quiet meta text.
   - Accent on about 10% of the area or less, only on the one live thing.
   - Hairlines instead of boxes.
   - The material carries the richness; the UI stays flat.
   - Measure meaningful-graphic contrast: an ink progress ring on foil is about 11:1, tangerine about 1.6:1, so the
     ring is ink and tangerine stays decorative.
7. **Present it.**
   - Render a presentation shot: the device on a soft backdrop, plus the hero asset enlarged with hairline annotations
     of its states.
   - Render the motion clip.
   - Make a before/after against the mockup route.

`stage` serves the page's folder over a local http server and drives Chrome through Playwright
(`pip install playwright`). It writes stills at `--dpr`, `frames/`, `motion.mp4` (an animated WebP without ffmpeg),
`motion_sheet.png` and `stage.json`. Real page errors (HTTP 4xx/5xx, console errors, exceptions) are printed and
recorded.

## Carry it into Compose

When the user approves a hybrid concept for an Android app, don't retype the motion from the JS. Extract it:
```bash
python3 $K stage hybrid/screen.html --compose app/src/main/java/com/acme/app/ui/hybrid \
  --package com.acme.app.ui.hybrid --prefix Hush [--author "Name"]
```
- It opens the page under `?capture`, calls `demo()` and reads `document.getAnimations()`. It writes `motion.json`
  (the neutral timeline: target id, rect at t=0, keyframes, delay, duration, easing; usable for SwiftUI too),
  `<Prefix>Motion.kt` (one `KeyframesSpec<Float>` per element and animated property: `translationX/Y`, `scaleX/Y`,
  `rotationZ`, `opacity` and every registered custom property such as `--hole` → `Hole`), `<Prefix>Geo.kt` (the
  page's `window.GEO`; `--init` pages load `geo.js` into it) and `<Prefix>MotionScreenshotTest.kt` (a Roborazzi
  template that pauses `mainClock` and captures each keyframe moment; move it to `src/test`).
- **Give every animated element an `id`.** The Kotlin names come from it (`#pill` → `pillTranslationX`); without one
  the kit falls back to `data-name` or a CSS path and warns.
- 1 CSS px = 1 dp: author the page at the device width. Properties that never change are not emitted.
- **Read the warnings.** Unsupported properties (`filter`, `clip-path`, `background-color`), `steps()` easing, skewed
  matrices, repeating or alternating iterations, and an effect-level easing over more than two keyframes (CSS spreads
  it over the whole animation; Compose applies it per segment) are listed, not silently dropped.
- Wire it up with `Animatable(...).animateTo(target, HushMotion.pillScaleX)` or `animateFloatAsState(..., HushMotion.pillScaleX)`
  inside `graphicsLayer { }`. A `--hole` mask becomes a `drawWithContent` circle cut with `BlendMode.DstOut` at the
  animated radius. The material PNGs go through `export` like any other raster asset.
- Run the Roborazzi test, then `compare` its frames with the `stage` stills of the same moments.
- The emitted Kotlin passes the kit's lint; it is type-checked with `kotlinc` against Compose stubs only when
  `kotlinc` is on PATH (`GPTKIT_KOTLINC`). The output says which ran.

## Rules learned from probing

- **An image edit re-renders everything.** Asked to change one region, `gpt-image-2.5-sunburst` still redrew the whole
  foil texture (mean drift 8.5–13.5 grey levels, no shift). Composite the region back with `state`; never ship the raw
  edit as a state frame.
- **Code-built screens beat whole-screen mockups on polish.** GPT's strength is the material, so paint only that and
  build the rest in code.
- **Capture a paused Web Animations timeline.** Seek it per frame; never record wall-clock playback. Gradient masks
  don't interpolate, but a registered `@property` length inside them does.
