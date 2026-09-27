---
name: gpt-ui-assets
description: Design platform-native app/web UIs with OpenAI image + vision models (gpt-image-2, gpt-6, or any OpenAI-compatible endpoint) and turn an APPROVED design into production assets redrawn from scratch 1:1 (never cropped from the screenshot) — icons as SVG/Android VectorDrawable, illustrations and backgrounds as clean PNG/WebP, launcher icons for Android/iOS/web, frame animations, Lottie and AnimatedVectorDrawable — then build and verify the real UI. Use whenever the user wants GPT/ChatGPT to design app screens, icons, images, illustrations or an app icon, approves a GPT design and says "go ahead"/"build it"/"give me the assets", asks for the assets "used in the UI", wants frames for an animated drawable, a Lottie or an animated vector, or complains that GPT cropped assets out of a mockup.
license: MIT
metadata:
  version: 2.0.0
---

# GPT UI assets

**You are the driver.** GPT is one instrument among several. Route every job to whoever does it best, and never
accept "GPT can't" as the end of the road:

| Job | Best done by |
|---|---|
| Screen mockups, painterly art, illustrations, photos, textures, raster frames | GPT image (`mockup`, `forge`, `frames`) |
| Asset inventory, design QA, icon-set style, vector keyframes | GPT vision (`manifest`, `critique`, `iconspec`, `vectorize`) |
| Flat/line icons as vectors | GPT vision writes SVG (`svg`, `icons`). If it stays weak after its retry, **you** write or fix the SVG by hand, then `svg --check` |
| Ranking, colour correction, safe zones, slicing, registration, QA | Local measurement in the kit. Trust numbers over the model's claims |
| Raster art → vectors, animatable parts | Local tracing and rigging (`trace`, `rig`) <!-- WAVE2 --> |
| Lottie / AnimatedVectorDrawable / motion | `vectorize` builds and validates them; **you** refine `layers.json` and `--rebuild` |
| Theme code, fonts, 1:1 check of the built screen | `tokens`, `fonts`, `compare` <!-- WAVE2 --> plus you, the engineer |
| Anything a human must judge | You: Read every review sheet before handing anything over |

`K` below is `scripts/gptkit.py` inside this skill's directory:
```bash
K="<this skill's directory>/scripts/gptkit.py"      # every command: python3 $K <cmd> -h
```

## Setup (once per machine)

```bash
pip install -r "<this skill's directory>/scripts/requirements.txt"    # requests, pillow, numpy (+ optional extras)
python3 $K doctor                                                      # key, endpoint, models, SVG renderer, aapt2
```
- **Config** comes from env vars or `~/.config/gptkit/env` (`KEY=value` lines):
  - `GPTKIT_API_KEY` (or `OPENAI_API_KEY`), required.
  - `GPTKIT_BASE_URL` (or `OPENAI_BASE_URL`), default `https://api.openai.com/v1`. Any OpenAI-compatible gateway that exposes
    `/images/generations`, `/images/edits` and `/responses` works.
  - `GPTKIT_IMAGE_MODEL` (default `gpt-image-2`) and `GPTKIT_TEXT_MODEL` (default `gpt-6`, which needs vision and structured outputs).
- **Optional tools**, each used when present:
  - Headless Chrome/Chromium renders SVG with real alpha and plays Lottie through lottie-web (`GPTKIT_CHROME` sets its path).
  - The Android SDK's `aapt2` (via `ANDROID_HOME` or `~/Library/Android/sdk`) validates every VectorDrawable, adaptive icon and AVD.
  - `xcrun actool` validates the iOS app icon.
- Never print or echo the API key.
- Image calls take 35–80 s and cost money or quota. Batch deliberately, run slow commands in the background, and run
  independent commands in parallel (e.g. `forge` and `svg`).

## Phase 1: design (the user drives)

**Know the platform first.** Ask the user if it isn't clear: `android` (Material 3 / Compose), `ios` (HIG / SwiftUI) or
`web`. `--platform` makes GPT design with that platform's native components, system UI, metrics, type scale and
icon family, for example the M3 navigation bar with its pill indicator vs the floating iOS tab bar. The choice is recorded
in `design.json`, and every later command inherits it.

```bash
python3 $K mockup "habit tracker home: date, progress summary, 5 habits with check + streak, add action, 4-tab nav; mint + warm white" \
  --platform android --out design/home --variants 2          # 1024x1536 portrait; --size 1536x1024 for web/tablet
python3 $K critique design/home/mockup_v1.png                # design QA -> mockup_v1_critique.png/json
python3 $K mockup "make the summary card taller" --edit design/home/mockup_v1.png --out design/home --name mockup_r2
# every further screen of the same app, in the approved screen's exact design language:
python3 $K mockup "Profile screen, Profile tab active: avatar + name, 3 stat tiles, weekly chart, settings list" \
  --style-ref design/home/mockup_r2_v1.png --out design/profile
python3 $K critique design/profile/mockup_v1.png --style-ref design/home/mockup_r2_v1.png --apply
```
- **Screens:**
  - Show the user each variant (Read the PNG) and iterate with `--edit` until they approve. A design made elsewhere
    (the ChatGPT app, a Figma export, a screenshot) works the same way: use its image path, and pass `--platform` to later commands.
  - **Design screen 2 onward with `--style-ref <approved screen>`, never text-only.** Describing the style in words drifts:
    a different wordmark, a different art style, text baked into the art. In the prompt, say which tab is active and which
    shared chrome the screen has. `layout_diff < 10` means GPT copied the reference; the kit retries once.
- **Critique:**
  - **Run `critique` on a variant before showing it**, then Read `<name>_critique.png` (red = high, orange = medium, blue = low).
    It checks text, alignment, icon consistency, contrast (measured locally: trust `[measured N:1]`), platform
    guidelines and implementability. With `--style-ref` it also catches a wrong app name, nav items, typeface or data.
  - `--apply` edits the fixes in and keeps the revision only if the issue weight drops. The verifier is lenient and the
    editor never sees the approved screen, so Read the revision. Give concrete follow-ups with `--edit`.
  - Almost every mockup is flagged "text under 11dp", because gpt-image draws small labels whatever the prompt says.
    Set real type sizes in code from `spec` rather than editing the mockup.

## Phase 2: "go ahead" → inventory and spec

```bash
python3 $K manifest design/home/mockup_v2.png --out build/home    # -> assets.json + manifest_overlay.png (platform inherited)
python3 $K spec     build/home                                    # palette, type (+ platform type style), radii, spacing,
                                                                  # components (+ the native widget for each); ~3 min, background
python3 $K iconspec build/home                                    # icon-set style + Material/SF symbol names (3+ icons)
```
**Review `manifest_overlay.png` yourself before generating anything.** Red boxes are PNG assets and green boxes are SVG.
Edit `assets.json` directly to fix the inventory:
- Remove anything code or the OS draws: native components (the tab bar itself, switches, chips, buttons, sheets),
  solid or gradient shapes, cards, dividers, text, shadows, and system UI (status bar, home indicator). `manifest` already
  excludes these and drops system glyphs it finds in the status-bar strip.
- Split merged graphics, merge duplicates, and tighten loose bboxes.
- Set `format` to `svg` for flat, line, monochrome or simple-gradient glyphs, and to `png` for painterly, glowing, 3D, photographic or detailed art.
- Give art behind UI `kind: background` and `transparent: false`. It gets redrawn with the UI removed.
- `material_symbol` / `sf_symbol` name the equivalent platform glyph. Ship the generated SVG (it is 1:1 with the
  design) unless the user asks for platform icons.
- Treat `dp` sizes as estimates, because image models don't draw to scale. Snap them to platform norms (24 dp icons,
  48 dp / 44 pt touch targets, an 8 dp grid). `spec` maps each component to the native widget to build it with
  (`native`) and each text style to the platform type scale (`platform_style`); build with those.

## Phase 3: produce assets (run `forge` and `svg` in parallel)

```bash
python3 $K forge build/home --candidates 3 --jobs 8   # PNG assets: parallel redraws, scored, colour-matched -> assets/, review/
python3 $K svg   build/home                           # SVG icons: scored vs the design, weak ones retried, 24x24 -> svg/
python3 $K svg   build/home --check                   # re-score svg/*.svg after you hand-edit them (no API calls)
python3 $K icons build/home "notifications, sleep timer, download" --vd app/src/main/res/drawable   # NEW icons in the set's style
```

**forge**
- It sends a **focused crop of the element** (60% margin, upscaled) as the single reference with a "recreate from
  scratch, 1:1" prompt. Candidate 2 adds a colour emphasis and candidate 3 a silhouette emphasis; `--same-prompt` turns this off.
- Backgrounds (`kind: background`, `transparent: false`) are redrawn from a tight crop of their own region.
- `fid` ranks candidates; lower is better. It combines silhouette `iou` (after alignment), CIELAB `de`, colour
  distribution `col` and pixel `diff`. Rough bands: about 16–24 is a close icon/illustration match, 30+ is visibly off and 45+ is a
  different style. Backgrounds score about 15–22.
- A candidate with any of these `flags` ranks last:
  - `COPY:*`: GPT returned the screenshot.
  - `OPAQUE`: transparency was lost.
  - `BACKDROP`: it kept the circle or card behind the icon.
  - `CUT-OFF`: the content touches the canvas edge.
- Warnings: `warn: silhouette` (iou < 0.5) means the shape is wrong. `warn: soft` is normal for smooth gradients.
- The best transparent candidate is then **colour-matched** to the design: a Lab transfer at strength 0.75, kept only if `fid` improves.
  The untouched copy stays in `raw/<name>_cN.png`. Use `--no-color-match` to keep the raw colours. Opaque backgrounds are never matched.
- **Always Read `review/forge_*.png`.** The metric ranks candidates, but your eyes decide. Then:
  ```bash
  python3 $K pick  build/home rain_drop 2                                 # choose another candidate
  python3 $K fix   build/home campfire --note "logs should be thinner"    # vision diff + one corrective edit
  python3 $K forge build/home --only campfire --force --candidates 3      # or re-roll
  ```
  `fix` keeps a result only if `fid` improves, because correction passes often over-correct. Pass `--accept` when the user asked for that exact
  change. After 2 unsuccessful rounds, tell the user what still differs rather than looping.

**svg / icons**
- Each icon prints `rel`: 0 is perfect and 1 is as bad as drawing nothing. Expect about 0.1 for filled glyphs, 0.2–0.35 for close
  thin strokes and about 0.4 on glows. It also prints `fit`, `ink` (above 1 means heavier than the design), `dE` and `!` issues. Weak or
  non-VectorDrawable-safe icons get one vision retry against a design|render side-by-side, and the better one is kept
  (`svg/raw/<name>_tryN.svg`).
- Read `review/svg_*.png`, which shows design | SVG on the same backdrop. If an issue survives the retry, **fix the SVG yourself**
  (you can read the design crop and write paths), then run `svg --check`. Icons end up on a 24x24 viewBox with their
  content in 2..22 and no transforms; `--grid 0` keeps the model's viewBox.
- `icons` measures the set's stroke, caps, joins and colour from existing SVGs, so run `svg` first. Read `icons/preview.png`:
  the new row must look like the same family. `--snap` forces the set's stroke, `--treatment filled` draws filled variants.

<!-- WAVE2: trace -->

## Phase 4: platform export

```bash
python3 $K vd build/home/svg/*.svg --out app/src/main/res/drawable --dp 24          # SVG -> VectorDrawable (ic_ prefix)
python3 $K export build/home/assets/campfire.png --dp 48 --res app/src/main/res --name img_campfire   # webp per density
python3 $K export <png> --dp 44  --platform ios --res App/Assets.xcassets          # @1x/@2x/@3x imageset
python3 $K export <png> --dp 320 --platform web --res public/img                   # 1x/2x webp
```
`vd`:
- Converts transforms to `<group>`, inherits root attributes and multiplies group opacity.
- Verifies each file: it renders the SVG and the VectorDrawable and compares them ("render match"), then compiles and
  links them with aapt2.
- Draws dashes solid, with a warning. Rejects masks, clip paths, filters, text and skews: re-run `svg --only <name>`
  or fix the SVG by hand.

Size background art to its container's aspect ratio and let code clip the corners. Never bake rounded corners or text into it.

## App launcher icon

```bash
python3 $K appicon concepts "Hush, a calm sleep app; dark navy and lavender; moon motif" \
  --mockup build/home/mockup.png --out icon --variants 3            # 1 image call per variant -> icon/review/concepts_*.png
python3 $K appicon build icon/concept_v2.png --out icon/build \
  --subject "the glowing crescent moon, its stars and the wind swoosh"   # 0 image calls + 1 gpt-6 call (themed glyph)
```
- **Concepts:**
  - Pass `--mockup` whenever an approved design exists. It's sent at low input fidelity with a "new icon, not the screen's
    illustration" prompt. Without it, concepts come out generic.
  - **Read the concepts sheet.** It shows each concept full-bleed, in circle/squircle/rounded/teardrop masks and at 48 px.
    Reject `BUSY-BACKGROUND` (a scene, not an icon), `SUBJECT-TOO-BIG` and `BLEEDS-OFF-EDGE`. Lower `palette_de` is closer to the
    design. Let the user pick.
- **Build** defaults are the measured best:
  - The foreground is matted locally against the background that actually ships, so the icon recomposes the concept almost exactly.
  - The background is a fitted VectorDrawable gradient (or a colour).
  - The subject is shrunk into the 66 dp safe circle only when measurement says so.
  - The monochrome (themed) layer is a gpt-6 glyph converted to a VectorDrawable.
  - Use `--bg image` (1 call) only for textured or scenic backgrounds; the report says when it's needed.
- **Read `review/appicon_*.png`**, and check `appicon.json` for `safe_zone.ok`, `ios.opaque`, `ios.actool: ok`, `web_maskable.ok`
  and `aapt2: compile+link ok`.
- **Install:**
  - Android: copy `android/res/*` into `app/src/main/res` and point the manifest at `@mipmap/ic_launcher` and `@mipmap/ic_launcher_round`.
    `android/play_store_512.png` is the Play listing icon.
  - iOS: copy `ios/AppIcon.appiconset` into `Assets.xcassets`.
  - Web: the files and the `manifest-icons.json` snippet go in `public/`.

## Phase 5: animation

Pick the route with the user:

| Route | Use when | Output |
|---|---|---|
| `frames --procedural bounce\|pulse\|float\|wiggle\|spin\|breathe` | the art moves rigidly; **no API calls**, art stays exactly 1:1 | frames, WebP, APNG, tiny transform Lottie, `motion.json` for code |
| `frames --moving … --static …` | the art changes shape (flame, ripples, blinking mascot) | registered frames, WebP/APNG/image-sequence Lottie, `animation-list` XML |
| `vectorize` | vector motion: morphs, transforms, glows | Lottie **and** looping AnimatedVectorDrawable |
<!-- WAVE2: rig -->

```bash
python3 $K frames build/home rain_drop --procedural bounce --amount 1.2
python3 $K frames build/home campfire --moving "the flame (outer + inner flame with glow)" \
  --static "the two crossed logs" --motion "a gentle campfire flicker" --count 8 --fps 12
python3 $K frames build/home rain_drop --moving "the concentric ripple rings" --static "the teardrop" \
  --motion "ripples expanding outward and fading, a new ring appearing" --anchor center
python3 $K vectorize build/home campfire --moving "the outer flame and the inner flame" \
  --static "the two crossed logs" --motion "a lively flicker; the warm glow pulses" --keyframes 4 --seconds 1.2
python3 $K vectorize build/home library_heart --moving "the whole heart" \
  --motion "a like pop: squash, overshoot with a slight tilt, settle, rest" --strength strong   # animates svg/library_heart.svg
python3 $K vectorize build/home campfire --rebuild      # after YOU edit layers.json: rebuild Lottie + AVD, no API call
```
- **frames:**
  - Read the printed QA line and `frames_review.png`, where flagged frames are labelled.
  - `style-drift`: run `--reuse all --match-colors` (no API calls).
  - `colour`/`size`/`drift`/`pop`: run `--reuse all --reroll`, which redraws only those frames, 1 call each, kept only if better.
    Fades and growth can flag legitimate frames, so look first.
  - `loop-seam`: re-roll the sheet.
  - `--count 12 --cols 4` gives more frames. `--formats gif,webp,apng,lottie` and `--px 480` control the outputs; the
    image-sequence Lottie costs about 280 KB per 480 px frame.
  - AnimationDrawable decodes every frame into memory, so ship the WebP or Lottie for long loops.
- **vectorize:**
  - Each layer can morph, move (anchor, dx/dy, sx/sy, rotate, opacity, easing) or both; `glows` are radial-gradient halos.
  - Read `keys_review.png`, which shows the real Lottie rendered by lottie-web. It prints each layer's movement against the `--strength` threshold
    (subtle 1.5%, normal 4%, strong 8%) and retries once if motion is too subtle.
  - The AVD: copy `<name>_avd.xml` to `res/drawable/` and call `(drawable as Animatable).start()`. It loops on its own and needs
    API 23+ framework (use AppCompat `app:srcCompat` below that); gradients need API 24+.
  - To refine motion, edit `layers.json` yourself and `--rebuild`.

## Phase 6: build the UI

<!-- WAVE2: tokens fonts compare -->
Build from `assets.json` (positions), `design_spec.json` (tokens, `native` widget per component, `platform_style` per
text style) and the exported assets, using the platform's native components. Multiply px by `px_to_dp`, then snap to the
grid. Follow the project's own conventions. At the end, compare a screenshot of the built screen with the mockup.

## Rules learned from probing (don't relearn them)

- **Never text-only prompt an asset of an approved design.** It comes out as a generic, different style. Always go through
  `forge`/`fix`/`frames`, which pass the design as the reference.
- **Exactly one reference image per edit call.** Multi-image edits returned the mockup itself.
- A single-reference edit designs a genuinely new screen when the prompt says "DIFFERENT screen … style guide only … do not
  copy its layout" (5/5 new layouts). A design-system board as the reference is worse than the screen itself.
- Some backends ignore `quality`, `size` and `n`, so the kit trims, crops and resizes locally and runs candidates in parallel.
  Output aspect ratio follows the prompt's wording, so `mockup` states it and retries on a mismatch.
- `background: transparent` works on the gpt-image models, with real alpha, and alpha edges need no defringing.
- Forged art drifts in colour (yellower, more saturated). The Lab colour match fixes that for transparent assets; never apply
  it to opaque backgrounds, because it washes out their accents.
- For app-icon concepts, a full-mockup reference at high fidelity copies the hero illustration, so use low fidelity. Never
  ask the image model for the foreground layer; the local matte is far more faithful.
- For animation, never ask for the whole object in every frame (static parts drift, grid lines appear). Generate only
  the moving layer. Use `--procedural` for rigid motion.
- gpt-6 animates too subtly by default, and the kit's motion check retries. AVD `pathType` keyframes crash the inflater, so the
  kit relays multi-key morphs across path copies.
- Full measurements are in `references/api-findings.md`.
