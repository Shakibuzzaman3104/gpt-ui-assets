---
name: gpt-ui-assets
description: Research the market and generate a feature blueprint (gpt-6-sol, high reasoning, web-searched and cited), design platform-native app/web UIs with OpenAI image + vision models (gpt-image-2, gpt-6, or any OpenAI-compatible endpoint) and turn an APPROVED design into production assets redrawn from scratch 1:1 (never cropped from the screenshot) — icons as SVG/Android VectorDrawable, illustrations and backgrounds as clean PNG/WebP, launcher icons for Android/iOS/web, frame animations (including layout-driven sequences where GPT renders every frame of smoke, drops or spray), Lottie and AnimatedVectorDrawable — then build and verify the real UI. Use whenever the user wants market research, competitor analysis or a feature list/blueprint for an app idea, wants GPT/ChatGPT to design app screens, icons, images, illustrations or an app icon, approves a GPT design and says "go ahead"/"build it"/"give me the assets", asks for the assets "used in the UI", wants frames for an animated drawable, a Lottie or an animated vector, wants smoke, steam, rain, spray or embers animated in an icon's own style, or complains that GPT cropped assets out of a mockup.
license: MIT
metadata:
  version: 2.2.1
---

# GPT UI assets

**You are the driver.** GPT is one instrument among several. Route every job to whoever does it best, and never
accept "GPT can't" as the end of the road:

| Job | Best done by |
|---|---|
| Market research (web-searched, cited) and the feature blueprint | GPT reasoning (`gpt-6-sol`, high/xhigh effort) with web search (`research`, `blueprint`) |
| Screen mockups, painterly art, illustrations, photos, textures, raster frames | GPT image (`mockup`, `forge`, `frames`) |
| Asset inventory, design QA, icon-set style, vector keyframes | GPT vision (`manifest`, `critique`, `iconspec`, `vectorize`) |
| Flat/line icons as vectors | GPT vision writes SVG (`svg`, `icons`). If it stays weak after its retry, **you** write or fix the SVG by hand, then `svg --check` |
| Ranking, colour correction, safe zones, slicing, registration, QA | Local measurement in the kit. Trust numbers over the model's claims |
| Raster art → vectors, animatable parts | Local tracing and rigging (`trace`, `rig`, `animate`); GPT can optionally isolate parts (`rig --route hybrid`) |
| Lottie / AnimatedVectorDrawable / motion | `vectorize` builds and validates them; **you** refine `layers.json` and `--rebuild` |
| Exact colours, fonts, geometry, theme code, 1:1 check of the built screen | Measurement from pixels (`tokens`, `fonts`, `redline`, `compare`) plus you, the engineer |
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
  - The Android SDK's `aapt2` (via `ANDROID_HOME` or `~/Library/Android/sdk`) validates every VectorDrawable and adaptive icon.
  - `xcrun actool` validates the iOS app icon.
  - `pip install vtracer scipy fonttools` add an extra trace candidate, faster region labelling and kerning in the font
    matcher. Everything works without them.
  - `pip install opencv-python-headless` gives `sequence` its motion-direction, seam and shimmer QA; without it the QA
    falls back to plain frame differences.
  - `fonts` needs network access to Google Fonts once; after that it runs from `~/.cache/gptkit`.
- Never print or echo the API key.
- Image calls take 35–80 s and cost money or quota. Batch deliberately, run slow commands in the background, and run
  independent commands in parallel (e.g. `forge` and `svg`).

## Phase 0: market research and feature blueprint (before design)

```bash
python3 $K research "Android budgeting app for young couples who share expenses" --platform android \
  --region "US + UK" --depth standard --out research/couples     # plan -> parallel web research -> cited report
python3 $K blueprint research/couples                            # features, RICE, stories, screens, roadmap
```
- **`research`** runs `gpt-6-sol` at `--effort high` (the synthesis at `--synth-effort xhigh`):
  1. It plans 5, 10 or 16 questions (`--depth quick|standard|deep`) across market size and trends, competitors (pricing,
     ratings, features), real user pain points from app-store reviews and forums, unmet needs, monetization benchmarks,
     platform/regulatory constraints, ASO and differentiation.
  2. It researches each question in parallel with web search and cites every claim.
  3. It synthesises `report.md` and `report.json`, plus `findings/*.md` and `sources.json`.
- **Read `report.md` yourself before the blueprint.** Challenge weak claims: `findings/` shows the raw evidence per question,
  and "could not verify" notes are deliberate. Re-run a narrower `research` with `--context` to fill a gap.
- **`blueprint`** turns the report into:
  - positioning, personas and jobs-to-be-done;
  - features with MoSCoW priority and RICE (the score is recomputed locally), each citing the pain or gap it solves;
  - user stories with acceptance criteria, flows, monetization, analytics events, a roadmap and non-goals;
  - a **screen inventory with a ready `mockup_brief` for each screen** (`screens.json`).
- Walk the user through `blueprint.md`, especially the MVP cut, then design each approved screen:
  `python3 $K mockup "<mockup_brief>" --platform <platform> --out design/<screen>`. Use `--style-ref` for screen 2 onward.
- Timing: the quick depth takes several minutes and deep can take 15+. Run it in the background. Each question is one
  web-searched reasoning call.

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
  - Vision QA is weak at spatial collisions: it can miss a FAB, snackbar or mini player covering content. **Check
    floating elements yourself**, and fix them with `--edit` (e.g. "move the FAB above the last row").
  - Don't chase absolute text sizes in the mockup. gpt-image draws everything small, and `spec` calibrates the real scale.

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
- **Scale.** GPT draws native components at about 0.65–0.7× platform size, for example a 56 dp nav bar where M3 uses 80 dp. `spec`
  measures its native anchors (the M3 navigation bar, FAB, top app bar and search bar; the iOS switch and segmented control),
  calibrates `px_to_dp` from them and rewrites the asset `dp` estimates. It prints that the design depicts a ~N dp-wide screen
  (`effective_width_dp`). All later commands (`tokens`, `fonts`, `redline`, `compare`) use the calibrated scale.
  - `spec --scale design` keeps the proportional width scale.
  - `spec DIR --recalibrate` re-runs the calibration on an existing spec with no API call.
  - Still snap to platform norms (24 dp icons, 48 dp / 44 pt touch targets, an 8 dp grid).
- `spec` maps each component to the native widget to build it with (`native`) and each text style to the platform type
  scale (`platform_style`). Build with those.

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

**Vector versions of raster art (optional).** GPT only writes reliable SVG for simple glyphs. For anything forged as a PNG, trace it locally:
```bash
python3 $K trace build/home/assets/*.png --out build/home/traced      # SVG + img_*.xml (VD) + *.layers.json + review/
```
- `--preset auto` (the default) traces with the flat, illustration and gradient presets (plus vtracer if installed). It renders
  each one at the source size with alpha and keeps the smallest whose display-size diff is within 10% of the best.
- Gradients are fitted per region, linear or radial, so VectorDrawable can express them. Glows are rebuilt as stacked translucent bands.
- **Read `review/<name>_trace.png`** (source | vector | diff heatmap), then follow `advice.prefer` in `<name>.trace.json`:
  - `vd`: use `img_<name>.xml`. It has already passed aapt2 and been rendered back. Gradients need API 24+ or AndroidX.
  - `svg/lottie`: the vector is good, but too big or heavy for a VectorDrawable (Android guidance is ≤200 dp and simple paths).
  - `png/webp`: tracing loses too much (photos, textures, star fields), so keep the raster and use `export`.
- Typical display-size diff: icons ≤0.5, illustrations 1.5–2.5, photographic backgrounds 7+ (keep those raster).

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
    Reject these flags:
    - `BUSY-BACKGROUND`: a scene, not an icon.
    - `COPIED-ILLUSTRATION:<asset>`: the concept is the screen's own illustration re-posed. This needs `assets.json` next to
      the mockup. Re-run with a stronger `--directions` ("a minimal emblem of X, not the mascot"), `--crop` a small motif,
      or drop `--mockup` and pass `--colors`.
    - `SUBJECT-TOO-BIG` and `BLEEDS-OFF-EDGE`.

    Lower `palette_de` is closer to the
    design. Let the user pick.
- **Build** defaults are the measured best:
  - The foreground is matted locally against the background that actually ships, so the icon recomposes the concept almost
    exactly. `--separate auto` (the default) switches to an image-edit foreground (1 call) when the matte leaves the subject
    mostly translucent, which happens with a light subject on a light background. Otherwise iOS dark/tinted and the silhouette would look ghosted.
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
| `frames --moving … --static …` | the art changes shape **in place** (flame, ripples, blinking mascot) | registered frames, WebP/APNG/image-sequence Lottie, `animation-list` XML |
| `sequence` (layout → render → propagate or build → reroll) | something **travels**: smoke, steam, falling drops, spray, embers, leaves | a seamless loop where code decides the motion and GPT paints the frames (keys + EbSynth, or every frame); animated WebP, frames, QA report |
| `vectorize` | GPT-authored vector motion: morphs, transforms, glows | Lottie **and** looping AnimatedVectorDrawable |
| `rig` + `animate` | the exact traced art must move (1:1 with the design, glow included); **no GPT needed** | layered Lottie with procedural keyframes |

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
  - `misregistered` (registration IoU < 0.3): the composites are wrong, usually because `static.png` still contains part of the
    moving element. Check it and describe `--moving`/`--static` more precisely, or run `--reuse static`. `moving_fNN.png`
    are still usable on their own.
  - `--procedural` moves the whole asset, including a baked ground shadow. Forge the shadow as its own asset, or `rig` the
    asset, when the shadow must stay put.
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

**Layout-driven sequences: code animates, GPT paints, EbSynth carries the paint.** An image model has no notion of
time. A `frames` sheet keeps the style but not the trajectory: drops jump between cells, the floor moves from row to
row, cells come back empty. So for anything that travels, decide the motion in code, have GPT paint keys from that
layout, and let EbSynth carry each key's real texture along the layout to every frame.
```bash
# 1. layout: a periodic physics plume (or --guides DIR: one guide PNG per frame, from any source)
python3 $K sequence build/player train_smoke layout --base train_body.png --sprites puff1.png,puff2.png \
  --vent 190,158 --loop 3 --fps 24 --chuff 0.75 --life 5.4 --velocity 0,-85 --drag 1.5 --wind 45,-3 --buoyancy -10 \
  --size 96 --grow 0.45,1.5
# 2. render the keys: one GPT edit per key from its guide (resumable, parallel, retries); every 6th of 72 frames = 12 calls
python3 $K sequence build/player train_smoke render --every 6 --moving "The smoke" --locked "The toy steam train" \
  --model gpt-image-2.5-sunburst
# 3. propagate: EbSynth synthesises all 72 frames from the two nearest keys, then WebP + preview + QA
python3 $K sequence build/player train_smoke propagate --every 6 --ref drawn_smoke.png --below 175
```
- **`propagate` is the default.** It needs the `ebsynth` binary (public domain, CPU is enough). Build it once with
  `git clone https://github.com/jamriska/ebsynth ~/.cache/gptkit/ebsynth && cd ~/.cache/gptkit/ebsynth && sh
  build-macos-cpu_only.sh` (`build-linux-cpu_only.sh` on Linux), or pass `--ebsynth PATH`.
  - Every frame, keys included, is synthesised from its two nearest keys. The appearance guide is the layout; plumes
    also get a positional guide that maps each pixel to where its puff was at the key. The two syntheses are
    cross-faded, and the result is clipped to where the layout has content.
  - Each output pixel is a patch of a real render, placed by the layout. That is why it neither stretches like a warp
    nor shimmers like independent frames.
  - About 7 minutes of CPU for 72 frames. `--every` must divide the frame count.
- **Every frame by GPT** (`render` without `--every`, then `build`): 72 calls instead of 12, and the frames shimmer.
  Keep it for when EbSynth is unavailable, and fix its outliers with `reroll` (2 candidates each, the best neighbour
  fit wins) before building again.
- **Look at `layout_review.png` before rendering.** The motion is decided there; GPT only adds the finish. Keys every
  quarter-second of a 3 s loop are 12 image calls (72 without `--every`), so get the motion right first.
- **The layout must loop.** `--loop` has to be a whole number of `--chuff` intervals; the plume's jitter repeats every
  loop, so frame 72 is frame 0. Custom guides from `--guides` need the same property.
- **Get the physics right, because that is what the viewer reads.**
  - Smoke leaves a chimney fast, is braked hard by the air (drag), then rises slowly (buoyancy) and is carried by the
    air (`--wind`). A moving engine's slipstream pushes it back more than buoyancy lifts it.
  - Drops swell out of the surface above them before they let go, then accelerate and splash.
  - Everything that goes out thins and fades; nothing reverses.
  - Err slow for calm contexts: on device a 68 u/s drift with 3.6 s puffs was judged too fast; the next render used
    45 u/s with 5.4 s puffs.
- **Keep emitted elements distinct.** GPT tracks separate, spaced elements (puffs a chuff apart). A dense stream reads
  as one generic plume, and GPT paints the same plume in every key, so the texture never travels: a car exhaust with
  about 9 overlapping puffs followed its layout 0.8%, while the same exhaust as distinct puffs followed it 16%.
- **The emitted element must be drawn in the art's own style.** Reusing another icon (a cloud icon for smoke) reads as
  wrong at once. Have GPT draw the isolated element, with a crop of the art's own element as the only reference.
- **Carry paint, don't warp pixels.** Optical-flow in-betweens between GPT keys stall and surge, because each key sits a
  little ahead of or behind its layout (it reads as the plume moving back and forth). Warping keys along the layout's
  own motion fixes the direction but leaves visible warp and cross-dissolve artefacts. EbSynth's patch synthesis,
  guided by the layout, has neither problem. Measured on the train smoke:

  | Route | Steps backwards | Shimmer |
  |---|---|---|
  | 72 GPT frames | 3 | 9.4 |
  | EbSynth, keys every 0.5 s | 2 | 5.6 |
  | EbSynth, keys every 0.25 s | 0 | 6.0 |

  Use keys every quarter second: the half-second keys stalled briefly mid-interval.
- **Read the QA line.**
  - `backwards` should be 0, and `seam` should be close to `median`.
  - `shimmer` is the texture change left after the motion is compensated. It is the open cost of rendering frames
    independently (about 9–10 grey levels on smoke), so compare it against your previous run rather than against zero.
  - `follows` (plumes) is the share of the frame-to-frame change that disappears when each frame is carried along the
    layout's exact motion. Measured: about 15–17% for a texture that travels with its plume, about 40% for a perfect
    render, and about 0 for one that boils in place (which `shimmer` and `backwards` do not catch). Below 5% the
    build says so.
  - `outliers` are frames that disagree with both neighbours, usually a render that ignored its layout. `reroll` renders
    candidates and keeps whichever fits its neighbours best, the current frame included. After `propagate`, fix an
    outlier through its nearest key instead: delete that key's `raw/NNN.png`, run `render --every 6` again (it renders
    only missing keys), then `propagate`.
- **Colour-match every frame (or key), not the loop once** (`--ref`, the approved picture of the moving part). Each render has its
  own exposure, and pulling each onto the reference evens that out: measured shimmer 9.4 per frame vs 10.5 for one shared
  transform and 9.7 with none.
- Ship the WebP as a frame sequence played on your own clock, and cross-fade the still art's drawn element into it, so
  play and pause never jump. `report.json` has the crop rect on the base canvas.
- If a video model with a first and last frame is available (Seedance, Veo, Kling through OpenRouter or Replicate), it is
  temporally coherent without a layout. Animate the approved still with first frame = last frame, matte
  against a known flat backing, and encode at the source frame rate. Per-frame rendering is the route when only image
  models are available.

**Rig the exact art into parts (vector, 1:1).** `vectorize` redraws the art in simplified form: keyframe 0 of GPT's
campfire scored 10.2 against the design, while a traced rig scores 2.3. When fidelity matters, rig the traced asset instead:
```bash
python3 $K rig build/home cozy_campfire --namer none \
  --parts "outer_flame=the outer flame; inner_flame=the inner flame; logs=the two crossed logs"   # bottom -> top
# Read rig/<name>/regions.png (numbered regions + connected groups c1..cN), then assign and animate:
python3 $K rig build/home cozy_campfire --parts "..." --assign "outer_flame=1,17,19; inner_flame=2,27; logs=rest" \
  --animate "outer_flame=flicker; inner_flame=flicker:1.3:0.35"          # glow_<part> follows its part
python3 $K animate build/home/traced/rain_drop.layers.json --part "art=squash:0.8; glow=squash:0.8@art"
```
- You can name the regions yourself from `regions.png`. Otherwise, omit `--assign` to have one vision call name them (98% agreement in
  testing), or use `--route hybrid`, where GPT isolates each part (1 image call per part) and the tool registers it back.
- Every part is extended under the parts above it, so moving parts never open gaps. Check `rig_preview.png`: the recomposite
  should match the source.
- Motions: sway, flicker, breathe, squash, bob, wave and ripple. The syntax is `part=motion[:amp[:phase]][@anchor]`; `@anchor` shares
  another layer's motion field. Frame 0 is always the approved design.
- **Read `anim/lottie_frames.png`** (rendered by lottie-web). In `animation_report.json`, `tear_px` should be about 0.
- Traced rigs make 150–250 KB Lotties; `--min-detail 0.02 --keyframes 6` shrinks them.

## Phase 6: build the UI 1:1 (measure, don't guess)

GPT's spec is a list of estimates. Replace them with measurements from the pixels before writing UI code:
```bash
python3 $K fonts   build/home     # Google Fonts match by rendering + measuring (network once; cached)
python3 $K redline build/home     # real edges, corner radii, insets and gaps in dp
python3 $K tokens  build/home --package com.acme.app.ui.theme --prefix Acme [--author "Name"] --verify
```
- **`fonts`** writes TTFs, Android `res/font` + font-family XML, `ios/UIAppFonts.plist`, `web/fonts.css`, `licenses/` and `fonts.json`
  (the fitted size, weight and tracking for each style).
  - **Read `fonts_review.png`.**
  - On GPT-drawn text, sans families often tie within 0.01; the tool says so, keeps the spec's family when it's close, and
    `--family` overrides it.
  - `! Style: fits X poorly; Y fits much better` means that style may really use a second family.
- **`redline`** writes surface boxes, corner radii (rect, pill or circle), insets and gaps in dp. Use these numbers for padding,
  spacing and shapes.
- **`tokens`** writes Compose `Color.kt`/`Type.kt`/`Shape.kt`/`Dimens.kt`, a SwiftUI tokens file, `web/tokens.css` and Tailwind config.
  - Every palette hex is first snapped to the mockup's pixels; `palette_check.png` shows the guess, the snapped value and the ΔE00.
  - `--verify` type-checks the Swift, loads the Tailwind config and parses the CSS.
  - If `tokens`/`fonts` print "missing: …", add `--vision` (one cached call to locate the text).

Build the screen with the platform's **native components**, using `spec`'s `native` widget per component and `platform_style` per
text style, plus these files and the exported assets. Follow the project's conventions. Then close the loop:
```bash
# screenshot the built screen (emulator/simulator/device, or headless Chrome at the target width), then:
python3 $K compare build/home shot.png [--vision] [--crop-top PX] [--crop-bottom PX]
```
- `compare` writes `compare/report.json`, `heatmap.png` (design | aligned implementation | ΔE heatmap with the worst regions boxed)
  and a ranked list of concrete fixes, for example:
  - `'Sleep Stories' 7% smaller (try 31.1sp)`
  - `'Breathe' looks like Nunito 400 rather than Lora 400`
  - `fill colour off by dE 4.8 (#A69BF7 -> #9A8BE6)`
  - `shifted +12dp vertically`
  - `element … missing`
- It detects a status bar and a taller screen automatically. "Differs inside the artwork" regions come from the forged assets, not
  your code. `--vision` adds qualitative descriptions; trust the measured numbers for sizes and positions.
- **Screenshot at the design's effective width** (`effective_width_dp`, printed by `spec` and `compare`) for a 1:1 check.
  Use a Compose `@Preview(widthDp = N)` rendered with Paparazzi/Roborazzi, an emulator display override, or a Chrome window N
  CSS px wide. A narrower real device reflows the layout, and `compare` will report that as differences.
- Fix, re-screenshot and re-run until the only remaining items are ones you accept.
- **Custom lettering** (a wordmark or handwritten caption) that `fonts` reports as having no reliable match: forge it as a PNG asset and
  `trace` it, or `svg` it, and ship it as a VectorDrawable rather than forcing a font.
- `redline` misses very low-contrast surfaces (white cards on cream, about ΔE 1.5). Take those from `spec`.

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
  ask the image model for the foreground layer when the subject contrasts with its background, because the local matte is far more
  faithful. For light-on-light concepts the kit switches to the image-edit foreground itself.
- For animation, never ask for the whole object in every frame (static parts drift, grid lines appear). Generate only
  the moving layer. Use `--procedural` for rigid motion.
- **GPT is the rendering artist, not the animator.** It cannot keep a trajectory across a sprite sheet. For anything that
  travels, lay the motion out in code and have GPT render keys from their guides with a brief that names the frame
  ("frame 10 of 72 of a seamless 3-second loop"), locks the static art and makes the layout binding. It kept the
  layout on every one of 12 keys, and on 67 of 72 frames on the first pass (the other 5 were fixed by `reroll`).
- Never write "keep its exact shape" about the moving part. 3dicon found it reads as an instruction to stay still (a
  video model returned the input 122 times). Lock the camera, the framing and the static art instead.
- To move a part that hides something (fan blades, a bird on a branch), have GPT redraw the base with the part removed
  and repaired, and keep the original pixels everywhere outside the part. Recomposite at rest and compare with the
  master before animating: a mean difference of about 0.2 when the part keeps its own pixels, about 3 when GPT had to
  redraw the part as well (the fan's blades).
- gpt-6 animates too subtly by default, and the kit's motion check retries. AVD `pathType` keyframes crash the inflater, so the
  kit relays multi-key morphs across path copies.
- Vision hex guesses drift (ΔE up to 5 on text ink and 10 on borders). `tokens` snaps them to pixels. Measure; never ask vision for numbers.
- Full measurements are in `references/api-findings.md`.
