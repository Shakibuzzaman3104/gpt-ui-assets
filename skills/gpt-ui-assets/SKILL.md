---
name: gpt-ui-assets
description: Research the market and generate a feature blueprint (gpt-6-sol, high reasoning, web-searched and cited), design platform-native app/web UIs with OpenAI image + vision models (gpt-image-2, gpt-6, or any OpenAI-compatible endpoint) and turn an APPROVED design into production assets redrawn from scratch 1:1 (never cropped from the screenshot) — icons as SVG/Android VectorDrawable, illustrations and backgrounds as clean PNG/WebP, launcher icons for Android/iOS/web, frame animations (including layout-driven sequences where GPT renders every frame of smoke, drops or spray), Lottie and AnimatedVectorDrawable — then build and verify the real UI. Use whenever the user wants market research, competitor analysis or a feature list/blueprint for an app idea, wants GPT/ChatGPT to design app screens, icons, images, illustrations or an app icon, approves a GPT design and says "go ahead"/"build it"/"give me the assets", asks for the assets "used in the UI", wants frames for an animated drawable, a Lottie or an animated vector, wants smoke, steam, rain, spray or embers animated in an icon's own style, or complains that GPT cropped assets out of a mockup.
license: MIT
metadata:
  version: 3.0.0
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
| Lottie / AnimatedVectorDrawable / motion | `vectorize` builds, validates (schema, aapt2, Lottie portability) and previews them; **you** refine `layers.json` and `--rebuild` |
| Exact colours, fonts, geometry, theme code, 1:1 check of the built screen | Measurement from pixels (`tokens`, `fonts`, `redline`, `compare`) plus you, the engineer |
| Anything a human must judge | You: Read every review sheet before handing anything over |

`K` below is `scripts/gptkit.py` inside this skill's directory:
```bash
K="<this skill's directory>/scripts/gptkit.py"      # every command: python3 $K <cmd> -h
```

## Setup (once per machine)

```bash
pip install -r "<this skill's directory>/scripts/requirements.txt"    # requests, pillow, numpy (+ optional extras)
python3 $K doctor                                                      # key, endpoint, models, SVG renderer, aapt2, EbSynth, caches
```
- **Config** comes from env vars or `~/.config/gptkit/env` (`KEY=value` lines):
  - `GPTKIT_API_KEY` (or `OPENAI_API_KEY`), required.
  - `GPTKIT_BASE_URL` (or `OPENAI_BASE_URL`), default `https://api.openai.com/v1`. OpenAI-compatible gateways work when they
    mirror `/images/generations` and `/images/edits` (multipart) and `/responses` including `reasoning`, the `web_search`
    tool and strict `json_schema` (LiteLLM does; the kit's measurements were made on a `sub2api` gateway). Azure OpenAI
    (`api-key` header, deployment URLs) and OpenRouter's native image API are not supported without changes. When a
    gateway rejects an optional field (`input_fidelity`, `quality`, `background`, `reasoning`, `tools`, strict schema)
    the kit says so once and retries without it.
  - `GPTKIT_IMAGE_MODEL` (default `gpt-image-2`) and `GPTKIT_TEXT_MODEL` (default `gpt-6`, which needs vision and structured outputs).
    `gpt-6` is an alias on some gateways; the official API lists `gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra` and `gpt-6.1-sol`; the `gpt-5.x` families (5.2 to 5.6) also work as text models. `doctor` shows
    what your endpoint lists: if it says `gpt-6: NOT LISTED`, set `GPTKIT_TEXT_MODEL=gpt-6-luna` (or `-sol`, `gpt-6.1-sol`). Image models: `gpt-image-1`, `-1.5`, `-2`, `-2.5-flare`, `-2.5-sunburst`. `gpt-image-1`
    retires on 2026-10-23: prefer `gpt-image-1.5` or `gpt-image-2`.
  - **Which image model for which job** (researched 2026-10-01 from OpenAI's 2.5 announcement and third-party API write-ups;
    the OpenAI docs host was unreachable from the sandbox, so confirm prices there). The kit applies this automatically:
    `generate` calls use the **concept** chain (Flare, Sunburst, 2, 1.5) and `edit` calls the **precise** chain (Sunburst, Flare, 2, 1.5);
    transparent requests push `gpt-image-2` to the end; the first model your endpoint lists wins. `--model` or a set
    `GPTKIT_IMAGE_MODEL` overrides it (`doctor` shows which mode is active):

    | Job | Model | Why |
    |---|---|---|
    | Default: mockups, backgrounds, illustrations, app-icon concepts, forge candidates | `gpt-image-2.5-flare` | Quality comparable to `gpt-image-2` at ~47% lower latency and the same token bill; supports `background: transparent`. |
    | Edits that must stay consistent: `sequence` frames, `frames` sheets, multi-reference edits, art with embedded text, final hero assets | `gpt-image-2.5-sunburst` | Highest quality, tuned for precise edits and consistency (still ~26% faster than `gpt-image-2`). Used for the layout-driven sequence measurements. |
    | Transparent PNG/WebP assets (forge, icons, sprites) | `gpt-image-2.5-flare` or `-sunburst`; else `gpt-image-1.5` | OpenAI documents `gpt-image-2` as accepting only `auto`/`opaque`, while 1.5 and 2.5 accept `transparent`. The kit saw real alpha from `gpt-image-2` on one gateway, but never rely on it: the kit errors with the model name when a transparent request returns opaque. |
    | Endpoint lists no 2.5 model | `gpt-image-2` for opaque art, `gpt-image-1.5` when you need transparency | 2 has the best text rendering of the older models; 1.5 is cheaper at medium/high quality and keeps `input_fidelity`. |
    | Draft or throwaway iterations | the chosen model at `quality: low` | Do not downgrade the model; low quality already costs ~$0.006 per 1024² image on 2/2.5. |
    | Never | `gpt-image-1`, `gpt-4o-*`, `codex-*` | `gpt-image-1` shuts down 2026-10-23; the others are not image generators for this kit. |

    Rules of thumb: a failed `transparent` call is a model problem (switch to 2.5 or 1.5), not a prompt problem; if a
    `sequence` or `frames` render drifts, move Flare to Sunburst before changing the prompt; try the other model on one asset (`--model`) and keep the
    better `score` before switching a whole project. In the kit's own campfire test all four models scored within
    0.3 fid of each other on text-only generation, so the extra cost of Sunburst only pays off in edits.
  - `GPTKIT_RETRIES` (default 2) retries a call on 408/429/5xx and network errors, honouring `Retry-After`; quota/billing 429s
    are not retried. Every failure is one line, `gptkit <cmd>: HTTP 400 /images/edits [moderation_blocked]: ... | prompt: '...'`;
    `GPTKIT_DEBUG=1` shows the traceback instead. `GPTKIT_MODERATION=low` relaxes image moderation where the org allows it.
- **Optional tools**, each used when present:
  - Headless Chrome/Chromium renders SVG with real alpha and plays Lottie through lottie-web (`GPTKIT_CHROME` sets its path).
    As root (containers, CI) Chrome needs `--no-sandbox`; the kit adds it automatically, or set `GPTKIT_CHROME_NO_SANDBOX=1`.
    `doctor` actually renders a probe page and prints `chrome (ok)` or Chrome's error line.
  - The Android SDK's `aapt2` (via `ANDROID_HOME` or `~/Library/Android/sdk`) compiles every VectorDrawable and adaptive icon
    and, with `platforms/android-*/android.jar` installed (`sdkmanager "platforms;android-35"`), links it (attribute names,
    enums, dimensions); without the jar the kit's offline VectorDrawable lint covers attribute names/enums. `doctor` says
    which level you have and the kit never prints "valid" for a compile-only run.
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
- **`research`** runs `gpt-6-sol` at `--effort high` (the synthesis at `--synth-effort xhigh`; `max` exists for both and is
  slower):
  1. It plans 5, 10 or 16 questions (`--depth quick|standard|deep`) across market size and trends, competitors (pricing,
     ratings, features), real user pain points from app-store reviews and forums, unmet needs, monetization benchmarks,
     platform/regulatory constraints, ASO and differentiation.
  2. It researches each question in parallel with web search and cites every claim.
  3. It synthesises `report.md` and `report.json`, plus `findings/*.md` and `sources.json`.
- **Read `report.md` yourself before the blueprint.** Challenge weak claims: `findings/` shows the raw evidence per question,
  and "could not verify" notes are deliberate. Re-run a narrower `research` with `--context` to fill a gap.
- A refusal, a truncated answer (`incomplete: max_output_tokens`) or a gateway that ignores strict `json_schema` stops the
  command with one line naming the cause and the missing key; lower `--effort`, shorten `--context`, or try `--model gpt-6-sol`
  on another gateway. `blueprint` refuses a `report.json` that is corrupt or missing sections.
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
  - The backend picks the aspect from the prompt's wording; `mockup` retries once on a mismatch and then saves the image
    with a `! variant N: backend returned WxH ...` warning. `--edit` / `--style-ref` images are sent downsized to a
    1536 px long side.
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
`manifest`, `spec` and `iconspec` take `--model` (default `GPTKIT_TEXT_MODEL`). Model bboxes are clipped to the mockup
before anything is written: `manifest` drops (and prints) assets missing a required field or lying mostly outside the
image, `spec` drops components with no box inside it, and `iconspec` names the icons the model did not describe.

**Review `manifest_overlay.png` yourself before generating anything.** Red boxes are PNG assets and green boxes are SVG.
Edit `assets.json` directly to fix the inventory, then run `spec` again (missing fields get defaults and are printed):
- Remove anything code or the OS draws: native components (the tab bar itself, switches, chips, buttons, sheets),
  solid or gradient shapes, cards, dividers, text, shadows, and system UI (status bar, home indicator). `manifest` already
  excludes these and drops system glyphs it finds in the status-bar strip.
- Split merged graphics, merge duplicates, and tighten loose bboxes.
- Set `format` to `svg` for flat, line, monochrome or simple-gradient glyphs, and to `png` for painterly, glowing, 3D, photographic or detailed art.
- Give art behind UI `kind: background` and `transparent: false`. It gets redrawn with the UI removed.
- `material_symbol` / `sf_symbol` name the equivalent platform glyph. Ship the generated SVG (it is 1:1 with the
  design) unless the user asks for platform icons.
- **Scale.** GPT draws native components at about 0.65–0.7× platform size, for example an 80 dp M3 navigation bar drawn
  at ~56 dp-equivalent on the proportional scale. `spec` measures its native anchors (M3: navigation bar 80 dp, FAB 56 dp,
  small top app bar 64 dp, search bar 56 dp, navigation rail 80 dp wide; iOS: tab bar 49 pt, inline navigation bar 44 pt,
  switch 31 pt, segmented control 32 pt; a large-title bar is never an anchor), calibrates `px_to_dp` from them and
  rewrites the asset `dp` estimates. It prints that the design depicts a ~N dp-wide screen
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
- It sends a **focused crop of the element** (60% margin, upscaled, padded with the design's backdrop colour so the target is
  always centred) as the single reference with a "recreate from scratch, 1:1" prompt. Candidate 2 adds a colour emphasis and
  candidate 3 a silhouette emphasis; `--same-prompt` turns this off. `--candidates N`/`--jobs N` must be ≥ 1.
- Backgrounds (`kind: background` or `kind: photo` with `transparent: false`) are redrawn opaque from a tight crop of their own region.
  `format: svg` assets are skipped unless `--include-svg` (raster redraw of an icon; normally run `svg` instead).
- `fid` ranks candidates; lower is better. It combines silhouette `iou` (after alignment), CIELAB `de`, colour
  distribution `col` and pixel `diff`. Indicative bands, per asset type: PNG illustrations about 16–24 close, 30+ visibly
  off, 45+ a different style; backgrounds 15–22; correct SVG icons have scored anywhere from 9 to 40, so judge icons by eye.
- `flags` outrank `fid`: candidates are sorted by the sum of flag weights first (`EMPTY` 1000, `COPY:*` 100, `OPAQUE` 10,
  `BACKDROP` 10, `CUT-OFF` 3), then by `fid`. An `EMPTY` or `COPY` winner is never written to `assets/`; the run prints
  `! <name>: no usable candidate` with the re-run command and exits 1.
  - `EMPTY`: nothing (or < 0.2% of the canvas) is opaque — a blank render.
  - `COPY:*`: GPT returned the screenshot or the reference crop.
  - `OPAQUE`: transparency was lost (a gateway that returns JPEG prints `! provider returned JPEG` once).
  - `BACKDROP`: more than half of the silhouette's outline has the colour of the design's surroundings, i.e. it kept the
    circle or card behind the icon. A light icon on a light page is not flagged.
  - `CUT-OFF`: the content touches the canvas edge.
- Warnings: `warn: silhouette` (iou < 0.5) means the shape is wrong. `warn: soft` is normal for smooth gradients.
- The best **unflagged** transparent candidate is then **colour-matched** to the design: a Lab transfer at strength 0.75
  (`--color-match STRENGTH`, 0–1), kept only if `fid` improves without adding a flag. The untouched copy stays in
  `raw/<name>_cN.png`. `--no-color-match` keeps the raw colours. Flagged candidates and opaque backgrounds are never matched
  (the skip reason is printed).
- Failures are per candidate (`! flame c2 failed: ...`); the other assets finish. The end line `forged K/N assets; failed: a, b`
  names the re-run, and `NEEDS ATTENTION` lists assets whose every candidate carried a flag: a later `forge DIR` without
  `--force` re-queues exactly those (plus assets without a final file); Ctrl-C keeps the scored candidates and the next run
  draws only the missing ones. `forge_report.json` stores dir-relative paths, so `fix`/`pick` work from any cwd and after moving the dir.
- **Always Read `review/forge_*.png`.** Each tile is labelled `cN fid=.. FLAGS` (N = the candidate number, `BEST` marks the
  chosen one; tiny design elements are shown enlarged with the bbox outlined). The metric ranks candidates, but your eyes decide. Then:
  ```bash
  python3 $K pick  build/home rain_drop 2                                 # choose tile c2 (or pass any PNG path)
  python3 $K fix   build/home campfire --note "logs should be thinner"    # vision diff + one corrective edit (--rounds N)
  python3 $K forge build/home --only campfire --force --candidates 3      # or re-roll
  python3 $K score build/home campfire some.png other.svg                 # JSON score of any file vs the design (no API)
  ```
  `fix` keeps a result only if `fid` improves, because correction passes often over-correct; it stops without an edit when the
  vision pass finds no difference. Pass `--accept` when the user asked for that exact change (an `EMPTY`/`COPY` result is still
  refused). After 2 unsuccessful rounds, tell the user what still differs rather than looping.

**svg / icons**
- Each icon prints `rel`: 0 is perfect and 1 is as bad as drawing nothing. Expect about 0.1 for filled glyphs, 0.2–0.35 for close
  thin strokes and about 0.4 on glows. It also prints `fit`, `ink` (above 1 means heavier than the design), `dE` and `!` issues. Weak or
  non-VectorDrawable-safe icons get one vision retry against a design|render side-by-side, and the better one is kept
  (`svg/raw/<name>_tryN.svg`). Scoring needs an SVG renderer (headless Chrome; as root the kit adds `--no-sandbox` itself):
  `doctor` shows `svg render`, and without one every icon prints `not scored` instead of a fake score.
- An asset the model omits is asked for once more; if it is still missing the run prints `! <name>: model returned no SVG`,
  records `{"error": "missing"}` in `svg_report.json` and exits 1 (re-run `svg DIR --only <name>`). One failed API batch is
  reported with its asset names and the other batches are still written. Files the model renames or invents are not written.
- Read `review/svg_*.png`, which shows design | SVG on the same backdrop. If an issue survives the retry, **fix the SVG yourself**
  (you can read the design crop and write paths), then run `svg --check` (a hand-edit with broken path data is reported as
  `unreadable path data`, not a crash). Icons end up on a 24x24 viewBox with transforms baked into the coordinates and no
  `%` units; icons of one size group (design sizes within 1.25x) share ONE scale: the largest fills the 2..22 live area and the
  smaller ones stay proportionally smaller, so one stroke weight serves the group. Each icon prints `use at Ndp` and that dp is
  saved per icon in `svg_report.json`; `vd` reads it when `--dp` is not given. An icon that cannot be normalised (nested
  `<svg>`/`<use>`, non-uniform scale on strokes) is kept as drawn with a `~ kept as drawn` note; `--grid 0` keeps the model's viewBox.
- `icons` measures the set's stroke, caps, joins and colour from existing SVGs, so run `svg` first (unreadable existing icons are
  named and left out). Read `icons/preview.png`: the new row must look like the same family. `--snap` forces the set's stroke on
  every stroke width (attribute, `style` and the inherited default), `--treatment filled` draws filled variants. Concepts come back
  under their slug (`sleep timer` -> `sleep_timer.svg`); a concept the model skips twice is printed as `! missing:`, a malformed one
  goes to `icons/raw/` only, and one that converts to VectorDrawable with an error is written but `kept out of --vd`; a VD that
  fails its lint or aapt2 makes `icons` exit 1 (the SVGs and the XML are still written).

**Vector versions of raster art (optional).** GPT only writes reliable SVG for simple glyphs. For anything forged as a PNG, trace it locally:
```bash
python3 $K trace build/home/assets/*.png --out build/home/traced      # SVG + img_*.xml (VD) + *.layers.json + review/
```
- `--preset auto` (the default) traces with the flat, illustration and gradient presets (plus vtracer if installed). It renders
  each one at the source size with alpha and keeps the smallest whose display-size diff is within 10% of the best.
- Gradients are fitted per region, linear or radial, so VectorDrawable can express them. Glows are rebuilt as stacked translucent bands.
- **Read `review/<name>_trace.png`** (source | vector | diff heatmap), then follow `advice.prefer` in `<name>.trace.json`:
  - Without a working headless Chrome there is no `review/*_trace.png`, only one preset is traced and `advice.prefer` is
    `unknown` with the reason (as root, Chrome needs `--no-sandbox`; the kit adds it).
  - `vd`: use `img_<name>.xml` (the file name is sanitised to `[a-z0-9_]`). `trace.json` `vd.aapt2` says what ran:
    `compile+link ok` only with `platforms/android-*/android.jar`; `compile ok; link skipped` means attribute names, enums and
    dimensions were not checked by aapt2 (the offline lint's findings are listed under `vd.lint`). `vs_svg_diff` is the kit's
    own VD->SVG converter rendered by Chrome, not Android. Gradients need API 24+ or AndroidX.
  - `svg/lottie`: the vector is good, but too big or heavy for a VectorDrawable (Android guidance is ≤200 dp and simple paths).
  - `png/webp`: tracing loses too much (photos, textures, star fields), so keep the raster and use `export`.
- Typical display-size diff: icons ≤0.5, illustrations 1.5–2.5, photographic backgrounds 7+ (keep those raster); measured with
  the padded Chrome capture (on Linux Chromium 141 an unpadded capture under-reports by 87 px of content).

## Phase 4: platform export

```bash
python3 $K vd build/home/svg/*.svg --out app/src/main/res/drawable   # SVG -> VectorDrawable (ic_ prefix); without --dp it uses the per-icon dp `svg` recorded in svg_report.json; pass --dp to override
python3 $K export build/home/assets/campfire.png --dp 48 --res app/src/main/res --name campfire   # webp per density
python3 $K export <png> --dp 44  --platform ios --res App/Assets.xcassets          # @1x/@2x/@3x imageset
python3 $K export <png> --dp 320 --platform web --res public/img                   # 1x/2x webp
```
`export`:
- Android names are sanitised to `[a-z0-9_]` starting with a letter (`My Icon-2` -> `my_icon_2`, `9lives` -> `img_9lives`;
  printed), because AGP rejects anything else. Web names become URL-safe (`[A-Za-z0-9._-]`, `My Icon-2` -> `My-Icon-2`:
  a space would split the `srcset`), iOS keeps the name except path separators.
- Densities mdpi/hdpi/xhdpi/xxhdpi/xxxhdpi = 1/1.5/2/3/4x, rounded like Android (`int(px + 0.5)`). `--res` is the `res/`
  dir itself, not `res/drawable` (warned). A source smaller than the xxxhdpi/@3x/@2x target is upscaled, with a warning.
- WebP: lossless when the image has <= 256 colours, else quality 90; `--quality 0-100` overrides (100 = lossless), encoder
  `method=6`. Transparent WebP needs API 18+, animated WebP API 28+ (`AnimatedImageDrawable`). `--format png` writes PNG for
  android and web; iOS imagesets are always PNG (`--format`/`--quality` are rejected there). A new `*.xcassets` gets its root
  `Contents.json`; web prints the `<img srcset>` snippet with width/height.
`vd`:
- Verifies each file: the offline VectorDrawable lint (attribute names, enums, dimensions, colours, pathData, gradient
  radius/stops), a render round-trip of the SVG against the VectorDrawable ("render match", or "render differs" when the mean
  diff exceeds 2 or more than 16 px differ strongly; "(render check skipped: ...)" without a renderer), then aapt2:
  "compile+link ok (android-NN)" needs `$ANDROID_HOME/platforms/android-NN/android.jar`; with build-tools only it prints
  "compile ok; link skipped (...)" and the lint stands in for link.
- Converts transforms to `<group>` (skews are baked into the path coordinates), inherits root attributes, multiplies group
  opacity, resolves `<style>` classes, `<use>`/`<symbol>`, clip-path (one per group; objectBoundingBox approximated),
  userSpaceOnUse and objectBoundingBox gradients incl. `gradientTransform`, CSS named/hsl colours. Draws dashes solid and drops
  markers, vector-effect, paint-order and focal points, each with a warning. Rejects masks, filters, text, images, patterns and
  non-uniformly scaled or skewed stroked shapes: re-run `svg --only <name>` or fix the SVG by hand.
- File names go through the Android resource-name rules (`ic_arrow_left.xml` from `Arrow-Left.svg`; printed when renamed);
  without `--dp` the size is the SVG's width, or the per-icon dp saved by `svg` in `svg_report.json`.

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
  - A blocked or failed variant prints `! variant N (<direction>) failed: ...` and the others are kept (exit 1 only when
    none succeeded). Reword that direction rather than re-running everything.
- **Build** defaults are the measured best:
  - The foreground is matted locally against the background that actually ships, so the icon recomposes the concept almost
    exactly. `--separate auto` (the default) switches to an image-edit foreground (1 call) when the matte leaves the subject
    mostly translucent, which happens with a light subject on a light background. Otherwise iOS dark/tinted and the silhouette would look ghosted.
  - The background is a fitted VectorDrawable gradient (or a colour).
  - The subject is shrunk into the 66 dp safe circle only when measurement says so.
  - The monochrome (themed) layer is a gpt-6 glyph converted to a VectorDrawable. It is referenced only when its render
    was measured and placed inside the safe zone (`glyph.ok: true`). A failed, empty, too-faint or unverifiable glyph
    (no working SVG renderer) falls back to the silhouette with a `!` line; a multi-colour glyph is flagged in `warnings`.
  - Use `--bg image` (1 call) only for textured or scenic backgrounds; the report says when it's needed.
  - The concept must be the opaque full-bleed square. A transparent input is flattened onto black and reported in `warnings`.
    `--name` must already be a resource name (`[a-z0-9_]`, starting with a letter); the error suggests one.
- **Read `review/appicon_*.png`**, and check `appicon.json`:
  - `safe_zone.ok`, `glyph.ok` (with `--mono glyph|both`), `web_maskable.ok`, `ios.opaque`, `ios.tinted_has_alpha`, and
    `warnings` (absent, or understood).
  - `ios.actool` starts with `ok`. `skipped (no Xcode actool)` means not checked (Linux).
  - `aapt2` starts with `aapt2 compile+link ok`. `aapt2 compile ok; link skipped (...)` means resource syntax compiled but
    attribute names and enums were NOT checked: install `sdkmanager "platforms;android-35"` and re-run.
- **Install:**
  - Android: copy `android/res/*` into `app/src/main/res` and point the manifest at `@mipmap/ic_launcher` and `@mipmap/ic_launcher_round`.
    One `mipmap-anydpi-v26/ic_launcher.xml` with an inline `<monochrome>` is correct (API 26-32 ignore the element; no
    `-v33` folder needed). Legacy 48 dp bitmaps cover mdpi-xxxhdpi.
    `android/play_store_512.png` is the Play listing icon: 512 px, opaque, well under the 1024 KB limit (`play_store.kb`).
  - iOS: copy `ios/AppIcon.appiconset` into `Assets.xcassets`. Light is opaque. Dark is the foreground on transparency (the
    system draws the dark backing). Tinted is grayscale with alpha, no baked black. This follows Apple's asset-catalog
    format for iOS 18; the set compiles with Xcode's actool (macOS, 2026-09-29), which does not check alpha.
  - Web: the files and the `manifest-icons.json` snippet go in `public/`.

## Phase 5: animation

Pick the route with the user:

| Route | Use when | Output |
|---|---|---|
| `frames --procedural bounce\|pulse\|float\|wiggle\|spin\|breathe` | the art moves rigidly; **no API calls**, art stays exactly 1:1 | frames, WebP, APNG, tiny transform Lottie, `motion.json` for code |
| `frames --moving … --static …` | the art changes shape **in place** (flame, ripples, blinking mascot) | registered frames, WebP/APNG/image-sequence Lottie, Android `animation-list` under `android/` |
| `sequence` (layout → plan → render → build, which propagates or not as planned → reroll) | something **travels**: smoke, steam, falling drops, spray, embers, leaves | a seamless loop where code decides the motion and GPT paints the frames (keys + EbSynth, or every frame); animated WebP, frames, QA report |
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
  - **Resolution cap:** the generated route's frames are the sheet's cells (about 250–350 px from a 4x2 sheet on a
    1536x1024 canvas), so they do not reach 1024-class asset size. For a full-size loop use `--procedural`, `sequence`
    or `rig`/`animate`.
  - **Android:** copy `android/drawable-nodpi/<name>_frame_NN.png` (pixel-sized, so `nodpi`) and
    `android/drawable/<name>_anim.xml` into `res/`. Names are sanitised to `[a-z0-9_]` (`Rain-Drop` becomes `rain_drop`,
    printed when it changes); aapt2 compiles the folder when it is installed (link needs a platform `android.jar`).
  - **Platforms:** animated WebP plays everywhere (Chrome, Firefox, Safari 14.1+, Android ImageDecoder/Glide,
    iOS 14+). APNG plays on the web and on Android (Glide or ImageDecoder, API 28+); iOS shows its first frame as a
    still, so ship WebP or Lottie there. Loop timing is exact per frame (a 24 fps loop is 1000 ms, not 24x42).
  - **Never silently wrong:** a sheet that comes back opaque (a gateway that ignores `background=transparent`) is
    saved as `sheet_raw.opaque.png` and the run stops; a sheet with fewer frames than asked is flagged `frame-count`;
    when one of the two calls fails the other result is kept (`--reuse static` re-runs only the sheet); an asset with
    no alpha exits, one with an opaque backdrop warns (the whole rectangle would move).
  - `--count 12 --cols 4` gives more frames. `--formats gif,webp,apng,lottie` and `--px 480` control the outputs; the
    image-sequence Lottie costs about 280 KB per 480 px frame.
  - AnimationDrawable decodes every frame into memory, so ship the WebP or Lottie for long loops.
- **vectorize:**
  - Each layer can morph, move (anchor, dx/dy, sx/sy, rotate, opacity, easing) or both; `glows` are radial-gradient halos.
  - Read `keys_review.png`. Its last output line states how it was rendered: `lottie-web` (the real Lottie in headless
    Chrome), `svg fallback (...)` or `NOT RENDERED (...)` with the reason (Chrome as root needs `GPTKIT_CHROME_NO_SANDBOX=1`,
    which the kit sets itself when it runs as root). The motion line is the area-weighted mean of each layer's
    `max(move, fade/4)` against the `--strength` threshold (subtle 1.5%, normal 4%, strong 8%); one retry when it is too
    subtle or nothing moves. An invalid answer exits 1 and leaves `layers.json` for editing.
  - `--svg FILE` (or `DIR/svg/NAME.svg`) needs no `assets.json`. Accepted: paths/shapes, groups, transforms (baked into the
    coordinates), `fill-rule`, fill/stroke opacity, caps/joins (mapped to the VD set), `currentColor`, named/rgb()/8-digit
    colours (alpha folded into opacity), gradients incl. `href`, percentages and objectBoundingBox units (rewritten as
    userSpaceOnUse). Rejected with one line before any API call: `<use>`, `<symbol>`, `<style>`/CSS classes, `clip-path`,
    `mask`, `filter`, `<text>`, `<image>` (flatten with svgo or re-run `svg --only NAME`).
  - The AVD (`<res_name>_avd.xml`) is compiled with aapt2 when it is installed (link too with `platforms/android-*/android.jar`;
    otherwise the message says enums were not checked). Copy it to `res/drawable/` and call `(drawable as Animatable).start()`.
    It loops on its own; framework AVD needs API 21 (24 with gradients or evenOdd fills; AppCompat `app:srcCompat` below
    that). Keep loops <= 5 s: the RenderThread subsamples longer keyframed animators. Alpha keyframes are clamped to 0..1.
  - The Lottie uses only shape layers with fills/strokes/gradients/transforms, no merge paths, expressions, masks or
    text, so lottie-android, lottie-ios and lottie-web render it alike (`vectorize` asserts this). Its loop is exactly
    `--seconds` (a fractional `fr` when `seconds * fps` is not an integer).
  - To refine motion, edit `layers.json` yourself and `--rebuild` (the edited file is kept as `layers.orig.json` when the
    kit normalises it).

**Layout-driven sequences: code animates, GPT paints, EbSynth carries the paint.** An image model has no notion of
time. A `frames` sheet keeps the style but not the trajectory: drops jump between cells, the floor moves from row to
row, cells come back empty. So for anything that travels, decide the motion in code, have GPT paint keys from that
layout, and let EbSynth carry each key's real texture along the layout to every frame.
```bash
# 1. layout: a periodic physics plume (or --guides DIR: one guide PNG per frame, from any source)
python3 $K sequence build/player train_smoke layout --base train_body.png --sprites puff1.png,puff2.png \
  --vent 190,158 --loop 3 --fps 24 --chuff 0.75 --life 5.4 --velocity 0,-85 --drag 1.5 --wind 45,-3 --buoyancy -10 \
  --size 96 --grow 0.45,1.5        # a leftward wind needs the = form: --wind=-40,-3 (argparse reads "-40,-3" as a flag)
#    layout ends by printing the plan: EbSynth keys (and how far apart) or GPT on every frame, and why.
#    layout starts fresh every time (old plume/route/QA are dropped, base and sprites are copied into
#    sequence/<name>/inputs/) and records a hash per guide: renders from an older layout are re-rendered by
#    `render` and refused by `build`/`propagate`, never reused silently
# 2. render what the plan asks for: the keys only (every 6th of 72 frames = 12 calls), or every frame
python3 $K sequence build/player train_smoke render --moving "The smoke" --locked "The toy steam train" \
  --model gpt-image-2.5-sunburst
# 3. build: keys only -> EbSynth propagates them to all 72 frames; every frame rendered -> built directly
python3 $K sequence build/player train_smoke build --ref drawn_smoke.png --below 175
```
- **The plan decides whether EbSynth runs** (`sequence ... plan` prints it again; `layout` prints it first). It is
  recorded as `route` in `sequence.json`, and `render`, `build` and `propagate` follow it unless you pass `--every`.
  `--engine` is remembered there too, and the plan is cached (keyed by the guides and the EbSynth install), so the
  later stages do not pay for it again (a 72-guide plan at 1024 px takes about 40 s and 1.1 GB).
  - **EbSynth, keys a quarter-second apart**, whenever it can carry the layout: the largest key interval of at most
    `round(fps / 4)` frames that divides the loop (14 fps allows 4 frames). That is 12 image calls instead of 72, with
    less shimmer (table below).
  - **Closer keys** when something between two keys is in neither of them. Plumes are checked element by element: every
    puff alive in an in-between must exist in one of its keys. Guide folders are checked by appearance, because EbSynth
    matches patches wherever they sit: travel, growth and jumps are carried, while a splash ring, a blink or a colour
    change that no key shows is not. Frames need 90% of their content carried (content sliced by the canvas edge
    doesn't count).
  - **GPT on every frame (EbSynth skipped)** when the binary isn't installed, the frame rate is below 6 fps
    (`round(fps / 4) < 2`: a quarter-second key is every frame; 6 and 7 fps plan keys every 2), no key interval
    divides the frame count, or the layout changes on every frame.
  - `--engine gpt` or `--engine ebsynth` forces the choice; `ebsynth` fails if the binary is missing or no interval
    can carry the layout.
  - The plan also warns when a plume is dense (more than about 3 elements stacked on a pixel). That fails with either
    engine, because GPT paints one generic plume; space the elements before rendering.
- **`propagate` is what the plan picks when it can.** It needs the `ebsynth` binary (public-domain code; the
  PatchMatch algorithm it implements is patented, so check that before redistributing it commercially; CPU is enough).
  Build it once with `git clone https://github.com/jamriska/ebsynth ~/.cache/gptkit/ebsynth && cd
  ~/.cache/gptkit/ebsynth && sh build-linux-cpu_only.sh` (`build-macos-cpu_only.sh` on macOS), or pass
  `--ebsynth PATH` / set `GPTKIT_EBSYNTH` (an explicit path must exist; there is no silent fallback).
  - Every frame, keys included, is synthesised from its two nearest keys. The appearance guide is the layout; plumes
    also get a positional guide that maps each pixel to where its puff was at the key. The two syntheses are
    cross-faded, and the result is clipped to where the layout has content.
  - Each output pixel is a patch of a real render, placed by the layout. That is why it neither stretches like a warp
    nor shimmers like independent frames.
  - The time scales with the moving part's box area: about 7 minutes for 72 frames of a 344x184 box, about 14
    minutes of EbSynth plus 5 minutes of QA for a 567x416 box (a 1024 canvas plume), with `--jobs 4`. `--every` must
    divide the frame count.
  - Each run is bounded by `--ebsynth-timeout` (600 s); a failed, hung or silent run stops the stage with the frame,
    the key and the exact command. Finished syntheses are named by a hash of their inputs, so a crashed or interrupted
    run resumes where it stopped (`--force` redoes them); the work files are removed after a successful build unless
    `--keep-work`.
- **Every frame by GPT** (the plan's fallback, or `--engine gpt`): 72 calls instead of 12, and the frames shimmer.
  Keep it for when EbSynth is unavailable or can't carry the layout, and fix its outliers with `reroll` (2 candidates each, the best neighbour
  fit wins) before building again.
- **Look at `layout_review.png` before rendering.** The motion is decided there; GPT only adds the finish. Keys every
  quarter-second of a 3 s loop are 12 image calls (72 without `--every`), so get the motion right first.
- **The layout must loop.** `--loop` has to be a whole number of `--chuff` intervals; the plume's jitter repeats every
  loop, so frame 72 is frame 0. Custom guides from `--guides` must be square and all one size, transparent (or pass
  the baked backdrop as `--base`, which `build` cuts away), naturally ordered (`0.png … 23.png` sorts by number, not
  text) and looping: `layout` prints the order it used, exits on size mismatches, warns on an opaque backdrop and warns
  when the step from the last guide back to the first is unlike any step inside the loop.
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
  - `outliers` are frames that disagree with both neighbours more than the layout itself does there, usually a render
    that ignored its layout. Every score is taken against the layout's own parts, so a puff being born, a splash the
    layout contains or a spray fan is not an outlier, `backwards` counts only steps the layout does not take, and a
    radial or oscillating layout (no dominant heading) gets no direction QA at all. `reroll` renders candidates and
    keeps whichever fits its neighbours best, the current frame included; the replaced render is kept as
    `raw/NNN.prev.png`. After `propagate`, `reroll` refuses a non-key frame and names its keys: fix it through the
    nearest key instead (delete that key's `raw/NNN.png`, run `render`, which renders only the missing keys, then
    `build`); the build output says so too.
  - `--ref`, `--below`, `--quality` and `--preview-bg` are remembered in `sequence.json`, so a `reroll` followed by a
    plain `build` keeps the same output ("using --ref X --below Y from the last build" is printed).
  - A render that comes back opaque (a gateway that ignores `background=transparent`) is saved as `raw/NNN.opaque.png`
    and counted as missing; a moderation refusal or another 4xx is printed once with its code and the prompt and is not
    retried (rephrase `--moving`/`--locked`/`--look`); `render` exits 1 while frames are missing. Without OpenCV the
    build says that `follows` was skipped.
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
- `rig DIR NAME` needs only `DIR/assets/NAME.png` (no `assets.json` or mockup); copy any PNG there to rig it. Output goes to
  `DIR/rig/NAME/`. A fully transparent asset, a part that gets no region, or an `--order` name missing from `--parts` exits with
  one line (`--allow-empty` keeps an empty part).
- You can name the regions yourself from `regions.png`. Otherwise, omit `--assign` to have one vision call name them (98% agreement in
  testing), or use `--route hybrid`, where GPT isolates each part (1 image call per part) and the tool registers it back.
  `--assign` tokens are checked: unknown regions/components, parts that are not in `--parts` and parts left without a region are
  printed; regions you did not assign are merged into their neighbours and listed, and `assign.json` marks them `"merged": true`.
  If a GPT isolation call fails (moderation, quota), the parts that succeeded stay cached in `parts_gpt/` and the run exits 1
  with the failing part's prompt; re-run to resume. If the vision namer fails you get the `--assign` hint instead of a traceback.
- Other rig flags: `--order a,b,c` (bottom→top, default = `--parts` order), `--glow part|shared` (each part owns the glow nearest to
  it, or one shared glow layer), `--preset flat|illustration|gradient` (default illustration), `--route gpt` (trace GPT's redraws;
  lower fidelity, ignores `--assign/--namer/--min-detail`), `--force` (redo cached GPT parts), `--model` / `--text-model`,
  `--preview-bg`. `rig --animate` always renders the Lottie at 512 px with linear easing; use `animate` on `rig/NAME/layers.json`
  for `--size`, `--ease keys`, `--seconds`, `--fps`, `--out`, `--name`.
- Every part is extended under the parts above it, so moving parts never open gaps. Check `rig_preview.png`: the recomposite
  should match the source. Without a working headless Chrome the rig is still written, but `rig_preview.png`, `keys_review.png`,
  `lottie_frames.png` and every fidelity number are skipped and the report says `not measured: <reason>` (never a guess).
- Motions: sway, flicker, breathe, squash, bob, wave and ripple. The syntax is `part=motion[:amp[:phase]][@anchor]`; `@anchor` shares
  another layer's motion field. Frame 0 is always the approved design. `animate` accepts `layers.json` from `rig`, `trace` and
  `vectorize` (relative commands, arcs, strokes and per-layer `motion` are kept); gradient fills ride along with the part they paint
  (animated Lottie `gf` start/end), so shading does not slide across a moving shape. A layer whose max travel is under 0.2 viewBox
  units is reported as invisible.
- **Read `anim/lottie_frames.png`** (rendered by lottie-web). In `animation_report.json`, `tear_px` per key counts pixels of the
  rest pose's interior (eroded by the largest travel + 1 px, glow bands excluded) that the key leaves uncovered: open gaps and
  enclosed holes both count, key 0 is 0 by construction, and `tear_frac_max` divides by the rest area. Values above ~0.5% mean a
  part slid out from under its neighbour: add the missing part to `--parts`, or animate the pair with a shared `@anchor`.
- Traced rigs make 150–250 KB Lotties; `--min-detail 0.02 --keyframes 6` shrinks them.
- Tracing notes (`trace`, Phase 3 above): `--work` (long side, default 512, never above the source: upscaling traces the pixel
  staircase), `--colors` (k-means colours), `--tol` (curve fit, work px), `--bands` (glow bands, 0 = none), `--display` (px the
  fidelity is measured at; default 3× the dp width, else ≥192 long / ≥96 short side), `--dp` (VD width; default 48 dp on the long
  side, so pass it for illustrations), `--name` (single input), `--prefix` (VD file prefix, default `img_`), `--bg` (review
  backdrop). VD file names are sanitised to `[a-z0-9_]` and printed when they change. `trace.json` `vd.aapt2` says exactly what
  ran: `compile+link ok (android-NN)` only with `platforms/android-*/android.jar` on `ANDROID_HOME`; `compile ok; link skipped`
  otherwise (attribute names, enums and dimensions unchecked). More than 400 colour regions is photographic: no SVG/VD is written and
  the advice is `png/webp`. `--preset vtracer` ignores `--colors/--tol`.

## Phase 6: build the UI 1:1 (measure, don't guess)

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

## Rules learned from probing (don't relearn them)

- **Never text-only prompt an asset of an approved design.** It comes out as a generic, different style. Always go through
  `forge`/`fix`/`frames`, which pass the design as the reference.
- **Exactly one reference image per edit call.** Multi-image edits returned the mockup itself on the `sub2api` gateway the
  kit was probed on; the official API accepts up to 16 reference images per edit, unverified here.
- A single-reference edit designs a genuinely new screen when the prompt says "DIFFERENT screen … style guide only … do not
  copy its layout" (5/5 new layouts). A design-system board as the reference is worse than the screen itself.
- Some backends ignore `quality`, `size` and `n` (measured on `sub2api`; the official API defines them), so the kit trims,
  crops and resizes locally and runs candidates in parallel. Output aspect ratio follows the prompt's wording, so `mockup`
  states it and retries on a mismatch. A gateway that returns JPEG instead of PNG has ignored `background: transparent`
  too; the kit re-encodes to PNG, records `source_format` and says so once.
- API errors carry the server's `error.code` and the prompt (`[moderation_blocked] ... | prompt: '...'`): read the code
  before rewording. Only `error.param` decides which optional parameter to drop; prose that mentions "background" does not.
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
- Clip every bbox a vision call returns to the image before using it (`critique`, `manifest`, `spec`, `--vision`); a box mostly
  outside the mockup drops the item with a printed reason instead of writing a garbage crop.
- A `background=transparent` call can still come back opaque (a gateway dropped the parameter): check opacity (> 95 % opaque
  or all four corners opaque) before using any generated raster as a transparent asset.
- Flags outrank `fid`: rank candidates by flag weight (`EMPTY` 1000 > `COPY` 100 > `OPAQUE`/`BACKDROP` 10 > `CUT-OFF` 3), never
  by flag count; an empty render has fewer flags than a real one.
- Sample the backdrop colour from a ring *outside* the bbox and test the silhouette *boundary*; testing all solid pixels against
  the bbox corners flags every light icon on a light page.
- Validate an imported file completely before the first paid call, and never feed import errors back to the model as if it had
  made them (a Lucide `currentColor` icon cost two calls and a wrong retry before).
- When a pool of paid API calls fails part-way, persist each result on arrival, name what is cached, catch `ApiError` per job
  (never blanket exceptions) and exit 1 only when nothing usable was produced.
- Files one command writes for another store dir-relative paths, are merged with re-read + temp + `os.replace` (parallel
  `--only` runs otherwise lose each other's entries) and record what they were made from (hash sidecars), so the next stage
  refuses stale inputs by name instead of reusing them silently.
- Headless Chrome (141, `--headless=new`) lays out 87 px less than `--window-size`: pad every capture and crop; as root add
  `--no-sandbox`. Never report a renderer as available without a probe render, and report a number that was not measured as
  `not measured: <reason>`; never guess `prefer`/`ok` from file size or silence.
- `aapt2 compile` alone checks well-formedness only: say `link skipped`, run the offline VD lint and never print `valid`;
  `compile+link ok` needs `platforms/android-*/android.jar` (`sdkmanager "platforms;android-35"`).
- Android resource names are enforced by AGP, not aapt2 (`[a-z0-9_]`, letter first, no Java keyword): every file that lands in
  `res/` is named through `_res_name`; pixel-sized frames belong in `drawable-nodpi`.
- Normalise an icon set per size group, never per glyph: per-glyph fitting to the live area changes stroke weights by up to
  27 % and then flags the set as inconsistent.
- SVG gradient attributes have defaults (`x2=100%`, `cx=cy=r=50%`) and percentages: resolve them before writing a
  VectorDrawable, or the drawable inflates with `gradientRadius=0` and crashes.
- Per-frame duration lists (`round((i+1)·1000/fps) − round(i·1000/fps)`) keep a loop exact; Pillow's animated WebP needs
  `method=6` (its default is 0).
- QA that compares neighbouring frames must subtract what the layout itself does at that frame, or every abrupt-but-intended
  change (a birth, a splash) becomes an "outlier".
- A gate on the mean of per-layer numbers is decided by the smallest hyperactive layer; weight by area.
- A themed-icon glyph is referenced only when its render was measured: check `glyph.ok`; `false` means the silhouette shipped.
- iOS tinted app icon = grayscale + alpha (no baked backdrop); dark = foreground on transparency; light = opaque. actool does not
  check alpha, so the kit's `ios.*` fields are the check.
- Never upscale a raster before tracing: the tracer follows the source's pixel staircase (11,855 vs 722 chars on the same ring).
- Accept a fitted gradient only when it is visible (stops > 6 ΔE apart) and clearly better (RMSE −2 absolute); a gradient costs API 24+.
- A tear metric must see open gaps: erode the rest pose by the travel and count what a key leaves uncovered.
- Compose theme code that compiles: namespace colours (`AcmeColors.X`; `object` members shadow top-level names), `FontWeight(n)`
  for the 450/550 weights GPT specs carry (`W{n}` exists only for hundreds), and one hex normalisation at load (`#fff`,
  `#rrggbbaa`, `rgb()` all occur in real specs and silently produced wrong `Color(...)` literals).
- Font-weight fits are renderer-specific (stem darkening): score candidates with and without emboldening and treat one-step
  ties as the spec's weight.
- A drop shadow makes "distance from the outside colour" useless for edges; use the largest colour step along the profile instead.
- Never cache a network failure (metadata, licences); cache only validated successes, written atomically.
- Full measurements are in `references/api-findings.md`.
