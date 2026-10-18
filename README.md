# gpt-ui-assets

A [Claude Code](https://claude.com/claude-code) skill that designs **platform-native** app screens with GPT image
models and turns an approved design into **production assets and a 1:1 build**.

Every icon, illustration and background is **redrawn from scratch at full resolution and matched 1:1**; nothing is
cropped out of the screenshot. You also get:
- SVG / Android VectorDrawable icons;
- launcher icons for Android, iOS and the web;
- frame animations, Lottie and AnimatedVectorDrawable;
- theme code and fonts measured from the pixels;
- a screenshot diff that tells you exactly what your built screen gets wrong.

Claude is the driver. GPT is one instrument, and each job goes to whatever does it best:
- **GPT image:** mockups, painterly art, raster frames.
- **GPT vision:** inventory, critique, SVG icons, vector keyframes.
- **Local measurement:** scoring, colour matching, tracing, rigging, font matching, redlines, diffs.
- **Claude:** reviews every result and writes the code.

<p align="center">
  <img src="docs/images/01-mockup.webp" width="260" alt="Approved mockup">
  <img src="docs/images/02-inventory.webp" width="260" alt="Asset inventory">
  <img src="docs/images/07-campfire.gif" width="200" alt="Generated animation">
</p>

## Why

When you ask GPT for "the assets of this design", it usually **crops them out of the mockup**. You get
low-resolution pixels with the card background baked in, neighbouring text bleeding in, and cut-off edges. Asking it to
redraw them from a text prompt gives **generic art in a different style**.

This skill does what a designer would do instead:
1. It inventories the assets.
2. It redraws each one from scratch, with a **focused crop of the element as the only reference**.
3. It scores candidates against the design with a perceptual metric.
4. It colour-corrects the winner in CIELAB.
5. It rejects copies and crops automatically.

<p align="center"><img src="docs/images/08-crop-vs-redraw.webp" width="460" alt="Upscaled crop vs redrawn asset"></p>

*Left: the element cropped from the mockup and upscaled. Right: the redrawn asset. It's 5–7× the resolution,
40–100× sharper and on a real transparent background.*

## What it does

| Phase | Commands | What you get |
|---|---|---|
| Research | `research`, `blueprint` | In-depth, web-searched, cited market report (competitors, pain points, pricing, gaps) and a feature blueprint (RICE, user stories, screens with mockup briefs, roadmap) using `gpt-6-sol` at high/xhigh reasoning |
| Design | `mockup --platform android\|ios\|web`, `--style-ref`, `critique` | Native-component screens; further screens in the approved design language; design QA with measured contrast |
| Inventory | `manifest`, `spec`, `iconspec` | Every asset with bbox, kind and format (native components and system UI excluded); tokens with the native widget per component |
| Raster assets | `forge`, `pick`, `fix` | 1:1 redraws: parallel candidates, perceptual `fid` ranking, Lab colour match, copy/crop rejection |
| Icons | `svg`, `icons`, `vd` | SVG icons scored against the design and retried when weak, on a 24-grid; new icons in the set's style; VectorDrawables validated by render round-trip + aapt2 |
| Vectors | `trace` | Raster art → gradient-aware SVG/VectorDrawable, with a vector-or-raster recommendation |
| App icon | `appicon concepts`, `appicon build` | Android adaptive (foreground, background, themed monochrome), legacy + Play 512, iOS 1024 + dark/tinted, favicons + PWA/maskable |
| Animation | `frames`, `sequence`, `vectorize`, `rig`, `animate` | Procedural motion (no API calls), generated frames with QA, layout-driven sequences (code animates, GPT paints keys, EbSynth carries them to every frame of smoke, drops or spray when the plan says it can, else GPT paints every frame), WebP/APNG/Lottie, Lottie **and** AnimatedVectorDrawable, rigged 1:1 vector animation |
| Hybrid screens | `material`, `state`, `stage` | Screens built in HTML/CSS/JS around GPT-painted materials; state variants pixel-identical outside the change; stills, a frame-exact motion clip and a presentation shot rendered through Chrome |
| Build 1:1 | `fonts`, `redline`, `tokens`, `compare` | Google Fonts matched by rendering; measured radii and gaps; Compose/SwiftUI/CSS/Tailwind theme code; screenshot-vs-design heatmap with ranked fixes |
| Export | `export`, `doctor` | Android densities, iOS imagesets, web 1x/2x; setup check |

**Platform-native design.** The same brief, `--platform android` vs `--platform ios`: M3 top app bar, cards, navigation
bar with the pill indicator and gesture handle vs a large title, inset-grouped list, floating Liquid Glass tab bar and
Dynamic Island.
<p align="center"><img src="docs/images/09-platform-native.webp" width="600" alt="Android vs iOS from one brief"></p>

**New screens in the approved design language** (`--style-ref`): same wordmark, tab bar, type and illustration style,
with a new layout.
<p align="center"><img src="docs/images/10-style-ref.webp" width="520" alt="Style-ref screens"></p>

**Forge and icon review sheets.**
<p align="center">
  <img src="docs/images/03-forge-review.webp" width="400" alt="forge review sheet">
  <img src="docs/images/04-svg-review.webp" width="220" alt="svg review sheet">
</p>

**App icon.** Adaptive masks, the 66 dp safe zone, the themed glyph, and legacy, iOS dark/tinted, maskable and favicon versions.
<p align="center"><img src="docs/images/11-app-icon.webp" width="800" alt="App icon review"></p>

**Tracing.** Gradient-aware vectors: display diff 2.25 vs 5.08 for vtracer.
<p align="center"><img src="docs/images/12-trace.webp" width="720" alt="Trace review"></p>

**Animation.** Registered raster frames, and vector motion as both Lottie and AnimatedVectorDrawable (morph, glow, a like-pop).
<p align="center"><img src="docs/images/05-frames.webp" width="720" alt="Animation frames"></p>
<p align="center"><img src="docs/images/13-vector-anim.webp" width="720" alt="Vector animations rendered by lottie-web"></p>

**Compare.** Implementation vs design, with every planted deviation found.
<p align="center"><img src="docs/images/14-compare.webp" width="720" alt="compare heatmap"></p>

## Install

**As a plugin** (recommended):
```
/plugin marketplace add Shakibuzzaman3104/gpt-ui-assets
/plugin install gpt-ui-assets@gpt-ui-assets
```

**Or as a plain skill**, linked so that `git pull` updates it:
```bash
git clone https://github.com/Shakibuzzaman3104/gpt-ui-assets.git ~/src/gpt-ui-assets
ln -s ~/src/gpt-ui-assets/skills/gpt-ui-assets ~/.claude/skills/gpt-ui-assets
```
A copied folder (`cp -r`) silently stays on the version you copied.

Then install the Python dependencies (Python 3.9+):
```bash
pip install -r ~/.claude/skills/gpt-ui-assets/scripts/requirements.txt   # requests, pillow, numpy, playwright
pip install vtracer scipy fonttools                                     # optional extras
```
These optional tools are used when present:
- **Headless Chrome/Chromium:** SVG rendering with alpha, lottie-web previews.
- **The Android SDK's `aapt2`:** validates every VectorDrawable, adaptive icon and AVD.
- **`xcrun actool`:** validates the iOS app icon.

## Configure

The skill works with the OpenAI API or any **OpenAI-compatible** endpoint that exposes `/images/generations`,
`/images/edits` and `/responses`. Set environment variables, or create `~/.config/gptkit/env`:

```ini
GPTKIT_API_KEY=sk-...
# optional
GPTKIT_BASE_URL=https://api.openai.com/v1
GPTKIT_IMAGE_MODEL=gpt-image-2
GPTKIT_TEXT_MODEL=gpt-6
```
`OPENAI_API_KEY` and `OPENAI_BASE_URL` are used as fallbacks. Keep the file private (`chmod 600`). Check the setup with:
```bash
python3 ~/.claude/skills/gpt-ui-assets/scripts/gptkit.py doctor
```

## Use

Just talk to Claude:

> Design an Android budgeting app home screen with GPT: this month's balance, spending by category, recent
> transactions and an add-expense button, in warm coral and cream with a friendly piggy-bank mascot.

> The second one. Now the budgets screen in the same style.

> Go ahead: give me all the assets, the app icon and the fonts, and build it in Compose.

> Make the mascot bounce, and animate the flame as an animated vector.

Claude designs with native components for your platform, critiques each variant before showing it, and on "go
ahead" does the rest: inventories and forges the assets, reviews every result, exports to your platform, matches the
fonts, generates theme code, builds the screen and diffs a screenshot of it against the design until it matches. You
can also start from a design made in the ChatGPT app or anywhere else: give Claude the image.

## How it stays 1:1

- **A focused reference beats the full mockup.** The element plus a 60% margin, upscaled, as the single reference
  image. Text-only prompts come out in a generic style.
- **One reference per call.** Multi-image edits were observed to return the mockup itself.
- **Perceptual ranking.** `fid` combines silhouette IoU after alignment, CIELAB distance and a colour-distribution
  distance. It fixed a case where plain pixel diff preferred the wrong candidate.
- **Colour correction.** Redraws drift yellower and more saturated, so a Lab transfer pulls them back to the design.
  It's kept only when the score improves.
- **Automatic rejection.** `COPY`, `OPAQUE`, `BACKDROP` and `CUT-OFF` candidates rank last.
- **Measure, don't guess.** Palette hexes are snapped to pixels (ΔE00 1.35 → 0.36), fonts are matched by rendering,
  radii are measured to within about 1 px, and vision is used for descriptions, never numbers.
- **Animation by layers.** Generate only the moving part, and register everything back to the approved asset.
  Rigged vector parts are completed underneath each other, so motion never tears.

The measurements behind every rule are in [`references/api-findings.md`](skills/gpt-ui-assets/references/api-findings.md).

## Cost and limits

- Each image call takes 35–80 s and is billed at your provider's image pricing. Paid commands print a
  `plan: ~N image calls … ≈ $X` line first; `forge`, `sequence render` and `research` take `--dry-run`, and
  `GPTKIT_BUDGET_USD` stops a command whose estimate is over budget. Every call is logged to
  `~/.cache/gptkit/ledger.jsonl`; `gptkit.py cost --by cmd` sums it at list prices (through a gateway or the
  Codex CLI backend that is plan quota, not money). `forge` with 2–3 candidates makes that
  many calls per raster asset. Icons take one vision call per 6 icons. Procedural animation, tracing, rigging, fonts,
  redline, tokens and compare make no image calls.
- Some backends ignore `quality`, `size` and `n`. The kit trims, crops and resizes locally and runs candidates in parallel.
- Image models don't draw to scale, so sizes are snapped to platform norms and type sizes come from measurement.
- The fidelity thresholds were tuned mainly on dark UI designs. Check the review sheets on very light or busy backgrounds.

## Layout

- `skills/gpt-ui-assets/SKILL.md` is a short router: the rules that hold in every phase and which reference to read.
  Each phase's details are in `skills/gpt-ui-assets/references/`, loaded only when Claude reaches that phase.
- `skills/gpt-ui-assets/scripts/gptkit.py` is the CLI (`python3 gptkit.py <cmd> -h`).
- `python3 gptkit.py status <dir>` shows where a project stands and the next command, with no API calls.
- `CONTRIBUTING.md` has the rules for changing the kit and how to run the tests.

## License

[MIT](LICENSE)
