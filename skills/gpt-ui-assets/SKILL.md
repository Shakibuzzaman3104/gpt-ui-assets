---
name: gpt-ui-assets
description: Drive OpenAI image + vision models (gpt-image-2, gpt-6, or any OpenAI-compatible endpoint) to design app/web UI mockups and then produce every asset of an APPROVED design as a separate file, redrawn from scratch 1:1 (never cropped from the screenshot) — icons as SVG/Android VectorDrawable, illustrations and backgrounds as clean PNG/WebP, animation frames (AnimationDrawable) and Lottie JSON — ready to build the UI in code. Use whenever the user wants GPT/ChatGPT to generate UI designs, icons, images or illustrations, approves a GPT design and says "go ahead"/"build it"/"give me the assets", asks for the assets "used in the UI", wants frames for an animated drawable or a Lottie, or complains that GPT cropped assets out of a mockup.
license: MIT
metadata:
  version: 1.0.0
---

# GPT UI assets

Claude drives GPT through `scripts/gptkit.py`, and the user approves designs. GPT is the illustrator; Claude is the
art director, the reviewer and the engineer who builds the real UI.

`K` below is `scripts/gptkit.py` inside this skill's directory:
```bash
K="<this skill's directory>/scripts/gptkit.py"      # every command: python3 $K <cmd> -h
```

## Setup (once per machine)

```bash
pip install -r "<this skill's directory>/scripts/requirements.txt"     # requests, pillow, numpy (+ optional cairosvg)
python3 $K doctor                                                       # checks key, endpoint, models, SVG renderer
```
Configuration comes from environment variables or `~/.config/gptkit/env` (`KEY=value` lines):
- `GPTKIT_API_KEY` (or `OPENAI_API_KEY`), required.
- `GPTKIT_BASE_URL` (or `OPENAI_BASE_URL`), default `https://api.openai.com/v1`. Any OpenAI-compatible gateway works if it
  exposes `/images/generations`, `/images/edits` and `/responses`.
- `GPTKIT_IMAGE_MODEL` (default `gpt-image-2`) and `GPTKIT_TEXT_MODEL` (default `gpt-6`, which needs vision and structured outputs).

If `doctor` reports a model as NOT LISTED, set the variable to one it does list. Never print or echo the API key.

Image calls take 35–80 s and cost real money or quota. Batch deliberately, run slow commands in the background,
and run independent commands in parallel (e.g. `forge` and `svg`).

## Phase 1: design (the user drives)

```bash
python3 $K mockup "meditation app home: greeting, featured card, 3 mood tiles, mini player, 4-tab bar; dark navy + lavender" \
  --out design/home --variants 2            # 1024x1536 portrait; --size 1536x1024 for web/tablet
python3 $K mockup "make the featured card taller and fill the tab bar icons" --edit design/home/mockup_v1.png --out design/home --name mockup_r2
```
Show the user each variant (Read the PNG) and iterate with `--edit` until they approve. A design made elsewhere
(the ChatGPT app, Figma export, a screenshot) works the same way: use its image path.

## Phase 2: "go ahead" → asset inventory

```bash
python3 $K manifest design/home/mockup_v2.png --out build/home --screen-dp 390   # -> assets.json + manifest_overlay.png
python3 $K spec build/home                                                        # palette, type, radii, spacing (~3 min; background)
```
**Review `manifest_overlay.png` yourself before generating anything.** Red boxes are PNG assets and green boxes are SVG.
Edit `assets.json` directly to fix the inventory:
- Remove anything code should draw: solid or gradient shapes, cards, dividers, plain text, shadows.
- Split merged graphics, merge duplicates, and tighten loose bboxes.
- Set `format` to `svg` for flat, line, monochrome or simple-gradient glyphs, and to `png` for painterly, glowing, 3D,
  photographic or detailed art.
- Give artwork that sits behind UI `kind: background` and `transparent: false`. It gets redrawn with the overlaid UI removed.
- Treat `dp` sizes as estimates, because image models don't draw to scale. Snap them to platform norms (24 dp
  toolbar/tab icons, 48 dp touch targets, an 8 dp grid).

## Phase 3: produce assets (run both in parallel)

```bash
python3 $K forge build/home --candidates 2 --jobs 8   # PNG assets: parallel redraws, scored, best kept -> assets/, review/
python3 $K svg   build/home                           # SVG assets -> svg/*.svg + review sheet
```
`forge` sends a **focused crop of the element** (60% margin, upscaled) as the single reference, with a
"recreate from scratch, 1:1, transparent" prompt, then scores each candidate:
- `diff` is the mean pixel difference from the design at design scale; lower is better. Around 12–16 is a close
  icon/illustration match. Backgrounds score 25+ because the design still has UI text on top.
- A candidate with any of these `flags` ranks last:
  - `COPY:returned-mockup` or `COPY:returned-reference`: GPT sent the screenshot back.
  - `OPAQUE`: the transparency was lost.
  - `BACKDROP`: the circle or card behind the icon was kept.
  - `CUT-OFF`: the content touches the canvas edge.
- `warn: soft` means low edge detail. That's normal for smooth gradients and only a problem if the asset also looks blurry.

**Always Read the review sheets** (`review/forge_*.png`, `review/svg_*.png`). The metric ranks candidates, but your
eyes decide. Then:
```bash
python3 $K pick  build/home rain_drop 2                                      # choose another candidate
python3 $K fix   build/home campfire --note "logs should be thinner"         # vision diff + one corrective edit
python3 $K forge build/home --only campfire --force --candidates 3           # or re-roll
```
`fix` keeps a result only if it scores better, because correction passes often over-correct. Pass `--accept` to keep it
anyway when the user asked for that exact change. If an asset is still off after 2 fix rounds, tell the user what
differs rather than looping.

## Phase 4: platform export

```bash
python3 $K vd build/home/svg/*.svg --out app/src/main/res/drawable --dp 24          # SVG -> VectorDrawable (ic_ prefix)
python3 $K export build/home/assets/campfire.png --dp 48 --res app/src/main/res --name img_campfire   # webp per density
python3 $K export <png> --dp 44  --platform ios --res App/Assets.xcassets          # @1x/@2x/@3x imageset
python3 $K export <png> --dp 320 --platform web --res public/img                   # 1x/2x webp
```
`vd` rejects what VectorDrawable can't express (transforms, masks, filters, text). When that happens, run
`svg --only <name>` again or fix the SVG by hand. Size background art to its container's aspect ratio and let code
clip the corners. Never bake rounded corners or text into it.

## Phase 5: animation

Pick the route with the user. Use **frames** for painterly or raster art (AnimationDrawable or an image sequence), and
**vector** for Lottie (small, scalable, smooth).

```bash
# raster frames: the moving layer as a sprite sheet, the static layer once, both registered to the approved asset
python3 $K frames build/home campfire --moving "the flame (outer + inner flame with glow)" \
  --static "the two crossed logs" --motion "a gentle campfire flicker" --count 8 --fps 12
#   -> frames/<name>/frame_01..08.png, preview.gif, frames_review.png, <name>_anim.xml (animation-list)

# vector keyframes -> Lottie (morph-compatible paths are validated, with one automatic retry)
python3 $K vectorize build/home campfire --moving "the outer flame and the inner flame" \
  --static "the two crossed logs" --motion "a gentle campfire flicker" --keyframes 4 --seconds 1.2
#   -> vector/<name>/<name>.json (Lottie), layers.json, keys_review.png, preview.html (lottie-web)
```
Read `frames_review.png` / `keys_review.png` before handing anything over. For frames, export every `frame_NN.png` with
`export`, using the drawable names that `<name>_anim.xml` references. The Lottie has no glow. If the design has one,
add a blurred copy of the moving layer or a radial-gradient ellipse to `layers.json` and re-run, or edit the Lottie
JSON directly. If the keyframes barely move, re-run with a stronger `--motion` (for example "pronounced flicker, tips sway ±15%").

## Phase 6: build the UI

Build from `assets.json` (element positions), `design_spec.json` (tokens) and the exported assets. Multiply px values
by `px_to_dp`, then snap to the grid. Follow the project's own conventions. At the end, compare a screenshot of your
screen against the mockup.

## Rules learned from probing (don't relearn them)

- **Never use a text-only prompt for an asset of an approved design.** It comes out as a generic, different style.
  Always go through `forge`/`fix`, which pass the design as the reference.
- **Send exactly one reference image per edit call.** Multi-image edits have returned the mockup itself.
- Some backends ignore `quality`, `size` and `n`, so the kit always trims, crops and resizes locally and runs candidates
  as parallel calls. Output aspect ratio follows the prompt's wording, so `mockup` states it explicitly and retries on a mismatch.
- `background: transparent` works on the gpt-image models, with real alpha.
- For animation, never ask for the whole object in every frame, because static parts drift and grid lines get drawn.
  Generate only the moving layer, with "empty space between cells, no dividers".
- Full measurements and prompt rationale are in `references/api-findings.md`.
