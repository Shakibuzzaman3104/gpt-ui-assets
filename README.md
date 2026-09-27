# gpt-ui-assets

A [Claude Code](https://claude.com/claude-code) skill that turns GPT-generated UI designs into **production assets**.
Every icon, illustration and background in an approved design is **redrawn from scratch at full resolution and
matched 1:1**. Nothing is cropped out of the screenshot. You also get SVG / Android VectorDrawable icons,
frame-by-frame animations and Lottie files, ready for Claude to build the real UI in code.

<p align="center">
  <img src="docs/images/01-mockup.webp" width="260" alt="Approved mockup">
  <img src="docs/images/02-inventory.webp" width="260" alt="Asset inventory">
  <img src="docs/images/07-campfire.gif" width="200" alt="Generated animation">
</p>

## Why

When you ask GPT to "give me the assets of this design", it usually **crops them out of the mockup**. That gets you
low-resolution pixels with the card background baked in, neighbouring text bleeding in, and cut-off edges. Asking
it to redraw them from a text prompt produces **generic art in a different style**.

This skill does what a designer would do instead. It inventories the assets, then redraws each one from scratch
using a **focused crop of the element as the only reference**. Candidates are scored against the design, and copies
or crops are rejected automatically.

<p align="center"><img src="docs/images/08-crop-vs-redraw.webp" width="460" alt="Upscaled crop vs redrawn asset"></p>

*Left: the element cropped from the mockup and upscaled. Right: the redrawn asset. It's 5–7× the resolution
and 40–100× sharper at the edges, on a real transparent background.*

## What it does

| Step | Command | Output |
|---|---|---|
| Design mockups, iterate with edits | `mockup` | `mockup_v*.png` |
| Inventory every asset of the approved design | `manifest` | `assets.json` + overlay (bbox, kind, svg/png, dp estimate) |
| Design tokens | `spec` | palette, type styles, radii, spacing |
| Redraw raster assets 1:1, parallel candidates, scored | `forge`, `pick`, `fix` | transparent PNGs + review sheets |
| Vector icons | `svg`, `vd` | SVG + Android VectorDrawable XML (gradients supported) |
| Frame animation | `frames` | registered frames, GIF preview, `animation-list` XML |
| Lottie | `vectorize` | morph-ready vector keyframes → Lottie JSON + lottie-web preview |
| Platform export | `export` | Android densities (webp), iOS imageset, web 1x/2x |
| Setup check | `doctor` | config, endpoint, models, SVG renderer |

<p align="center">
  <img src="docs/images/03-forge-review.webp" width="400" alt="forge review sheet">
  <img src="docs/images/04-svg-review.webp" width="220" alt="svg review sheet">
</p>
<p align="center"><img src="docs/images/05-frames.webp" width="720" alt="Animation frames"></p>
<p align="center"><img src="docs/images/06-lottie.webp" width="520" alt="Lottie keyframes rendered by lottie-web"></p>

## Install

**As a plugin** (recommended):
```
/plugin marketplace add Shakibuzzaman3104/gpt-ui-assets
/plugin install gpt-ui-assets@gpt-ui-assets
```

**Or as a plain skill:**
```bash
git clone https://github.com/Shakibuzzaman3104/gpt-ui-assets.git
cp -r gpt-ui-assets/skills/gpt-ui-assets ~/.claude/skills/
```

Then install the Python dependencies (Python 3.9+):
```bash
pip install -r ~/.claude/skills/gpt-ui-assets/scripts/requirements.txt   # or the plugin's skills/gpt-ui-assets/scripts/
```

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
`OPENAI_API_KEY` and `OPENAI_BASE_URL` are used as fallbacks. Keep the file private (`chmod 600`).

Check the setup:
```bash
python3 ~/.claude/skills/gpt-ui-assets/scripts/gptkit.py doctor
```

## Use

Just talk to Claude:

> Design a meditation app home screen with GPT: greeting, featured sleep-story card, three mood tiles, a mini
> player and a 4-tab bar, in dark navy and lavender.

> The second one. Go ahead: give me all the assets and build it in Compose.

> Animate the campfire as a Lottie, a gentle flicker.

Claude generates the design, you approve it, and on "go ahead" Claude inventories the assets and forges them.
It reviews every result against the design itself, exports to your platform and builds the screen. You can also
start from a design made in the ChatGPT app or anywhere else: give Claude the image.

## How it stays 1:1

- **A focused reference beats the full mockup.** The element plus a 60% margin, upscaled, as the single reference
  image scored a 13.8 mean pixel difference against the design, against 21.8 with the full mockup. Text-only prompts
  come out in a generic style.
- **One reference per call.** Multi-image edits were observed to return the mockup itself.
- **Automatic rejection.** Candidates that return the screenshot or reference (`COPY`), lose transparency (`OPAQUE`),
  keep the icon's backdrop (`BACKDROP`) or touch the canvas edge (`CUT-OFF`) are ranked last.
- **Corrections only when they help.** `fix` asks the vision model for concrete differences and applies one
  corrective edit, but keeps it only if it scores better. In testing, correction passes often over-corrected.
- **Animation by layers.** The moving part is generated as a sprite sheet and the static part once. Both are
  registered to the approved asset by silhouette matching, so the logs never drift while the flame flickers.

The measurements are in [`references/api-findings.md`](skills/gpt-ui-assets/references/api-findings.md).

## Cost and limits

- Each image call takes 35–80 s and is billed at your provider's image pricing. `forge` with 2 candidates makes two
  calls per raster asset. Icons go through a single vision call per 6 icons.
- Some backends ignore `quality`, `size` and `n`. The kit handles that by trimming, cropping and resizing locally,
  and by running candidates in parallel.
- Image models don't draw to scale, so dp estimates need snapping to platform norms (Claude does this).
- The Lottie output has no glow or blur layers yet. Keyframe motion is subtle unless you ask for more.
- SVG previews use `cairosvg` if the cairo library is installed, then `rsvg-convert`, then macOS Quick Look.

## License

[MIT](LICENSE)
