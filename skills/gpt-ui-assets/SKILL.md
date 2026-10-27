---
name: gpt-ui-assets
description: Design app, tablet, foldable, desktop-app and website screens with OpenAI's GPT image and vision models (API key or the local Codex CLI), then turn an approved design into production assets redrawn 1:1, never cropped: SVG/VectorDrawable icons, PNG/WebP illustrations, launcher icons, frame, Lottie and AVD animation, smoke/steam/rain loops, theme code and fonts measured from the pixels, a screenshot diff of the built UI, and App Store / Google Play screenshot sets composited from real captures. Also writes GPT-researched feature blueprints. Use whenever the user wants GPT/ChatGPT to design screens, icons, illustrations or an app icon; approves a GPT design ("go ahead", "build it", "give me the assets"); wants assets used in the UI; wants an animated drawable, Lottie or AVD from existing art; wants store screenshots or a Play feature graphic; or complains that GPT cropped assets out of a mockup. Not for go/no-go idea validation, or design direction without GPT.
license: MIT
metadata:
  version: 5.4.0
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

`K` is `scripts/gptkit.py` inside this skill's base directory (the folder holding this SKILL.md; Claude Code prints it
as "Base directory for this skill" when the skill loads, whether it was installed as a plugin or copied into
`~/.claude/skills`). Every command documents itself with `python3 $K <cmd> -h`.
```bash
K="<this skill's base directory>/scripts/gptkit.py"
```

## Before you start

- **First run, new machine or a failing call:** run `python3 $K doctor` and read `references/setup.md`. When no backend
  is ready, `python3 $K setup` shows both options. Ask the user which they want, in plain words: **an API key** (OpenAI or
  any compatible endpoint; billed per call; best for pixel-exact asset work) or **the local Codex CLI** on their
  ChatGPT plan (no key, plan quota; fine for design, critique and research, but its image tool has no size or fidelity
  control). Then run `setup --backend api|cli`; for a key, have the user pipe it into `setup --api-key-stdin` themselves
  rather than pasting it into the chat. The setup also covers gateways and optional tools (Chrome, Playwright, aapt2,
  EbSynth).
- **Resuming a project** (after a break, a context reset or in a new session): run `python3 $K status <dir>`. It lists
  each unit's finished, missing and stale steps, the problems the kit's reports recorded, and the exact next command.
  No API calls.
- **Cost.** Image calls take 35–80 s each and cost money or quota. Paid commands print a `plan: ~N image calls … ≈ $X`
  line before the first call (on the Codex CLI it counts plan quota instead). `GPTKIT_QUALITY=low` makes cheap drafts. `forge`, `sequence render` and `research` take `--dry-run`. `GPTKIT_BUDGET_USD` stops any
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
| 1. Design | The user wants screens designed | `mockup "<brief>" --platform P [--device tablet\|foldable\|desktop] --out design/home`, `critique`, `--edit`, `--style-ref`, `--set tablet,desktop --anchor <approved>` | `references/design.md` |
| 1b. Hybrid | The concept must look portfolio-grade or move | `material`, `state`, `stage --init`, `stage`, `stage --compose` | `references/hybrid.md` |
| 2–4. Assets | The user approves a design ("go ahead") | `manifest`, `spec`, `iconspec`, `forge`, `svg`, `icons`, `pick`, `fix`, `trace`, `vd`, `export` | `references/assets.md` |
| App icon | A launcher icon is needed | `appicon concepts`, `appicon build` | `references/appicon.md` |
| 5. Animation | Art should move in place, pop, flicker or loop | `frames`, `vectorize`, `rig`, `animate` | `references/animation.md` |
| 5. Sequences | Something travels: smoke, steam, drops, spray, embers | `sequence layout → plan → render → build → reroll` | `references/sequence.md` |
| 6. Build 1:1 | Writing the UI code from the design | `fonts`, `redline`, `tokens`, then `compare` against a screenshot | `references/build-ui.md` |
| 7. Store screenshots | App Store / Play screenshots or a feature graphic for a built app | `store capture`, `store research`, `store plan`, `store art`, `store build`, `store check`, `store feature`, `store scene`, `store compose` | `references/store.md` |

A typical "design a screen and give me the assets" run:
```bash
python3 $K mockup "habit tracker home: date, progress, 5 habits with check + streak, 4-tab nav; mint + warm white" \
  --platform android --out design/home --variants 2    # 9:19.5 phone canvas, house style, aspect-checked, auto-critiqued
# first screen of a new app: add --directions so the variants differ in palette, type and signature, not just details
# Read each mockup_v*_critique.png, then show the user the variants
# ...iterate with --edit until the user locks one; screen 2 onward uses --style-ref <approved screen>
python3 $K critique design/home/mockup_v1.png --fit     # on the locked design: offer its fixes, apply the accepted ones
python3 $K manifest design/home/mockup_v1.png --out build/home   # Read manifest_overlay.png; fix assets.json by hand
python3 $K spec build/home                              # calibrated scale, palette, type, native widget per component
python3 $K forge build/home --candidates 3 &         # forge and svg in parallel (each image call takes 35-80 s)
python3 $K svg build/home & wait                      # then Read review/forge_*.png and review/svg_*.png
python3 $K vd build/home/svg/*.svg --out app/src/main/res/drawable
python3 $K export build/home/assets/hero.png --dp 240 --res app/src/main/res --name hero
python3 $K fonts build/home
python3 $K redline build/home
python3 $K tokens build/home --package com.acme.ui.theme --verify
# build the screen with spec's native widgets and the tokens' padding, gaps and radii (never default ones),
# screenshot it at the design's effective width, then:
python3 $K compare build/home shot.png               # also flags wrong corner radii, padding and gaps, with the token
```

## Rules that hold in every phase

1. **Know the platform and form factor first** (`android`, `ios`, `web`, `macos`, `windows`, `linux`; phone, tablet,
   foldable or desktop). Ask if it isn't clear. `--platform` and `--device` drive native components, system UI or
   window chrome, canvas, metrics and icon family, and every later command inherits them from `design.json`. Once a
   phone screen is approved, derive the other form factors with `--set ... --anchor <approved>`, not fresh prompts.
2. **Read every review sheet yourself before the user sees the result.** These include `*_critique.png`,
   `manifest_overlay.png`, `review/forge_*.png`, `review/svg_*.png`, `icons/preview.png`, the `appicon` sheets,
   `frames_review.png`, `keys_review.png`, `layout_review.png`, `fonts_review.png` and `compare/heatmap.png`. The
   metrics rank candidates, but your eyes decide.
3. **Never crop an asset out of the mockup, and never text-only prompt an asset of an approved design.** A text-only
   prompt gives a generic piece in a different style. `forge`, `fix`, `frames`, `sequence` and `material` pass the
   design as the reference.
4. **Design screen 2 onward with `--style-ref <approved screen>`.** Describing the style in words drifts.
5. **House style and canvas are automatic.** Every new screen is a clean, decluttered, modern UI with artwork only
   where it adds meaning, and no ads unless the brief asks; the user's `GPTKIT_HOUSE_STYLE` replaces this and
   `--no-house-style` drops it for one run. Phone screens render on a 9:19.5 (iPhone-proportion) canvas, and every
   other device on its own canvas. Each render
   is checked against it and fitted locally (empty gaps trimmed or extended, nothing scaled) when it comes back off;
   `aspect=OFF` means it could not be fitted. Never hand over an `OFF` variant as final.
6. **Measure, don't guess.** Vision hex guesses drift by up to ΔE 10, and GPT draws native components at ~0.7× scale.
   Take colours, sizes, radii and fonts from `spec`, `tokens`, `redline` and `fonts`, not from the model's prose.
7. **Flags outrank scores.** `EMPTY` and `COPY` are never shipped. After two unsuccessful `fix` rounds, tell the user
   what still differs instead of looping.
8. **Read the error code before rewording a prompt.** `[moderation_blocked]` needs a rephrase; `exhausted` and 5xx need
   a retry later; a model error needs `doctor`.
9. **Never claim a check that did not run.** `link skipped`, `not measured: <reason>` and `render check skipped` mean
   unverified. Say so.
10. **An image edit re-renders the whole image.** For a state change, use `state`, which composites only the changed
   region back. Never ship a raw edit as a state frame.
11. **GPT renders; it does not animate.** Code or the kit decides motion (`--procedural`, `sequence` layouts, `rig`),
    and GPT paints only the moving layer.
12. **Store listings get an art direction invented for that app.** There is no common template and no house look for
    store scenes: the world, audience and brand of the app decide the picture, and every example is an illustration,
    never a template. Never present a store shot with BAD-PERSPECTIVE, SMALL-PHONE, OCCLUDED or LOW-FIT; read every
    shot at half size first. Painted scenes are App Store only: Play asks for the interface without device frames or
    added scenery, so its set always comes from `store build` (frameless 9:16).

## Working with the user

- If the `attachment-preview` mod is installed (it ships in this repo under `mods/`), every picture a gptkit command
  prints lands in its pane as one grid: variants side by side, picks compared A–D, transparent assets on a checker,
  light or dark backdrop, icons as a contact sheet, with aspect and QA per picture. Point the user at it
  (`/attachments`) when showing variants; `/attachments add <folder>` shows a whole output folder.

- **Design checkpoints.** Stop at each one and wait for the user's answer:
  1. **Before a new design**, ask how many variants they want (default 2) and whether the variants should take
     different directions (`--directions`: palette, type, shape and signature differ) or stay close to one idea. Skip
     the question when the request already says, and for screen 2 onward, which follows the approved screen with
     `--style-ref`.
  2. **Show the variants** with what their critiques found, let the user pick one, and iterate with `--edit` until
     they lock it.
  3. **When the user locks a design, critique the locked file**: `critique <file>`, plus `--fit` on a phone screen and
     `--style-ref <approved screen>` for screen 2 onward. Read the sheet, drop the issues you judge wrong, and present
     the rest as suggested improvements, one line each with its fix. Apply only the ones the user accepts (`--apply`
     or a targeted `--edit`) and show the result. If nothing is worth fixing, say so.
  4. **After the user approves**, list the next steps and let them choose: the next screens (`--style-ref`), other
     form factors (`--set ... --anchor`), the assets (phases 2–4), the app icon, the 1:1 build, or, once the app
     runs, the store screenshots (phase 7).
- **Store listing style checkpoint.** Before planning or painting any store shot, show the reference sheets in
  `references/store-styles/` (`full-creative.jpg`, `flat-normal.jpg`, `mixed.jpg`, `mixed-bedside.jpg`) and ask which style the user wants:
  **full creative** (every slot a painted scene), **flat & normal** (upright phones on clean backgrounds), **mixed**
  (both in one set), or **something else** they describe. Skip it only when the request already says. The style
  shapes the App Store set; the Play set stays frameless in the same type and colours. Then show the
  hero in the chosen style before painting the rest of the set.
- Walk the user through `blueprint.md`, especially the MVP cut, before designing every screen.
- App-icon concepts, animation routes and whether to ship platform icons (Material Symbols / SF Symbols) instead of the
  generated SVGs are the user's call. Present the options with the review sheet.
- Hand over files with their destinations (`res/drawable`, `Assets.xcassets`, `public/`) and say what was verified and
  what was not.

## Pairs well with (if installed)

- An art-direction or design-reference skill before a `mockup`: put the chosen direction (palette, type, shape, style
  family) into the mockup brief.
- An idea-validation or PRD skill before `research`, when the user needs a go/no-go verdict or a full specification.
- A store-listing skill after `appicon build` and `store build`: `android/play_store_512.png` is the listing icon, and
  `store` makes the screenshots and feature graphic from real captures; the listing text is the other skill's job.

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
| `references/store.md` | `store` (capture, research, plan, art, build, check, feature, scene, compose) |
| `references/api-findings.md` | The raw measurements behind every rule; read only to settle a doubt |
