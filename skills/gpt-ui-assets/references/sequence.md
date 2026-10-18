# Layout-driven sequences: code animates, GPT paints, EbSynth carries the paint

Read before `sequence layout|plan|render|build|propagate|reroll`. `$K` is `scripts/gptkit.py` in this skill's directory.

An image model has no notion of
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

## Rules learned from probing

- **Lay the motion out in code, then have GPT render.** A brief that names the frame ("frame 10 of 72 of a seamless
  3-second loop"), locks the static art and makes the layout binding kept the layout on every one of 12 keys, and on
  67 of 72 frames on the first pass (the other 5 were fixed by `reroll`).
- Never write "keep its exact shape" about the moving part (see `animation.md`).
