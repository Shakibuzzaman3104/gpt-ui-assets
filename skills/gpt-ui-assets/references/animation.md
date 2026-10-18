# Phase 5: animation (frames, vectorize, rig, animate)

Read before any animation command. For something that travels (smoke, drops, spray, embers) read `sequence.md` instead. `$K` is `scripts/gptkit.py` in this skill's directory.

## Pick the route

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

## Rig the exact art into parts (vector, 1:1)

`vectorize` redraws the art in simplified form: keyframe 0 of GPT's
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
- Tracing notes (`trace`, see `assets.md`): `--work` (long side, default 512, never above the source: upscaling traces the pixel
  staircase), `--colors` (k-means colours), `--tol` (curve fit, work px), `--bands` (glow bands, 0 = none), `--display` (px the
  fidelity is measured at; default 3× the dp width, else ≥192 long / ≥96 short side), `--dp` (VD width; default 48 dp on the long
  side, so pass it for illustrations), `--name` (single input), `--prefix` (VD file prefix, default `img_`), `--bg` (review
  backdrop). VD file names are sanitised to `[a-z0-9_]` and printed when they change. `trace.json` `vd.aapt2` says exactly what
  ran: `compile+link ok (android-NN)` only with `platforms/android-*/android.jar` on `ANDROID_HOME`; `compile ok; link skipped`
  otherwise (attribute names, enums and dimensions unchecked). More than 400 colour regions is photographic: no SVG/VD is written and
  the advice is `png/webp`. `--preset vtracer` ignores `--colors/--tol`.

## Rules learned from probing

- **Never ask for the whole object in every frame.** Static parts drift and grid lines appear. Generate only the moving
  layer, and use `--procedural` for rigid motion.
- **GPT is the rendering artist, not the animator.** It cannot keep a trajectory across a sprite sheet. For anything
  that travels, use `sequence` (`sequence.md`).
- **Never write "keep its exact shape" about the moving part.** It reads as an instruction to stay still (a video model
  returned the input 122 times). Lock the camera, the framing and the static art instead.
- **To move a part that hides something** (fan blades, a bird on a branch), have GPT redraw the base with the part
  removed and repaired, and keep the original pixels everywhere outside the part. Recomposite at rest and compare with
  the master before animating: a mean difference of about 0.2 when the part keeps its own pixels, about 3 when GPT had
  to redraw the part as well.
- gpt-6 animates too subtly by default; the kit's motion check retries once. Raise `--strength` before rewording.
