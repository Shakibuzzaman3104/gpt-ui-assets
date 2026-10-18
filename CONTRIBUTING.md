# Contributing to gptkit

Rules for changing `skills/gpt-ui-assets/scripts/gptkit.py`. They were learned from probing the APIs and from bugs that
shipped. Rules for *using* the kit live in the skill (`SKILL.md` and `references/`). Full measurements are in
`skills/gpt-ui-assets/references/api-findings.md`.

## Layout

`scripts/gptkit.py` holds only the imports and a loader. The code is in `scripts/gptkit_parts/NN_area.py`, one file per
area (`04_forge.py`, `08_sequence.py`, `23_cli.py` for argparse, …), executed in name order into `gptkit.py`'s own
namespace. At runtime it is still one module: a part uses names from earlier parts as plain globals, `import gptkit`
exposes everything, and the tests' `K.generate = fake` patching reaches every part. So:
- edit the part, never paste code back into `gptkit.py`;
- a name used at import time (a constant, a default argument, a decorator) must be defined in an earlier part;
  names used inside functions can live anywhere;
- a new subcommand needs its `cmd_*` function in its area's part and its parser in `23_cli.py`.

Run the tests from `skills/gpt-ui-assets`: `for t in tests/test_*.py; do python3 $t || echo FAIL $t; done`. Every file
runs on its own without pytest, offline, with the API mocked. Keep Python 3.9 compatibility (macOS system Python).

## API behaviour

- **Exactly one reference image per edit call.** Multi-image edits returned the mockup itself on the OpenAI-compatible gateway
  the kit was probed on; the official API accepts up to 16 reference images per edit, unverified here.
- A single-reference edit designs a genuinely new screen when the prompt says "DIFFERENT screen … style guide only … do
  not copy its layout" (5/5 new layouts). A design-system board as the reference is worse than the screen itself.
- Some backends ignore `quality`, `size` and `n` (measured on a gateway; the official API defines them), so the kit
  trims, crops and resizes locally and runs candidates in parallel. Output aspect ratio follows the prompt's wording, so
  `mockup` states it and retries on a mismatch. A gateway that returns JPEG instead of PNG has ignored
  `background: transparent` too; the kit re-encodes to PNG, records `source_format` and says so once.
- API errors carry the server's `error.code` and the prompt (`[moderation_blocked] ... | prompt: '...'`). Only
  `error.param` decides which optional parameter to drop; prose that mentions "background" does not.
- A `background=transparent` call can still come back opaque (a gateway dropped the parameter): check opacity (> 95 %
  opaque or all four corners opaque) before using any generated raster as a transparent asset. Alpha edges from the
  gpt-image models need no defringing.
- Forged art drifts in colour (yellower, more saturated). The Lab colour match fixes that for transparent assets; never
  apply it to opaque backgrounds, because it washes out their accents.
- For app-icon concepts, send a full-mockup reference at low fidelity (high fidelity copies the hero illustration).
  Never ask the image model for the foreground layer when the subject contrasts with its background: the local matte is
  far more faithful. Light-on-light concepts switch to the image-edit foreground.
- gpt-6 animates too subtly by default, so the motion check retries. AVD `pathType` keyframes crash the inflater, so
  multi-key morphs are relayed across path copies.
- Vision hex guesses drift (ΔE up to 5 on text ink and 10 on borders). Snap colours to pixels; never ask vision for numbers.

## Robustness

- Clip every bbox a vision call returns to the image before using it (`critique`, `manifest`, `spec`, `--vision`); a
  box mostly outside the mockup drops the item with a printed reason instead of writing a garbage crop.
- Flags outrank `fid`: rank candidates by flag weight (`EMPTY` 1000 > `COPY` 100 > `OPAQUE`/`BACKDROP` 10 > `CUT-OFF` 3),
  never by flag count; an empty render has fewer flags than a real one.
- Sample the backdrop colour from a ring *outside* the bbox and test the silhouette *boundary*; testing all solid pixels
  against the bbox corners flags every light icon on a light page.
- Validate an imported file completely before the first paid call, and never feed import errors back to the model as if
  it had made them (a Lucide `currentColor` icon cost two calls and a wrong retry before).
- When a pool of paid API calls fails part-way, persist each result on arrival, name what is cached, catch `ApiError`
  per job (never blanket exceptions) and exit 1 only when nothing usable was produced.
- Files one command writes for another store dir-relative paths, are merged with re-read + temp + `os.replace` (parallel
  `--only` runs otherwise lose each other's entries) and record what they were made from (hash sidecars), so the next
  stage refuses stale inputs by name instead of reusing them silently.
- Never cache a network failure (metadata, licences); cache only validated successes, written atomically.
- Every failure is one stderr line. Nothing else may write to stderr on the error path (third-party import warnings
  included).

## Rendering and validation

- Headless Chrome (141, `--headless=new`) lays out 87 px less than `--window-size`: pad every capture and crop; as root
  add `--no-sandbox`. Never report a renderer as available without a probe render, and report a number that was not
  measured as `not measured: <reason>`; never guess `prefer`/`ok` from file size or silence.
- `aapt2 compile` alone checks well-formedness only: say `link skipped`, run the offline VD lint and never print
  `valid`; `compile+link ok` needs `platforms/android-*/android.jar`.
- Android resource names are enforced by AGP, not aapt2 (`[a-z0-9_]`, letter first, no Java keyword): every file that
  lands in `res/` is named through `_res_name`; pixel-sized frames belong in `drawable-nodpi`.
- SVG gradient attributes have defaults (`x2=100%`, `cx=cy=r=50%`) and percentages: resolve them before writing a
  VectorDrawable, or it inflates with `gradientRadius=0` and crashes.
- Accept a fitted gradient only when it is visible (stops > 6 ΔE apart) and clearly better (RMSE −2 absolute); a
  gradient costs API 24+.
- Never upscale a raster before tracing: the tracer follows the source's pixel staircase (11,855 vs 722 chars on the
  same ring).

## Measurement

- Normalise an icon set per size group, never per glyph: per-glyph fitting to the live area changes stroke weights by up
  to 27 % and then flags the set as inconsistent.
- Per-frame duration lists (`round((i+1)·1000/fps) − round(i·1000/fps)`) keep a loop exact; Pillow's animated WebP
  needs `method=6` (its default is 0).
- QA that compares neighbouring frames must subtract what the layout itself does at that frame, or every
  abrupt-but-intended change (a birth, a splash) becomes an "outlier".
- A gate on the mean of per-layer numbers is decided by the smallest hyperactive layer; weight by area.
- A tear metric must see open gaps: erode the rest pose by the travel and count what a key leaves uncovered.
- Font-weight fits are renderer-specific (stem darkening): score candidates with and without emboldening and treat
  one-step ties as the spec's weight. Tests must not depend on one renderer's fitted size.
- A drop shadow makes "distance from the outside colour" useless for edges; use the largest colour step along the profile.

## Generated code

- Compose theme code must compile: namespace colours (`AcmeColors.X`; `object` members shadow top-level names),
  `FontWeight(n)` for the 450/550 weights GPT specs carry (`W{n}` exists only for hundreds), and one hex normalisation
  at load (`#fff`, `#rrggbbaa`, `rgb()` all occur in real specs).
