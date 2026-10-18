# Phases 2-4: inventory, spec, asset production and platform export

Read when the user approves a design ("go ahead", "give me the assets"): `manifest`, `spec`, `iconspec`, `forge`, `fix`, `pick`, `svg`, `icons`, `trace`, `vd`, `export`. `$K` is `scripts/gptkit.py` in this skill's directory.

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
