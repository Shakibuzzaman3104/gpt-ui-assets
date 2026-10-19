# Changelog

All notable changes to gpt-ui-assets. Versions follow [semantic versioning](https://semver.org).

## 5.2.0 (2026-10-19)

Design to code that keeps the design's geometry, and smoother animation.

### Added
- **Layout checks in `compare`.** Corner radii, padding and gaps are measured on both screens with `redline`'s edge
  finder and reported with the token to use, for example `corner radius 8dp, design 16dp, token PlanCardShape` or
  `gap above Start button (below Plan card) 300dp, design 320dp`. A wrong gap is named once, and the elements it pushes
  down get no `shifted` hints. On 12 GPT mockups rebuilt faithfully in HTML there were no false layout hints beyond
  the rebuilds' own mistakes. Of 4-8dp errors injected into them, it caught 20/22 gaps, 26/33 radii and 9/22 paddings.
  `build-ui.md` says to build with the tokens' padding, gaps and radii and to keep unusual values as designed.
- **`--tween N`** on `frames` (generated route) and `sequence build`/`propagate`. It adds N-1 in-between frames of the
  moving layer after every frame, the last→first pair included. The generated frames and the static pixels stay
  exact, and the loop length holds. Engines:
  - `flow` (default): built-in optical flow, needs OpenCV.
  - `rife`: rife-ncnn-vulkan, optional, MIT; `setup.md` has the 35 MB install.
  - `blend`: a crossfade.

  Measured on held-out frames of two synthetic loops (premultiplied error, a procedural smoke plume | a swaying warp
  of a GPT campfire): flow 3.0 | 2.6, RIFE 3.1 | 2.6, crossfade 4.3 | 5.6, repeating a frame 8.2 | 8.2. On real GPT
  flame frames the in-betweens keep the keys' edge sharpness, where a crossfade doubles the flame tip. `sequence`: lay
  out at 12 fps and tween to 24 fps with half the frames to render; GPT renders whose texture changes frame to frame
  were not measured. `doctor` reports RIFE.
- **`mockup --directions`**: a distinct art direction per variant (palette, type, radii, spacing, one signature
  device) with a different accent hue each; colours named in the brief win.

### Changed
- **Fitted easing for `rig`/`animate`** (`--ease fit`, the new default). Each key segment's bezier ease follows the
  motion, instead of a constant speed that jumps at every key. Against the exact motion at 8 keys, speed error went
  from 20% to 0.6% for sway, breathe, squash, bob and ripple, and from 47% to 37% for flicker. In lottie-web the
  in-between frames came 4.6× (sway) and 2.3× (breathe) closer to a 96-key reference. `--ease linear` keeps the old
  output.
- **Flow screens on phones** (onboarding, sign-in, players, camera, checkout, compose, edit forms) show a back or close
  control instead of the tab bar unless the brief asks for it, and `critique` flags a tab bar on them. Measured on 9
  briefs × 2 variants: 0 of 10 flow screens drew a tab bar and 8 of 8 detail screens kept theirs. An earlier wording
  left the bar on 4 of 4 iOS payment screens.
- Animated WebP encodes at libwebp method 5 instead of 6: within 2% of the size at 1/13 to 1/80 of the time.

## 5.1.1 (2026-10-19)

### Fixed
- **Square card among rounded ones.** `critique` now measures this locally instead of asking vision, which never
  reported it on a planted case. It raised no false alarm on 33 real mockups, and it caught 27 of 28 planted square
  cards plus both made by GPT edits.
- `SKILL.md` metadata version now matches the plugin (it still said 4.0.0).

## 5.1.0 (2026-10-18)

Spacing, shape and restraint. Measured on 24 probe mockups and planted defects: prompt wording alone barely moves
GPT's spacing, so the gains come from QA that sees spacing and from a measured token audit.

### Added
- **Spacing and shape QA.** `critique` (and the automatic QA after `mockup`) flags content within 16 units of the
  screen's side edges, text or controls within 8 units of their container's edge, ambiguous grouping, uneven gaps,
  more than three corner radii, non-concentric nested corners and cards inside cards (new `spacing` and `shape`
  categories). On planted defects it caught 12 of 15 root causes, against 0 of 20 before; a square-cornered card
  among rounded ones is still missed.
- **Issue scope.** Each critique issue is `local` or `screen`. Screen-wide issues (margins, density, section rhythm)
  are not sent to `--apply`, since an image edit can't re-flow a screen; the kit says to re-run `mockup` with the
  fix in the brief.
- **Token audit.** `tokens` warns about spacing off the 4 dp grid and about more than four non-pill corner radii, and
  records both under `audit` in `tokens.json`. `--snap-grid` rounds spacing tokens to the grid, a deliberate
  departure from the mockup that `compare` will show.
- **Quality-cap warning.** Image calls warn once when the endpoint answers a lower quality than requested. A sub2api
  gateway returned `medium` or `low` for `quality: high` with every image model.

### Changed
- **House style.** Stat tiles and icon backgrounds share the one accent instead of a different tint each, and
  section headers and labels are in sentence case, never ALL-CAPS.

## 5.0.0 (2026-10-18)

### Added
- **Form factors.** `mockup --device phone|tablet|foldable|desktop` with `--orientation`, `--posture` and `--page`,
  plus the `macos`, `windows` and `linux` platforms. Each device has its own canvas, logical size, system chrome,
  native components and critique checks: no stretched phone layouts on tablets, nothing across a foldable's hinge,
  correct window chrome and pointer density on desktop, above-the-fold hierarchy on the web.
- **Device sets.** `mockup --set tablet,desktop --anchor <approved.png>` derives each form factor from one approved
  screen, keeping the information architecture and data while recomposing the layout.
- **Local Codex CLI backend.** `GPTKIT_BACKEND=api|cli|auto`: design, critique and research through the Codex CLI on
  a ChatGPT login, with no API key. `setup` picks and configures a backend; `doctor` checks both.
- **The attachment-preview mod** (`mods/attachment-preview`): a terminal pane that shows gptkit's output as variant
  grids, compares up to four picks side by side, draws transparent assets on checker, light or dark backdrops, and
  shows aspect ratio, size and design QA per picture.
- **House style.** Every new design follows a default art direction (clean, decluttered, modern; artwork only where
  it carries meaning; no ads unless asked). `GPTKIT_HOUSE_STYLE` replaces it and `--no-house-style` skips it.
- **Automatic design QA.** `mockup` critiques every variant as soon as it is rendered (`--no-critique` skips it).
- `GPTKIT_QUALITY=low|medium|high` for cheap drafts.

### Fixed
- Chrome is never launched inside the Codex CLI's macOS sandbox, where it aborts on start (SIGABRT in
  TransformProcessType) and leaves a crash report; the kit falls back to another renderer and says why.
- Generated screens invent a product name instead of borrowing a real brand.
- A contrast issue whose measured ratio already meets its target drops to low and says "check by eye".
- `mockup --set` writes and prints each device under its final name and keeps the folder's other screen records.
- A `--style-ref` from another platform no longer turns a desktop or web screen phone-shaped.
- The offline tests no longer write mocked calls into the user's cost ledger or reach a real Codex CLI.

### Changed
- New phone screens render on a 9:19.5 canvas (1152x2496, current iPhone proportions) instead of 2:3. Every render is
  measured against its canvas and fitted locally (empty gaps trimmed or extended, nothing scaled) when the backend
  returns another shape. An endpoint that rejects the size falls back once to the nearest classic size.
- The text model is picked from the endpoint's model list (`gpt-6.1-sol`, `gpt-6-sol`, …) instead of defaulting to
  the gateway alias `gpt-6`.
- The kit is a regular Python package (`scripts/gptkit/`) with explicit imports; `scripts/gptkit.py` stays the entry
  point.
- Playwright is optional (only `stage` needs it).
- Web mockups calibrate to the device's logical width (previously 390 px for every web page).
- The longest functions (`cmd_compare`, `svg_to_vd`, `cmd_fonts`, `_status_scan`, `cmd_appicon_build`, the argparse
  tree) are split into named helpers, with no change in behaviour or `-h` output.
- On the Codex CLI backend the `plan:` line counts plan quota instead of dollars.

## 4.0.0 (2026-10-18)

- Cost ledger, `cost` command, a `plan:` estimate line before paid work, `GPTKIT_BUDGET_USD` and `--dry-run`.
- `status`: offline project progress, recorded problems and the next command.
- `stage --compose`: carries a hybrid page's timeline into Compose.
- SKILL.md became a short router; phase details load from `references/` on demand.
- The CLI was split into `scripts/gptkit_parts/`.

## 3.2.0 (2026-10-02)

- Hybrid screens: `material`, `state` and `stage` build screens in HTML/CSS/JS around GPT-painted materials.

## 3.1.0 (2026-10-01)

- Per-task image model routing (gpt-image-2.5 flare for concepts, sunburst for precise edits), transparency-aware,
  with a pinned override.

## 3.0.0 (2026-09-28)

- Engine plan for sequences and a production-readiness pass across every command.

## 2.2.1 and earlier

See the git history.
