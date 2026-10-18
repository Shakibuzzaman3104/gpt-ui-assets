# Changelog

All notable changes to gpt-ui-assets. Versions follow [semantic versioning](https://semver.org).

## Unreleased

### Added
- **Spacing and shape QA.** `critique` (and the automatic QA after `mockup`) now flags content too close to the
  screen or container edge, ambiguous grouping, uneven gaps, more than three corner radii, non-concentric nested
  corners and cards inside cards. On planted defects it caught 12 of 15, against 0 of 20 before.
- Each critique issue has a `scope`. Screen-wide issues (margins, density, section rhythm) are not sent to
  `--apply`, since an image edit can't re-flow a screen; the kit says to re-run `mockup` with the fix instead.
- `tokens` warns about spacing off the 4 dp grid and about more than four corner radii; `--snap-grid` rounds the
  spacing tokens (a deliberate departure from the mockup that `compare` will show).

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
