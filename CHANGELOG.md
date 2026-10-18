# Changelog

All notable changes to gpt-ui-assets. Versions follow [semantic versioning](https://semver.org).

## 5.0.0 (unreleased)

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
