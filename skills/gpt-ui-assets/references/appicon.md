# App launcher icon

Read before `appicon concepts` or `appicon build`. `$K` is `scripts/gptkit.py` in this skill's directory.

```bash
python3 $K appicon concepts "Hush, a calm sleep app; dark navy and lavender; moon motif" \
  --mockup build/home/mockup.png --out icon --variants 3            # 1 image call per variant -> icon/review/concepts_*.png
python3 $K appicon build icon/concept_v2.png --out icon/build \
  --subject "the glowing crescent moon, its stars and the wind swoosh"   # 0 image calls + 1 gpt-6 call (themed glyph)
```
- **Concepts:**
  - Pass `--mockup` whenever an approved design exists. It's sent at low input fidelity with a "new icon, not the screen's
    illustration" prompt. Without it, concepts come out generic.
  - **Read the concepts sheet.** It shows each concept full-bleed, in circle/squircle/rounded/teardrop masks and at 48 px.
    Reject these flags:
    - `BUSY-BACKGROUND`: a scene, not an icon.
    - `COPIED-ILLUSTRATION:<asset>`: the concept is the screen's own illustration re-posed. This needs `assets.json` next to
      the mockup. Re-run with a stronger `--directions` ("a minimal emblem of X, not the mascot"), `--crop` a small motif,
      or drop `--mockup` and pass `--colors`.
    - `SUBJECT-TOO-BIG` and `BLEEDS-OFF-EDGE`.

    Lower `palette_de` is closer to the
    design. Let the user pick.
  - A blocked or failed variant prints `! variant N (<direction>) failed: ...` and the others are kept (exit 1 only when
    none succeeded). Reword that direction rather than re-running everything.
- **Build** defaults are the measured best:
  - The foreground is matted locally against the background that actually ships, so the icon recomposes the concept almost
    exactly. `--separate auto` (the default) switches to an image-edit foreground (1 call) when the matte leaves the subject
    mostly translucent, which happens with a light subject on a light background. Otherwise iOS dark/tinted and the silhouette would look ghosted.
  - The background is a fitted VectorDrawable gradient (or a colour).
  - The subject is shrunk into the 66 dp safe circle only when measurement says so.
  - The monochrome (themed) layer is a gpt-6 glyph converted to a VectorDrawable. It is referenced only when its render
    was measured and placed inside the safe zone (`glyph.ok: true`). A failed, empty, too-faint or unverifiable glyph
    (no working SVG renderer) falls back to the silhouette with a `!` line; a multi-colour glyph is flagged in `warnings`.
  - Use `--bg image` (1 call) only for textured or scenic backgrounds; the report says when it's needed.
  - The concept must be the opaque full-bleed square. A transparent input is flattened onto black and reported in `warnings`.
    `--name` must already be a resource name (`[a-z0-9_]`, starting with a letter); the error suggests one.
- **Read `review/appicon_*.png`**, and check `appicon.json`:
  - `safe_zone.ok`, `glyph.ok` (with `--mono glyph|both`), `web_maskable.ok`, `ios.opaque`, `ios.tinted_has_alpha`, and
    `warnings` (absent, or understood).
  - `ios.actool` starts with `ok`. `skipped (no Xcode actool)` means not checked (Linux).
  - `aapt2` starts with `aapt2 compile+link ok`. `aapt2 compile ok; link skipped (...)` means resource syntax compiled but
    attribute names and enums were NOT checked: install `sdkmanager "platforms;android-35"` and re-run.
- **Install:**
  - Android: copy `android/res/*` into `app/src/main/res` and point the manifest at `@mipmap/ic_launcher` and `@mipmap/ic_launcher_round`.
    One `mipmap-anydpi-v26/ic_launcher.xml` with an inline `<monochrome>` is correct (API 26-32 ignore the element; no
    `-v33` folder needed). Legacy 48 dp bitmaps cover mdpi-xxxhdpi.
    `android/play_store_512.png` is the Play listing icon: 512 px, opaque, well under the 1024 KB limit (`play_store.kb`).
  - iOS: copy `ios/AppIcon.appiconset` into `Assets.xcassets`. Light is opaque. Dark is the foreground on transparency (the
    system draws the dark backing). Tinted is grayscale with alpha, no baked black. This follows Apple's asset-catalog
    format for iOS 18; the set compiles with Xcode's actool (macOS, 2026-09-29), which does not check alpha.
  - Web: the files and the `manifest-icons.json` snippet go in `public/`.

## Rules learned from probing

- A full-mockup reference at high fidelity copies the hero illustration into the concept; the kit sends it at low
  fidelity. If a concept still copies it (`COPIED-ILLUSTRATION`), change the directions rather than the fidelity.
- A themed-icon glyph ships only when its render was measured: check `glyph.ok`; `false` means the silhouette shipped.
- iOS: light is opaque, dark is the foreground on transparency, tinted is grayscale with alpha. actool does not check
  alpha, so the kit's `ios.*` fields are the check.
