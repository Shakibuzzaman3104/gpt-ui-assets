# Phase 1: design screens with GPT (mockup, critique)

Read before the first `mockup`, and again when iterating with `--edit`, `--style-ref` or `critique`. `$K` is `scripts/gptkit.py` in this skill's directory.

**Know the platform first.** Ask the user if it isn't clear: `android` (Material 3 / Compose), `ios` (HIG / SwiftUI) or
`web`. `--platform` makes GPT design with that platform's native components, system UI, metrics, type scale and
icon family, for example the M3 navigation bar with its pill indicator vs the floating iOS tab bar. The choice is recorded
in `design.json`, and every later command inherits it.

```bash
python3 $K mockup "habit tracker home: date, progress summary, 5 habits with check + streak, add action, 4-tab nav; mint + warm white" \
  --platform android --out design/home --variants 2          # 1152x2496 (9:19.5) phone canvas; web defaults to 1536x1024
# each variant is aspect-checked and critiqued automatically -> mockup_v1_critique.png/json (--no-critique skips)
python3 $K mockup "make the summary card taller" --edit design/home/mockup_v1.png --out design/home --name mockup_r2
# every further screen of the same app, in the approved screen's exact design language:
python3 $K mockup "Profile screen, Profile tab active: avatar + name, 3 stat tiles, weekly chart, settings list" \
  --style-ref design/home/mockup_r2_v1.png --out design/profile
python3 $K critique design/profile/mockup_v1.png --style-ref design/home/mockup_r2_v1.png --apply
```
- **Screens:**
  - Show the user each variant (Read the PNG) and iterate with `--edit` until they approve. A design made elsewhere
    (the ChatGPT app, a Figma export, a screenshot) works the same way: use its image path, and pass `--platform` to later commands.
  - **Design screen 2 onward with `--style-ref <approved screen>`, never text-only.** Describing the style in words drifts:
    a different wordmark, a different art style, text baked into the art. In the prompt, say which tab is active and which
    shared chrome the screen has. `layout_diff < 10` means GPT copied the reference; the kit retries once.
  - `--edit` / `--style-ref` images are sent downsized to a 1536 px long side.
- **Canvas and validation:**
  - New phone screens render at `1152x2496`, the 9:19.5 proportion of current iPhones (and close to most Android
    phones). `--size` overrides it; `--edit` keeps the edited screen's canvas; web defaults to `1536x1024`.
  - The backend often ignores `size`. `mockup` measures each render, retries once when its ratio is more than 2% off,
    then fits it locally: rows are removed from, or repeated inside, the flat gaps between sections, so nothing is
    stretched. The untouched render is kept as `<name>_vN_raw.png`. Each output line ends `aspect=ok`,
    `aspect=fitted ...` or `aspect=OFF (...)`. Fitting stops at 15% of the height; an `OFF` variant needs a re-run.
- **House style:** every `mockup` and `--style-ref` prompt carries the house art direction: clean, decluttered, modern;
  one focal point and one accent; artwork only where it carries meaning (hero, empty state, onboarding); real content,
  no ads or promos unless the brief asks for them. `--edit` does not add it (an edit changes only what it names).
  Set `GPTKIT_HOUSE_STYLE` in `~/.config/gptkit/env` to replace it with the owner's own direction, `off` to drop it,
  or pass `--no-house-style` for one run. The brief always wins over the house style.
- **Critique:**
  - `mockup` critiques every variant automatically (in parallel) and writes `<name>_critique.png/json`; a failed call
    prints `critique NOT run`, so run `critique` on that file by hand. Read `<name>_critique.png` (red = high,
    orange = medium, blue = low) before showing a variant.
    It checks text, alignment, icon consistency, contrast (measured locally: trust `[measured N:1]`), platform
    guidelines and implementability. With `--style-ref` it also catches a wrong app name, nav items, typeface or data.
  - `--apply` edits the fixes in and keeps the revision only if the issue weight drops. The verifier is lenient and the
    editor never sees the approved screen, so Read the revision. Give concrete follow-ups with `--edit`.
  - Vision QA is weak at spatial collisions: it can miss a FAB, snackbar or mini player covering content. **Check
    floating elements yourself**, and fix them with `--edit` (e.g. "move the FAB above the last row").
  - Don't chase absolute text sizes in the mockup. gpt-image draws everything small, and `spec` calibrates the real scale.
