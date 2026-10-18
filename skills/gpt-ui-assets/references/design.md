# Phase 1: design screens with GPT (mockup, critique)

Read before the first `mockup`, and again when iterating with `--edit`, `--style-ref` or `critique`. `$K` is `scripts/gptkit.py` in this skill's directory.

**Know the platform first.** Ask the user if it isn't clear: `android` (Material 3 / Compose), `ios` (HIG / SwiftUI) or
`web`. `--platform` makes GPT design with that platform's native components, system UI, metrics, type scale and
icon family, for example the M3 navigation bar with its pill indicator vs the floating iOS tab bar. The choice is recorded
in `design.json`, and every later command inherits it.

```bash
python3 $K mockup "habit tracker home: date, progress summary, 5 habits with check + streak, add action, 4-tab nav; mint + warm white" \
  --platform android --out design/home --variants 2          # 1024x1536 portrait; --size 1536x1024 for web/tablet
python3 $K critique design/home/mockup_v1.png                # design QA -> mockup_v1_critique.png/json
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
  - The backend picks the aspect from the prompt's wording; `mockup` retries once on a mismatch and then saves the image
    with a `! variant N: backend returned WxH ...` warning. `--edit` / `--style-ref` images are sent downsized to a
    1536 px long side.
- **Critique:**
  - **Run `critique` on a variant before showing it**, then Read `<name>_critique.png` (red = high, orange = medium, blue = low).
    It checks text, alignment, icon consistency, contrast (measured locally: trust `[measured N:1]`), platform
    guidelines and implementability. With `--style-ref` it also catches a wrong app name, nav items, typeface or data.
  - `--apply` edits the fixes in and keeps the revision only if the issue weight drops. The verifier is lenient and the
    editor never sees the approved screen, so Read the revision. Give concrete follow-ups with `--edit`.
  - Vision QA is weak at spatial collisions: it can miss a FAB, snackbar or mini player covering content. **Check
    floating elements yourself**, and fix them with `--edit` (e.g. "move the FAB above the last row").
  - Don't chase absolute text sizes in the mockup. gpt-image draws everything small, and `spec` calibrates the real scale.
