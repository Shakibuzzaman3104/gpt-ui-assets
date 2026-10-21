# Phase 7: store screenshots (App Store and Google Play)

Read before any `store` command, or when the user wants App Store / Play screenshots or a feature graphic. `$K` is
`scripts/gptkit.py` in this skill's directory.

The UI in a store shot must be the app's own pixels: GPT redraws any screen it is shown (text, icons and layout drift),
and Apple wants the app in use (2.3.3). So `store` never sends a screen through an image model. Real captures are
composited pixel-exact in HTML and rendered in Chrome; GPT paints only the **art plate** behind them (an edit of the
slot's own capture, so it keeps the app's style) and, optionally, the **display lettering** (drawn transparent, then
read back by OCR and re-rolled until it spells the caption exactly). Captions, sizes, contrast and UI fidelity are
checked locally.

```bash
python3 $K store capture store --name hunt                # current screen of the booted simulator / device
python3 $K store capture store --from ~/Desktop/shots      # or import real captures (stems become names)
python3 $K store research store --term "word game"         # optional: what the category's top listings do
python3 $K store plan store --app "WordFight: live 1v1 word duels for casual players"
python3 $K store plan store --lint                         # after editing storyboard.json
python3 $K store art store --feature                       # plates (+ --lettering for display captions)
python3 $K store build store                               # ios-phone + play-phone (+ tablets when captured)
python3 $K store check store                               # store rules; exits 1 on a FAIL
python3 $K store feature store --logo brand/logo.png       # Play feature graphic 1024x500
```

## 1. Capture

- `store capture DIR --name NAME` shoots the one booted iOS simulator (`xcrun simctl`) or adb device, with
  `--platform`, `--udid` or `--serial` to choose. It needs neither argent nor Maestro: drive the app to the screen
  yourself (or with `--url` for a deep link and `--wait` seconds), then capture.
- The status bar is cleaned for the shot and restored after, also when the shot fails. iOS uses a 9:41 override with
  full Wi-Fi and battery (`discharging` at 100; `charged` draws a bolt). Android uses demo mode. `--no-status-bar`
  skips this.
- Captures land in `captures/<platform>-<device>[-landscape]/NAME.png` and in `captures.json`, with the status bar
  height that frameless cards crop. The same NAME on an iPhone, an iPad and an Android device is one screen that each
  set picks from.
- `--from FOLDER` imports existing captures. It refuses image-model sizes (1024x1536 and the like; real captures only)
  and warns on iOS sizes that are not native. Pass `--platform` and `--kind` when the size can't tell.
- **Fictional data** (Apple 2.3.9): sign in as a demo persona before capturing. `plan` lists any real-looking names,
  handles or emails it sees, and `check` repeats them as warnings.
- Stage each screen at its best moment: mid-action, a winning state, no expiring timers or error toasts. Read every
  capture before planning.
- Launching an iOS app from another app leaves a "◀ App" breadcrumb in the status bar. Press home, then launch.

## 2. Research (optional)

`store research DIR --term "<category words>"` pulls the top App Store results from the iTunes Search API, downloads
slots 1-3 of each (`--limit`, `--per-app`) into labelled `research/competitors_N.jpg` sheets, and has one vision call
code them into `playbook.json`/`.md`: slot-1 type, caption voice and case, frame style, background, palette, callouts,
what to copy and what to avoid. `plan` reads the playbook when it exists. `--no-ai` stops after the sheets. Play has no
public listing API, so research covers the App Store only.

What the research behind this feature found (308 top apps, Phiture/SplitMetrics data):
- About 13% of visitors scroll the screenshots. Slots 1-3 are the ad, and search shows them. Lead with the hook.
- One idea per slot: a 2-6 word benefit caption plus one UI moment. A themed background beats a flat colour, and
  zoomed callouts of the key element test well.
- Games mostly run frameless; utilities and finance use a device frame. Categories differ, so brand first, then
  category cues.

## 3. Plan

`store plan DIR --app "<name: what it does, for whom>"` sends the captures (and the playbook) to one vision call and
writes `storyboard.json` + `.md`. Per slot it holds the capture, `caption`/`sub` for iOS, `caption_play`/`sub_play` for
Play, `arch`, `ink`, callout or card boxes (fractions of the capture), `plate_brief` and `alt`. It also writes the brand
block (colours, `ink_dark`/`ink_light`, Google Fonts, art and lettering style), the feature subtitle, privacy findings
and notes on missing screens. Edit the JSON freely, then `store plan DIR --lint`.

Archetypes:
- `framed`: the whole screen in a device (iOS) or a frameless card (Play).
- `callout`: framed, plus 1-2 enlarged crops of the element that proves the caption. Each sits over its own place on
  the screen, 1.2x, ringed and shadowed. Full-width bands read best.
- `fan`: 2-3 screens as tilted cards; `cards[0]` is the hero. Use it once at most.
- Tablets reuse the phone boxes unless a slot has `calls_tablet` / `cards_tablet`. Play reads `calls_play` /
  `cards_play` first, for an Android layout that differs from the iOS one.

Floats (any archetype): up to 3 pieces of the app's own art (stickers, 3D icons, mascots; transparent PNGs) drift
around the screen, as on many top listings. Each slot's `floats` (or `floats_play` / `floats_tablet`) is a list of
`{"src", "x", "y", "w", "rot", "layer", "blur"}`: `src` is a path from the project (files outside it are copied into
`floats/`), `x`/`y` the centre as fractions of the canvas, `w` the width as a fraction of it, `layer` `back` (behind
the screen, the default) or `front`, and `blur` px for depth. Keep them off the caption and on the screen's edges:
`check` warns when one touches the caption or when front floats hide more than 3% of the screen. Use the app's art
only, 1-3 per slot, and leave a slot or two without.

Copy lint (also run by `check`):
- **FAIL**: prices (Apple 2.3.7), another platform's names on iOS (2.3.10), alt text over 140 characters.
- **WARN**: claims (#1, best, award, millions) unless `--claims` lists them (2.3.14); Play's ranking and promo words
  (best, #1, top, new, free, sale) and calls to action (download, play now); captions over 6 words; alt text starting
  "Image of".

## 4. Art

`store art DIR` paints `art/plate_<slot>.png` for every slot with a `plate_brief`. Each plate is an edit of that slot's
capture, so the palette, light and rendering match the app.
- The top 30% stays calm for the caption, and the middle soft because the device covers it.
- Plates are flagged `TEXT` (a vision check; one re-roll, `--no-text-check` skips it) and `BUSY-TOP` (measured locally).
- `--feature` adds the wide `plate_feature.png`. `--lettering` draws each caption as display lettering, trimmed to its
  alpha box and OCR-checked, up to `--tries` (6/6 read back exactly on the first try in the WordFight probe).
- Read `art/review.jpg`. A slot with no plate falls back to a brand gradient. Lettering that never passed OCR falls back
  to the HTML caption. `--dry-run` prints the calls and cost.
- For an app with no illustration style, write an abstract `plate_brief` (soft gradient in the brand colours) or none.

## 5. Build

`store build DIR` renders every slot for each target:

| Target | Output px | Canvas (CSS px × DPR) |
|---|---|---|
| `ios-phone` | 1206x2622 (the required Dynamic Island medium slot; App Store Connect scales it down) | 402x874 × 3 |
| `ios-phone-large` (`--iphone-large`) | 1320x2868 (optional large slot) | 440x956 × 3 |
| `ios-tablet` / `-landscape` | 2064x2752 / 2752x2064 (13") | 1032x1376 / 1376x1032 × 2 |
| `play-phone` | 1440x2560 (9:16, ≥1080 px for promotion) | 360x640 × 4 |
| `play-tablet` / `-landscape` | 1440x2560 / 2560x1440 | 720x1280 / 1280x720 × 2 |

- iOS sets use only iOS captures (2.3.10), in an iPhone frame with the Dynamic Island, or in an iPad frame.
  `--frame card` makes them frameless.
- Play is always frameless (Play asks for no device imagery). It uses Android captures and falls back to iOS ones with
  a note, so capture Android for release.
- Tablet sets appear when tablet captures exist (`--devices phone,tablet` to choose).
- `--variant lettering` uses the OCR-checked lettering. `--locale de` builds from `storyboard.de.json` into `build-de/`.
- Every shot is rendered three times: as shipped, with the text hidden (`bare`, for contrast) and with the overlays
  hidden (`fid`, for fidelity).
  - Caption contrast is WCAG against the worst 5% of the background under it. Below 4.5:1 the caption moves onto a
    pill (`--no-pill` to keep it bare).
  - UI fidelity is CIELAB ΔE of the visible screen against the capture: mean ≤ 2.5 and p99 ≤ 20. Resampling at text
    edges alone gives a p99 of 8-16. WordFight's real captures measured a mean of 0.5-1.2.
- Output is 24-bit RGB with no alpha, as PNG, or JPEG above Play's 8 MB. Each target also gets `alt.txt` (Play alt
  text), `sheet.jpg` and, for `ios-phone`, `search_row.jpg`: the first three shots at search-result size. **Read the
  sheets and the search row before handing the set over.**

## 6. Check

`store check DIR` reads `build/report.json` and the files, and exits 1 on any FAIL:

- **Images**: exact size per target, RGB mode, ≤ 8 MB, and Play's 320-3840 px and 2:1 limits.
- **Counts**: App Store 1-10. Play 2-8, plus a promotion check: 4 shots at 1080 px+, 3 for games.
- **Quality**: fidelity and headline contrast fail; subhead contrast warns; the iOS search headline should be 9 pt or
  more (CSS px × 0.261).
- **Copy**: Play captions should cover under 20% of the image, plus the copy lint, privacy notes and the feature checks.

## 7. Feature graphic

`store feature DIR --logo logo.png` renders the 1024x500 Play feature graphic:
- `art/plate_feature.png` (or the brand gradient) with a soft centre glow, the logo, or `--title` in the headline
  font, and the storyboard's subtitle. `--icon appicon.png` adds the app icon as a rounded tile above them, which
  ties the graphic to the icon users see in search.
- It warns when the art reads as plain white, black or flat, or when the content leaves the centre, since Play crops
  the edges.
