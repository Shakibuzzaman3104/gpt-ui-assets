# Phase 7: store screenshots (App Store and Google Play)

Read before any `store` command, or when the user wants App Store / Play screenshots or a feature graphic. `$K` is
`scripts/gptkit.py` in this skill's directory.

The UI in a store shot must be the app's own pixels: GPT redraws any screen it is shown (text, icons and layout drift),
and Apple wants the app in use (2.3.3). So `store` never sends a screen through an image model. Real captures are
composited pixel-exact in HTML and rendered in Chrome; GPT paints only the **art plate** behind them (an edit of the
slot's own capture, so it keeps the app's style) and, optionally, the **display lettering** (drawn transparent, then
read back by OCR and re-rolled until it spells the caption exactly). Captions, sizes, contrast and UI fidelity are
checked locally.

There are two routes to a listing. The first composites the captures in HTML behind GPT-painted art plates (`art`,
`build`). The second has GPT paint the whole picture around a chroma-green screen that the kit fills with the real
capture (`scene`, `compose`, section 5b).

**Ask for the style first.** Before `plan`, `art` or `scene`, show the user the reference sheets in
`references/store-styles/` and ask which style they want. Skip the question only when the request already says.

| Style | Sheet | Route |
|---|---|---|
| Full creative | `full-creative.jpg` | `scene` + `compose` for every slot; mix front-facing and angled phones (`scene_view`) |
| Flat & normal | `flat-normal.jpg` | `art` + `build`: upright phones on clean or plated backgrounds, headline above |
| Mixed | `mixed.jpg`, `mixed-bedside.jpg` | Both in one set, alternating painted and flat slots (the hero is usually painted) |
| Something else | none | Whatever the user describes; the two routes combine freely |

For a mixed set, render both routes at the same size per store. `compose` makes Play shots at 1080x2160, because a
9:16 crop cuts a painted phone; fit the flat Play shots to 1080x2160 as well (for example by extending the plain
band above the headline) so the carousel does not jump between aspect ratios. Show the hero in the chosen style
and get a yes before painting the rest.

```bash
python3 $K store capture store --name hunt                # current screen of the booted simulator / device
python3 $K store capture store --from ~/Desktop/shots      # or import real captures (stems become names)
python3 $K store research store --term "word game"         # optional: what the category's top listings do
python3 $K store plan store --app "WordFight: live 1v1 word duels for casual players"
python3 $K store plan store --lint                         # after editing storyboard.json
python3 $K store art store --feature                       # plates (+ --lettering for display captions)
python3 $K store build store                               # ios-phone + play-phone (+ tablets when captured)
python3 $K store check store                               # store rules; exits 1 on a FAIL
python3 $K store check store --idea                        # + can the pictures alone say what the app does?
python3 $K store feature store --logo brand/logo.png       # Play feature graphic 1024x500
python3 $K store scene store                               # painted scenes, one per slot with a scene_brief
python3 $K store compose store                             # scenes per target, with fit and overlap checks
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

The set is an ad, not a manual:
- `promise` is one line: what the app does, the outcome its users want, and for whom. Slot 1 says it, so a shopper
  who sees only slot 1 knows what the app is; slots 2-3 back it up as a story (the key moment, then the payoff).
- Captions sell the outcome in words users would say (calmer nights, money that arrives); the sub names the feature
  that delivers it. Keep the app's voice, and no outcome the app can't back.
- Measured on 7 apps × 2 runs, judged blind by a model that wrote none of the copy: slot 1 alone told a shopper what
  the app does 4.93/5 (4.57 with the earlier wording), the judge would install from the new set in 11 of the 13 pairs
  it called the same way in both orders, and personality held (7 to 6).

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
  (best, #1, top, new, free, sale) and calls to action (download, play now); tutorial copy ("tap the", "swipe
  to", "step 1": show what the user gets, not how to use the app); captions over 6 words; alt text starting "Image of".

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
- Headlines share one size per set: the largest at which every caption fits in two lines above the screen, up to 13%
  of the canvas width, and on Play under 18% of the image. Shorter captions buy bigger headlines (HushNest: 42 → 51 px
  on iPhone).
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

## 5b. Painted scenes (green screen)

The second route paints the whole picture. `store scene DIR` has GPT paint each slot's listing image as one scene: the
app's world, props and a big tilted phone whose display is flat chroma green (`#00FF00`) and carries no text. Locally
the kit finds the green screen, fits its four edges in true perspective (sides may converge; each edge is its own
fitted line and the corners are where they meet), warps the app's real capture into the fitted quad so objects painted
in front of the phone stay in front, removes green spill, and sets the headline in real type from Google Fonts. The
screen is still the app's own pixels and is never sent through the image model, so the rule at the top of this file
holds.

```bash
python3 $K store scene store --variants 2 --style-ref art/scene_hero_v1.png
python3 $K store scene store --slots hero,duel --pick hero=2   # re-pick without API calls
python3 $K store scene store --set night --look "moonlit night, cool rim light, wet stone"
python3 $K store compose store                                 # ios-phone, ios-phone-large, play-phone
python3 $K store compose store --targets ios-phone,play-phone
python3 $K store compose store --set night --captures captures/ios-phone-dark
python3 $K store compose store --clean-top                     # inpaint painted status-bar glyphs
```

`store scene DIR` paints `scenes/<set>/<slot>_v<n>.png` for every slot with a `scene_brief`. Each variant is checked
for a fittable green screen: the fit is the IoU of the fitted quad against the green area and must be at least 0.85,
and an edge cut by the canvas is an error. It flags `GREEN-SPILL` (green outside the screen), `BUSY-TOP` (a busy
headline band), `BAD-PERSPECTIVE` (an impossible screen view), `COPIES-REF` (a copied reference composition) and `SMALL-PHONE` (a screen that is too small), auto-picks the best passing variant, and writes `scenes/<set>/scenes.json` and
`scenes/<set>/review.jpg`, each variant shown with the real capture already keyed in. `--pick hero=2` overrides a pick
with no API calls. `--slots a,b` narrows the run; `--jobs`, `--force`, `--model` and `--dry-run` work as elsewhere. A
non-main set with `--look "<re-light prompt>"` (for example `--set night --look "moonlit night ..."`) edits each slot's
picked main scene into a variant look with the same composition, and the look is saved in the storyboard.

`store compose DIR` renders each picked scene per target: `ios-phone` 1206x2622, `ios-phone-large` 1320x2868,
`ios-phone-65` 1242x2688 (optional) and `play-phone` 1080x2160 (Play's 2:1 limit; a 9:16 crop cuts the painted
phone). `--targets` chooses them, and the crop keeps the top and every screen whole. It writes
`scenes/<set>/build/<target>/NN_slot.png`, `sheet.jpg` and `report.json` with the fit and the flags `OVERLAP` (the
headline runs into the phone), `LOW-FIT`, `LOW-CONTRAST` (under 3:1), `BAD-PERSPECTIVE` and `SMALL-PHONE`. `--captures` (or the storyboard's
`scenes.sets.<set>.captures`) points at another capture folder, for example dark-mode captures for a night set.
`--slots` and `--out DIR` choose the slots and the output folder. `--clean-top` inpaints status-bar glyphs a model
painted into the top strip. Scenes need OpenCV (`pip install opencv-python-headless`); the rest of the kit does not.

`storyboard.json` gains a `scenes` block:

```json
"scenes": {
  "style": "<this app's own art direction for the whole set>",
  "headroom": 0.22,
  "style_ref": "optional/path.png",
  "type": {"font": "Outfit", "weight": 900, "sub_font": "DM Sans", "sub_weight": 700, "color": "#FFFFFF",
           "accent": "#FFC23D", "accent_from": 1, "shadow": "#1F1B24", "glow": true, "align": "left",
           "size": 0.127, "sub_size": 0.041, "top": 0.042, "margin": 0.064, "line_gap": 0.94, "case": "none"},
  "sets": {"night": {"look": "...", "captures": "captures/ios-phone-dark"}}
}
```

Per slot: `scene_brief`, `scene_view` (`"front"` or `"angled"`), `scene_lines`, `scene_sub` and `scene_captures` (2+
screens, keyed left to right).
`scene_lines` holds strings, or `{"text", "size", "color"}` where `size` is a multiplier and `color` is `accent`,
`color` or a `#hex`, for a hero number such as a big "4,500" over "solo levels". Every `type` key is optional: the
numbers above are the defaults, and fonts and colours default to the brand block. Sizes are fractions of the width,
`top` of the height, and lines from index `accent_from` on take the accent colour.

**Every app gets its own style of listing.** There is no common template and no house look for scenes. The driver
invents the art direction per app from what the app is: its world, its audience and its brand. A word duel game might
get sandstone letter cubes tumbling through a painted canyon with a VS medal; a sleep app gets something entirely
different that fits it. Vary the composition between slots (tilt, scale, props, one or two screens, a hero number
slot), choose headline typography per app, and treat every example here as an illustration, never a template. There is
always room for creativity.

What a real app taught:

- No impossible perspective. The fit only says the quad matches the green; it does not say a camera could see a rectangle that way. The kit recovers the screen's true shape from the four corners (Zhang-He rectangle recovery) and fails a variant whose implied aspect is off by more than ~15% or that no camera can produce (BAD-PERSPECTIVE); compose refuses such a shot. Before presenting any set, also look at each shot at half size: the UI's rows and the phone's edges must agree.
- A style reference carries the look, not the layout. Edits of a reference tend to repeat its composition, so the kit tells the model to paint a new camera and pose and fails variants whose screen sits where the reference's does (COPIES-REF). Still read the review: two slots that look alike waste a slot.
- Props may cover a corner, never a whole screen edge. A prop across an edge becomes that edge for the fitter and skews the UI; the prompt says so, and the perspective check catches what slips through.
- The phone is big. Its screen should span about 55-60% of the image height (about a quarter of the image area);
  the world frames it, never shrinks it. Briefs that say a size are not enough, the model still shrinks the phone,
  so the kit asks for a big phone in every prompt and flags `SMALL-PHONE` (screen under 42% of the height or 16%
  of the area) and never auto-picks a small one when a big one passed. For slots with two or more screens, the
  height threshold is 40%. Re-roll small ones.
- Mix front-facing and angled shots. A set of only tilted phones reads as decoration; give two or more slots
  `"scene_view": "front"` (the phone square to the camera, screen flat and fully readable) and keep the angled ones for
  energy. The hero may be either.
- Write `scene_brief` per slot as a scene: the phone, props that echo that slot's caption, the background and the
  light. Say "whole phone inside the frame" and keep the headroom clear. It is in the prompt, but briefs that crowd the
  top fail `BUSY-TOP`.
- Paint with `--style-ref` (the first approved scene, or a concept image the user liked) so the set keeps one look.
  Describing a style in words drifts.
- Read `review.jpg` before composing. Re-roll a slot whose fit fails (a cut edge, a screen covered by props) rather
  than forcing it. A prop may overlap a corner; the fitter tolerates that.
- Prefer the app's real moments in the captures (mid-action, a win, a staged letter about to score) and stage them
  deliberately. A dull capture undoes a beautiful scene.
- Dark mode: capture the app in dark appearance into its own folder, then `store scene DIR --set night --look "..."`
  and `store compose DIR --set night --captures <folder>`. The night scenes keep the composition, so the light and dark
  sets match.
- Copy honesty still applies: a caption must not claim "live" opponents over a screen that shows a labelled bot or a
  practice match.
- Play: frameless cards remain the safest route for Play, which asks for no device imagery in screenshots.
  Painted-phone scenes are a creative choice, and the 1080x2160 output stays within Play's 2:1 limit.
- `store check` validates the HTML `build` route. The compose report covers scenes, whose sizes are exact by
  construction.
- The route was proven on WordFight: perspective fit IoU 0.94-0.98 on 8 scenes, and a light set and a moonlit dark set
  from one composition.

## 6. Check

`store check DIR` reads `build/report.json` and the files, and exits 1 on any FAIL:

- **Images**: exact size per target, RGB mode, ≤ 8 MB, and Play's 320-3840 px and 2:1 limits.
- **Counts**: App Store 1-10. Play 2-8, plus a promotion check: 4 shots at 1080 px+, 3 for games.
- **Quality**: fidelity and headline contrast fail; subhead contrast warns; the iOS search headline should be 9 pt or
  more (CSS px × 0.261).
- **Copy**: Play captions should cover under 20% of the image, plus the copy lint, privacy notes and the feature checks.
- **Idea** (`--idea`, 2 text calls): the hide-the-copy test. One vision call sees only the captionless first three
  shots (`build/bare/`, written by every build) at search size and says what the app does and what each shot shows. A
  second, text-only call compares that reading with the storyboard's `promise` and captions, and WARNs on a mismatch:
  people look before they read, so the pictures must carry the idea and the caption only sharpen it. On HushNest it
  read "soothing sounds to relax or sleep" in 3 of 3 runs and flagged that nothing shows babies, the app's audience.
  Told the app was a general sleep-sounds app, it passed 2 of 2.

## 7. Feature graphic

`store feature DIR --logo logo.png` renders the 1024x500 Play feature graphic:
- `art/plate_feature.png` (or the brand gradient) with a soft centre glow, the logo, or `--title` in the headline
  font, and the storyboard's subtitle. `--icon appicon.png` adds the app icon as a rounded tile above them, which
  ties the graphic to the icon users see in search.
- It warns when the art reads as plain white, black or flat, or when the content leaves the centre, since Play crops
  the edges.
