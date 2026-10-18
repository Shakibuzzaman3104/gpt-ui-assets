# Setup, endpoints and optional tools

Read this on a new machine, when `doctor` reports a problem, or when a gateway misbehaves. Which model to use for
which job lives in `models.md`.

```bash
pip install -r "<this skill's directory>/scripts/requirements.txt"    # requests, pillow, numpy, playwright (+ optional extras)
python3 $K doctor                                                      # key, endpoint, models, SVG renderer, aapt2, EbSynth, caches
```

## Config

Config comes from env vars or `~/.config/gptkit/env` (`KEY=value` lines):
- `GPTKIT_BACKEND=auto|api|cli` (default `auto`): with an API key, auto uses the API; without one, it uses a ready
  Codex CLI login on the user's ChatGPT plan. `python3 $K setup` reports readiness; `setup --backend api|cli|auto`
  saves the selection. `setup --api-key-stdin` reads a key without echoing it.
- `GPTKIT_API_KEY` (or `OPENAI_API_KEY`) is required for the API backend. Never print or echo it.
- `GPTKIT_QUALITY=low|medium|high` (default `high`) sets API image output quality; low and medium reduce draft cost.
- `GPTKIT_BASE_URL` (or `OPENAI_BASE_URL`), default `https://api.openai.com/v1`. OpenAI-compatible gateways work when they
  mirror `/images/generations` and `/images/edits` (multipart) and `/responses` including `reasoning`, the `web_search`
  tool and strict `json_schema` (LiteLLM does; the kit's measurements were made on an OpenAI-compatible gateway). Azure OpenAI
  (`api-key` header, deployment URLs) and OpenRouter's native image API are not supported without changes. When a
  gateway rejects an optional field (`input_fidelity`, `quality`, `background`, `reasoning`, `tools`, strict schema)
  the kit says so once and retries without it. A gateway that accepts `quality: high` but answers `medium` or `low`
  (measured on a sub2api gateway, with every image model) gets a one-time warning: it caps quality and the kit can't lift it.
- `GPTKIT_IMAGE_MODEL` and `GPTKIT_TEXT_MODEL`: see `models.md`. Setting `GPTKIT_IMAGE_MODEL` pins one model and turns
  off per-job routing.
- `GPTKIT_RETRIES` (default 2) retries a call on 408/429/5xx and network errors, honouring `Retry-After`; quota/billing 429s
  are not retried. Every failure is one line, `gptkit <cmd>: HTTP 400 /images/edits [moderation_blocked]: ... | prompt: '...'`;
  `GPTKIT_DEBUG=1` shows the traceback instead. `GPTKIT_MODERATION=low` relaxes image moderation where the org allows it.

### Local Codex CLI

Install with `npm i -g @openai/codex`, sign in with `codex login`, then run `python3 $K setup --backend cli`. The kit
uses `codex exec` with an ephemeral read-only workspace and no API key. Text, vision, and web research use your ChatGPT
login; image generation uses Codex's `image_gen` tool. The local image tool has no fidelity, exact-size, quality, or
candidate-count controls, so prefer the API for pixel-exact asset work (`forge`, `fix`, `sequence`). Parallel variants stay apart (each run's image is read from its own thread folder under `$CODEX_HOME/generated_images`). `$CODEX_HOME` can point at a non-default Codex home; `GPTKIT_CODEX` names the binary (`none` disables the backend). When the plan refuses a model, the kit switches to `gpt-6-luna` for the rest of the run. Calls are logged to the ledger at $0 (plan quota).

## Optional tools (each used when present)

- **Headless Chrome/Chromium** renders SVG with real alpha and plays Lottie through lottie-web (`GPTKIT_CHROME` sets its path).
  As root (containers, CI) Chrome needs `--no-sandbox`; the kit adds it automatically, or set `GPTKIT_CHROME_NO_SANDBOX=1`.
  `doctor` actually renders a probe page and prints `chrome (ok)` or Chrome's error line.
- **Playwright** (`pip install playwright`, in requirements.txt) drives that Chrome for `stage`. Without an installed Chrome,
  also run `playwright install chromium`.
- **Android SDK `aapt2`** (via `ANDROID_HOME` or `~/Library/Android/sdk`) compiles every VectorDrawable and adaptive icon
  and, with `platforms/android-*/android.jar` installed (`sdkmanager "platforms;android-35"`), links it (attribute names,
  enums, dimensions); without the jar the kit's offline VectorDrawable lint covers attribute names/enums. `doctor` says
  which level you have and the kit never prints "valid" for a compile-only run.
- **`xcrun actool`** validates the iOS app icon.
- `pip install vtracer scipy fonttools` add an extra trace candidate, faster region labelling and kerning in the font
  matcher. Everything works without them.
- `pip install opencv-python-headless` gives `sequence` its motion-direction, seam and shimmer QA; without it the QA
  falls back to plain frame differences.
- **EbSynth** for `sequence propagate`: see `sequence.md`.
- `fonts` needs network access to Google Fonts once; after that it runs from `~/.cache/gptkit`.

## Cost and time

Image calls take 35–80 s each and cost money or quota. Batch deliberately, run slow commands in the background, and run
independent commands in parallel (e.g. `forge` and `svg`).

- **Plan line.** `mockup`, `material`, `forge`, `frames`, `sequence render`, `appicon concepts` and `research` print
  `plan: ~N image calls (<model>) + M text calls ≈ $X list price` before the first call. It is deliberately rough:
  high-quality output scaled by area, plus the reference image for edits; research allows for reasoning tokens and
  about four web searches per question. It exists to catch a 10× surprise, not to bill.
- **`--dry-run`** on `forge`, `sequence render` and `research` prints the plan and what would be rendered, then exits
  with no API call.
- **`GPTKIT_BUDGET_USD=1.50`** stops any command whose plan exceeds it before anything is called. Use it whenever the
  user mentions a budget, and lower `--candidates`/`--variants` or narrow `--only` to fit.
- **Ledger.** Every successful call appends a line to `~/.cache/gptkit/ledger.jsonl` (`GPTKIT_LEDGER=<path>` moves it,
  `GPTKIT_LEDGER=off` disables it): command, project dir, model, the API's usage, web-search calls and the list-price
  estimate (`estimated: true` when the gateway stripped `usage`).
- **`python3 $K cost [--since 7d|24h|YYYY-MM-DD] [--by cmd|model|day|dir] [--json]`** sums it. Through a subscription
  OpenAI-compatible gateway the dollars are a list-price equivalent of quota, not money billed; the output says so.
