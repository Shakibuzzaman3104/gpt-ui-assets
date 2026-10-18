# Models: which one for which job, prices, retirements

Facts checked on 2026-10-18 against OpenAI's pricing and deprecations pages
(developers.openai.com/api/docs/pricing, /api/docs/deprecations). Re-check them when a date below has passed.

## Image models (the kit routes these itself)

`generate` calls use the **concept** chain (Flare, Sunburst, 2, 1.5) and `edit` calls the **precise** chain (Sunburst,
Flare, 2, 1.5). Transparent requests push `gpt-image-2` to the end. The first model your endpoint lists wins. `--model`
or a set `GPTKIT_IMAGE_MODEL` overrides the routing (`doctor` shows which mode is active).

| Job | Model | Why |
|---|---|---|
| Default: mockups, backgrounds, illustrations, app-icon concepts, forge candidates | `gpt-image-2.5-flare` | Quality comparable to `gpt-image-2` at ~47% lower latency, same token prices; supports `background: transparent`. |
| Edits that must stay consistent: `sequence` frames, `frames` sheets, `state`, art with embedded text, final hero assets | `gpt-image-2.5-sunburst` | Highest quality, tuned for precise edits and consistency (still ~26% faster than `gpt-image-2`). |
| Transparent PNG/WebP assets (forge, icons, sprites) | `gpt-image-2.5-flare` or `-sunburst` | OpenAI documents `gpt-image-2` as accepting only `auto`/`opaque`. The kit errors with the model name when a transparent request comes back opaque. |
| Endpoint lists no 2.5 model | `gpt-image-2` for opaque art; `gpt-image-1.5` for transparency **until 2026-12-01** | 1.5 shuts down on 2026-12-01; after that a gateway without 2.5 cannot make transparent assets. |
| Draft or throwaway iterations | the chosen model at `quality: low` | Don't downgrade the model; low quality is ~$0.006 per 1024² image. |
| Never | `gpt-image-1`, `gpt-4o-*`, `codex-*` | `gpt-image-1` shuts down 2026-10-23; the others are not image generators for this kit. |

Rules of thumb:
- A failed `transparent` call is a model problem (switch to 2.5), not a prompt problem.
- If a `sequence` or `frames` render drifts, move Flare to Sunburst before changing the prompt.
- Try the other model on one asset (`--model`) and keep the better `score` before switching a whole project. In the
  kit's campfire test all four models scored within 0.3 fid of each other on text-only generation, so Sunburst only
  pays off in edits.

## Text models

`GPTKIT_TEXT_MODEL` pins the text model. Otherwise the API backend picks the first readable `/models` match from
`gpt-6.1-sol`, `gpt-6-sol`, `gpt-6`, `gpt-6-luna`, falling back to `gpt-6-sol`. The local Codex backend uses the first
of `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna` listed in `~/.codex/models_cache.json`, falling back to `gpt-6-luna`.
`research`/`blueprint` default to `gpt-6-sol` with high effort.
The `gpt-5.x` families also work, but several are retiring (below).

## Prices (USD, standard tier)

| Model | Text in / 1M | Image in / 1M | Image out / 1M | Text out / 1M |
|---|---|---|---|---|
| `gpt-image-2.5-flare`, `-sunburst`, `gpt-image-2` | 5.00 | 8.00 | 30.00 | — |
| `gpt-image-1.5` | 5.00 | 8.00 | 32.00 | 10.00 |
| `gpt-6-sol`, `gpt-6.1-sol` (≤272K context) | 2.00 | — | — | 10.00 |
| `gpt-6-luna` | 0.10 | — | — | 0.50 |

Web search: $10 per 1,000 calls plus the search content tokens at the model's rates. OpenAI's calculator gives an
output-only estimate for a 1024×1024 image of about $0.006 (low), $0.013 (medium) and $0.053 (high); reference images and
prompt tokens add to that. Through an OpenAI-compatible gateway these are list-price equivalents of quota, not money billed.
`GPTKIT_QUALITY` sets API image quality (`low`, `medium`, or `high`); the Codex CLI image tool does not expose quality.

## Retirements

| Model | Shuts down | Replacement |
|---|---|---|
| `gpt-image-1` | 2026-10-23 | `gpt-image-2.5-sunburst` or `-flare` |
| `gpt-image-1.5`, `gpt-image-1-mini`, `chatgpt-image-latest` | 2026-12-01 | `gpt-image-2.5-sunburst` or `-flare` |
| `gpt-5-2025-08-07`, `gpt-5-mini-…`, `gpt-5-nano-…`, `gpt-5-pro-2025-10-06` | 2026-12-11 | `gpt-6-sol` / `gpt-6-luna` |
| `gpt-5.1`, `gpt-5.3-codex` | 2027-04-01 | `gpt-6-sol` |
| `gpt-5.4-nano` | 2027-04-01 | `gpt-6-luna` |

`doctor` warns when a configured model is retired or retires within 60 days.
