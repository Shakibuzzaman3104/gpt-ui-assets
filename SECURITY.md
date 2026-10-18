# Security policy

## Your API key

- gptkit reads the key from `GPTKIT_API_KEY` / `OPENAI_API_KEY` or from `~/.config/gptkit/env`. `setup` creates that
  file with mode 600; keep it that way and never commit it.
- The key is sent only to the endpoint in `GPTKIT_BASE_URL` (default `https://api.openai.com/v1`). Point it only at an
  endpoint you trust: it receives your prompts, your mockups and the key.
- With the Codex CLI backend no key is stored by gptkit: the Codex CLI uses its own login, and gptkit never reads its
  credential files.
- The cost ledger (`~/.cache/gptkit/ledger.jsonl`) records models, token counts and estimates, never prompts or keys.

## What leaves your machine

Prompts, reference images (mockups, crops of them) and screenshots you pass to `compare` are sent to the configured
image and text models. Fonts and lottie-web are downloaded from Google Fonts and cdnjs. Nothing else is uploaded.

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/Shakibuzzaman3104/gpt-ui-assets/security/advisories/new)
rather than a public issue. You should get a reply within a week.
