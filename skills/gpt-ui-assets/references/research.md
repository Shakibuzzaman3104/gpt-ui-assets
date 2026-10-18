# Phase 0: market research and feature blueprint

Read before `research` or `blueprint`. `$K` is `scripts/gptkit.py` in this skill's directory.

```bash
python3 $K research "Android budgeting app for young couples who share expenses" --platform android \
  --region "US + UK" --depth standard --out research/couples     # plan -> parallel web research -> cited report
python3 $K blueprint research/couples                            # features, RICE, stories, screens, roadmap
```
- **`research`** runs `gpt-6-sol` at `--effort high` (the synthesis at `--synth-effort xhigh`; `max` exists for both and is
  slower):
  1. It plans 5, 10 or 16 questions (`--depth quick|standard|deep`) across market size and trends, competitors (pricing,
     ratings, features), real user pain points from app-store reviews and forums, unmet needs, monetization benchmarks,
     platform/regulatory constraints, ASO and differentiation.
  2. It researches each question in parallel with web search and cites every claim.
  3. It synthesises `report.md` and `report.json`, plus `findings/*.md` and `sources.json`.
- **Read `report.md` yourself before the blueprint.** Challenge weak claims: `findings/` shows the raw evidence per question,
  and "could not verify" notes are deliberate. Re-run a narrower `research` with `--context` to fill a gap.
- A refusal, a truncated answer (`incomplete: max_output_tokens`) or a gateway that ignores strict `json_schema` stops the
  command with one line naming the cause and the missing key; lower `--effort`, shorten `--context`, or try `--model gpt-6-sol`
  on another gateway. `blueprint` refuses a `report.json` that is corrupt or missing sections.
- **`blueprint`** turns the report into:
  - positioning, personas and jobs-to-be-done;
  - features with MoSCoW priority and RICE (the score is recomputed locally), each citing the pain or gap it solves;
  - user stories with acceptance criteria, flows, monetization, analytics events, a roadmap and non-goals;
  - a **screen inventory with a ready `mockup_brief` for each screen** (`screens.json`).
- Walk the user through `blueprint.md`, especially the MVP cut, then design each approved screen:
  `python3 $K mockup "<mockup_brief>" --platform <platform> --out design/<screen>`. Use `--style-ref` for screen 2 onward.
- Timing: the quick depth takes several minutes and deep can take 15+. Run it in the background. Each question is one
  web-searched reasoning call.
