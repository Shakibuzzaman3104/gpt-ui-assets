# gptkit part 21_research: research, blueprint.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---- market research + feature blueprint

RESEARCH_MODEL = "gpt-6-sol"
DEPTHS = {"quick": 5, "standard": 10, "deep": 16}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def think(prompt, schema=None, name="result", model=None, effort="high", search=False):
    """Reasoning call (optionally with web search). Returns (text or parsed JSON, [url citations]); a refusal,
    truncation, empty or non-JSON answer raises ApiError (see _responses_text)."""
    if effort not in EFFORTS:
        raise ValueError(f"think: effort must be one of {', '.join(EFFORTS)}, got {effort!r}")
    chosen_model = model or RESEARCH_MODEL
    _warn_retired_model(chosen_model)
    body = {"model": chosen_model, "store": False, "reasoning": {"effort": effort},
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    if search:
        body["tools"] = [{"type": "web_search"}]
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}
    try:
        return _responses_text(_responses(body, timeout=1200), schema)
    except ApiError as e:
        e.prompt = e.prompt or prompt[:200]
        raise


def _check_keys(data, schema, what):
    """Exit when a structured response lacks a top-level key the schema requires (a gateway ignoring strict mode)."""
    missing = [k for k in schema["required"] if not isinstance(data, dict) or k not in data]
    if missing:
        sys.exit(f"gptkit: the {what} response is missing {', '.join(missing)}: the endpoint may not honour strict "
                 "json_schema; try --model gpt-6-sol or another gateway")
    return data


def _mdcell(s):
    """A Markdown table cell: pipes escaped, newlines collapsed."""
    return " ".join(str(s).replace("|", "\\|").split())


def _obj(props, **extra):
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props, **extra}


_S, _N, _B = {"type": "string"}, {"type": "number"}, {"type": "boolean"}
_SA = {"type": "array", "items": _S}

PLAN_SCHEMA = _obj({"product_frame": _S, "questions": {"type": "array", "items": _obj(
    {"id": _S, "area": _S, "question": _S, "why": _S})}})

REPORT_SCHEMA = _obj({
    "executive_summary": _S,
    "market": _obj({"size_and_growth": _S, "trends": _SA, "target_segments": _SA}),
    "competitors": {"type": "array", "items": _obj({
        "name": _S, "platforms": _S, "positioning": _S, "pricing": _S, "rating_and_scale": _S,
        "key_features": _SA, "strengths": _SA, "weaknesses": _SA})},
    "user_pain_points": {"type": "array", "items": _obj({"pain": _S, "evidence": _S, "severity": {"type": "string", "enum": ["high", "medium", "low"]}})},
    "unmet_needs_and_gaps": _SA,
    "monetization": _obj({"benchmarks": _S, "recommended_model": _S, "price_points": _S}),
    "regulatory_and_platform": _SA,
    "risks": _SA,
    "differentiation_opportunities": _SA,
    "open_questions": _SA,
    "confidence_notes": _S})

BLUEPRINT_SCHEMA = _obj({
    "product": _obj({"name_suggestion": _S, "one_liner": _S, "positioning": _S, "north_star_metric": _S,
                     "differentiators": _SA}),
    "personas": {"type": "array", "items": _obj({"name": _S, "description": _S, "jobs_to_be_done": _SA, "pains": _SA})},
    "features": {"type": "array", "items": _obj({
        "id": _S, "name": _S, "description": _S, "personas": _SA, "solves": _S, "evidence": _S,
        "priority": {"type": "string", "enum": ["must", "should", "could", "wont"]},
        "rice": _obj({"reach": _N, "impact": _N, "confidence": _N, "effort": _N, "score": _N}),
        "release": {"type": "string", "enum": ["mvp", "v1", "v2", "later"]},
        "user_stories": {"type": "array", "items": _obj({"story": _S, "acceptance_criteria": _SA})},
        "platform_notes": _S, "dependencies": _SA, "risks": _SA})},
    "screens": {"type": "array", "items": _obj({
        "id": _S, "name": _S, "purpose": _S, "features": _SA, "native_components": _SA, "content": _SA,
        "mockup_brief": _S, "states": _SA})},
    "flows": {"type": "array", "items": _obj({"name": _S, "steps": _SA})},
    "monetization": _obj({"model": _S, "tiers": _SA, "paywall_placement": _S}),
    "analytics_events": _SA,
    "roadmap": _obj({"mvp": _SA, "v1": _SA, "v2": _SA}),
    "non_goals": _SA,
    "open_risks": _SA})


def _md_cites(cites):
    seen, out = set(), []
    for c in cites:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)
    return out


def cmd_research(a):
    n = DEPTHS[a.depth]
    _plan(0, n + 2, model=a.model, tokens_in=15000, tokens_out=15000, web_calls=n * 4)  # reasoning-heavy, ~4 searches each
    if getattr(a, "dry_run", False):
        print(f"would make 1 planning call, {n} web-searched questions, and 1 synthesis call")
        return
    out = Path(a.out)
    (out / "findings").mkdir(parents=True, exist_ok=True)
    ctx = (f"Product idea: {a.idea}\nTarget platform(s): {a.platform or 'unspecified'}\nRegion/market: {a.region}\n"
           + (f"Extra context: {a.context}\n" if a.context else ""))
    print(f"planning {n} research questions ({a.model}, effort {a.effort})...")
    plan, _ = think(ctx + f"\nYou are a senior product strategist. Plan an in-depth market study for this idea as exactly "
                    f"{n} precise, answerable web-research questions covering: market size/growth and trends; the top "
                    "direct and indirect competitors (names, pricing, ratings, downloads, features); real user pain points "
                    "and complaints from app-store reviews, Reddit and forums; unmet needs; monetization and pricing "
                    "benchmarks; platform/regulatory constraints; acquisition channels and ASO keywords; differentiation "
                    "angles. Also restate the product frame in 2 sentences.", PLAN_SCHEMA, "research_plan",
                    a.model, a.effort)
    _check_keys(plan, PLAN_SCHEMA, "research plan")
    (out / "plan.json").write_text(json.dumps(plan, indent=2))

    def run(q):
        text, cites = think(ctx + f"\nResearch question ({q['area']}): {q['question']}\nWhy it matters: {q['why']}\n"
                            "Search the web thoroughly (several queries; prefer primary sources, app-store listings, "
                            "official pricing pages, recent reviews and reputable reports). Answer in concise markdown "
                            "with concrete facts, numbers, names and dates, and cite every claim. Say explicitly what "
                            "you could not verify.", model=a.model, effort=a.effort, search=True)
        cites = _md_cites(cites)
        (out / "findings" / f"{q['id']}.md").write_text(f"# {q['question']}\n\n{text}\n\n## Sources\n" +
                                                        "".join(f"- [{c['title'] or c['url']}]({c['url']})\n" for c in cites))
        print(f"  done {q['id']}: {q['question'][:70]} ({len(cites)} sources)")
        return q, text, cites
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        found = list(ex.map(run, plan["questions"]))
    corpus = "\n\n".join(f"## {q['id']} [{q['area']}] {q['question']}\n{t}" for q, t, _ in found)
    sources = _md_cites([c for _, _, cs in found for c in cs])
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    print(f"synthesising report from {len(found)} findings, {len(sources)} sources...")
    report, _ = think(ctx + "\nYou are a senior product strategist. Synthesise these research findings into a decision-grade "
                      "market report. Use ONLY facts in the findings (keep their numbers and names), mark uncertainty, "
                      "and make the gaps/opportunities specific and actionable.\n\nFINDINGS:\n" + corpus,
                      REPORT_SCHEMA, "market_report", a.model, a.synth_effort)
    _check_keys(report, REPORT_SCHEMA, "market report")
    report["idea"], report["platform"], report["region"] = a.idea, a.platform, a.region
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(_report_md(report, sources))
    print(f"-> {out / 'report.md'}, report.json, findings/, sources.json")


def _report_md(r, sources):
    L = [f"# Market research: {r['idea']}", "", f"_Platform: {r['platform'] or 'unspecified'} · Region: {r['region']}_", "",
         "## Executive summary", r["executive_summary"], "", "## Market", r["market"]["size_and_growth"], "",
         "**Trends**", *[f"- {x}" for x in r["market"]["trends"]], "", "**Target segments**",
         *[f"- {x}" for x in r["market"]["target_segments"]], "", "## Competitors", "",
         "| App | Platforms | Pricing | Rating / scale | Positioning |", "|---|---|---|---|---|",
         *["| " + " | ".join(_mdcell(c[k]) for k in ("name", "platforms", "pricing", "rating_and_scale", "positioning")) + " |"
           for c in r["competitors"]], ""]
    for c in r["competitors"]:
        L += [f"### {c['name']}", "**Key features:** " + "; ".join(c["key_features"]),
              "**Strengths:** " + "; ".join(c["strengths"]), "**Weaknesses:** " + "; ".join(c["weaknesses"]), ""]
    L += ["## User pain points", *[f"- **{p['pain']}** ({p['severity']}): {p['evidence']}" for p in r["user_pain_points"]],
          "", "## Unmet needs and gaps", *[f"- {x}" for x in r["unmet_needs_and_gaps"]], "",
          "## Monetization", r["monetization"]["benchmarks"], "", f"**Recommended:** {r['monetization']['recommended_model']}",
          f"**Price points:** {r['monetization']['price_points']}", "", "## Platform and regulatory",
          *[f"- {x}" for x in r["regulatory_and_platform"]], "", "## Risks", *[f"- {x}" for x in r["risks"]], "",
          "## Differentiation opportunities", *[f"- {x}" for x in r["differentiation_opportunities"]], "",
          "## Open questions", *[f"- {x}" for x in r["open_questions"]], "", "## Confidence", r["confidence_notes"], "",
          "## Sources", *[f"{i + 1}. [{s['title'] or s['url']}]({s['url']})" for i, s in enumerate(sources)]]
    return "\n".join(L) + "\n"


def cmd_blueprint(a):
    d = Path(a.dir)
    report = (_read_json(d / "report.json", "research report", required=REPORT_SCHEMA["required"])
              if (d / "report.json").exists() else None)
    idea = a.idea or (report or {}).get("idea")
    if not idea:
        sys.exit("gptkit: give --idea or a research dir with report.json")
    platform = a.platform or (report or {}).get("platform")
    native = (f"Target platform: {PLATFORMS[platform]['label']}; name each screen's native components from: "
              f"{PLATFORMS[platform]['components']}.") if platform in PLATFORMS else "Platform: unspecified."
    print(f"generating the feature blueprint ({a.model}, effort {a.effort})...")
    bp, _ = think(f"Product idea: {idea}\n{native}\n" + (f"Constraints: {a.constraints}\n" if a.constraints else "") +
                  ("MARKET RESEARCH REPORT (ground every feature in it; cite the pain point or gap it addresses in "
                   "'evidence'):\n" + json.dumps(report) if report else "No research report: reason from first principles "
                   "and mark evidence as 'assumption'.") +
                  "\n\nYou are a principal product manager. Produce a complete, buildable feature blueprint: product "
                  "positioning, personas with jobs-to-be-done, a prioritised feature list (MoSCoW + RICE with reach per "
                  "quarter, impact 0.25-3, confidence 0-1, effort in person-weeks, score = R*I*C/E; release mvp/v1/v2), "
                  "user stories with testable acceptance criteria, a screen inventory where every screen has a precise "
                  "one-paragraph mockup_brief a designer can render directly (layout, content, states, native "
                  "components), key flows, monetization, analytics events, a roadmap and non-goals. Keep the MVP "
                  "small and differentiated.", BLUEPRINT_SCHEMA, "feature_blueprint", a.model, a.effort)
    _check_keys(bp, BLUEPRINT_SCHEMA, "blueprint")
    for f in bp["features"]:
        r = f["rice"]
        r["score"] = round(r["reach"] * r["impact"] * r["confidence"] / max(r["effort"], 0.1), 1)
    bp["features"].sort(key=lambda f: (["mvp", "v1", "v2", "later"].index(f["release"]), -f["rice"]["score"]))
    bp["idea"], bp["platform"] = idea, platform
    out = Path(a.out) if a.out else d
    out.mkdir(parents=True, exist_ok=True)
    (out / "blueprint.json").write_text(json.dumps(bp, indent=2))
    (out / "screens.json").write_text(json.dumps([{"id": s["id"], "name": s["name"], "platform": platform,
                                                   "mockup_brief": s["mockup_brief"]} for s in bp["screens"]], indent=2))
    (out / "blueprint.md").write_text(_blueprint_md(bp))
    print(f"-> {out / 'blueprint.md'}, blueprint.json, screens.json ({len(bp['features'])} features, "
          f"{len(bp['screens'])} screens; MVP: {sum(f['release'] == 'mvp' for f in bp['features'])} features)")


def _blueprint_md(b):
    p = b["product"]
    L = [f"# Feature blueprint: {p['name_suggestion']}", "", f"**{p['one_liner']}**", "", p["positioning"], "",
         f"**North-star metric:** {p['north_star_metric']}", "", "**Differentiators**", *[f"- {x}" for x in p["differentiators"]],
         "", "## Personas"]
    for x in b["personas"]:
        L += [f"### {x['name']}", x["description"], "**Jobs to be done:** " + "; ".join(x["jobs_to_be_done"]),
              "**Pains:** " + "; ".join(x["pains"]), ""]
    L += ["## Features", "", "| ID | Feature | Priority | Release | RICE | Solves |", "|---|---|---|---|---|---|",
          *["| " + " | ".join(_mdcell(v) for v in (f["id"], f["name"], f["priority"], f["release"], f["rice"]["score"], f["solves"])) + " |"
            for f in b["features"]], ""]
    for f in b["features"]:
        L += [f"### {f['id']} {f['name']} ({f['release']}, {f['priority']})", f["description"], "",
              f"_Evidence:_ {f['evidence']}", ""]
        for us in f["user_stories"]:
            L += [f"- {us['story']}", *[f"  - [ ] {c}" for c in us["acceptance_criteria"]]]
        if f["platform_notes"]:
            L += ["", f"_Platform:_ {f['platform_notes']}"]
        L.append("")
    L += ["## Screens"]
    for s in b["screens"]:
        L += [f"### {s['id']} {s['name']}", s["purpose"], "", "**Native components:** " + ", ".join(s["native_components"]),
              "**States:** " + ", ".join(s["states"]), "", f"> Mockup brief: {s['mockup_brief']}", ""]
    L += ["## Flows", *[f"- **{x['name']}:** " + " → ".join(x["steps"]) for x in b["flows"]], "", "## Monetization",
          b["monetization"]["model"], *[f"- {t}" for t in b["monetization"]["tiers"]],
          f"Paywall: {b['monetization']['paywall_placement']}", "", "## Roadmap",
          "**MVP:** " + "; ".join(b["roadmap"]["mvp"]), "**v1:** " + "; ".join(b["roadmap"]["v1"]),
          "**v2:** " + "; ".join(b["roadmap"]["v2"]), "", "## Analytics events", *[f"- `{e}`" for e in b["analytics_events"]],
          "", "## Non-goals", *[f"- {x}" for x in b["non_goals"]], "", "## Open risks", *[f"- {x}" for x in b["open_risks"]]]
    return "\n".join(L) + "\n"
