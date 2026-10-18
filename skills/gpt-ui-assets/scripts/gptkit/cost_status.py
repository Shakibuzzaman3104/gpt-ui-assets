"""Cost and status."""
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

from PIL import Image

from .core import _config
from .forge import _incomplete

def cmd_cost(a):
    """Summarise the local JSONL ledger by call, model or time bucket."""
    ledger = a.ledger or os.environ.get("GPTKIT_LEDGER", str(Path.home() / ".cache/gptkit/ledger.jsonl"))
    if ledger == "off":
        print("no ledger entries: ledger disabled")
        return
    path = Path(ledger).expanduser()
    if not path.exists():
        print(f"no ledger entries: {path}")
        return
    since = None
    if a.since:
        try:
            if re.fullmatch(r"\d+[dh]", a.since):
                n, unit = int(a.since[:-1]), a.since[-1]
                since = datetime.now(timezone.utc) - (timedelta(days=n) if unit == "d" else timedelta(hours=n))
            else:
                since = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            sys.exit("gptkit cost: --since must be 7d, 24h or YYYY-MM-DD")
    rows, malformed = [], 0
    for line in path.read_text(encoding="utf8").splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("ts"), str):
                raise ValueError("not a ledger object")
            if row.get("usage") is not None and not isinstance(row.get("usage"), dict):
                raise ValueError("invalid usage")
            if not isinstance(row.get("est_usd", 0), (int, float)):
                raise ValueError("invalid estimate")
            ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
            if since is None or ts >= since:
                rows.append(row)
        except (ValueError, KeyError, TypeError, AttributeError):
            malformed += 1
    if not rows and not malformed:
        print(f"no ledger entries: {path}")
        return
    groups = {}
    for row in rows:
        key = {"cmd": row.get("cmd") or "unknown", "model": row.get("model") or "unknown",
               "day": row.get("ts", "")[:10], "dir": row.get("dir") or "(none)"}[a.by]
        g = groups.setdefault(key, {"calls": 0, "image_calls": 0, "text_calls": 0, "tokens_in": 0,
                                    "tokens_out": 0, "est_usd": 0.0})
        image = str(row.get("endpoint", "")).startswith("/images/")
        usage = row.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        tokens_in = usage.get("input_tokens") or details.get("text_tokens") or 0
        tokens_out = usage.get("output_tokens") or 0
        if not isinstance(tokens_in, (int, float)):
            tokens_in = 0
        if not isinstance(tokens_out, (int, float)):
            tokens_out = 0
        g["calls"] += 1
        g["image_calls" if image else "text_calls"] += 1
        g["tokens_in"] += tokens_in
        g["tokens_out"] += tokens_out
        g["est_usd"] += row.get("est_usd", 0) or 0
    total = {k: sum(g[k] for g in groups.values()) for k in next(iter(groups.values()))} if groups else {
        "calls": 0, "image_calls": 0, "text_calls": 0, "tokens_in": 0, "tokens_out": 0, "est_usd": 0.0}
    note = "list-price equivalent; through a gateway or a plan-quota backend this is quota, not money billed" \
        if "api.openai.com" not in _config()["base"] else None
    if a.json:
        print(json.dumps({"groups": groups, "total": total, "malformed": malformed, "note": note}, indent=2))
    else:
        print(f"{a.by:24} calls image text tokens in/out est USD")
        for key, g in sorted(groups.items()):
            print(f"{key:24} {g['calls']:5} {g['image_calls']:5} {g['text_calls']:4} "
                  f"{g['tokens_in']:9}/{g['tokens_out']:<9} ${g['est_usd']:.4f}")
        print(f"{'TOTAL':24} {total['calls']:5} {total['image_calls']:5} {total['text_calls']:4} "
              f"{total['tokens_in']:9}/{total['tokens_out']:<9} ${total['est_usd']:.4f}")
        if malformed:
            print(f"skipped {malformed} malformed ledger lines")
        if note:
            print(note)


def _status_json(path):
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError, UnicodeError) as e:
        return None, f"{path.name}: invalid JSON ({type(e).__name__})"


def _status_sha(path):
    """Hash of the decoded pixels: manifest re-encodes the mockup it copies, so the file bytes differ."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return hashlib.sha1(repr(im.size).encode() + im.tobytes()).hexdigest()
    except (OSError, ValueError):
        return None


def _status_unit(root, kind, label, checks, problems=None, next_=None):
    return {"type": kind, "path": str(root), "name": label,
            "checks": checks, "problems": problems or [], "next": next_ or []}


def _status_arg(path):
    """A path for a printed command: relative to the cwd when inside it (short and still runnable), else absolute."""
    path = Path(path)
    try:
        rel = os.path.relpath(path, os.getcwd())
        path = rel if not rel.startswith("..") else path
    except ValueError:  # another drive on Windows
        pass
    return shlex.quote(str(path))


def _status_scan_directories(base):
    """Walk the bounded project tree in stable order."""
    base = Path(base).resolve()
    dirs = [base]
    for here, names, files in os.walk(base):
        p = Path(here)
        depth = len(p.relative_to(base).parts)
        names[:] = [n for n in names if n not in ("node_modules", ".git", "raw", "frames")]
        if depth >= 4:
            names[:] = []
        dirs.append(p)
    dirs = sorted(set(dirs), key=lambda p: (len(p.parts), str(p)))
    return dirs


def _status_scan_build_mockups(dirs):
    """Index build directories by the approved mockup bytes they contain."""
    # manifest copies the approved mockup into the build dir: matching bytes link a design to its build
    build_mockups = {}
    for d in dirs:
        if (d / "assets.json").exists() and (d / "mockup.png").exists():
            build_mockups.setdefault(_status_sha(d / "mockup.png"), []).append(d)
    return build_mockups


def _status_scan_add(units, seen, d, kind, label, checks, problems=None, nxt=None):
    """Append a discovered status unit only once per directory and kind."""
    key = (str(d), kind)
    if key not in seen:
        seen.add(key)
        units.append(_status_unit(d, kind, label, checks, problems, nxt))


def _status_scan_project_units(d, children, names, build_mockups, units, seen):
    """Record research, blueprint, design, critique and hybrid work in one directory."""
    if {"report.json", "sources.json"} <= names or "findings" in {p.name for p in d.iterdir() if p.is_dir()}:
        probs, checks = [], []
        for n in ("report.json", "report.md", "findings", "sources.json"):
            exists = (d / n).exists()
            checks.append({"name": n, "status": "done" if exists else "missing"})
            if n.endswith(".json") and exists:
                _, err = _status_json(d / n)
                if err: probs.append(err)
        nxt = [] if (d / "report.json").exists() else [f"python3 $K research \"<idea>\" --out {_status_arg(d)}"]
        if (d / "report.json").exists() and not (d / "screens.json").exists() and not (d / "blueprint.md").exists():
            nxt = [f"python3 $K blueprint {_status_arg(d)}"]
        _status_scan_add(units, seen, d, "research", d.name, checks, probs, nxt)
    if "blueprint.md" in names or "screens.json" in names:
        checks = [{"name": n, "status": "done" if (d / n).exists() else "missing"}
                  for n in ("blueprint.md", "screens.json")]
        probs = []
        for n in ("screens.json",):
            if (d / n).exists():
                _, err = _status_json(d / n)
                if err: probs.append(err)
        _status_scan_add(units, seen, d, "blueprint", d.name, checks, probs, [] if all(c["status"] == "done" for c in checks)
            else [f"python3 $K blueprint {_status_arg(d)}"])
    mockups = [p for p in children if p.suffix.lower() == ".png" and p.name.startswith("mockup")]
    if (mockups or "design.json" in names) and "assets.json" not in names:  # a build dir keeps its own mockup.png
        design = d / "design.json"
        checks = [{"name": "mockup PNGs", "status": "done" if mockups else "missing"},
                  {"name": "design.json", "status": "done" if design.exists() else "missing"}]
        probs = []
        if design.exists():
            _, err = _status_json(design)
            if err: probs.append(err)
        latest = max(mockups, key=lambda p: p.stat().st_mtime) if mockups else None
        built = [b for m in mockups for b in build_mockups.get(_status_sha(m), [])]
        if built:
            checks.append({"name": "manifest (" + ", ".join(_status_arg(b) for b in built) + ")", "status": "done"})
        nxt = [f"python3 $K mockup \"<brief>\" --out {_status_arg(d)}"] if not latest else [] if built else \
            [f"once the user approves a variant: python3 $K manifest {_status_arg(latest)} --out <build dir>"]
        _status_scan_add(units, seen, d, "design", d.name, checks, probs, nxt)
    critiques = [p for p in children if p.name.endswith("_critique.json")]
    for p in critiques:
        data, err = _status_json(p)
        probs = [err] if err else []
        issues = []
        if isinstance(data, dict):
            for rnd in data.get("rounds", []):
                if isinstance(rnd, dict):
                    for batch in (rnd.get("issues"), rnd.get("new_issues")):
                        if isinstance(batch, list): issues.extend(x for x in batch if isinstance(x, dict))
        high = sum(1 for x in issues if isinstance(x, dict) and x.get("severity") == "high")
        if high: probs.append(f"{high} high-severity critique issue(s)")
        _status_scan_add(units, seen, d, "critique", p.stem, [{"name": p.name, "status": "done"}], probs, [])
    if "stage.json" in names or "material.json" in names or "stage" in names:
        probs, checks = [], []
        for n in ("material.json", "stage.json", "frames", "motion_sheet.png"):
            exists = (d / n).exists()
            checks.append({"name": n, "status": "done" if exists else "missing"})
            if n == "material.json" and exists:
                data, err = _status_json(d / n)
                if err: probs.append(err)
                elif isinstance(data, dict):
                    for name, entry in data.items():
                        if not isinstance(entry, dict):
                            probs.append(f"material.json: damaged entry {name}"); continue
                        candidates = entry.get("candidates") or {}
                        flags = sorted({flag for c in candidates.values() if isinstance(c, dict)
                                        for flag in c.get("flags", [])}) if isinstance(candidates, dict) else []
                        if flags: probs.append(f"{name}: {' '.join(flags)}")
        if (d / "stage.json").exists():
            data, err = _status_json(d / "stage.json")
            if err: probs.append(err)
            elif isinstance(data, dict): probs.extend(str(x) for x in data.get("errors", []))
        _status_scan_add(units, seen, d, "hybrid", d.name, checks, probs, [])


def _status_scan_build_unit(d, names, build_mockups, units, seen):
    """Inspect a manifest build and report missing, stale or pending assets."""
    if "assets.json" in names:
        probs, checks = [], []
        core = ("assets.json", "manifest_overlay.png", "design_spec.json", "forge_report.json", "assets/", "svg_report.json",
                "svg/", "tokens/tokens.json", "compare/report.json")
        for n in core + ("icon_spec.json", "review/", "icons/", "traced/", "rig/", "anim/", "vector/", "frames/"):
            npath = d / n.rstrip("/")
            is_dir = n.endswith("/")
            exists = (npath.is_dir() and any(npath.iterdir())) if is_dir else npath.exists()  # an empty dir is not done
            if not exists and n not in core:
                continue  # optional outputs are listed only once they exist
            state = "done" if exists else "missing"
            checks.append({"name": n, "status": state})
            if n.endswith(".json") and exists:
                data, err = _status_json(npath)
                if err: probs.append(err)
                if n == "forge_report.json" and isinstance(data, dict):
                    flagged, incomplete = [], []
                    for name, entry in data.items():
                        if not isinstance(entry, dict):
                            probs.append(f"forge_report.json: damaged entry {name}"); continue
                        # the same test forge uses to re-queue: flagged winner, unfinished run or no final file
                        if entry.get("needs_attention"): flagged.append(name)
                        elif _incomplete(entry) or not (d / "assets" / f"{name}.png").exists(): incomplete.append(name)
                    if flagged: probs.append("NEEDS ATTENTION: " + ", ".join(flagged))
                    if incomplete: probs.append("no final file: " + ", ".join(incomplete))
                if n == "svg_report.json" and isinstance(data, dict):
                    for name, entry in data.items():
                        if isinstance(entry, dict) and entry.get("error"): probs.append(f"svg {name}: {entry['error']}")
        def stale(output, inputs, label):
            op = d / output
            ins = [d / x for x in inputs if (d / x).exists()]
            if op.exists() and ins and op.stat().st_mtime < max(x.stat().st_mtime for x in ins):
                probs.append(f"STALE: {label} older than " + ", ".join(x.name for x in ins if x.stat().st_mtime > op.stat().st_mtime))
                for c in checks:
                    if c["name"] == output: c["status"] = "STALE"
        stale("design_spec.json", ["assets.json"], "design_spec.json")
        stale("svg", ["design_spec.json"], "svg/")
        stale("tokens/tokens.json", ["design_spec.json", "fonts/fonts.json"], "tokens")
        stale("compare/report.json", ["tokens/tokens.json"], "compare")
        manifest, _ = _status_json(d / "assets.json")
        report, _ = _status_json(d / "forge_report.json") if (d / "forge_report.json").exists() else ({}, None)
        report = report if isinstance(report, dict) else {}
        pngs = [x.get("name") for x in (manifest or {}).get("assets", []) if isinstance(x, dict) and x.get("format") == "png"] \
            if isinstance(manifest, dict) else []
        unforged = [n for n in pngs if n and n not in report and not (d / "assets" / f"{n}.png").exists()]
        if unforged and (d / "forge_report.json").exists():
            probs.append("not forged yet: " + ", ".join(unforged))
        svgs = [x.get("name") for x in (manifest or {}).get("assets", []) if isinstance(x, dict) and x.get("format") == "svg"] \
            if isinstance(manifest, dict) else []
        nxt = []
        if not (d / "design_spec.json").exists() or any(c["name"] == "design_spec.json" and c["status"] == "STALE" for c in checks):
            nxt = [f"python3 $K spec {_status_arg(d)}"]
        elif (pngs and not (d / "forge_report.json").exists()) or any("NEEDS ATTENTION" in x or "no final file" in x for x in probs):
            flagged = next((x.partition(": ")[2] for x in probs if x.startswith("NEEDS ATTENTION:")), "")
            nxt = ([f"read {_status_arg(d / 'review')}/forge_*.png, then python3 $K pick {_status_arg(d)} <name> <N>, "
                    f"python3 $K fix {_status_arg(d)} <name>, or python3 $K forge {_status_arg(d)} --only "
                    f"{flagged.replace(', ', ',')} --force --candidates 3"] if flagged else []) + \
                  ([f"python3 $K forge {_status_arg(d)}"] if not flagged or any(x.startswith(("no final file", "not forged")) for x in probs) else [])
            if svgs and not (d / "svg_report.json").exists():
                nxt.append(f"python3 $K svg {_status_arg(d)}   (in parallel with forge)")
        elif unforged:
            nxt = [f"python3 $K forge {_status_arg(d)}"] + ([f"python3 $K svg {_status_arg(d)}   (in parallel with forge)"]
                                                           if svgs and not (d / "svg_report.json").exists() else [])
        elif svgs and not (d / "svg_report.json").exists(): nxt = [f"python3 $K svg {_status_arg(d)}"]
        elif not (d / "tokens" / "tokens.json").exists(): nxt = [f"python3 $K tokens {_status_arg(d)}"]
        elif not (d / "compare" / "report.json").exists(): nxt = [f"python3 $K compare {_status_arg(d)} <screenshot>"]
        else: nxt = ["review reports; no pending kit step"]
        _status_scan_add(units, seen, d, "build", d.name, checks, probs, nxt)


def _status_scan_misc_units(d, children, names, units, seen):
    """Record app icon, sequence, token, comparison, font and animation outputs."""
    if "appicon.json" in names:
        data, err = _status_json(d / "appicon.json")
        probs = [err] if err else []
        if isinstance(data, dict):
            for key in ("safe_zone", "glyph"):
                if isinstance(data.get(key), dict) and data[key].get("ok") is False: probs.append(f"{key}.ok is false")
            probs.extend(str(x) for x in data.get("warnings", []))
        _status_scan_add(units, seen, d, "appicon", d.name, [{"name": "appicon.json", "status": "done"}], probs, [])
    concept_pngs = [p for p in children if p.suffix.lower() == ".png" and re.match(r"concept(?:_v\d+)?\.png$", p.name)]
    if "concepts" in names or concept_pngs:
        checks = [{"name": "concept PNGs", "status": "done" if concept_pngs else "missing"},
                  {"name": "review/", "status": "done" if (d / "review").is_dir() and any((d / "review").iterdir()) else "missing"}]
        chosen = concept_pngs[0] if len(concept_pngs) == 1 else None
        nxt = [f"python3 $K appicon build {_status_arg(chosen)} --out {_status_arg(d / 'build')}" ] if chosen else []
        _status_scan_add(units, seen, d, "appicon concepts", d.name, checks, [], nxt)
    if "sequence.json" in names:
        meta, err = _status_json(d / "sequence.json")
        probs = [err] if err else []
        checks = [{"name": "sequence.json", "status": "done"}]
        if isinstance(meta, dict) and isinstance(meta.get("frames"), int) and 0 <= meta["frames"] <= 10000:
            if meta.get("route"):
                checks.append({"name": "route", "status": str(meta["route"].get("engine", meta["route"]))})
            missing = []
            for i in range(meta["frames"]):
                raw = d / "raw" / f"{i:03d}.png"
                side = raw.with_suffix(".json")
                if not raw.exists(): missing.append(i)
                elif side.exists():
                    side_data, _ = _status_json(side)
                    guide = meta.get("guide_sha1") or []
                    if not isinstance(side_data, dict) or i >= len(guide) or side_data.get("guide_sha1") != guide[i]:
                        probs.append(f"stale frame {i}")
            if missing: probs.append(f"frames missing: {missing[:10]}")
            checks.append({"name": "raw frames", "status": "missing" if missing else "done"})
            qa, qe = _status_json(d / "report.json") if (d / "report.json").exists() else (None, None)
            if qe: probs.append(qe)
            elif isinstance(qa, dict):
                if qa.get("route"): checks.append({"name": "route", "status": str(qa["route"])})
                q = qa.get("outliers") or (qa.get("sequence", {}).get("outliers", []) if isinstance(qa.get("sequence"), dict) else [])
                if q: probs.append(f"outlier frames: {q}")
        nxt = [f"python3 $K sequence {_status_arg(d)} <name> render"] if any("missing" in p for p in probs) else []
        _status_scan_add(units, seen, d, "sequence", d.name, checks, probs, nxt)
    if "tokens.json" in names and d.name == "tokens":
        data, err = _status_json(d / "tokens.json")
        probs = [err] if err else []
        if isinstance(data, dict):
            probs.extend(f"verify {k}: {v}" for k, v in (data.get("verify") or {}).items()
                         if v is False or (isinstance(v, dict) and not v.get("ok", True))
                         or (isinstance(v, str) and v.startswith("FAILED")))
        check = {"name": "tokens.json", "status": "done"}
        inputs = [p for p in (d.parent / "design_spec.json", d.parent / "fonts" / "fonts.json") if p.exists()]
        if inputs and (d / "tokens.json").stat().st_mtime < max(p.stat().st_mtime for p in inputs):
            check["status"] = "STALE"
            probs.append("STALE: tokens.json older than " + ", ".join(p.name for p in inputs if p.stat().st_mtime > (d / "tokens.json").stat().st_mtime))
        _status_scan_add(units, seen, d, "tokens", d.name, [check], probs, [f"python3 $K tokens {_status_arg(d.parent)}"] if check["status"] == "STALE" else [])
    if "report.json" in names and d.name == "compare":
        data, err = _status_json(d / "report.json")
        probs = [err] if err else []
        score = data.get("overall", {}).get("score") if isinstance(data, dict) and isinstance(data.get("overall"), dict) else None
        if score is not None: probs.append(f"overall.score {score}/100")
        check = {"name": "report.json", "status": "done"}
        tokens = d.parent / "tokens" / "tokens.json"
        if tokens.exists() and (d / "report.json").stat().st_mtime < tokens.stat().st_mtime:
            check["status"] = "STALE"
            probs.append("STALE: compare report.json older than tokens.json")
        _status_scan_add(units, seen, d, "compare", d.name, [check], probs,
            [f"python3 $K compare {_status_arg(d.parent)} <screenshot>"] if check["status"] == "STALE" else [])
    if "fonts.json" in names and d.name == "fonts":
        _, err = _status_json(d / "fonts.json")
        _status_scan_add(units, seen, d, "fonts", d.name, [{"name": "fonts.json", "status": "done"}], [err] if err else [], [])
    if "redline.json" in names and d.name == "redline":
        _, err = _status_json(d / "redline.json")
        _status_scan_add(units, seen, d, "redline", d.name, [{"name": "redline.json", "status": "done"}], [err] if err else [], [])
    if "animation_report.json" in names:
        data, err = _status_json(d / "animation_report.json")
        probs = [err] if err else []
        if isinstance(data, dict):
            for name, motion in (data.get("motion") or {}).items():
                if isinstance(motion, dict) and motion.get("max", 1) < 0.2:
                    probs.append(f"{name}: max travel {motion.get('max')} (< 0.2)")
        _status_scan_add(units, seen, d, "animation", d.name, [{"name": "animation_report.json", "status": "done"}], probs, [])
    for p in (p for p in children if p.name.endswith(".trace.json")):
        data, err = _status_json(p)
        probs = [err] if err else []
        if isinstance(data, dict):
            advice = data.get("advice")
            if isinstance(advice, dict) and advice.get("prefer") in ("png", "webp"):
                probs.append(f"advice.prefer: {advice['prefer']}")
        _status_scan_add(units, seen, d, "trace", p.stem, [{"name": p.name, "status": "done"}], probs, [])
    if "layers.json" in names:
        _, err = _status_json(d / "layers.json")
        _status_scan_add(units, seen, d, "layers", d.name, [{"name": "layers.json", "status": "done"}], [err] if err else [], [])
    if "frames_review.png" in names:
        _status_scan_add(units, seen, d, "frames", d.name, [{"name": "frames_review.png", "status": "done"}], [], [])


def _status_scan_units(dirs, build_mockups):
    """Inspect each discovered directory and assemble its status units."""
    units = []
    seen = set()
    for d in dirs:
        # Ignore individual oversized artifacts and JSONs which cannot be read.
        children = []
        try:
            children = [p for p in d.iterdir() if p.is_file() and p.stat().st_size <= 50 * 1024 * 1024]
        except OSError:
            continue
        names = {p.name for p in children}
        _status_scan_project_units(d, children, names, build_mockups, units, seen)
        _status_scan_build_unit(d, names, build_mockups, units, seen)
        _status_scan_misc_units(d, children, names, units, seen)
    return units



def _status_scan(base):
    """Discover bounded project dirs; status never opens images or makes network calls."""
    base = Path(base).resolve()
    dirs = _status_scan_directories(base)
    build_mockups = _status_scan_build_mockups(dirs)
    return _status_scan_units(dirs, build_mockups)


def cmd_status(a):
    base = Path(a.dir or os.getcwd()).resolve()
    units = _status_scan(base)
    result = {"dir": str(base), "units": units}
    if not units:
        result["next"] = [f"python3 $K mockup \"<brief>\" --out {_status_arg(base)}",
                           f"python3 $K research \"<idea>\" --out {_status_arg(base / 'research')}"]
        if a.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"no gptkit artifacts under {base}")
            print("next: " + " | ".join(result["next"]))
        return
    result["next"] = [command for unit in units for command in unit["next"]]
    if a.json:
        print(json.dumps(result, indent=2))
        return
    for unit in units:
        print(f"{unit['type']} {_status_arg(unit['path'])}")
        for c in unit["checks"]:
            print(f"  {c['status']:7s} {c['name']}")
        for problem in unit["problems"]:
            print(f"  ! {problem}")
        for i, command in enumerate(unit["next"] or ["nothing pending in the kit: review the reports or continue to the next phase"]):
            print(("  next: " if i == 0 else "        ") + command)
