"""Local OpenAI Codex CLI backend and setup command."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

from .core import ApiError, _config, _ledger_record, _png_bytes
from .research import RESEARCH_MODEL, _check_keys

_CODEX_READY = None
_CODEX_NOTICE = False
_CODEX_REFUSED_MODEL = False
_CODEX_EDIT_WARNED = False
_CODEX_MODEL_CACHE = None


def _codex_bin():
    """The Codex CLI to run: GPTKIT_CODEX (a path or command name; "none" disables the backend), else `codex`."""
    return _config().get("codex") or "codex"


def _codex_home():
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()


def _codex_readiness():
    """Cache Codex readiness for this process; never inspect credential files."""
    global _CODEX_READY
    if _CODEX_READY is None:
        if _codex_bin().lower() in ("none", "off", "0"):
            _CODEX_READY = False
            return _CODEX_READY
        try:
            a = subprocess.run([_codex_bin(), "--version"], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=15)
            b = subprocess.run([_codex_bin(), "login", "status"], stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=15) if a.returncode == 0 else None
            _CODEX_READY = bool(a.returncode == 0 and b and b.returncode == 0)
        except (OSError, subprocess.TimeoutExpired):
            _CODEX_READY = False
    return _CODEX_READY


def _selected_backend():
    global _CODEX_NOTICE
    c = _config()
    backend = c.get("backend", "auto")
    if backend == "api":
        return "api"
    if backend == "cli":
        if not _codex_readiness():
            _setup_card()
        return "cli"
    if c.get("key"):
        return "api"
    if _codex_readiness():
        if not _CODEX_NOTICE:
            _CODEX_NOTICE = True
            print("  (backend: Codex CLI on your ChatGPT plan; no API key found. Set GPTKIT_BACKEND=api to change)",
                  file=sys.stderr)
        return "cli"
    _setup_card()


def _setup_card():
    sys.exit("gptkit: no backend ready. Pick one:\n"
             "  API key:   python3 <kit> setup --backend api   (or set GPTKIT_API_KEY in ~/.config/gptkit/env)\n"
             "  Local CLI: npm i -g @openai/codex && codex login, then python3 <kit> setup --backend cli")


def _codex_default_model():
    global _CODEX_MODEL_CACHE
    if _CODEX_MODEL_CACHE is not None:
        return _CODEX_MODEL_CACHE
    candidate = "gpt-6-luna"
    try:
        data = json.loads((_codex_home() / "models_cache.json").read_text())
        models = data if isinstance(data, list) else data.get("models", [])
        slugs = {m.get("slug") for m in models if isinstance(m, dict)}
        candidate = next((m for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna") if m in slugs), candidate)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    _CODEX_MODEL_CACHE = candidate
    return candidate


def _codex_error(text, prompt=""):
    low = text.lower()
    if re.search(r"rate limit|usage limit|quota|429|insufficient_quota|try again in|resource_exhausted", low):
        code = "quota"
    elif re.search(r"not logged in|authenticat", low):
        code = "login"
    elif "not supported when using codex with a chatgpt account" in low:
        code = "plan_model"
    else:
        code = "exhausted"
    return ApiError(0, f"Codex CLI [{code}]: {text[:500] or 'command failed'}", code=code, prompt=prompt[:200] or None)


def _codex(prompt, images=(), schema=None, model=None, effort=None, search=False, timeout=900):
    """Run Codex in an ephemeral read-only workspace; return answer, usage, thread id, actual model."""
    global _CODEX_REFUSED_MODEL
    selected = model or _config().get("text_model") or os.environ.get("GPTKIT_TEXT_MODEL") or _codex_default_model()
    chosen = "gpt-6-luna" if _CODEX_REFUSED_MODEL and selected != "gpt-6-luna" else selected
    with tempfile.TemporaryDirectory(prefix="gptkit-codex-") as td:
        temp = Path(td)
        work = Path.home() / ".cache/gptkit/codex-work"
        work.mkdir(parents=True, exist_ok=True)
        output = temp / "last.txt"
        argv = [_codex_bin(), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
                "--ignore-rules", "-s", "read-only", "-C", str(work), "-m", chosen,
                "-c", 'approval_policy="never"']
        if effort:
            argv += ["-c", f'model_reasoning_effort="{"low" if effort in ("none", "minimal") else effort}"']
        if search:
            argv += ["-c", 'web_search="live"']
        schema_path = None
        if schema is not None:
            schema_path = temp / "schema.json"
            schema_path.write_text(json.dumps(schema))
            argv += ["--output-schema", str(schema_path)]
        argv += ["-o", str(output)]
        for image in images:
            argv += ["-i", str(Path(image).resolve())]
        argv += ["--", prompt]
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ApiError(0, f"Codex CLI timed out after {timeout}s", code="exhausted", prompt=prompt[:200])
        except OSError as e:
            raise ApiError(0, f"Codex CLI could not run: {e}", code="exhausted", prompt=prompt[:200])
        thread_id, answer, agent_answer, usage, failure, errors, completed = None, "", "", None, "", [], False
        for line in (result.stdout or "").splitlines():
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            typ = row.get("type")
            if typ == "thread.started":
                thread_id = row.get("thread_id")
            elif typ == "item.completed":
                item = row.get("item") or {}
                if item.get("type") == "agent_message":
                    agent_answer = item.get("text") or agent_answer
            elif typ == "turn.completed":
                completed = True
                u = row.get("usage") or {}
                usage = {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}
            elif typ == "turn.failed":
                failure = ((row.get("error") or {}).get("message") or "turn failed")
            elif typ == "error":
                errors.append((row.get("message") or row.get("error") or "error"))
        stderr = result.stderr or ""
        failure_text = failure or ("; ".join(map(str, errors)) if errors and not completed else "")
        if "not supported when using codex with a chatgpt account" in (stderr + " " + str(failure_text)).lower() and chosen != "gpt-6-luna":
            _CODEX_REFUSED_MODEL = True
            return _codex(prompt, images=images, schema=schema, model="gpt-6-luna", effort=effort,
                          search=search, timeout=timeout)
        if result.returncode != 0 or failure_text or (errors and not completed):
            raise _codex_error(stderr + " " + str(failure_text), prompt)
        try:
            saved_answer = output.read_text()
            answer = saved_answer if saved_answer.strip() else agent_answer
        except OSError:
            answer = agent_answer
        if not answer.strip():
            raise ApiError(0, "Codex CLI returned no answer", code="empty", prompt=prompt[:200])
        _ledger_record("codex exec", chosen, {"usage": usage} if usage else {}, backend="cli")
        return answer, usage, thread_id, chosen


def _cli_ask(parts, schema=None, name="result", model=None):
    texts, paths = [], []
    with tempfile.TemporaryDirectory(prefix="gptkit-ask-") as td:
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            else:
                raw = part if isinstance(part, bytes) else _png_bytes(part)
                path = Path(td) / f"image-{len(paths) + 1}.png"
                path.write_bytes(raw)
                paths.append(path)
        preface = "Answer directly from the attached images and text. Do not run commands or edit files."
        prompt = preface + "\n\n" + "\n\n".join(texts)
        if paths:
            prompt += "\n\n" + " ".join(f"Image {i} is attached as image {i}." for i in range(1, len(paths) + 1))
        answer, _, _, _ = _codex(prompt, images=paths, schema=schema, model=model)
    if schema:
        try:
            data = json.loads(answer)
        except ValueError as e:
            raise ApiError(0, f"Codex CLI answer was not JSON: {e}", code="not_json", prompt=prompt[:200])
        if schema and isinstance(schema, dict) and "required" in schema:
            return _check_keys(data, schema, name)
        return data
    return answer


def _cli_think(prompt, schema=None, name="result", model=None, effort="high", search=False):
    instr = prompt
    if search:
        instr += "\n\nWhen using web search, cite sources as inline Markdown links [title](https://...)."
    answer, _, _, _ = _codex(instr, schema=schema, model=model or RESEARCH_MODEL, effort=effort, search=search, timeout=1200)
    cites = []
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", answer):
        item = {"title": title, "url": url}
        if item not in cites:
            cites.append(item)
    if schema:
        try:
            answer = json.loads(answer)
        except ValueError as e:
            raise ApiError(0, f"Codex CLI answer was not JSON: {e}", code="not_json", prompt=prompt[:200])
    return answer, cites


def _generated_snapshot():
    root = _codex_home() / "generated_images"
    files = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    files[str(p.resolve())] = p.stat().st_mtime_ns
                except OSError:
                    pass
    return root, files


def _pick_codex_image(root, before, started, thread_id=None):
    """The image this run's image_gen call saved. Codex files them under generated_images/<thread id>/, which keeps
    parallel variants apart; without a thread id, the newest new file anywhere under the root."""
    candidates = []
    if thread_id and (root / thread_id).is_dir():
        root = root / thread_id
    if root.exists():
        for p in root.rglob("*"):
            if p.suffix.lower() not in (".png", ".webp", ".jpg", ".jpeg"):
                continue
            try:
                stamp = p.stat().st_mtime_ns
                if stamp >= started and (str(p.resolve()) not in before or stamp > before[str(p.resolve())]):
                    candidates.append((stamp, p))
            except OSError:
                pass
    if not candidates:
        raise ApiError(0, "Codex image_gen created no new image", code="no_image")
    return max(candidates, key=lambda pair: pair[0])[1]


def _image_from_codex(path):
    try:
        im = Image.open(path)
        im.load()
        fmt, size = im.format, im.size
        return _png_bytes(im.convert("RGBA" if "A" in im.getbands() else "RGB")), {
            "quality": "n/a", "size": f"{size[0]}x{size[1]}", "source_format": fmt, "backend": "cli"}
    except (OSError, ValueError) as e:
        raise ApiError(0, f"Codex image_gen output is unreadable: {e}", code="bad_image")


def _image_prompt(prompt, size, background, ref=None):
    try:
        w, h = [int(x) for x in size.split("x")]
    except (ValueError, AttributeError):
        w, h = 1024, 1024
    ratio = float(w) / max(1, h)
    if .95 <= ratio <= 1.05:
        aspect = "square canvas, 1:1 aspect ratio"
    elif ratio < 1:
        aspect = f"tall portrait canvas, 9:{9 / ratio:.1f} aspect ratio"
    else:
        aspect = f"wide landscape canvas, {ratio:.2f}:1 aspect ratio"
    transparent = " Create a transparent background." if background == "transparent" else ""
    if ref:
        return (f"Use the image_gen tool exactly once to create this image, then reply DONE. {aspect}.{transparent} "
                f"Call image_gen with referenced_image_paths=[{str(Path(ref).resolve())!r}]. Image prompt: {prompt}")
    return (f"Use the image_gen tool exactly once to create this image, then reply DONE. {aspect}.{transparent} "
            f"Image prompt: {prompt}")


def _cli_generate(prompt, size="1024x1024", background="auto", model=None, task="concept"):
    root, before = _generated_snapshot()
    started = time.time_ns()
    thread_id = _codex(_image_prompt(prompt, size, background), timeout=600)[2]
    return _image_from_codex(_pick_codex_image(root, before, started, thread_id))


def _cli_edit(ref_png, prompt, size="1024x1024", background="auto", model=None, fidelity="high", task="precise"):
    global _CODEX_EDIT_WARNED
    if (fidelity == "high" or task == "precise") and not _CODEX_EDIT_WARNED:
        _CODEX_EDIT_WARNED = True
        print("  ! the Codex CLI image tool has no fidelity or size control; use the API backend for pixel-exact asset work (forge, fix, sequence)", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="gptkit-edit-") as td:
        ref = Path(td) / "reference.png"
        ref.write_bytes(ref_png)
        root, before = _generated_snapshot()
        started = time.time_ns()
        thread_id = _codex(_image_prompt(prompt, size, background, ref=ref), images=[ref], timeout=600)[2]
        return _image_from_codex(_pick_codex_image(root, before, started, thread_id))


def _backend_parsers(sub):
    p = sub.add_parser("setup", help="configure the API key or local Codex CLI backend")
    p.add_argument("--backend", choices=["api", "cli", "auto"])
    p.add_argument("--api-key-stdin", action="store_true", help="read API key from stdin without echoing it")
    p.add_argument("--base-url", help="API base URL")
    p.set_defaults(fn=cmd_setup)


def _backend_status():
    c = _config()
    cli = _codex_readiness()
    api = bool(c.get("key"))
    backend = c.get("backend", "auto")
    selected = ("api" if api else "cli" if cli else "none") if backend == "auto" else backend
    reason = "API key set" if api else "Codex CLI installed and logged in" if cli else "no API key or ready Codex CLI"
    return api, cli, selected, reason


def cmd_setup(a):
    global _CODEX_READY
    api, cli, selected, reason = _backend_status()
    if not a.backend and not a.api_key_stdin and not a.base_url:
        print(f"API backend : {'✓ ready' if api else '✗ not ready — set GPTKIT_API_KEY in ~/.config/gptkit/env'}")
        print(f"Local CLI   : {'✓ ready' if cli else '✗ not ready — npm i -g @openai/codex && codex login'}")
        print(f"Recommendation: {selected} ({reason})")
        return
    path = Path(_config()["config_path"])
    values = {}
    if a.backend:
        values["GPTKIT_BACKEND"] = a.backend
    if a.base_url:
        values["GPTKIT_BASE_URL"] = a.base_url
    if a.api_key_stdin:
        key = sys.stdin.readline().strip()
        if not key:
            sys.exit("gptkit: no API key received on stdin")
        values["GPTKIT_API_KEY"] = key
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text().splitlines() if path.exists() else []
    for key, value in values.items():
        found = False
        for i, line in enumerate(existing):
            if line.strip().partition("=")[0].strip() == key:
                existing[i] = f"{key}={value}"
                found = True
                break
        if not found:
            existing.append(f"{key}={value}")
    path.write_text("\n".join(existing) + "\n")
    os.chmod(path, 0o600)
    _config.cache_clear()
    _CODEX_READY = None
    api, cli, selected, reason = _backend_status()
    print(f"Saved {path} (mode 600)")
    print(f"API backend : {'✓ ready' if api else '✗ not ready — set GPTKIT_API_KEY in ~/.config/gptkit/env'}")
    print(f"Local CLI   : {'✓ ready' if cli else '✗ not ready — npm i -g @openai/codex && codex login'}")
    print(f"Recommendation: {selected} ({reason})")
