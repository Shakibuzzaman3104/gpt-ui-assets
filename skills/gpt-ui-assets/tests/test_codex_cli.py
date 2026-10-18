"""Offline tests for the Codex CLI backend. Run: python3 tests/test_codex_cli.py"""
import argparse
import contextlib
import io
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import gptkit as K  # noqa: E402
os.environ["GPTKIT_CODEX"] = "codex"  # subprocess.run is faked below; argv[0] is the default binary name
os.environ.setdefault("GPTKIT_LEDGER", "off")  # never append mocked calls to the user's cost ledger


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    K._config.cache_clear()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        K._config.cache_clear()


class Result:
    def __init__(self, stdout="", stderr="", code=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, code


def _rows(answer="ok", usage=True):
    end = {"type": "turn.completed"}
    if usage:
        end["usage"] = {"input_tokens": 11, "output_tokens": 7}
    return "junk\n{bad json\n" + "\n".join(json.dumps(x) for x in [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": answer}}, end])


def run_tests():
    # Auto-selection and the three-line setup card happen before any model call.
    saved_ready = K._codex_readiness
    try:
        with env(GPTKIT_CONFIG="/nonexistent/gptkit-test", GPTKIT_API_KEY="sk-test", GPTKIT_BACKEND="auto"):
            assert K._selected_backend() == "api"
        with env(GPTKIT_CONFIG="/nonexistent/gptkit-test", GPTKIT_API_KEY=None, OPENAI_API_KEY=None,
                 GPTKIT_BACKEND="auto"):
            K._codex_readiness = lambda: True
            assert K._selected_backend() == "cli"
        with env(GPTKIT_CONFIG="/nonexistent/gptkit-test", GPTKIT_API_KEY=None, OPENAI_API_KEY=None,
                 GPTKIT_BACKEND="auto"):
            K._codex_readiness = lambda: False
            try:
                K._selected_backend()
                assert False
            except SystemExit as e:
                assert "no backend ready" in str(e) and "API key:" in str(e) and "Local CLI:" in str(e)
    finally:
        K._codex_readiness = saved_ready

    # Argv, attachment ordering, schema file, search and malformed JSONL handling.
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_LEDGER="off", CODEX_HOME=td, HOME=td):
        a, b = Path(td) / "a.png", Path(td) / "b.png"
        a.write_bytes(b"a"); b.write_bytes(b"b")
        seen = {}
        def fake(argv, **kw):
            seen["argv"], seen["kw"] = argv, kw
            seen["schema"] = json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text())
            Path(argv[argv.index("-o") + 1]).write_text("schema-answer")
            return Result(_rows("message-answer"))
        old_run = K.subprocess.run
        K.subprocess.run = fake
        try:
            answer, usage, tid, used_model = K._codex("the prompt", images=[a, b], schema={"type": "object"},
                                                       effort="minimal", search=True, model="gpt-6-sol")
        finally:
            K.subprocess.run = old_run
        argv = seen["argv"]
        assert answer == "schema-answer" and usage == {"input_tokens": 11, "output_tokens": 7} and tid == "thread-1"
        assert used_model == "gpt-6-sol" and argv[:3] == ["codex", "exec", "--json"]
        assert "--ephemeral" in argv and "--ignore-user-config" in argv and "--ignore-rules" in argv
        assert argv[argv.index("-s") + 1] == "read-only" and argv.index("-i") < argv.index("--")
        assert argv.count("-i") == 2 and argv[-2:] == ["--", "the prompt"]
        assert "model_reasoning_effort=\"low\"" in argv and 'web_search="live"' in argv
        assert seen["schema"] == {"type": "object"} and seen["kw"]["stdin"] == K.subprocess.DEVNULL

    # Failure code classification and one process-wide fallback after a plan-model refusal.
    for phrase, code in [("429 usage limit", "quota"), ("authentication required", "login"),
                         ("generic failure", "exhausted")]:
        assert K._codex_error(phrase).code == code
    assert K._codex_error("not supported when using Codex with a ChatGPT account").code == "plan_model"
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_LEDGER="off", CODEX_HOME=td, HOME=td):
        calls = []
        def retry_run(argv, **kw):
            calls.append(argv[argv.index("-m") + 1])
            if len(calls) == 1:
                return Result('{"type":"turn.failed","error":{"message":"Model not supported when using Codex with a ChatGPT account"}}', code=1)
            Path(argv[argv.index("-o") + 1]).write_text("done")
            return Result(_rows("done"))
        old_run, old_refused = K.subprocess.run, K._CODEX_REFUSED_MODEL
        K.subprocess.run, K._CODEX_REFUSED_MODEL = retry_run, False
        try:
            assert K._codex("p", model="gpt-6-sol")[0] == "done"
            assert calls == ["gpt-6-sol", "gpt-6-luna"]
        finally:
            K.subprocess.run, K._CODEX_REFUSED_MODEL = old_run, old_refused

    # ask() writes strict schema and image attachment; response is parsed and required keys checked.
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_CONFIG="/nonexistent", GPTKIT_BACKEND="cli", GPTKIT_LEDGER="off"):
        K._codex_readiness = lambda: True
        seen = {}
        def ask_run(prompt, **kw):
            seen.update(prompt=prompt, **kw)
            assert len(kw["images"]) == 1 and Path(kw["images"][0]).suffix == ".png"
            assert "Answer directly from the attached images and text" in prompt
            assert kw["schema"]["required"] == ["x"]
            return ('{"x":2}', None, None, "gpt-6-luna")
        old = K._codex
        K._codex = ask_run
        try:
            assert K.ask([Image.new("RGB", (3, 3)), "read it"], {"type": "object", "required": ["x"]}) == {"x": 2}
        finally:
            K._codex = old

    # generate picks only a new generated image and reports actual size/format.
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_CONFIG="/nonexistent", GPTKIT_BACKEND="cli", GPTKIT_LEDGER="off", CODEX_HOME=td):
        K._codex_readiness = lambda: True
        root = Path(td) / "generated_images"
        root.mkdir()
        old_image = root / "old.png"
        Image.new("RGB", (2, 2), "red").save(old_image)
        def gen_run(*args, **kwargs):
            Image.new("RGB", (5, 7), "blue").save(root / "new.webp", "WEBP")
            return ("DONE", None, None, "gpt-6-luna")
        old = K._codex
        K._codex = gen_run
        try:
            raw, meta = K.generate("sunrise", size="9x19")
            assert Image.open(io.BytesIO(raw)).format == "PNG" and meta["size"] == "5x7"
            assert meta["source_format"] == "WEBP" and meta["quality"] == "n/a" and meta["backend"] == "cli"
        finally:
            K._codex = old

    # edit warning is emitted once per process.
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_CONFIG="/nonexistent", GPTKIT_BACKEND="cli", GPTKIT_LEDGER="off", CODEX_HOME=td):
        K._codex_readiness = lambda: True
        calls = []
        def edit_run(prompt, **kw):
            calls.append((prompt, kw))
            root = Path(td) / "generated_images"; root.mkdir(exist_ok=True)
            Image.new("RGB", (3, 3)).save(root / f"{len(calls)}.png")
            return ("DONE", None, None, "gpt-6-luna")
        old, warned = K._codex, K._CODEX_EDIT_WARNED
        K._codex, K._CODEX_EDIT_WARNED = edit_run, False
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                K.edit(b"reference", "revise", fidelity="high", task="precise")
                K.edit(b"reference", "revise", fidelity="high", task="precise")
            assert err.getvalue().count("no fidelity or size control") == 1
            assert "referenced_image_paths" in calls[0][0] and len(calls[0][1]["images"]) == 1
        finally:
            K._codex, K._CODEX_EDIT_WARNED = old, warned

    # setup preserves keys/other lines and secures the config.
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_CONFIG=str(Path(td) / "env"), GPTKIT_LEDGER="off"):
        cfg = Path(td) / "env"
        cfg.write_text("OTHER=keep\nGPTKIT_BACKEND=api\n")
        K._codex_readiness = lambda: True
        with contextlib.redirect_stdout(io.StringIO()):
            K.cmd_setup(argparse.Namespace(backend="cli", api_key_stdin=False, base_url=None))
        assert "OTHER=keep" in cfg.read_text() and "GPTKIT_BACKEND=cli" in cfg.read_text()
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o600

    # doctor reports both readiness states and does not fail when the CLI is ready without an API key.
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_CONFIG="/nonexistent", GPTKIT_API_KEY=None,
                                                  OPENAI_API_KEY=None, GPTKIT_BACKEND="auto"):
        names = ("_backend_status", "_renderer_name", "_chrome", "_android_sdk", "_find_ebsynth",
                 "_lottie_js_cache", "_cache_dir", "_lottie_js_ok")
        saved = {n: getattr(K, n) for n in names}
        K._backend_status = lambda: (False, True, "cli", "Codex CLI installed and logged in")
        K._renderer_name, K._chrome = lambda: None, lambda: None
        K._android_sdk, K._find_ebsynth = lambda: None, lambda _: None
        K._lottie_js_cache, K._cache_dir, K._lottie_js_ok = lambda: Path(td) / "none.js", lambda: Path(td), lambda _: False
        try:
            with contextlib.redirect_stdout(io.StringIO()) as doctor:
                K.cmd_doctor(argparse.Namespace())
            out = doctor.getvalue()
            assert "backend     : cli (auto: Codex CLI installed and logged in)" in out
            assert "✗ API backend" in out and "✓ Local CLI" in out
        finally:
            for n, value in saved.items():
                setattr(K, n, value)

    # API text routing, quality request body and plan annotation.
    with env(GPTKIT_CONFIG="/nonexistent", GPTKIT_BACKEND="api", GPTKIT_API_KEY="sk-test", GPTKIT_QUALITY="low",
             GPTKIT_LEDGER="off"):
        saved_list, saved_post = K._listed_models, K.requests.post
        K._listed_models = lambda base, key: frozenset(["gpt-6-sol", "gpt-6-luna"])
        calls = []
        K.requests.post = lambda url, **kw: (calls.append(kw) or type("R", (), {
            "status_code": 200, "json": lambda self: {"data": [{"b64_json": __import__("base64").b64encode(
                (lambda b: (Image.new("RGB", (2, 2)).save(b, "PNG"), b.getvalue())[1])(io.BytesIO())).decode()}]}})())
        try:
            assert K.pick_text_model() == "gpt-6-sol"
            with contextlib.redirect_stdout(io.StringIO()) as out:
                K._plan(1, 0, image_model="gpt-image-2")
            assert "quality=low" in out.getvalue()
            K.generate("draft")
            assert calls[0]["json"]["quality"] == "low"
        finally:
            K._listed_models, K.requests.post = saved_list, saved_post



def test_plan_on_the_cli_backend_is_quota_not_money():
    with env(GPTKIT_CONFIG="/nonexistent", GPTKIT_BACKEND="cli", GPTKIT_BUDGET_USD="0.01"):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert K._plan(3, 1, image_model="gpt-image-2") == 0.0
    assert "Codex CLI (ChatGPT plan quota, no API cost)" in out.getvalue() and "$" not in out.getvalue()

if __name__ == "__main__":
    run_tests()
    test_plan_on_the_cli_backend_is_quota_not_money()
    print("ok")
