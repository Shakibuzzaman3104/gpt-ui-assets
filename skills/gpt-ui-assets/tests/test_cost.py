"""API spend ledger and budget helpers; fully offline. Run: python3 tests/test_cost.py."""
import argparse
import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import gptkit as K


@contextlib.contextmanager
def env(**values):
    values.setdefault("GPTKIT_CONFIG", "/nonexistent/gptkit-env")
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


def png():
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(b, "PNG")
    return b.getvalue()


def test_image_records_usage_and_fallback():
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_LEDGER=str(Path(td) / "l.jsonl")):
        K._LEDGER_CMD, K._LEDGER_DIR = "forge", td
        raw = png()
        response = {"data": [{"b64_json": base64.b64encode(raw).decode()}], "usage": {
            "input_tokens": 20, "input_tokens_details": {"text_tokens": 10, "image_tokens": 10},
            "output_tokens": 100}, "quality": "high", "size": "1024x1024"}
        saved = K._call
        K._call = lambda *args, **kwargs: response
        try:
            K.generate("one")
            K._call = lambda *args, **kwargs: {"data": [{"b64_json": base64.b64encode(raw).decode()}]}
            K.generate("two")
        finally:
            K._call = saved
        rows = [json.loads(x) for x in (Path(td) / "l.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and rows[0]["usage"]["output_tokens"] == 100
        assert rows[0]["cmd"] == "forge" and rows[0]["dir"] == str(Path(td).resolve())
        assert rows[0]["estimated"] is False and rows[1]["estimated"] is True


def test_responses_and_web_search_usage():
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_LEDGER=str(Path(td) / "l.jsonl")):
        K._LEDGER_CMD = "research"
        body = {"model": "gpt-6-sol", "input": "hello"}
        response = {"usage": {"input_tokens": 400, "output_tokens": 100}, "output": [
            {"type": "web_search_call"}, {"type": "message", "content": []}, {"type": "web_search_call"}]}
        saved = K._call
        K._call = lambda *args, **kwargs: response
        try:
            assert K._responses(body) is response
        finally:
            K._call = saved
        row = json.loads((Path(td) / "l.jsonl").read_text())
        assert row["endpoint"] == "/responses" and row["web_search_calls"] == 2
        assert row["usage"]["input_tokens"] == 400 and not row["estimated"]


def test_parallel_lines_and_off():
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_LEDGER=str(Path(td) / "l.jsonl")):
        def write(i):
            K._ledger_record("/responses", "gpt-6-sol", {"usage": {"input_tokens": i, "output_tokens": 1}})
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(write, range(100)))
        lines = (Path(td) / "l.jsonl").read_text().splitlines()
        assert len(lines) == 100 and all(isinstance(json.loads(x), dict) for x in lines)
        with env(GPTKIT_LEDGER="off"):
            K._ledger_record("/responses", "gpt-6-sol", {})
        assert len((Path(td) / "l.jsonl").read_text().splitlines()) == 100


def test_cost_json_and_malformed():
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_BASE_URL="https://api.openai.com/v1", GPTKIT_CONFIG="/none"):
        path = Path(td) / "ledger.jsonl"
        path.write_text(json.dumps({"ts": "2026-10-18T00:00:00+00:00", "cmd": "forge", "dir": td,
                                    "endpoint": "/images/generations", "model": "gpt-image-2", "usage": {
                                        "input_tokens": 4, "input_tokens_details": {"text_tokens": 3}, "output_tokens": 12},
                                    "est_usd": 0.01, "web_search_calls": 0}) + "\n{bad\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            K.cmd_cost(argparse.Namespace(ledger=str(path), since=None, by="model", json=True))
        obj = json.loads(out.getvalue())
        assert obj["total"]["calls"] == 1 and obj["total"]["image_calls"] == 1
        assert obj["total"]["est_usd"] == 0.01 and obj["malformed"] == 1


def test_budget_exceeded_and_dry_run_no_calls():
    calls = []
    with tempfile.TemporaryDirectory() as td, env(GPTKIT_BUDGET_USD="0.01", GPTKIT_IMAGE_MODEL="gpt-image-2"):
        saved = K.generate
        K.generate = lambda *args, **kwargs: calls.append(args)
        try:
            K.cmd_mockup(argparse.Namespace(edit=None, style_ref=None, size="1024x1536", out=td,
                                            platform=None, raw=False, prompt="test", model=None,
                                            variants=1, name="mockup", no_critique=False,
                                            no_house_style=False))
        except SystemExit as e:
            assert "exceeds GPTKIT_BUDGET_USD" in str(e)
        else:
            raise AssertionError("budget should reject")
        finally:
            K.generate = saved
        assert not calls
    # A dry-run command exits after planning without reaching its API function.
    with tempfile.TemporaryDirectory() as td:
        saved = K.think
        K.think = lambda *args, **kwargs: calls.append(args)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                K.cmd_research(argparse.Namespace(depth="standard", model="gpt-6-sol", dry_run=True,
                                                  idea="test app", platform=None, region="global", context=None,
                                                  out=td, effort="high", synth_effort="xhigh", jobs=2))
        finally:
            K.think = saved
        assert "plan:" in out.getvalue() and not calls


def test_plan_prices_each_call_by_its_size():
    with env(GPTKIT_IMAGE_MODEL="gpt-image-2", GPTKIT_BUDGET_USD=None):
        with contextlib.redirect_stdout(io.StringIO()):
            square = K._plan(2, 0, edits=True)
            wide = K._plan(2, 0, edits=True, size=["1536x1024", "1536x1024"])
            mixed = K._plan(2, 0, edits=True, size=["1536x1024", "1024x1024"])
    assert square < mixed < wide, (square, mixed, wide)


def run():
    tests = [test_plan_prices_each_call_by_its_size, test_image_records_usage_and_fallback, test_responses_and_web_search_usage,
             test_parallel_lines_and_off, test_cost_json_and_malformed,
             test_budget_exceeded_and_dry_run_no_calls]
    for test in tests:
        test()
        print("ok", test.__name__)


if __name__ == "__main__":
    run()
