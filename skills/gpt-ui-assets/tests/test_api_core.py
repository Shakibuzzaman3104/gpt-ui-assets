"""API layer, config, doctor, research/blueprint and main() error handling (contract C2), all offline: requests.post is
mocked, no network. Run: python3 tests/test_api_core.py (or pytest)."""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import gptkit as K  # noqa: E402

FIX = HERE / "api_core"
MODERATION = {"error": {"message": "Your request was rejected by the safety system.", "type": "user_error",
                        "param": None, "code": "moderation_blocked"}}


class R:
    """A requests.Response stand-in."""

    def __init__(self, status, body=None, text=None, headers=None):
        self.status_code, self.headers = status, headers or {}
        self.text = text if text is not None else json.dumps(body)
        self.content = self.text.encode()

    def json(self):
        return json.loads(self.text)


@contextlib.contextmanager
def env(**kv):
    """Temporarily set env vars (None removes) with the config cache cleared on both sides."""
    keep = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    K._config.cache_clear()
    try:
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        K._config.cache_clear()


@contextlib.contextmanager
def api(responses, sleeps=None):
    """Mock requests.post with a queue of R (or exceptions) and time.sleep with a recorder; yields the posted calls."""
    calls, queue = [], list(responses)
    saved = K.requests.post, K.time.sleep

    def post(url, headers=None, json=None, files=None, data=None, timeout=None):
        import copy
        calls.append({"url": url, "json": copy.deepcopy(json), "data": copy.deepcopy(data), "files": files, "headers": headers})
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    K.requests.post = post
    K.time.sleep = lambda s: (sleeps if sleeps is not None else []).append(s)
    try:
        with env(GPTKIT_API_KEY="sk-test-key-123456", GPTKIT_CONFIG="/nonexistent/gptkit-env", GPTKIT_RETRIES=None,
                 GPTKIT_MODERATION=None, GPTKIT_DEBUG=None):
            yield calls
    finally:
        K.requests.post, K.time.sleep = saved


def raises(fn, exc=K.ApiError):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def message(text, **extra):
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}], **extra}


def png_bytes(size=(8, 8), color=(255, 0, 0, 255), fmt="PNG"):
    buf = io.BytesIO()
    im = Image.new("RGBA", size, color)
    (im.convert("RGB") if fmt == "JPEG" else im).save(buf, fmt)
    return buf.getvalue()


def b64(raw):
    import base64
    return base64.b64encode(raw).decode()


# ---- config

def test_config_precedence_and_parsing():
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "env"
        cfg.write_text('GPTKIT_BASE_URL="https://gw.example/v1/?a=b=c"\n# GPTKIT_IMAGE_MODEL=commented\n'
                       "GPTKIT_TEXT_MODEL='from-file'\nGPTKIT_API_KEY=sk-file\nGPTKIT_RETRIES=5\nGPTKIT_DEBUG=yes\n")
        with env(GPTKIT_CONFIG=str(cfg), GPTKIT_TEXT_MODEL="from-env", GPTKIT_API_KEY=None, OPENAI_API_KEY=None,
                 GPTKIT_BASE_URL=None, OPENAI_BASE_URL=None, GPTKIT_IMAGE_MODEL=None, GPTKIT_RETRIES=None, GPTKIT_DEBUG=None):
            c = K._config()
            assert c["base"] == "https://gw.example/v1/?a=b=c" and c["key"] == "sk-file"
            assert c["text_model"] == "from-env" and c["image_model"] == "gpt-image-2"
            assert c["retries"] == 5 and c["debug"] is True and c["moderation"] is None
        with env(GPTKIT_CONFIG=str(cfg), GPTKIT_RETRIES="0", GPTKIT_DEBUG="0"):
            assert K._config()["retries"] == 0 and K._config()["debug"] is False
        with env(GPTKIT_CONFIG=str(cfg), GPTKIT_RETRIES="lots"):
            assert "GPTKIT_RETRIES" in str(raises(K._config, SystemExit))
        with env(GPTKIT_CONFIG=str(cfg), GPTKIT_RETRIES=None, GPTKIT_MODERATION="high"):
            assert "GPTKIT_MODERATION" in str(raises(K._config, SystemExit))
        with env(GPTKIT_CONFIG=str(cfg), GPTKIT_MODERATION="low"):
            assert K._config()["moderation"] == "low"


def test_config_dir_and_unreadable():
    with tempfile.TemporaryDirectory() as td:
        with env(GPTKIT_CONFIG=td):
            e = raises(K._config, SystemExit)
            assert "directory" in str(e) and str(Path(td) / "env") in str(e)
        bad = Path(td) / "env"
        bad.write_bytes(b"GPTKIT_API_KEY=\xff\xfe\n")
        with env(GPTKIT_CONFIG=str(bad)):
            e = raises(K._config, SystemExit)
            assert "can't read config file" in str(e) and str(bad) in str(e)
        bad.write_text("GPTKIT_API_KEY=x\n")
        saved = K.Path.read_text

        def denied(self, *a, **k):
            raise PermissionError(13, "Permission denied", str(self))
        K.Path.read_text = denied
        try:
            with env(GPTKIT_CONFIG=str(bad)):
                assert "can't read config file" in str(raises(K._config, SystemExit))
        finally:
            K.Path.read_text = saved


def test_auth_no_key_exit_and_small_helpers():
    with env(GPTKIT_API_KEY=None, OPENAI_API_KEY=None, GPTKIT_CONFIG="/nonexistent/env"):
        e = raises(K._auth, SystemExit)
        assert "no API key" in str(e) and "/nonexistent/env" in str(e)
    assert K._size_for(1536, 1024) == "1536x1024" and K._size_for(1000, 1300) == "1024x1536"
    assert K._size_for(1000, 1100) == "1024x1024" and K._size_for(1200, 1000) == "1024x1024"
    raw = K._png_bytes(Image.new("RGBA", (3, 2), (1, 2, 3, 4)))
    assert raw[:8] == b"\x89PNG\r\n\x1a\n" and Image.open(io.BytesIO(raw)).size == (3, 2)
    url = K._data_url(raw)
    assert url.startswith("data:image/png;base64,") and K._data_url(Image.new("RGBA", (1, 1))).startswith("data:image/png;base64,")


# ---- _call

def test_retry_after_honoured():
    sleeps = []
    with api([R(429, {"error": {"message": "slow down", "code": "rate_limit_exceeded"}}, headers={"Retry-After": "3"}),
              R(200, {"ok": 1})], sleeps) as calls:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            assert K._call("/images/edits", json_body={"a": 1}) == {"ok": 1}
    assert sleeps == [3.0] and len(calls) == 2 and "retry 1/2 in 3.0 s" in err.getvalue()
    sleeps = []
    with api([R(503, text="<html>bad gateway</html>", headers={"retry-after-ms": "1500"}), R(200, {"ok": 2})], sleeps):
        with contextlib.redirect_stderr(io.StringIO()):
            assert K._call("/responses", json_body={})["ok"] == 2
    assert sleeps == [1.5]
    sleeps = []
    with api([R(500, {"error": {"message": "boom"}}), R(502, {"error": {"message": "boom"}}), R(200, {"ok": 3})], sleeps):
        with contextlib.redirect_stderr(io.StringIO()):
            assert K._call("/responses", json_body={})["ok"] == 3
    assert len(sleeps) == 2 and 4 <= sleeps[0] < 5 and 8 <= sleeps[1] < 9, sleeps
    sleeps = []
    with api([R(429, {"error": {"message": "x"}}, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), R(200, {})], sleeps):
        with contextlib.redirect_stderr(io.StringIO()):
            K._call("/responses", json_body={})
    assert 4 <= sleeps[0] < 5


def test_quota_not_retried():
    for code in ("insufficient_quota", "billing_hard_limit_reached"):
        sleeps = []
        with api([R(429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": code}})],
                 sleeps) as calls:
            e = raises(lambda: K._call("/images/generations", json_body={}))
        assert e.status == 429 and e.code == code and len(calls) == 1 and sleeps == []
        assert str(e).startswith(f"HTTP 429 /images/generations [{code}]: You exceeded")


def test_exhausted_and_transport():
    sleeps = []
    with api([R(503, {"error": {"message": "overloaded", "code": "server_error"}})] * 3, sleeps) as calls:
        with contextlib.redirect_stderr(io.StringIO()):
            e = raises(lambda: K._call("/responses", json_body={}))
    assert e.status == 503 and e.code == "exhausted" and len(calls) == 3 and len(sleeps) == 2
    assert "after 3 attempts" in str(e) and "server_error" in str(e) and "overloaded" in str(e)
    with api([K.requests.ConnectionError("dns failed")] * 3, []) as calls:
        with contextlib.redirect_stderr(io.StringIO()):
            e = raises(lambda: K._call("/responses", json_body={}))
    assert e.status == 0 and e.code == "exhausted" and "ConnectionError" in str(e) and len(calls) == 3
    with api([R(503, {"error": {"message": "x"}})] * 3, []) as calls:
        with env(GPTKIT_RETRIES="0"):
            e = raises(lambda: K._call("/responses", json_body={}))
    assert e.code == "exhausted" and len(calls) == 1 and "after 1 attempt)" in str(e)
    with api([R(503, {"error": {"message": "x"}})] * 2, []) as calls:
        with contextlib.redirect_stderr(io.StringIO()):
            raises(lambda: K._call("/responses", json_body={}, retries=1))
    assert len(calls) == 2


def test_not_json_and_error_fields():
    with api([R(200, text="<html>login</html>")], []) as calls:
        e = raises(lambda: K._call("/responses", json_body={}))
    assert e.code == "not_json" and e.status == 200 and "not JSON" in str(e) and len(calls) == 1
    with api([R(400, {"error": {"message": "Unknown parameter: 'moderation'.", "type": "invalid_request_error",
                                "param": "moderation", "code": "unknown_parameter"}})], []):
        e = raises(lambda: K._call("/images/edits", json_body={}))
    assert (e.status, e.code, e.etype, e.param) == (400, "unknown_parameter", "invalid_request_error", "moderation")
    assert e.body["error"]["param"] == "moderation" and str(e) == "HTTP 400 /images/edits [unknown_parameter]: Unknown parameter: 'moderation'."
    with api([R(401, text="Unauthorized")], []):
        e = raises(lambda: K._call("/responses", json_body={}))
    assert e.status == 401 and e.code is None and e.body is None and "Unauthorized" in str(e)
    with api([R(404, {"error": {"message": "The model `gpt-6` does not exist", "code": "model_not_found", "param": "model"}})], []):
        e = raises(lambda: K._call("/responses", json_body={}))
    assert "gpt-6-luna" in str(e) and "doctor" in str(e)


def test_bearer_redacted():
    leak = "upstream said: Authorization: Bearer sk-secret-abc123 and key sk-test-key-123456 rejected"
    with api([R(400, {"error": {"message": leak, "code": "bad"}})], []):
        e = raises(lambda: K._call("/responses", json_body={}))
    s = str(e)
    assert "sk-secret" not in s and "sk-test-key" not in s and "Bearer [redacted]" in s
    with api([R(500, text="Bearer sk-secret-xyz\nline two")] * 3, []):
        with contextlib.redirect_stderr(io.StringIO()):
            e = raises(lambda: K._call("/responses", json_body={}))
    assert "sk-secret" not in str(e) and "\n" not in str(e)


# ---- _with_fallback

def test_fallback_uses_param():
    rejected = {"error": {"message": "Unknown parameter: 'input_fidelity'.", "param": "input_fidelity", "code": "unknown_parameter"}}
    K._DROPPED.discard("input_fidelity")
    with api([R(400, rejected), R(200, {"ok": 1}), R(400, rejected), R(200, {"ok": 1})], []) as calls:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            out = K._with_fallback(lambda p: K._call("/images/edits", json_body=p), {"input_fidelity": "high", "quality": "high"})
            K._with_fallback(lambda p: K._call("/images/edits", json_body=p), {"input_fidelity": "high"})
    assert out == {"ok": 1} and calls[1]["json"] == {"quality": "high"} and "input_fidelity" not in calls[1]["json"]
    assert err.getvalue().count("provider rejected 'input_fidelity'") == 1, err.getvalue()
    # a moderation 400 whose prose mentions "background" must NOT drop the background param
    with api([R(400, {"error": {"message": "inappropriate background depicted", "code": "moderation_blocked", "param": None}})], []) as calls:
        e = raises(lambda: K._with_fallback(lambda p: K._call("/images/generations", json_body=p), {"background": "transparent"}))
    assert e.code == "moderation_blocked" and len(calls) == 1
    # message substring only when param is absent and the code does not say otherwise
    with api([R(400, {"error": {"message": "Unsupported parameter: 'quality'"}}), R(200, {"ok": 2})], []) as calls:
        with contextlib.redirect_stderr(io.StringIO()):
            assert K._with_fallback(lambda p: K._call("/x", json_body=p), {"quality": "high", "size": "1024x1024"})["ok"] == 2
    assert calls[1]["json"] == {"size": "1024x1024"}
    # param outside `optional`, or a non-400, propagates unchanged
    with api([R(400, {"error": {"message": "bad size", "param": "size"}})], []) as calls:
        e = raises(lambda: K._with_fallback(lambda p: K._call("/x", json_body=p), {"size": "9x9", "quality": "high"}))
    assert e.param == "size" and len(calls) == 1
    with api([R(500, rejected)] * 3, []):
        with contextlib.redirect_stderr(io.StringIO()):
            e = raises(lambda: K._with_fallback(lambda p: K._call("/x", json_body=p), {"input_fidelity": "high"}))
    assert e.code == "exhausted"
    # nested param names drop their root, restricted to `optional`
    with api([R(400, {"error": {"message": "x", "param": "reasoning.effort"}}), R(200, {"ok": 3})], []) as calls:
        with contextlib.redirect_stderr(io.StringIO()):
            K._with_fallback(lambda p: K._call("/responses", json_body=p), {"reasoning": {"effort": "high"}, "store": False},
                             K.RESPONSES_OPTIONAL)
    assert calls[1]["json"] == {"store": False}


# ---- images

def test_image_result_shapes():
    e = raises(lambda: K._image_result({"data": []}))
    assert e.code == "no_image" and e.status == 200 and "data: []" in str(e)
    assert raises(lambda: K._image_result({})).code == "no_image"
    assert raises(lambda: K._image_result({"data": [{}]})).code == "no_image"
    K._NOTED.discard("source_format")
    with contextlib.redirect_stderr(io.StringIO()) as err:
        raw, meta = K._image_result({"data": [{"b64_json": b64(png_bytes((5, 7), fmt="JPEG"))}], "size": "1024x1024"})
        K._image_result({"data": [{"b64_json": b64(png_bytes((5, 7), fmt="JPEG"))}]})
    im = Image.open(io.BytesIO(raw))
    assert im.format == "PNG" and im.size == (5, 7) and meta["source_format"] == "JPEG" and meta["size"] == "1024x1024"
    assert err.getvalue().count("provider returned JPEG") == 1
    raw, meta = K._image_result({"data": [{"b64_json": b64(png_bytes((4, 4)))}], "usage": {"total_tokens": 9}})
    assert raw == png_bytes((4, 4)) and meta["source_format"] == "PNG" and meta["usage"] == {"total_tokens": 9}
    e = raises(lambda: K._image_result({"data": [{"b64_json": b64(b"<html>nope</html>")}]}))
    assert e.code == "bad_image" and "<html" in str(e)
    assert raises(lambda: K._image_result({"data": [{"b64_json": "not base64!!"}]})).code == "bad_image"
    saved = K.requests.get
    try:
        K.requests.get = lambda url, timeout=None: type("G", (), {"content": png_bytes((2, 3)), "raise_for_status": lambda s: None})()
        raw, meta = K._image_result({"data": [{"url": "https://img.example/x.png"}]})
        assert Image.open(io.BytesIO(raw)).size == (2, 3)

        def fail(url, timeout=None):
            raise K.requests.HTTPError("404 Client Error")
        K.requests.get = fail
        assert raises(lambda: K._image_result({"data": [{"url": "https://img.example/x.png"}]})).code == "no_image"
    finally:
        K.requests.get = saved


def test_prompt_in_error():
    with api([R(400, MODERATION)], []) as calls:
        e = raises(lambda: K.generate("a violent scene " * 30, "1024x1024", "transparent"))
    assert e.code == "moderation_blocked" and e.prompt == ("a violent scene " * 30)[:200]
    assert "| prompt: 'a violent scene" in str(e) and str(e).startswith("HTTP 400 /images/generations [moderation_blocked]: Your request")
    assert calls[0]["json"]["background"] == "transparent" and "moderation" not in calls[0]["json"]
    with api([R(400, MODERATION)], []) as calls:
        e = raises(lambda: K.edit(b"png", "redraw the flame", fidelity="low"))
    assert "prompt: 'redraw the flame'" in str(e) and calls[0]["data"]["input_fidelity"] == "low" and calls[0]["files"][0][0] == "image"
    with api([R(400, MODERATION)], []):
        e = raises(lambda: K.ask([Image.new("RGBA", (2, 2)), "describe the icon", "more"], {"type": "object"}))
    assert e.prompt == "describe the icon" and "/responses" in str(e)
    with api([R(200, {"data": []})], []):
        e = raises(lambda: K.generate("blank"))
    assert e.code == "no_image" and e.prompt == "blank"
    # GPTKIT_MODERATION rides along as an optional param
    with api([R(200, {"data": [{"b64_json": b64(png_bytes())}]})], []) as calls:
        with env(GPTKIT_MODERATION="low"):
            K.generate("x")
    assert calls[0]["json"]["moderation"] == "low"


# ---- /responses

def test_ask_refusal_incomplete_empty():
    schema = {"type": "object", "properties": {}, "additionalProperties": False, "required": []}
    refusal = {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "I can't help with that."}]}]}
    shapes = [("refusal", refusal, None), ("refusal", refusal, schema),
              ("empty", {"output": [{"type": "reasoning", "summary": []}]}, schema),
              ("incomplete", {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}, None),
              ("not_json", message("Sure! Here's a summary: ..."), schema)]
    for code, resp, sch in shapes:
        with api([R(200, resp)], []):
            e = raises(lambda: K.ask(["describe this"], sch))
        assert e.code == code, (code, e.code)
        assert re.search("refus|incomplete|empty|not JSON", str(e)), str(e)
        if code == "incomplete":
            assert "max_output_tokens" in str(e) and "--effort" in str(e)
    with api([R(200, message("plain answer"))], []):
        assert K.ask(["q"]) == "plain answer"
    with api([R(200, message('{"a": 1}'))], []) as calls:
        assert K.ask(["q"], schema) == {"a": 1} and calls[0]["json"]["text"]["format"]["strict"] is True
    with api([R(200, message('{"b": 2}'))], []) as calls:  # deprecated json_object mode still works
        assert K.ask(["q"], "json") == {"b": 2} and calls[0]["json"]["text"]["format"] == {"type": "json_object"}
    ok = {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "x", "annotations": [{"type": "url_citation", "title": "T", "url": "https://a", "start_index": 0, "end_index": 1},
                                                              {"type": "file_citation", "url": "ignored"}]}]}]}
    with api([R(200, ok)], []):
        text, cites = K.think("q", effort="max")
    assert text == "x" and cites == [{"title": "T", "url": "https://a", "start_index": 0, "end_index": 1}]
    assert "effort" in str(raises(lambda: K.think("q", effort="ultra"), ValueError))


def test_think_drops_reasoning_on_rejection():
    K._DROPPED.discard("reasoning")
    with api([R(400, {"error": {"message": "Unrecognized request argument supplied: reasoning", "param": "reasoning"}}),
              R(200, message('{"k": 1}'))], []) as calls:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            out, _ = K.think("q", {"type": "object"}, search=True)
    assert out == {"k": 1} and "reasoning" in calls[0]["json"] and "reasoning" not in calls[1]["json"]
    assert calls[1]["json"]["tools"] == [{"type": "web_search"}] and "provider rejected 'reasoning'" in err.getvalue()
    # strict json_schema rejected -> one retry with strict false
    K._NOTED.discard("strict")
    with api([R(400, {"error": {"message": "strict is not supported", "param": "text.format.strict"}}),
              R(200, message('{"k": 2}'))], []) as calls:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            assert K.ask(["q"], {"type": "object"}) == {"k": 2}
    assert calls[0]["json"]["text"]["format"]["strict"] is True and calls[1]["json"]["text"]["format"]["strict"] is False
    assert "strict" in err.getvalue()
    with api([R(400, {"error": {"message": "bad schema", "param": "text.format.schema"}})] * 2, []) as calls:
        with contextlib.redirect_stderr(io.StringIO()):
            e = raises(lambda: K.ask(["q"], {"type": "object"}))
    assert e.status == 400 and len(calls) == 2


# ---- research / blueprint

def cells(row):
    return re.split(r"(?<!\\)\|", row.strip())[1:-1]


def test_mdcell_tables():
    assert K._mdcell("a|b\nc") == "a\\|b c" and K._mdcell(3.5) == "3.5"
    r = json.loads((FIX / "report.json").read_text())
    r.update({"idea": "idea", "platform": None, "region": "US"})
    r["competitors"][0]["pricing"] = "free; Pro $3/mo | Family $5/mo"
    r["competitors"][0]["name"] = "Bud|get\nApp"
    md = K._report_md(r, [{"title": "T", "url": "https://a"}])
    rows = [l for l in md.splitlines() if l.startswith("| ")]
    assert len(cells(rows[0])) == 5 and all(len(cells(x)) == 5 for x in rows[1:]), rows
    assert "Family $5/mo" in md and "\\|" in md and "[T](https://a)" in md
    b = json.loads((FIX / "blueprint.json").read_text())
    b["features"][0]["name"], b["features"][0]["solves"] = "Split | merge", "x|y"
    md = K._blueprint_md(b)
    rows = [l for l in md.splitlines() if l.startswith("| ")]
    assert all(len(cells(x)) == 6 for x in rows), rows


def test_blueprint_missing_key_and_corrupt_report():
    bp = json.loads((FIX / "blueprint.json").read_text())
    report = json.loads((FIX / "report.json").read_text())
    report.update({"idea": "Budget app", "platform": "android", "region": "US"})
    saved = K.think
    ns = lambda d, **kw: argparse.Namespace(dir=d, idea=None, platform=None, constraints=None, out=None,
                                            model=K.RESEARCH_MODEL, effort="xhigh", **kw)
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "report.json").write_text(json.dumps(report))
            short = {k: v for k, v in bp.items() if k != "personas"}
            K.think = lambda *a, **k: (short, [])
            with contextlib.redirect_stdout(io.StringIO()):
                e = raises(lambda: K.cmd_blueprint(ns(td)), SystemExit)
            assert "personas" in str(e) and "json_schema" in str(e) and not (d / "blueprint.json").exists()
            K.think = lambda *a, **k: (bp, [])
            with contextlib.redirect_stdout(io.StringIO()):
                K.cmd_blueprint(ns(td))
            assert (d / "blueprint.md").exists() and json.loads((d / "screens.json").read_text())[0]["platform"] == "android"
            (d / "report.json").write_text("{not valid json,,,")
            with contextlib.redirect_stdout(io.StringIO()):
                e = raises(lambda: K.cmd_blueprint(ns(td)), SystemExit)
            assert "report.json" in str(e) and "not valid JSON" in str(e)
            del report["risks"]
            (d / "report.json").write_text(json.dumps(report))
            e = raises(lambda: K.cmd_blueprint(ns(td)), SystemExit)
            assert "report.json" in str(e) and "risks" in str(e)
            # research: the plan and the report are checked right after think()
            K.think = lambda *a, **k: ({"product_frame": "x"}, [])
            with contextlib.redirect_stdout(io.StringIO()):
                e = raises(lambda: K.cmd_research(argparse.Namespace(idea="i", out=str(d / "r"), platform=None, region="global",
                                                                     context=None, depth="quick", model="m", effort="high",
                                                                     synth_effort="xhigh", jobs=2)), SystemExit)
            assert "questions" in str(e) and "json_schema" in str(e)
    finally:
        K.think = saved


# ---- doctor

def test_doctor_lines():
    saved_chrome, saved_get = K._chrome, K.requests.get
    ns = argparse.Namespace()
    try:
        with env(GPTKIT_API_KEY=None, OPENAI_API_KEY=None, GPTKIT_CONFIG="/nonexistent/env", GPTKIT_RETRIES="3", GPTKIT_DEBUG="1"):
            K._chrome = lambda: "/bin/false"
            K._chrome_probe.cache_clear()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                assert raises(lambda: K.cmd_doctor(ns), SystemExit).code == 1
            text = out.getvalue()
            assert "retries     : 3" in text and "debug       : on" in text and "api key     : MISSING" in text
            assert "svg render  : none: chrome found at /bin/false but failed: /bin/false: exit code 1" in text, text
            aapt = next(l for l in text.splitlines() if l.startswith("aapt2"))
            assert ("compile only" in aapt and "android.jar" in aapt) or "link checks: android-" in aapt or "not found" in aapt
            assert not ("valid" in aapt)
        K._chrome = saved_chrome
        K._chrome_probe.cache_clear()
        if K._chrome():  # a real Chrome must pass the probe with the shared flags (root gets --no-sandbox)
            ok, why = K._chrome_probe()
            assert ok, why
            with env(GPTKIT_API_KEY=None, OPENAI_API_KEY=None, GPTKIT_CONFIG="/nonexistent/env"):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    raises(lambda: K.cmd_doctor(ns), SystemExit)
            assert "svg render  : chrome (ok)" in out.getvalue()
        ids = {"data": [{"id": i} for i in ("gpt-6-sol", "gpt-6-luna", "gpt-image-1.5", "gpt-5.4", "gpt-5.3-codex-spark", "gpt-4o-audio-preview")]}
        K.requests.get = lambda url, headers=None, timeout=None: type("G", (), {"raise_for_status": lambda s: None, "json": lambda s: ids})()
        with env(GPTKIT_API_KEY="sk-test", GPTKIT_CONFIG="/nonexistent/env", GPTKIT_TEXT_MODEL="gpt-6", GPTKIT_IMAGE_MODEL="gpt-image-2"):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                K.cmd_doctor(ns)
        text = out.getvalue()
        assert "models      : 4 available" in text
        assert "text  gpt-6: NOT LISTED (set GPTKIT_TEXT_MODEL to one of: gpt-5.4, gpt-6-luna, gpt-6-sol)" in text, text
        assert "image gpt-image-2: NOT LISTED (set GPTKIT_IMAGE_MODEL to one of: gpt-image-1.5)" in text, text
        with env(GPTKIT_API_KEY="sk-test", GPTKIT_CONFIG="/nonexistent/env", GPTKIT_TEXT_MODEL="gpt-6-sol", GPTKIT_IMAGE_MODEL="gpt-image-1.5"):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                K.cmd_doctor(ns)
        assert "text  gpt-6-sol: ok" in out.getvalue() and "image gpt-image-1.5: ok" in out.getvalue()
    finally:
        K._chrome, K.requests.get = saved_chrome, saved_get
        K._chrome_probe.cache_clear()


# ---- main(): one line, exit 1, no traceback

RUNNER = """
import json, sys
sys.path.insert(0, {scripts!r})
import gptkit
class R:
    status_code, headers = 400, {{}}
    text = json.dumps({moderation!r})
    def json(self): return json.loads(self.text)
gptkit.requests.post = lambda *a, **k: R()
{extra}
sys.argv = ["gptkit"] + {argv!r}
gptkit.main()
"""


def run_cli(argv, cwd, extra="", **envx):
    e = {**os.environ, "GPTKIT_API_KEY": "sk-test", "GPTKIT_CONFIG": "/nonexistent/env", "GPTKIT_DEBUG": "0", **envx}
    code = RUNNER.format(scripts=str(SCRIPTS), moderation=MODERATION, extra=extra, argv=argv)
    return subprocess.Popen([sys.executable, "-c", code], cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_main_exits_one_line():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        Image.new("RGB", (64, 96), "#223").save(d / "mockup.png")
        assets = [{"name": "x", "kind": "icon", "format": "svg", "bbox": {"x": 4, "y": 4, "w": 16, "h": 16}, "description": "an icon",
                   "colors": ["#FFFFFF"], "transparent": True},
                  {"name": "y", "kind": "illustration", "format": "png", "bbox": {"x": 8, "y": 40, "w": 40, "h": 40},
                   "description": "a flame", "colors": ["#FF6600"], "transparent": True}]
        (d / "assets.json").write_text(json.dumps({"mockup": "mockup.png", "width": 64, "height": 96, "screen_dp": 390,
                                                   "platform": "android", "assets": assets}))
        cases = {
            "mockup": run_cli(["mockup", "a calm screen", "--out", str(d / "m")], td),
            "critique": run_cli(["critique", str(d / "mockup.png"), "--out", str(d / "c")], td),
            "svg": run_cli(["svg", td, "--only", "x"], td),
            "forge": run_cli(["forge", td, "--only", "y", "--candidates", "1", "--jobs", "1"], td),
            "research": run_cli(["research", "an idea", "--out", str(d / "r"), "--depth", "quick"], td),
            "keyerror": run_cli(["doctor"], td, extra="gptkit.cmd_doctor = lambda a: {}['personas']"),
            "interrupt": run_cli(["doctor"], td, extra="def _ki(a): raise KeyboardInterrupt\ngptkit.cmd_doctor = _ki"),
            "debug": run_cli(["research", "an idea", "--out", str(d / "r2")], td, GPTKIT_DEBUG="1"),
            "cfgdir": run_cli(["doctor"], td, GPTKIT_CONFIG=td),
        }
        res = {k: (p.wait(timeout=120), *p.communicate()) for k, p in cases.items()}
    for name in ("mockup", "critique", "svg", "research"):
        rc, out, err = res[name]
        lines = [l for l in err.splitlines() if l.strip()]
        assert rc == 1 and "Traceback" not in err, (name, rc, err)
        assert len(lines) == 1 and lines[0].startswith(f"gptkit {name}: HTTP 400 /") and "moderation_blocked" in lines[0], (name, err)
        assert "prompt:" in lines[0], (name, lines[0])
    rc, out, err = res["forge"]
    assert "Traceback" not in err and "Traceback" not in out, err
    assert "moderation_blocked" in out + err, (out, err)
    rc, out, err = res["keyerror"]
    assert rc == 1 and err.strip() == "gptkit doctor: missing key 'personas' (older JSON file or non-conforming model response?)", err
    assert res["interrupt"][0] == 130
    rc, out, err = res["debug"]
    assert rc == 1 and "Traceback" in err and "ApiError" in err, err
    rc, out, err = res["cfgdir"]
    assert rc == 1 and "Traceback" not in err and "did you mean" in err and err.count("\n") == 1, err


if __name__ == "__main__":
    import time
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            t = time.time()
            fn()
            print(f"ok {k} ({time.time() - t:.1f}s)")


def test_image_model_routing():
    saved = K.requests.get
    ids = {"data": [{"id": i} for i in ("gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst")]}
    try:
        K.requests.get = lambda url, headers=None, timeout=None: type("G", (), {"raise_for_status": lambda s: None, "json": lambda s: ids})()
        K._listed_models.cache_clear()
        with env(GPTKIT_API_KEY="sk-test", GPTKIT_CONFIG="/nonexistent/env", GPTKIT_IMAGE_MODEL=None):
            K._config.cache_clear()
            assert K.pick_image_model(task="concept") == "gpt-image-2.5-flare"
            assert K.pick_image_model(task="precise", background="transparent") == "gpt-image-2.5-sunburst"
            assert K.pick_image_model("mine", "precise") == "mine"
            assert K.image_chain("precise", "transparent")[-1] == "gpt-image-2"
            ids["data"] = [{"id": "gpt-image-2"}, {"id": "gpt-image-1.5"}]
            K._listed_models.cache_clear()
            assert K.pick_image_model(task="precise") == "gpt-image-2"
            assert K.pick_image_model(task="precise", background="transparent") == "gpt-image-1.5"
        with env(GPTKIT_API_KEY="sk-test", GPTKIT_CONFIG="/nonexistent/env", GPTKIT_IMAGE_MODEL="pinned"):
            K._config.cache_clear()
            assert K.pick_image_model(task="precise") == "pinned"
    finally:
        K.requests.get = saved
        K._listed_models.cache_clear()
        K._config.cache_clear()
