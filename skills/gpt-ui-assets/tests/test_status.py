"""Offline project status checks: python3 tests/test_status.py."""
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gptkit as g  # noqa: E402


class TempDir:
    """Temporary fixture path."""
    def __enter__(self):
        return tempfile.mkdtemp()

    def __exit__(self, *exc):
        return False


def call(*args):
    return run_cli("status", *args)


def run_cli(*args):
    old = sys.argv
    out = io.StringIO()
    sys.argv = ["gptkit.py", *map(str, args)]
    try:
        with contextlib.redirect_stdout(out):
            try:
                g.main()
            except SystemExit as e:
                assert e.code in (None, 0), out.getvalue()
    finally:
        sys.argv = old
    return out.getvalue()


def put(path, data="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    return path


def unit(doc, kind):
    return next(x for x in doc["units"] if x["type"] == kind)


def test_manifest_to_spec_next():
    with TempDir() as td:
        d = Path(td) / "build" / "home"
        put(d / "assets.json", '{"assets": []}')
        put(d / "manifest_overlay.png", "fake png")
        text = call(d)
        assert f"next: python3 $K spec {d}" in text


def test_stale_spec():
    with TempDir() as td:
        d = Path(td)
        spec = put(d / "design_spec.json")
        time.sleep(0.02)
        put(d / "assets.json", '{"assets": []}')
        doc = json.loads(call(d, "--json"))
        b = unit(doc, "build")
        assert any(c["name"] == "design_spec.json" and c["status"] == "STALE" for c in b["checks"])
        assert b["next"] == [f"python3 $K spec {d}"]


def test_forge_attention():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        put(d / "design_spec.json")
        put(d / "forge_report.json", '{"a": {"candidates": [{"path": "review/a.png", "flags": ["OPAQUE"]}]}}')
        doc = json.loads(call(d, "--json"))
        b = unit(doc, "build")
        assert any("NEEDS ATTENTION: a" in p for p in b["problems"])
        assert b["next"] == [f"python3 $K forge {d} --only a --force"]


def test_corrupt_forge_report():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        put(d / "forge_report.json", "{")
        doc = json.loads(call(d, "--json"))
        assert any("forge_report.json: invalid JSON" in p for p in unit(doc, "build")["problems"])


def test_sequence_missing_frames():
    with TempDir() as td:
        d = Path(td) / "sequence" / "rain"
        put(d / "sequence.json", '{"frames": 3, "guide_sha1": ["a", "b", "c"]}')
        doc = json.loads(call(Path(td), "--json"))
        seq = unit(doc, "sequence")
        assert "frames missing: [0, 1, 2]" in seq["problems"]
        assert seq["next"] == [f"python3 $K sequence {d} <name> render"]


def test_empty_dir():
    with TempDir() as td:
        out = call(td)
        assert f"no gptkit artifacts under {Path(td).resolve()}" in out
        assert "python3 $K mockup" in out


def test_json_shape():
    with TempDir() as td:
        d = Path(td)
        put(d / "assets.json", '{"assets": []}')
        doc = json.loads(call(d, "--json"))
        assert set(("dir", "units", "next")) <= set(doc)
        b = unit(doc, "build")
        assert set(("checks", "problems", "next")) <= set(b)


def test_real_offline_kit_output():
    """Run the real offline SVG -> VectorDrawable command, then inspect that project tree."""
    with TempDir() as td:
        d = Path(td) / "build" / "home"
        put(d / "assets.json", '{"assets": []}')
        put(d / "design_spec.json")
        svg = put(d / "svg" / "mark.svg",
                  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
                  '<path fill="#000000" d="M3 3h18v18H3z"/></svg>')
        run_cli("vd", svg, "--out", d / "platform" / "drawable", "--no-check")
        assert list((d / "platform" / "drawable").glob("*.xml"))
        out = call(d)
        assert f"build {d}" in out and "design_spec.json" in out


def run():
    for fn in (test_manifest_to_spec_next, test_stale_spec, test_forge_attention,
               test_corrupt_forge_report, test_sequence_missing_frames, test_empty_dir, test_json_shape,
               test_real_offline_kit_output):
        fn()
        print(f"ok {fn.__name__}")


if __name__ == "__main__":
    run()
