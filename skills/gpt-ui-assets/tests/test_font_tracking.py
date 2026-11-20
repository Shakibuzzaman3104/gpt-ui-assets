"""Offline checks for the font matcher's tracking bounds: python3 tests/test_font_tracking.py."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GPTKIT_CODEX", "none")
os.environ.setdefault("GPTKIT_LEDGER", "off")
from gptkit import fonts  # noqa: E402


def test_tracking_bounds_follow_guideline_ranges():
    assert fonts._tracking_ok(0.03, 14) and fonts._tracking_ok(-0.02, 57)
    assert not fonts._tracking_ok(0.3, 14)
    assert not fonts._tracking_ok(0.06, 40)
    assert not fonts._tracking_ok(-0.12, 22)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
