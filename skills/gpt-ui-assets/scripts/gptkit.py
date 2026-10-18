#!/usr/bin/env python3
"""gptkit's command line: `python3 gptkit.py -h` lists the commands and the configuration variables.

The code is the `gptkit` package next to this file (scripts/gptkit/); this script only starts its CLI.
"""
import sys
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("gptkit needs Python 3.9 or newer")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gptkit import main  # noqa: E402

if __name__ == "__main__":
    main(standalone=True)
