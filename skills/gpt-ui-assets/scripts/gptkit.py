#!/usr/bin/env python3
"""gptkit: design-to-assets pipeline on any OpenAI-compatible API — UI mockups, asset inventory, 1:1 from-scratch
redraws, SVG icons, Android VectorDrawables, animation frames, Lottie and app launcher icon sets. Run `gptkit.py -h`.

Config (environment variables win over the config file):
  GPTKIT_BASE_URL / OPENAI_BASE_URL   default https://api.openai.com/v1 (or a gateway that mirrors /images/* multipart
                                      and /responses with reasoning, web_search and strict json_schema, e.g. LiteLLM)
  GPTKIT_API_KEY  / OPENAI_API_KEY    required
  GPTKIT_IMAGE_MODEL                  default gpt-image-2
  GPTKIT_TEXT_MODEL                   default gpt-6 (vision + structured outputs; `doctor` lists the ids your endpoint
                                      offers, e.g. gpt-6-luna / gpt-6-sol / gpt-6.1-sol when the bare alias is not listed)
  GPTKIT_CONFIG                       config file path, default ~/.config/gptkit/env (KEY=value lines)
  GPTKIT_RETRIES                      retries per API call on 408/429/5xx and transport errors, default 2
  GPTKIT_DEBUG                        1: re-raise errors with the traceback instead of the one-line "gptkit <cmd>: ..."
  GPTKIT_MODERATION                   low | auto: `moderation` for image calls (dropped when a gateway rejects it)
  GPTKIT_CHROME                       Chrome/Chromium binary for SVG/Lottie/CSS rendering (none disables)
  GPTKIT_CHROME_NO_SANDBOX            1: pass --no-sandbox to headless Chrome (added automatically when running as root)
  GPTKIT_EBSYNTH                      ebsynth binary for `sequence propagate`
"""
import argparse
import base64
import concurrent.futures as cf
import functools
import hashlib
import http.server
import io
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from datetime import date, datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import numpy as np
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat

# The kit's code lives in gptkit_parts/, one file per area, executed here in name order into this module's own
# namespace. Splitting the file this way keeps a single module at runtime: every function still sees every other as
# a global, `import gptkit` exposes all of them, and patching gptkit.<name> (as the tests do) reaches every part.
_PARTS = Path(__file__).resolve().parent / "gptkit_parts"
for _part in sorted(_PARTS.glob("[0-9][0-9]_*.py")):
    exec(compile(_part.read_text(encoding="utf-8"), str(_part), "exec"), globals())
del _part


if __name__ == "__main__":
    main()
