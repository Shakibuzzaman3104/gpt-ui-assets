"""gptkit: design-to-assets pipeline on any OpenAI-compatible API or the Codex CLI.

The code is split by area into the modules below and loaded in this order. Each module imports what it uses from the
modules before it at the top, and from the few modules after it inside the functions that need them (those modules
import it back). Every name of every module is also re-exported here, so `import gptkit` gives the whole kit, and
assigning `gptkit.<name> = value` (as the tests do to stub the API) rebinds that name in every module that holds it.
"""
import sys
import types
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import requests  # noqa: F401
except ImportError as _e:  # pragma: no cover - exercised by hand
    sys.exit(f"gptkit: missing Python package {_e.name!r}. Install the requirements: "
             f"pip install -r {__file__.rsplit('/', 2)[0]}/requirements.txt")

from . import (  # noqa: E402  (order matters: each module imports from the ones before it)
    core,
    imaging,
    prompts,
    design,
    forge,
    vectordrawable,
    svg_icons,
    tween,
    frames,
    sequence,
    vectorize,
    previews,
    export,
    appicon,
    trace_rig,
    measure,
    fit,
    tokens,
    fonts,
    redline,
    compare,
    hybrid,
    doctor,
    research,
    store,
    scenes,
    cost_status,
    cli,
    codex_cli,
)

MODULES = (core, imaging, prompts, design, forge, vectordrawable, svg_icons, tween, frames, sequence, vectorize, previews, export, appicon, trace_rig, measure, fit, tokens, fonts, redline, compare, hybrid, doctor, research, store, scenes, cost_status, cli, codex_cli)

# Module state a function rebinds with `global` (warn-once flags, cached probe results). A copy here would go stale,
# so these are not re-exported: reading gptkit.<name> falls through to __getattr__ and reads the owning module.
STATE = {"_CHROME_FAILED_REASON": trace_rig, "_CHROME_PROBE_WARNED": tokens, "_CODEX_EDIT_WARNED": codex_cli,
         "_CODEX_MODEL_CACHE": codex_cli, "_CODEX_NOTICE": codex_cli, "_CODEX_READY": codex_cli,
         "_CODEX_REFUSED_MODEL": codex_cli, "_LAST_BATCH_ERROR": svg_icons, "_LEDGER_CMD": core, "_LEDGER_DIR": core,
         "_LEDGER_WARNED": core, "_LOTTIE_JS_REASON": previews, "_LOTTIE_REASON": previews, "_SHOT_REASON": previews,
         "_WOFF2_WHY": fonts, "_warned_chrome": forge, "_warned_render": forge, "_warned_shot": previews}

for _m in MODULES:
    for _k, _v in vars(_m).items():
        if not _k.startswith("__") and _k not in STATE:
            globals()[_k] = _v
del _m, _k, _v


def __getattr__(name):
    if name in STATE:
        return getattr(STATE[name], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _Kit(types.ModuleType):
    """The package module: setting an attribute also rebinds that name in every module that defines or imports it."""

    def __setattr__(self, name, value):
        if name in STATE:
            setattr(STATE[name], name, value)
            return
        super().__setattr__(name, value)
        for mod in MODULES:
            if name in vars(mod):
                vars(mod)[name] = value


sys.modules[__name__].__class__ = _Kit
