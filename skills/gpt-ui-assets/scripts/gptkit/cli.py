"""gptkit: design-to-assets pipeline on any OpenAI-compatible API or the local Codex CLI: UI mockups for phone, tablet,
foldable, desktop and web, asset inventory, 1:1 redraws, SVG icons, Android VectorDrawables, animation frames, Lottie
and app launcher icon sets. `setup` picks a backend; `doctor` checks the setup; `status <dir>` shows a project.

Config (environment variables win over the config file ~/.config/gptkit/env, KEY=value lines; `setup` writes it):
  GPTKIT_BACKEND                      api | cli | auto (default): the API with a key, else the Codex CLI when signed in
  GPTKIT_BASE_URL / OPENAI_BASE_URL   default https://api.openai.com/v1 (or a gateway that mirrors /images/* multipart
                                      and /responses with reasoning, web_search and strict json_schema)
  GPTKIT_API_KEY  / OPENAI_API_KEY    the API key (not needed with the Codex CLI backend)
  GPTKIT_IMAGE_MODEL                  pin an image model (default: routed per task from the endpoint's model list)
  GPTKIT_TEXT_MODEL                   pin a text model (default: routed from the endpoint's model list)
  GPTKIT_QUALITY                      low | medium | high (default high): image quality; low for cheap drafts
  GPTKIT_HOUSE_STYLE                  default art direction added to every new design; "off" drops it
  GPTKIT_BUDGET_USD                   refuse a command whose plan estimate exceeds this
  GPTKIT_LEDGER                       cost ledger path, default ~/.cache/gptkit/ledger.jsonl
  GPTKIT_CONFIG                       config file path, default ~/.config/gptkit/env
  GPTKIT_RETRIES                      retries per API call on 408/429/5xx and transport errors, default 2
  GPTKIT_DEBUG                        1: re-raise errors with the traceback instead of the one-line "gptkit <cmd>: ..."
  GPTKIT_MODERATION                   low | auto: `moderation` for image calls (dropped when a gateway rejects it)
  GPTKIT_CODEX                        Codex CLI binary for the cli backend, default codex (none disables it)
  GPTKIT_CHROME                       Chrome/Chromium binary for SVG/Lottie/CSS rendering (none disables)
  GPTKIT_CHROME_NO_SANDBOX            1: pass --no-sandbox to headless Chrome (added automatically when running as root)
  GPTKIT_EBSYNTH                      ebsynth binary for `sequence propagate`
  GPTKIT_RIFE                         rife-ncnn-vulkan binary for `--tween-engine rife`
"""
import argparse
import os
import re
import sys
from pathlib import Path

from .core import ApiError, _TRUE, _config, _pos_float, _pos_int, _unit_float
from .prompts import MOTION_STRENGTH, PLATFORMS
from .design import PHONE_SIZE, cmd_critique, cmd_manifest, cmd_mockup, cmd_spec
from .forge import cmd_fix, cmd_forge
from .vectordrawable import cmd_vd
from .svg_icons import cmd_icons, cmd_iconspec, cmd_svg
from .tween import TWEEN_ENGINES
from .frames import MOTIONS, cmd_frames
from .sequence import cmd_sequence
from .previews import cmd_vectorize
from .export import cmd_export
from .appicon import cmd_appicon_build, cmd_appicon_concepts, cmd_pick
from .trace_rig import _add_trace_rig_parsers
from .hybrid import _design_to_code_parsers, _hybrid_parsers
from .doctor import _color_args, cmd_doctor, cmd_score
from .research import DEPTHS, RESEARCH_MODEL, cmd_blueprint, cmd_research
from .store import _store_parsers
from .cost_status import cmd_cost, cmd_status

def _color_arg(s):
    """argparse type: a colour PIL and CSS both read (#RGB, #RRGGBB, #RRGGBBAA or a CSS colour name)."""
    from PIL import ImageColor
    try:
        ImageColor.getrgb(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a colour such as #1b1d33 or white, got {s!r}")
    if not re.fullmatch(r"#[0-9a-fA-F]{3,8}|[a-zA-Z]+", s):
        raise argparse.ArgumentTypeError(f"expected a colour such as #1b1d33 or white, got {s!r}")
    return s


def _research_parsers(sub):
    """Add research parsers to the command line."""
    efforts = ["high", "xhigh", "max"]
    p = sub.add_parser("cost", help="summarise API usage and estimated list-price cost")
    p.add_argument("--since", help="7d, 24h or YYYY-MM-DD")
    p.add_argument("--by", choices=["cmd", "model", "day", "dir"], default="cmd")
    p.add_argument("--json", action="store_true")
    p.add_argument("--ledger", help="ledger JSONL path (default GPTKIT_LEDGER or ~/.cache/gptkit/ledger.jsonl)")
    p.set_defaults(fn=cmd_cost)
    p = sub.add_parser("research", help="in-depth market research with web search (gpt-6-sol, high effort) -> report.md")
    p.add_argument("idea", help='e.g. "Android budgeting app for young couples"')
    p.add_argument("--out", required=True, help="output dir: report.md, report.json, plan.json, findings/, sources.json")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform (also carried into blueprint)")
    p.add_argument("--region", default="global", help='market/region to study, e.g. "US + UK" (default global)')
    p.add_argument("--context", help="extra constraints or known facts")
    p.add_argument("--depth", choices=list(DEPTHS), default="standard", help="5 / 10 / 16 research questions")
    p.add_argument("--model", default=RESEARCH_MODEL, help=f"reasoning model with web search (default {RESEARCH_MODEL})")
    p.add_argument("--effort", choices=efforts, default="high", help="reasoning effort for the plan and each question")
    p.add_argument("--synth-effort", choices=efforts, default="xhigh", help="effort for the final synthesis")
    p.add_argument("--jobs", type=_pos_int, default=4, help="parallel research questions"); p.set_defaults(fn=cmd_research)
    p.add_argument("--dry-run", action="store_true", help="show the planned research calls and estimated cost only")
    p = sub.add_parser("blueprint", help="feature blueprint (features, RICE, stories, screens with mockup briefs) from research")
    p.add_argument("dir", help="research dir with report.json (or an empty dir with --idea)")
    p.add_argument("--idea", help="product idea when DIR has no report.json (first-principles blueprint)")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform for native components (default: the report's)")
    p.add_argument("--constraints", help="constraints the blueprint must respect (team, timeline, tech, budget)")
    p.add_argument("--out", help="output dir (default DIR): blueprint.md, blueprint.json, screens.json")
    p.add_argument("--model", default=RESEARCH_MODEL, help=f"reasoning model (default {RESEARCH_MODEL})")
    p.add_argument("--effort", choices=efforts, default="xhigh", help="reasoning effort"); p.set_defaults(fn=cmd_blueprint)


def _design_parsers(sub):
    """Add design parsers to the command line."""
    p = sub.add_parser("doctor", help="check config, API access, models and optional tools (Chrome probe, aapt2 level)")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("status", help="offline project progress, recorded problems and next commands")
    p.add_argument("dir", nargs="?", help="project root (default: current directory)")
    p.add_argument("--json", action="store_true", help="print a machine-readable status object")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("mockup", help="generate UI mockup variants (or --edit an existing one)")
    p.add_argument("prompt"); p.add_argument("--out", required=True); p.add_argument("--name", default="mockup")
    p.add_argument("--size", help=f"WIDTHxHEIGHT canvas (default depends on --platform/--device; phone: {PHONE_SIZE}; "
                   "desktop: 2304x1440; --edit keeps the edited screen's canvas). Every render is checked against it and "
                   "fitted locally when it comes back off")
    p.add_argument("--variants", type=_pos_int, default=1, help="images generated in parallel")
    p.add_argument("--directions", action="store_true", help="give each variant its own art direction (palette, type, shape, signature) drawn from a shortlist of style families and one accent hue per variant; one extra text call. New screens only: not with --edit, --style-ref, --anchor or --raw")
    p.add_argument("--no-critique", action="store_true", help="skip the automatic design QA of each variant")
    p.add_argument("--no-house-style", action="store_true",
                   help="leave out the house style (clean, decluttered, modern; set GPTKIT_HOUSE_STYLE to change it)")
    p.add_argument("--edit", help="existing mockup PNG to revise with the prompt"); p.add_argument("--raw", action="store_true")
    p.add_argument("--style-ref", help="approved screen PNG: design a NEW screen in its exact design language")
    p.add_argument("--platform", choices=list(PLATFORMS), help="target platform: native components, system UI, metrics")
    p.add_argument("--device", choices=["phone", "tablet", "foldable", "desktop"], help="target form factor (default: phone for Android/iOS; desktop for web/desktop OSes)")
    p.add_argument("--orientation", choices=["portrait", "landscape"], help="tablet orientation (default: landscape)")
    p.add_argument("--posture", choices=["flat", "tabletop", "folded"], help="foldable posture (default: flat)")
    p.add_argument("--page", choices=["app", "landing"], help="web page type (default: app)")
    p.add_argument("--set", dest="set", help="comma-separated form factors to render from an approved anchor")
    p.add_argument("--anchor", help="approved screen PNG used as the style reference for --set")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_mockup)
    p = sub.add_parser("critique", help="vision design QA of a mockup; --apply edits the fixes in, kept only if better")
    p.add_argument("mockup"); p.add_argument("--out", help="output dir (default: the mockup's dir)")
    p.add_argument("--style-ref", help="approved screen of the same app: also check consistency with it")
    p.add_argument("--platform", choices=list(PLATFORMS), help="default: design.json next to the mockup, else android")
    p.add_argument("--screen-dp", type=_pos_float, help="logical screen width (default: the platform's)")
    p.add_argument("--note", help="extra focus for the review")
    p.add_argument("--apply", action="store_true", help="edit the issues' fixes into the mockup, verify, keep if better")
    p.add_argument("--min-severity", choices=["high", "medium", "low"], default="medium")
    p.add_argument("--rounds", type=_pos_int, default=1, help="max --apply rounds (each kept only if the weight drops)")
    p.add_argument("--fit", action="store_true", help="phone screens: which single-line text runs out of room on a small "
                   "phone or at large text sizes (one extra vision call; measured from the pixels)")
    p.add_argument("--model", default=None); p.set_defaults(fn=cmd_critique)
    p = sub.add_parser("manifest", help="inventory assets of an approved mockup -> DIR/assets.json")
    p.add_argument("mockup"); p.add_argument("--out", required=True)
    p.add_argument("--platform", choices=list(PLATFORMS), help="default: design.json written by mockup")
    p.add_argument("--screen-dp", type=_pos_float, help="logical screen width (default: the platform's, else 390)")
    p.add_argument("--model", default=None, help="text/vision model (default GPTKIT_TEXT_MODEL)")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("spec", help="extract palette/typography/spacing/native widgets -> DIR/design_spec.json")
    p.add_argument("dir")
    p.add_argument("--scale", choices=["auto", "native", "design"], default="auto",
                   help="px->dp from native widget sizes (auto: when GPT drew them >12%% off) or the screen width")
    p.add_argument("--recalibrate", action="store_true", help="only redo the px->dp scale of an existing spec (no API)")
    p.add_argument("--model", default=None, help="text/vision model (default GPTKIT_TEXT_MODEL)")
    p.set_defaults(fn=cmd_spec)


def _asset_parsers(sub):
    """Add asset parsers to the command line."""
    p = sub.add_parser("forge", help="redraw png assets from scratch, 1:1, with candidates + scoring")
    p.add_argument("dir", help="project dir with assets.json (from manifest)")
    p.add_argument("--only", help="comma-separated asset names (default: every png-format asset without a final file)")
    p.add_argument("--candidates", type=_pos_int, default=2, help="redraws per asset, best one wins (default 2)")
    p.add_argument("--jobs", type=_pos_int, default=6, help="parallel image calls (default 6)")
    p.add_argument("--dry-run", action="store_true", help="show pending candidate calls and estimated cost only")
    p.add_argument("--force", action="store_true", help="redo assets that already have a final file")
    p.add_argument("--include-svg", action="store_true", help="also raster-forge svg-format assets")
    p.add_argument("--same-prompt", action="store_true", help="no per-candidate colour/shape emphasis")
    _color_args(p); p.add_argument("--model", default=None, help="image model (default GPTKIT_IMAGE_MODEL)"); p.set_defaults(fn=cmd_forge)
    p = sub.add_parser("fix", help="vision-diff the chosen asset against the design and apply one correction")
    p.add_argument("dir"); p.add_argument("name", help="asset name from assets.json")
    p.add_argument("--rounds", type=_pos_int, default=1, help="diff+edit rounds (default 1); stops early when nothing differs")
    p.add_argument("--note", help="extra user feedback to include"); p.add_argument("--accept", action="store_true",
                                                                                   help="keep the result even if the score is worse")
    _color_args(p); p.add_argument("--model", default=None, help="image model (default GPTKIT_IMAGE_MODEL)"); p.set_defaults(fn=cmd_fix)
    p = sub.add_parser("svg", help="write svg-format assets as SVG: scored vs the design, retried, 24-grid, previews")
    p.add_argument("dir"); p.add_argument("--only")
    p.add_argument("--no-retry", dest="retry", action="store_false", help="skip the vision-guided retry of weak icons")
    p.add_argument("--grid", type=float, default=24, help="normalise icon viewBoxes to this grid (0 = keep as drawn)")
    p.add_argument("--pad", type=float, default=2, help="grid padding around the icon's content")
    p.add_argument("--check", action="store_true", help="only score existing DIR/svg files (no API calls)")
    p.set_defaults(fn=cmd_svg)
    p = sub.add_parser("iconspec", help="extract the icon-set style spec + Material/SF symbol names -> DIR/icon_spec.json")
    p.add_argument("dir"); p.add_argument("--model", default=None, help="text/vision model (default GPTKIT_TEXT_MODEL)")
    p.set_defaults(fn=cmd_iconspec)
    p = sub.add_parser("icons", help="draw NEW icons (not in the mockup) in the design's icon style as SVG")
    p.add_argument("dir"); p.add_argument("concepts", help='comma-separated, e.g. "notifications, sleep timer, download"')
    p.add_argument("--out", help="default DIR/icons")
    p.add_argument("--treatment", choices=["auto", "outlined", "filled"], default="auto")
    p.add_argument("--stroke", type=_pos_float, help="stroke width on the 24 grid (default: measured from the set)")
    p.add_argument("--color", help="hex colour (default: the set's most common)")
    p.add_argument("--snap", action="store_true", help="force every stroke-width (attribute, style or inherited default) to the set's stroke")
    p.add_argument("--vd", help="also write VectorDrawables to this res/drawable dir")
    p.add_argument("--dp", type=_pos_float, help="VectorDrawable size in dp for --vd (default: the SVG's width)")
    p.set_defaults(fn=cmd_icons)
    p = sub.add_parser("vd", help="convert SVG files (or directories of them) to Android VectorDrawable XML")
    p.add_argument("svgs", nargs="+", help="SVG files or directories (*.svg)")
    p.add_argument("--out", required=True, help="output dir, e.g. app/src/main/res/drawable")
    p.add_argument("--dp", type=_pos_float, help="android:width in dp (default: the SVG's width, or svg_report.json's per-icon dp)")
    p.add_argument("--prefix", default="ic_", help="resource name prefix (names are lower-cased to [a-z0-9_])")
    p.add_argument("--no-check", action="store_true", help="skip the render round-trip and aapt2 validation")
    p.set_defaults(fn=cmd_vd)


def _tween_args(p, scope="", remembered=False):
    p.add_argument("--tween", type=_pos_int, default=None if remembered else 1,
                   help=f"{scope}N-1 in-between frames after each frame (N x the frames and fps, same loop length; "
                        f"default 1 = none{', remembered' if remembered else ''})")
    p.add_argument("--tween-engine", choices=TWEEN_ENGINES, default=None if remembered else "flow",
                   help="flow (default): built-in optical flow (needs OpenCV); rife: rife-ncnn-vulkan; blend: crossfade")
    p.add_argument("--rife", help="rife-ncnn-vulkan binary (else $GPTKIT_RIFE, PATH, ~/.cache/gptkit/rife)")


def _animation_parsers(sub):
    """Add animation parsers to the command line."""
    p = sub.add_parser("frames", help="frame-by-frame animation from a forged asset (moving layer sprite sheet, "
                                      "or --procedural local motion)")
    p.add_argument("dir"); p.add_argument("name"); p.add_argument("--moving", help="part that moves (generated route)")
    p.add_argument("--static", help="part that stays still (generated once, composited)")
    p.add_argument("--static-order", choices=["front", "back"], default="front")
    p.add_argument("--motion"); p.add_argument("--count", type=_pos_int, help="frames, at least 2 (default 8; procedural: one loop)")
    p.add_argument("--cols", type=_pos_int, default=4, help="sheet columns (clamped to --count)")
    p.add_argument("--anchor", choices=["bottom", "center"], default="bottom")
    p.add_argument("--fps", type=_pos_float, help="default 12 (procedural 24, or 12 for slow loops)")
    p.add_argument("--procedural", choices=sorted(MOTIONS), help="no API calls: eased affine motion of the asset")
    p.add_argument("--amount", type=_pos_float, default=1.0, help="procedural motion strength multiplier")
    p.add_argument("--reuse", choices=["all", "static"], help="reuse sheet_raw/static_raw.png from a previous run")
    p.add_argument("--reroll", action="store_true", help="redraw QA-flagged frames from a good neighbour (1 call each)")
    p.add_argument("--match-colors", action="store_true", help="shift frame colours onto the approved asset (Lab)")
    _tween_args(p, "generated route: ")
    p.add_argument("--formats", default="gif,webp,apng,lottie", help="comma list of gif,webp,apng,lottie")
    p.add_argument("--px", type=_pos_int, default=480, help="long side of the animated WebP/APNG/Lottie/GIF and the "
                                                            "android/drawable-nodpi frames")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="backdrop of preview.gif and preview.html")
    p.add_argument("--model", default=None, help="image model (default $GPTKIT_IMAGE_MODEL)"); p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("sequence", help="layout-driven animation: code lays the motion out, GPT renders frames "
                                        "(stages: layout, plan, render, build | propagate, reroll)")
    p.add_argument("dir"); p.add_argument("name")
    p.add_argument("stage", choices=["layout", "plan", "render", "build", "propagate", "reroll"])
    p.add_argument("--guides", help="layout: use these guide frames (one PNG per frame, square, one size, transparent, "
                                    "naturally ordered 0.png..N.png) instead of a plume")
    p.add_argument("--base", help="layout: the locked artwork on its own canvas; build cuts it away")
    p.add_argument("--sprites", help="layout: comma list of the plume's element sprites, drawn in the art's own style")
    p.add_argument("--vent", help="layout: emission point x,y on the base canvas")
    p.add_argument("--loop", type=_pos_float, default=3.0, help="layout: loop length in seconds")
    p.add_argument("--fps", type=_pos_float, default=24)
    p.add_argument("--chuff", type=_pos_float, default=0.75, help="layout: seconds between emissions (the loop must be a multiple)")
    p.add_argument("--life", type=_pos_float, default=5.4, help="layout: seconds each element lives")
    p.add_argument("--velocity", default="0,-85", help="layout: launch velocity x,y in canvas units per second")
    p.add_argument("--spread", default="12,18", help="layout: launch jitter x,y")
    p.add_argument("--drag", type=_pos_float, default=1.5, help="layout: air drag per second toward --wind")
    p.add_argument("--wind", default="45,-3", help="layout: air velocity x,y (a moving engine's slipstream, a breeze); "
                                                  "a negative x needs the = form: --wind=-40,-3")
    p.add_argument("--buoyancy", type=float, default=-10, help="layout: vertical acceleration; negative rises")
    p.add_argument("--size", type=_pos_float, default=96, help="layout: sprite width at scale 1, canvas units")
    p.add_argument("--grow", default="0.45,1.5", help="layout: scale at birth,death")
    p.add_argument("--front", action="store_true", help="layout: draw the plume in front of the base")
    p.add_argument("--moving", default="The moving element", help="render: what the layout shows, e.g. 'The smoke'")
    p.add_argument("--locked", default="The rest of the artwork", help="render: what must not change, e.g. 'The toy steam train'")
    p.add_argument("--look", help="render: the finish to aim for (default: the same finish as the locked artwork)")
    p.add_argument("--jobs", type=_pos_int, default=12, help="render/reroll: parallel image calls; propagate: parallel EbSynth runs")
    p.add_argument("--force", action="store_true", help="render: redo frames that are already rendered; propagate: redo "
                                                        "finished syntheses")
    p.add_argument("--every", type=_pos_int, help="render/propagate: GPT renders every Nth frame only (the keys) and "
                                                  "propagate synthesises the rest with EbSynth (default: the plan's; "
                                                  "N must divide the frame count)")
    p.add_argument("--engine", choices=["auto", "ebsynth", "gpt"], default=None,
                   help="plan: auto picks EbSynth keys when they can carry the layout, else GPT renders every frame; "
                        "remembered in sequence.json for the later stages")
    p.add_argument("--ebsynth", help="propagate: path to the ebsynth binary (else $GPTKIT_EBSYNTH, PATH, ~/.cache/gptkit)")
    p.add_argument("--ebsynth-timeout", type=_pos_float, help="propagate: seconds per EbSynth run before it counts as "
                                                              "hung (default 600)")
    p.add_argument("--keep-work", action="store_true", help="propagate: keep propagate/syn_* and pos_* after a successful "
                                                            "run (a later run then makes no EbSynth runs)")
    p.add_argument("--guide-weight", type=_pos_float, default=2.0, help="propagate: weight of the appearance guide")
    p.add_argument("--position-weight", type=_pos_float, default=1.5, help="propagate: weight of the positional guide (plumes)")
    p.add_argument("--ref", help="build/propagate: approved picture of the moving part; each frame (or key) is "
                                 "colour-matched to it (remembered for later builds)")
    p.add_argument("--below", type=_pos_float, help="build/propagate: drop anything below this canvas row (under a chimney "
                                                    "mouth; remembered for later builds)")
    p.add_argument("--quality", type=_pos_int, help="build: WebP quality 1-100 (default 80, remembered)")
    _tween_args(p, "build/propagate: ", remembered=True)
    p.add_argument("--frames", help="reroll: comma list of frames (default: the outliers build reported)")
    p.add_argument("--candidates", type=_pos_int, default=2, help="reroll: renders per frame to choose from")
    p.add_argument("--preview-bg", type=_color_arg, help="build: backdrop of preview.gif and contact.png (default #a0aabe, remembered)")
    p.add_argument("--model", default=None, help="render/reroll: image model (default $GPTKIT_IMAGE_MODEL)")
    p.add_argument("--dry-run", action="store_true", help="render: list pending frames and estimated cost only")
    p.set_defaults(fn=cmd_sequence)
    p = sub.add_parser("vectorize", help="vector layers (morphs, transforms, glows) -> Lottie + AnimatedVectorDrawable")
    p.add_argument("dir", help="design dir (assets.json) or any dir for the outputs with --svg / --rebuild")
    p.add_argument("name", help="asset name: DIR/assets/NAME.png or DIR/svg/NAME.svg; outputs in DIR/vector/NAME/")
    p.add_argument("--moving", help="what moves, e.g. 'the flame and its glow' (required unless --rebuild)")
    p.add_argument("--static", help="what stays still, e.g. 'the logs'")
    p.add_argument("--motion", help="how it moves, e.g. 'gentle flicker, a slow pulse'")
    p.add_argument("--keyframes", type=_pos_int, default=4, help="morph keyframes per animated layer (>= 2)")
    p.add_argument("--size", type=_pos_int, default=512, help="Lottie canvas long side in px")
    p.add_argument("--seconds", type=_pos_float, default=1.2, help="loop length (AVD loops longer than 5 s are subsampled)")
    p.add_argument("--fps", type=_pos_int, default=30, help="Lottie frame rate")
    p.add_argument("--strength", choices=list(MOTION_STRENGTH), default="normal", help="how much the moving parts move")
    p.add_argument("--min-motion", type=_unit_float, help="retry once below this share of a layer's size (default per --strength)")
    p.add_argument("--svg", help="animate this SVG (default DIR/svg/NAME.svg when DIR/assets/NAME.png is missing); no assets.json needed")
    p.add_argument("--dp", type=_pos_float, help="AnimatedVectorDrawable width in dp (default: the asset's dp, else 48)")
    p.add_argument("--rebuild", action="store_true", help="rebuild outputs from an edited layers.json, no API call")
    p.add_argument("--no-fit", action="store_true", help="keep the viewBox even if the motion leaves it (clips)")
    p.add_argument("--review-frames", type=_pos_int, default=8, help="frames in keys_review.png")
    p.add_argument("--preview-bg", type=_color_arg, default="#1b1d33", help="background colour of keys_review.png and preview.html")
    p.set_defaults(fn=cmd_vectorize)


def _export_parsers(sub):
    """Add image export parsers to the command line."""
    p = sub.add_parser("export", help="export a PNG to android densities / ios imageset / web")
    p.add_argument("file", help="source image (PNG with alpha; any size)"); p.add_argument("--dp", type=_pos_float, required=True, help="width in dp/pt/css px")
    p.add_argument("--hdp", type=_pos_float, help="height in dp/pt/css px (default: keeps the aspect ratio)")
    p.add_argument("--name", help="asset name (default: the file stem; sanitised to [a-z0-9_] for Android)")
    p.add_argument("--res", required=True, help="android: res/ dir; ios: *.xcassets dir; web: output dir")
    p.add_argument("--platform", choices=["android", "ios", "web"], default="android")
    p.add_argument("--format", choices=["webp", "png"], help="android/web file format (default webp; ios is always png)")
    p.add_argument("--quality", type=int, help="WebP quality 0-100; 100 = lossless (default: lossless for <= 256 colours, else 90)")
    p.set_defaults(fn=cmd_export)


def _appicon_parsers(sub):
    """Add app icon parsers to the command line."""
    p = sub.add_parser("appicon", help="app launcher icons: concepts, then Android adaptive / iOS / web icon sets")
    isub = p.add_subparsers(dest="icmd", required=True)
    q = isub.add_parser("concepts", help="square full-bleed icon concepts + mask review sheet")
    q.add_argument("brief", help='brand brief, e.g. "Hush, a calm sleep app, moon motif"'); q.add_argument("--out", required=True)
    q.add_argument("--mockup", help="approved design to take the visual language from (sent as the single reference)")
    q.add_argument("--crop", help="x,y,w,h region of --mockup to use as the reference instead of the whole screen")
    q.add_argument("--variants", type=_pos_int, default=3, help="number of concepts (1 image call each)")
    q.add_argument("--directions", help="'|'-separated concept directions")
    q.add_argument("--text", help="wordmark/letter to include (default: no text)"); q.add_argument("--colors")
    q.add_argument("--name", default="concept", help="file stem of the concepts: <name>_v1.png, ...")
    q.add_argument("--jobs", type=_pos_int, default=2, help="concurrent image calls")
    q.add_argument("--fidelity", choices=["high", "low"], default="low"); q.add_argument("--model", default=None)
    q.set_defaults(fn=cmd_appicon_concepts)
    q = isub.add_parser("build", help="Android adaptive (fg/bg/monochrome) + legacy + Play, iOS AppIcon, web icons")
    q.add_argument("concept", help="chosen square concept PNG"); q.add_argument("--out", required=True)
    q.add_argument("--subject", default="the main subject", help='what the foreground is, e.g. "the crescent moon and stars"')
    q.add_argument("--separate", choices=["auto", "local", "api"], default="auto",
                   help="foreground: local matte against the shipped background (exact, 0 calls), an image-edit redraw "
                        "(1 call), or auto: local unless it leaves the subject mostly semi-transparent")
    q.add_argument("--bg", choices=["auto", "color", "gradient", "model", "image"], default="auto",
                   help="background layer: colour resource, VectorDrawable gradient, fitted bitmap, or an image-edit "
                        "plate (1 call; auto picks it only for busy backgrounds)")
    q.add_argument("--bg-color", help="hex for --bg color (default: sampled)")
    q.add_argument("--mono", choices=["glyph", "silhouette", "both", "none"], default="glyph",
                   help="themed-icon layer: gpt-6 glyph -> VectorDrawable (1 text call, falls back to the silhouette), "
                        "the foreground silhouette bitmap, or both (glyph referenced)")
    q.add_argument("--fg-scale", type=_pos_float, default=1.0, help="fraction of the 66dp safe circle the subject may fill")
    q.add_argument("--ios-from", choices=["concept", "composite"], default="concept")
    q.add_argument("--name", default="ic_launcher", help="Android resource name ([a-z0-9_]): <name>, <name>_round, ...")
    q.add_argument("--no-validate", dest="validate", action="store_false", help="skip the aapt2 / actool checks")
    q.add_argument("--model", default=None); q.set_defaults(fn=cmd_appicon_build)
    p = sub.add_parser("pick", help="override the chosen candidate (the cN number from the review-sheet label, or an image file)")
    p.add_argument("dir"); p.add_argument("name", help="asset name from assets.json")
    p.add_argument("candidate", help="N of the tile labelled cN in review/forge_*.png, or a path to any PNG")
    _color_args(p); p.set_defaults(fn=cmd_pick)


def _integration_parsers(sub, backend_parsers):
    """Add design-to-code, hybrid, backend, score, trace and rig parsers."""
    _design_to_code_parsers(sub)
    _hybrid_parsers(sub)
    backend_parsers(sub)
    p = sub.add_parser("score", help="score arbitrary PNG or SVG files against a manifest asset (JSON per file, no API)")
    p.add_argument("dir"); p.add_argument("name", help="asset name from assets.json"); p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_score)
    _add_trace_rig_parsers(sub)


def build_parser():
    """Every subcommand and flag, as (parser, subparsers); main() parses with it."""
    from .codex_cli import _backend_parsers  # deferred: codex_cli imports this module
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _research_parsers(sub)
    _design_parsers(sub)
    _asset_parsers(sub)
    _animation_parsers(sub)
    _export_parsers(sub)
    _appicon_parsers(sub)
    _integration_parsers(sub, _backend_parsers)
    _store_parsers(sub)
    return ap, sub

def main(standalone=False, argv=None):
    """The CLI. `standalone` is True when run as a script (gptkit.py), False when called from Python."""
    ap, _ = build_parser()
    a = ap.parse_args(argv)
    from . import core  # the ledger reads these from core's globals
    core._LEDGER_CMD = ("sequence " + a.stage) if a.cmd == "sequence" else (
        "appicon " + a.icmd if a.cmd == "appicon" else "store " + a.scmd if a.cmd == "store" else a.cmd)
    dir_arg = getattr(a, "dir", None) or getattr(a, "out", None)
    core._LEDGER_DIR = str(Path(dir_arg).expanduser().resolve()) if dir_arg else None
    try:
        a.fn(a)
    except KeyboardInterrupt:
        if standalone:  # the CLI: pool threads still inside a 900 s API call would hold the exit until they return
            try:  # the interrupt may have landed inside a print: never let a flush error skip the exit
                sys.stdout.flush()
                sys.stderr.flush()
            except BaseException:
                pass
            finally:
                os.write(2, f"gptkit {a.cmd}: interrupted\n".encode())
                os._exit(130)
        print(f"gptkit {a.cmd}: interrupted", file=sys.stderr)
        sys.exit(130)
    except (ApiError, RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError,
            ArithmeticError, MemoryError) as e:
        debug = os.environ.get("GPTKIT_DEBUG", "").lower() in _TRUE
        try:
            debug = debug or _config()["debug"]
        except SystemExit:
            pass
        if debug:
            raise
        msg = (f"missing key {e.args[0]!r} (older JSON file or non-conforming model response?)" if isinstance(e, KeyError)
               else " ".join(str(e).split()) or type(e).__name__)
        if isinstance(e, (FileExistsError, NotADirectoryError)):
            msg += " (a file is where the kit needs a directory: check --out/--res and the project dir)"
        if isinstance(e, MemoryError):
            msg = "out of memory (an input image, or a size/--px/--dp flag, is far larger than the kit needs)"
        elif not isinstance(e, (ApiError, RuntimeError, OSError, ValueError, KeyError)):  # a bug or a JSON file of the wrong shape
            msg = (f"unexpected {type(e).__name__}: {msg} (a hand-edited or older JSON file in the project? re-create it; "
                   "else re-run with GPTKIT_DEBUG=1 and report the traceback)")
        sys.exit(f"gptkit {a.cmd}: {msg}")
