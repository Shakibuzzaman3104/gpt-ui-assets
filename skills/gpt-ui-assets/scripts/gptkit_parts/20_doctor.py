# gptkit part 20_doctor: doctor.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

def cmd_doctor(a):
    c = _config()
    api_ready, cli_ready, selected, reason = _backend_status()
    print(f"backend     : {selected} (auto: {reason})" if c["backend"] == "auto" else f"backend     : {selected}")
    print(f"  {'✓' if api_ready else '✗'} API backend: " + ("API key set" if api_ready else "set GPTKIT_API_KEY in ~/.config/gptkit/env (chmod 600) or run `setup`"))
    print(f"  {'✓' if cli_ready else '✗'} Local CLI: " + ("Codex installed and logged in" if cli_ready else "npm i -g @openai/codex && codex login"))
    print(f"config file : {c['config_path']} ({'found' if Path(c['config_path']).exists() else 'not found'})")
    print(f"base url    : {c['base']}")
    print(f"api key     : {'set' if c['key'] else 'MISSING'}")
    print(f"image model : {c['image_model']}" + (" (pinned)" if c["image_model_pinned"] else
          f" (default; auto-routed per task: concept {image_chain('concept')[0]}, precise {image_chain('precise')[0]}, first listed wins)"))
    text_model = c["text_model"] or pick_text_model()
    print(f"text model  : {text_model}")
    today = _today()
    for label, mid in (("image", c["image_model"]), ("text", text_model)):
        retirement = RETIREMENTS.get(mid)
        if retirement:
            days = (date.fromisoformat(retirement[0]) - date.fromisoformat(today)).days
            if days <= 60:
                status = "retired" if days < 0 else f"retires in {days} days"
                print(f"  ! {label} model {mid} {status} (shutdown {retirement[0]}; replacement: {retirement[1]})")
    print(f"retries     : {c['retries']}")
    print(f"debug       : {'on' if c['debug'] else 'off'}")
    renderer, exe = _renderer_name(), _chrome()
    if renderer == "chrome" or (renderer is None and exe):
        ok, why = _chrome_probe()
        hint = " (running as root needs --no-sandbox: set GPTKIT_CHROME_NO_SANDBOX=1)" if "sandbox" in why.lower() else ""
        line = "chrome (ok)" if ok else f"none: chrome found at {exe} but failed: {why}{hint}"
    elif renderer:
        line = renderer + (" (no alpha: matted from two renders)" if renderer == "qlmanage" else "")
    else:
        line = "none (install Chrome/Chromium and set GPTKIT_CHROME, or pip install cairosvg)"
    print(f"svg render  : {line}")
    sdk = _android_sdk()
    if sdk and sdk[1]:
        print(f"aapt2       : {sdk[0]} (link checks: {Path(sdk[1]).parent.name})")
    elif sdk:
        print(f"aapt2       : {sdk[0]} (compile only: no platforms/android-*/android.jar; attribute names/enums not checked)")
    else:
        print("aapt2       : not found (set ANDROID_HOME to validate VectorDrawables)")
    env_eb = os.environ.get("GPTKIT_EBSYNTH")
    if env_eb and not Path(env_eb).exists():
        eb_line = f"GPTKIT_EBSYNTH={env_eb} does not exist (sequence propagate refuses to run until it does)"
    else:
        eb = _find_ebsynth(None)
        eb_line = eb or ("not found: git clone https://github.com/jamriska/ebsynth ~/.cache/gptkit/ebsynth && cd ~/.cache/gptkit/ebsynth"
                         f" && sh build-{'macos' if sys.platform == 'darwin' else 'linux'}-cpu_only.sh, or set GPTKIT_EBSYNTH "
                         "(sequence then renders every frame with GPT)")
    print(f"ebsynth     : {eb_line}")
    lj = _lottie_js_cache()
    try:
        txt = lj.read_text() if lj.exists() else ""
    except OSError as e:
        txt, lj_line = "", f"cannot read {lj}: {e}"
    else:
        lj_line = (f"cached ({lj}, {len(txt) // 1024} kB)" if txt and _lottie_js_ok(txt) else
                   f"{lj} is not lottie-web ({len(txt)} bytes): refetched on the next preview" if txt else
                   f"not cached: fetched from {LOTTIE_JS} on the first vectorize/rig preview (SVG fallback without it)")
    print(f"lottie-web  : {lj_line}")
    gf = _cache_dir() / "gf_metadata.json"
    if gf.exists():
        try:
            fams = json.loads(gf.read_text())["familyMetadataList"]
            age = (time.time() - gf.stat().st_mtime) / 86400
            gf_line = (f"cached ({gf}: {len(fams)} families, {age:.0f} days old"
                       + ("; refreshed by the next online fonts run)" if age > 7 else ")"))
        except (OSError, ValueError, KeyError, TypeError) as e:
            gf_line = f"{gf} is corrupt ({type(e).__name__}): the next fonts run deletes and refetches it"
    else:
        gf_line = f"not cached: fonts/compare need {GF_METADATA} once (7-day cache)"
    print(f"fonts cache : {gf_line}")
    if not c["key"]:
        if not cli_ready:
            print("set GPTKIT_API_KEY in ~/.config/gptkit/env (chmod 600) or run `setup`")
            sys.exit(1)
        return
    base, headers = _auth()
    try:
        r = requests.get(base + "/models", headers=headers, timeout=30)
        r.raise_for_status()
        ids = {m["id"] for m in r.json().get("data", [])}
    except Exception as e:
        sys.exit(f"models      : request failed ({_redact(e)})")
    print(f"models      : {len(ids)} available")
    for label, mid, like in (("image", c["image_model"], lambda i: "image" in i), ("text", text_model, _is_text_model)):
        if mid in ids:
            print(f"  {label:5s} {mid}: ok")
        else:
            alt = sorted(i for i in ids if like(i))
            print(f"  {label:5s} {mid}: NOT LISTED (set GPTKIT_{label.upper()}_MODEL"
                  + (f" to one of: {', '.join(alt[:6])})" if alt else " to an id this endpoint lists)"))
    images = sorted(i for i in ids if "image" in i)
    if images:
        print("  image models on this endpoint:", ", ".join(images))


_NON_TEXT_MODEL = ("image", "audio", "realtime", "codex", "transcribe", "tts")


def _is_text_model(i):
    """Chat/vision-capable ids in the gpt-6 / gpt-5 families (skips image, audio, realtime and codex variants)."""
    return i.startswith(("gpt-6", "gpt-5")) and not any(t in i for t in _NON_TEXT_MODEL)


def cmd_score(a):
    d, m, mock = _load(a.dir)
    asset = _asset(m, a.name)
    ref = ref_crop(mock, asset["bbox"])
    js = lambda o: json.dumps(o, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    for f in a.files:
        p = _need(f, "file")
        if p.suffix.lower() == ".svg":
            r = render_svgs([p.read_text()], 256)[0]
            if r is None:
                print(f"{f}: not renderable (no working SVG renderer: see the 'svg render' line of doctor)")
                continue
            sc = svg_fidelity(mock, asset["bbox"], r)
            print(f, js(sc), *(f"\n  ! {i}" for i in _svg_issues(sc)))
        else:
            print(f, js(score(mock, asset["bbox"], _open_image(p, "file"), ref, asset["transparent"])))


def _color_args(p):
    p.add_argument("--color-match", type=_unit_float, default=0.75, metavar="STRENGTH",
                   help="pull the final file's colours toward the design's (0-1, default 0.75); kept only if it scores better")
    p.add_argument("--no-color-match", dest="color_match", action="store_const", const=0.0, help="keep the raw colours")
