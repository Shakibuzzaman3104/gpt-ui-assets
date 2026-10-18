# gptkit part 08_sequence: sequence (layout, plan, render, build, propagate, reroll).
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---- layout-driven sequences: code lays the motion out, GPT renders every frame

# An image model has no notion of time. Asked for a sprite sheet it keeps the style but not the trajectory: drops
# jump between cells, the floor moves from row to row, cells come back empty. So the motion is decided in code
# (a periodic layout, one guide per frame) and GPT only renders each frame's finish, as a rendering artist working
# over an animator's layout. Nothing is interpolated: in-betweens synthesised from independently rendered keys
# stall and surge (see references/api-findings.md, "Layout-driven sequences").
SEQUENCE_BRIEF = (
    "You are the lighting and rendering artist on a premium animated app-icon production. The attached image is "
    "frame {n} of {total} of a seamless {seconds:g}-second looping animation at {fps:g} frames per second, as laid out "
    "by the animator. {locked} is final artwork and is locked. {moving} is the animator's rough layout: it shows "
    "exactly where every part of it is on this frame and how big it is.\n"
    "Render {moving_lc} as the finished, high-end frame: {look}\n"
    "This is one frame of a moving sequence, so the layout is binding: keep every part at exactly the position and "
    "size the layout gives it. Do not add, remove, merge away, move or resize anything, and do not tidy it into a "
    "different shape. Only the finish changes. {locked}, the camera, the framing, the scale and the lighting stay "
    "exactly as they are. Fully transparent background, no ground, no cast shadow, no text."
)
SEQUENCE_LOOK = ("with exactly the same finish as the locked artwork: the same materials, palette, lighting and level "
                 "of detail, with gentle self-shadowing where parts overlap (pass --look to describe it, e.g. "
                 "'stylised 3D, soft light from the upper left')")


def _seq_dir(a):
    d = Path(a.dir) / "sequence" / a.name
    if getattr(a, "stage", "layout") != "layout" and not (d / "sequence.json").exists():
        sys.exit(_seq_no_layout(a, d))  # before any mkdir: a typo in DIR or NAME must not leave empty folders behind
    for sub in ("guides", "raw", "frames", "inputs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _seq_no_layout(a, d):
    return (f"sequence {getattr(a, 'stage', '')}: no layout in {d / 'sequence.json'}: run `sequence {a.dir} {a.name} layout` first "
            "(--guides DIR, or --base/--sprites/--vent)")


def _seq_meta(d, **update):
    p = d / "sequence.json"
    meta = _read_json(p, "sequence.json") if p.exists() else {}
    if not isinstance(meta, dict):
        sys.exit(f"gptkit: sequence.json {p} is not a JSON object (hand-edited?): re-run `sequence ... layout`, which "
                 "starts a new one")
    if meta.get("frames") is not None:
        ok = {"frames": isinstance(meta["frames"], int) and not isinstance(meta["frames"], bool) and meta["frames"] >= 1,
              "fps": _num_ok(meta.get("fps")) and meta["fps"] > 0,
              "canvas": isinstance(meta.get("canvas"), int) and not isinstance(meta.get("canvas"), bool) and meta["canvas"] >= 1}
        bad = [f"{k}={meta.get(k)!r}" for k, v in ok.items() if not v]
        if bad:
            sys.exit(f"gptkit: sequence.json {p}: {', '.join(bad)} (frames and canvas must be positive integers, fps a "
                     "positive number; hand-edited or older file?): fix it or re-run `sequence ... layout`")
    if update:
        meta.update(update)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, p)
    return meta


def _seq_path(d, p):
    """A path stored in sequence.json (relative to the sequence dir since the inputs are copied there)."""
    p = Path(p)
    return p if p.is_absolute() else d / p


def _seq_sprites(d, meta):
    return [_rgba(_seq_path(d, s)) for s in meta["plume"]["sprites"]]


def _sha1(data):
    import hashlib
    return hashlib.sha1(data).hexdigest()


def _seq_raw_state(d, meta, i, name=None):
    """'ok' | 'missing' | 'stale' for raw/NNN.png: stale when its sidecar names a guide hash the layout no longer has
    (a raw without a sidecar is from an older kit and is trusted)."""
    p = d / "raw" / (name or f"{i:03d}.png")
    if not p.exists():
        return "missing"
    side = p.with_suffix(".json")
    if side.exists():
        try:
            want = (meta.get("guide_sha1") or [None] * meta["frames"])[i]
            if json.loads(side.read_text()).get("guide_sha1") != want:
                return "stale"
        except (OSError, ValueError, IndexError):
            return "stale"
    return "ok"


def _seq_stale(d, meta, idx):
    stale = [i for i in idx if _seq_raw_state(d, meta, i) == "stale"]
    if stale:
        names = ", ".join(f"raw/{i:03d}.png" for i in stale[:6]) + (" ..." if len(stale) > 6 else "")
        sys.exit(f"sequence: {names} {'was' if len(stale) == 1 else 'were'} rendered from an older layout: run "
                 "`sequence ... render` again (it re-renders stale frames)")


def _vec(text, n=2, flag="value"):
    try:
        vals = [float(v) for v in str(text).split(",")]
    except ValueError:
        vals = []
    if len(vals) != n or any(math.isnan(v) or math.isinf(v) for v in vals):
        raise SystemExit(f"sequence layout: {flag} expects x,y numbers (a negative x needs {flag}=-40,-3), got {text!r}")
    return vals


def _natural(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).stem)]


def plume_state(age, k, p):
    """Closed-form flight of emission k at `age`: linear drag toward the wind plus buoyancy, so the terminal
    velocity is wind + (0, buoyancy / drag). The per-emission jitter repeats every loop, which makes the layout
    periodic by construction: layout(t + loop) == layout(t)."""
    per_loop = round(p["loop"] / p["chuff"])
    rnd = np.random.default_rng(k % per_loop)
    jx, jy, js = rnd.uniform(-1, 1, 3)
    v0 = (p["velocity"][0] + p["spread"][0] * jx, p["velocity"][1] + p["spread"][1] * jy)
    w = (p["wind"][0], p["wind"][1] + p["buoyancy"] / p["drag"])
    e = (1 - math.exp(-p["drag"] * age)) / p["drag"]
    x = p["vent"][0] + w[0] * age + (v0[0] - w[0]) * e
    y = p["vent"][1] + w[1] * age + (v0[1] - w[1]) * e
    u = age / p["life"]
    scale = (p["grow"][0] + (p["grow"][1] - p["grow"][0]) * u) * (1 + 0.08 * js)
    alpha = min(1, age / 0.08) * min(1, (1 - u) / 0.4)
    return x, y, scale, max(0.0, alpha), k % max(1, len(p["sprites"]))


def plume_layout(t, p):
    """Every live emission at absolute time t, keyed by emission number (identity survives the loop wrap)."""
    out = {}
    for k in range(math.floor((t - p["life"]) / p["chuff"]), math.floor(t / p["chuff"]) + 1):
        age = t - k * p["chuff"]
        if 0 <= age < p["life"]:
            out[k] = plume_state(age, k, p)
    return out


def plume_guide(t, p, base=None, sprites=None, canvas=512):
    """One guide frame: the plume's sprites at time t, oldest first, behind (or in front of) the locked base art.
    `sprites` are the loaded sprite images (paths in p are relative to the sequence dir)."""
    sprites = sprites or [_rgba(s) for s in p["sprites"]]
    layer = Image.new("RGBA", (canvas, canvas))
    for _, (x, y, s, al, v) in sorted(plume_layout(t, p).items()):
        sp = sprites[v]
        w = p["size"] * s
        h = w * sp.height / sp.width
        im = sp.resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS)
        arr = np.array(im)
        arr[..., 3] = (arr[..., 3] * al).astype(np.uint8)
        layer.alpha_composite(Image.fromarray(arr), (round(x - w / 2), round(y - h / 2)))
    if base is None:
        return layer
    out = Image.new("RGBA", (canvas, canvas))
    for part in ((base, layer) if p.get("front") else (layer, base)):
        out.alpha_composite(part)
    return out


def _seq_save_guide(d, i, g):
    big = Image.new("RGBA", (1024, 1024))
    big.alpha_composite(g if g.size == (1024, 1024) else g.resize((1024, 1024), Image.LANCZOS))
    p = d / "guides" / f"{i:03d}.png"
    big.save(p, compress_level=3)                       # transient 1024 px files: half the encode time of level 6
    return _sha1(p.read_bytes())


def _seq_check_layout(d, meta):
    """The layout must have moving content, and a guide set must loop (the wrap step is a step like any other)."""
    if meta.get("plume"):
        p = meta["plume"]
        if not any(st[3] > 0 for i in range(meta["frames"]) for st in plume_layout(i / meta["fps"], p).values()):
            sys.exit("sequence layout: the plume has no live element on any frame (raise --life or --chuff)")
        return
    static = _seq_static_mask(d, meta)
    parts = [_seq_layout_part(d, meta, i, static).resize((256, 256), Image.LANCZOS) for i in range(meta["frames"])]
    if not any((np.asarray(p)[..., 3] > 3).any() for p in parts):
        sys.exit("sequence layout: the layout has no moving content: every guide is empty"
                 + (" after cutting --base" if meta.get("base") else "") + " (check the guides and --base)")
    if meta["frames"] < 4:
        return
    # the wrap step must look like a step the loop already contains (measured: non-looping sets score 4x+ the largest
    # in-loop motion-compensated residual, looping ones with in-loop jumps or splashes at most 1x)
    steps = [_seq_step(parts[i], parts[(i + 1) % len(parts)])[1] for i in range(len(parts))]
    ref = max(max(steps[:-1]), 0.5)
    if steps[-1] > 2 * ref:
        print(f"  ! the guides do not loop: the step from the last guide back to the first is {steps[-1] / ref:.1f}x "
              "the largest step inside the loop (frame 0 must follow the last frame as the next frame would)")


def _seq_fresh_meta(d, **fresh):
    """Layout starts a new sequence.json: only the render-side text (moving, locked, look) survives, never the old
    plume, route, qa, rect or propagated state."""
    try:
        old = _seq_meta(d)
    except SystemExit as e:  # layout is how a broken sequence.json gets replaced: never let it block that
        print(f"  ! {str(e.code)[:200]}; starting a new one")
        old = {}
    keep = {k: old[k] for k in ("moving", "locked", "look") if old.get(k)}
    p = d / "sequence.json"
    p.write_text(json.dumps({**keep, **fresh}, indent=1))
    return _seq_meta(d)


def cmd_sequence_layout(a):
    d = _seq_dir(a)
    for sub in ("guides",):
        for old in (d / sub).glob("*.png"):
            old.unlink()
    if a.guides:
        folder = _need(a.guides, "guide folder")
        paths = sorted(folder.glob("*.png"), key=_natural)
        if not paths:
            sys.exit(f"sequence layout: no PNG guides in {folder}")
        guides = [_open_image(p, "guide") for p in paths]
        sizes = sorted({g.size for g in guides})
        if len(sizes) != 1 or sizes[0][0] != sizes[0][1]:
            got = "; ".join(f"{w}x{h}: " + ", ".join(p.name for p, g in zip(paths, guides) if g.size == (w, h))[:60]
                            for w, h in sizes[:4])
            sys.exit(f"sequence layout: guides must be square and all one size, got {got}")
        base = None
        if a.base:
            base = _open_image(a.base, "--base")
            shutil.copyfile(a.base, d / "inputs" / "base.png")
        opaque = float(np.mean([(np.asarray(g.getchannel("A")) == 255).mean() for g in guides]))
        if opaque > 0.5 and not a.base:
            print(f"  ! the guides have an opaque backdrop ({opaque:.0%} of their pixels are opaque): pass it as --base "
                  "so build can cut it away, or supply transparent guides")
        hashes = [_seq_save_guide(d, i, g) for i, g in enumerate(guides)]
        names = [p.name for p in paths]
        print(f"  guides in order: {', '.join(names[:3])}"
              + (f" ... {', '.join(names[-3:])}" if len(names) > 6 else (", " + ", ".join(names[3:]) if len(names) > 3 else "")))
        meta = _seq_fresh_meta(d, frames=len(paths), fps=a.fps, seconds=len(paths) / a.fps, canvas=guides[0].width,
                               base="inputs/base.png" if base is not None else None, source=str(folder.resolve()),
                               guide_sha1=hashes)
    else:
        if not (a.base and a.sprites and a.vent):
            sys.exit("sequence layout: a plume layout needs --base, --sprites and --vent (or pass --guides DIR)")
        base = _open_image(a.base, "--base")
        sprite_paths = [_need(s, "sprite") for s in a.sprites.split(",") if s]
        sprites = [_open_image(s, "sprite") for s in sprite_paths]
        shutil.copyfile(a.base, d / "inputs" / "base.png")
        for i, sp in enumerate(sprite_paths):
            shutil.copyfile(sp, d / "inputs" / f"sprite_{i}.png")
        p = dict(loop=a.loop, chuff=a.chuff, life=a.life, vent=_vec(a.vent, flag="--vent"),
                 velocity=_vec(a.velocity, flag="--velocity"), spread=_vec(a.spread, flag="--spread"), drag=a.drag,
                 wind=_vec(a.wind, flag="--wind"), buoyancy=a.buoyancy, size=a.size, grow=_vec(a.grow, flag="--grow"),
                 sprites=[f"inputs/sprite_{i}.png" for i in range(len(sprites))], front=a.front)
        if not (math.isfinite(a.buoyancy) and abs(a.buoyancy) <= 10000):
            sys.exit(f"sequence layout: --buoyancy {a.buoyancy:g} must be a number between -10000 and 10000 (canvas units "
                     "per second squared; negative rises)")
        if min(p["grow"]) <= 0:
            sys.exit(f"sequence layout: --grow {a.grow} must be two scales > 0 (at birth, at death), e.g. 0.45,1.5")
        per_loop = p["loop"] / p["chuff"]
        if abs(per_loop - round(per_loop)) > 1e-6:
            sys.exit(f"sequence layout: --loop {a.loop} must be a whole number of --chuff {a.chuff} intervals, "
                     "or the loop cannot repeat exactly")
        n = round(a.loop * a.fps)
        if n < 2:
            sys.exit(f"sequence layout: --loop {a.loop} at {a.fps:g} fps is {n} frame(s): raise --loop or --fps")
        if n > 1000:
            sys.exit(f"sequence layout: --loop {a.loop:g} at {a.fps:g} fps is {n} frames (one image call each; the limit is "
                     "1000): lower --loop or --fps")
        if a.life / a.chuff > 2000:
            sys.exit(f"sequence layout: --life {a.life:g} / --chuff {a.chuff:g} keeps {a.life / a.chuff:.0f} elements alive "
                     "(limit 2000): use values on the scale of the defaults (--life 5.4 --chuff 0.75)")
        if a.size > 8 * base.width:
            sys.exit(f"sequence layout: --size {a.size:g} is over 8x the {base.width}px --base canvas (default 96)")
        hashes = [_seq_save_guide(d, i, plume_guide(i / a.fps, p, base, sprites, canvas=base.width)) for i in range(n)]
        meta = _seq_fresh_meta(d, frames=n, fps=a.fps, seconds=a.loop, canvas=base.width, base="inputs/base.png",
                               plume=p, guide_sha1=hashes)
    _seq_check_layout(d, meta)
    picks = list(range(0, meta["frames"], max(1, meta["frames"] // 12)))[:12]
    sheet([[(f"f{i}", _rgba(d / "guides" / f"{i:03d}.png")) for i in picks[r:r + 6]] for r in (0, 6)],
          d / "layout_review.png", T=200)
    print(f"{meta['frames']} guide frames ({meta['seconds']:g} s at {meta['fps']:g} fps) -> {d / 'guides'}\n"
          f"  review {d / 'layout_review.png'} before rendering: the motion is decided here, not by GPT")
    seq_plan(d, meta, a)


def _seq_prompt(meta, i):
    moving = meta.get("moving") or "The moving element"
    return SEQUENCE_BRIEF.format(n=i + 1, total=meta["frames"], seconds=meta["seconds"], fps=meta["fps"],
                                 locked=meta.get("locked") or "The rest of the artwork", moving=moving,
                                 moving_lc=moving[:1].lower() + moving[1:], look=meta.get("look") or SEQUENCE_LOOK)


def _seq_render_one(d, meta, i, out, model):
    """Render frame i from its guide into `out` (temp + rename, with a sidecar naming the guide it came from)."""
    raw = edit(_png_bytes(_rgba(d / "guides" / f"{i:03d}.png")), _seq_prompt(meta, i), "1024x1024", "transparent", model)[0]
    img = _decode_result(raw, f"raw/{out.name}")
    err = _opaque_error(img, f"raw/{out.name}", model)
    if err:
        out.with_name(out.stem + ".opaque.png").write_bytes(raw)
        raise RuntimeError(f"{err}; saved as raw/{out.stem}.opaque.png, not used")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, out)
    out.with_suffix(".json").write_text(json.dumps({"guide_sha1": (meta.get("guide_sha1") or [None] * (i + 1))[i]}))
    return out


def cmd_sequence_render(a):
    d = _seq_dir(a)
    meta = _seq_meta(d, moving=a.moving, locked=a.locked, look=a.look or SEQUENCE_LOOK)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    if a.every is None:
        a.every = _seq_route(d, meta, a)["every"]
    elif meta["frames"] % a.every:
        sys.exit(f"sequence render: --every {a.every} must divide the {meta['frames']} frames so the keys repeat with the "
                 "loop (propagate needs that)")
    todo = [i for i in range(0, meta["frames"], a.every) if a.force or _seq_raw_state(d, meta, i) != "ok"]
    _plan(len(todo), 0, image_model=a.model, task="precise", background="transparent", edits=True)
    if getattr(a, "dry_run", False):
        print("would render frames: " + (", ".join(str(i) for i in todo) or "none"))
        return
    print(f"rendering {len(todo)} of {meta['frames']} frames, {a.jobs} at a time (resumable: finished frames are kept)")
    printed = set()

    def job(i):
        for attempt in range(3):
            try:
                return _seq_render_one(d, meta, i, d / "raw" / f"{i:03d}.png", a.model) and i
            except (ApiError, RuntimeError) as e:
                again = _retryable(e) and attempt < 2
                _fail_note(f"frame {i}" + (f" (attempt {attempt + 1}, retrying)" if again else ""), e, printed,
                           _seq_prompt(meta, i))
                if not again:
                    return None
                time.sleep(4 * (attempt + 1))
        return None
    done = _pool(a.jobs, job, todo)
    # count only the frames asked for (keys with --every), by name, so reroll candidates never count as frames
    missing = sum(1 for i in range(0, meta["frames"], a.every) if _seq_raw_state(d, meta, i) != "ok")
    print(f"  {len(done)} rendered" + (f", {missing} still missing" if missing else ""))
    if missing:
        sys.exit(f"sequence render: {missing} of {len(range(0, meta['frames'], a.every))} frames still missing (the "
                 "errors are above): fix the cause and run render again; finished frames are kept")


def _seq_extract(d, meta, raw_path, i, static):
    """Register a render onto its guide and keep only the moving part (the locked art is cut away)."""
    canvas = meta["canvas"]
    guide = _rgba(d / "guides" / f"{i:03d}.png").resize((canvas, canvas), Image.LANCZOS)
    arr = np.array(_register(_rgba(raw_path).resize((canvas, canvas), Image.LANCZOS), guide))
    if static is not None:
        arr[static] = 0
    return Image.fromarray(arr)


def _seq_static_mask(d, meta, grow=2):
    if not meta.get("base"):
        return None
    base = _rgba(_seq_path(d, meta["base"])).resize((meta["canvas"], meta["canvas"]), Image.LANCZOS)
    m = base.getchannel("A").point(lambda v: 255 if v > 8 else 0).filter(ImageFilter.MaxFilter(grow * 2 + 1))
    return np.asarray(m) > 0


def _grey(img):
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    return arr[..., :3].mean(-1) * arr[..., 3] / 255


def _seq_step(a, b):
    """(motion vector, motion-compensated residual) from frame a to b; needs OpenCV, else a plain difference."""
    ga, gb = _grey(a), _grey(b)
    m = ga > 30
    if not m.any():
        return (0.0, 0.0), 0.0
    try:
        import cv2
    except ImportError:
        return (0.0, 0.0), float(np.abs(gb - ga)[m].mean())
    fl = cv2.calcOpticalFlowFarneback(ga.astype(np.uint8), gb.astype(np.uint8), None, 0.5, 4, 21, 4, 7, 1.5, 0)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(gb, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR)
    return (float(fl[..., 0][m].mean()), float(fl[..., 1][m].mean())), float(np.abs(warped - ga)[m].mean())


def _qa_size(frames, cap=384):
    """Copies with the long side at most `cap` px: flow QA cost grows with area and 384 px is plenty for it."""
    w, h = frames[0].size
    if max(w, h) <= cap:
        return frames
    k = cap / max(w, h)
    return [f.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS) for f in frames]


def _seq_qa(frames, layout=None):
    """Direction, seam and shimmer of a loop. A frame that disagrees with both neighbours much more than they agree
    with each other is an outlier (usually a render that ignored the layout): reroll it. With `layout` (the layout
    parts, cropped like the frames) every score is taken relative to what the layout itself does, so the birth of a
    puff or a splash the layout contains is not an outlier and a radial/oscillating layout gets no direction QA."""
    frames = _qa_size(frames)
    n = len(frames)
    steps = [_seq_step(frames[i], frames[(i + 1) % n]) for i in range(n)]
    vecs = np.array([s[0] for s in steps])
    heading = vecs.mean(0)
    norm = float(np.hypot(*heading)) or 1.0
    along = vecs @ (heading / norm)
    resid = np.array([s[1] for s in steps])
    excess = []
    for i in range(n):
        r_in, r_out = resid[(i - 1) % n], resid[i]
        _, r_skip = _seq_step(frames[(i - 1) % n], frames[(i + 1) % n])
        dev = abs(along[(i - 1) % n] - np.median(along)) + abs(along[i] - np.median(along))
        excess.append(r_in + r_out - r_skip + 4 * dev)
    excess = np.array(excess)
    base = _seq_qa(layout) if layout else None
    score = excess - np.array(base["excess"]) if base else excess
    order = sorted(range(n), key=lambda i: -score[i])
    med, mad = float(np.median(score)), float(np.median(np.abs(score - np.median(score))))
    # robust: well clear of the typical frame by both ratio and spread, so a clean loop reports none; against a
    # layout, a frame must also add more than 60% of a typical frame's own excess on top of what the layout does
    floor = 0.6 * float(np.median(excess)) if base else 0.0
    outliers = [i for i in order[:max(1, n // 12)]
                if score[i] > max(med * 1.6, floor) and score[i] > med + 4 * mad]
    backwards = int((along < 0).sum()) if norm > 0.05 else 0
    directional = bool(base is None or np.hypot(*base["heading_px"]) >= 0.3)
    if base:
        backwards = max(0, backwards - base["backwards"]) if directional else 0
    return {
        "frames": n,
        "heading": [round(float(v), 3) for v in heading / norm],
        "heading_px": [round(float(v), 3) for v in heading],
        "speed_along": [round(float(x), 2) for x in along],
        "backwards": backwards,
        "seam_step": round(float(along[-1]), 2),
        "median_step": round(float(np.median(along)), 2),
        "shimmer_mean": round(float(resid.mean()), 2),
        "shimmer_max": round(float(resid.max()), 2),
        "excess": [round(float(x), 2) for x in excess],
        "outliers": outliers,
        "outlier_scores": {i: round(float(score[i]), 1) for i in order[:8]},
        **({"vs_layout": True, "layout_backwards": base["backwards"], "directional": directional} if base else {}),
    }


def _seq_moving(d, meta, a, i, raw_path, static):
    """One render reduced to its moving part: registered, the locked art cut away, clipped below --below."""
    f = _seq_extract(d, meta, raw_path, i, static)
    if a.below is not None:
        arr = np.array(f)
        arr[int(a.below):] = 0
        f = Image.fromarray(arr)
    return f


def cmd_sequence_build(a):
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    n = meta["frames"]
    raws = [d / "raw" / f"{i:03d}.png" for i in range(n)]
    _seq_stale(d, meta, range(n))
    missing = [i for i, p in enumerate(raws) if not p.exists()]
    if missing:
        route = _seq_route(d, meta, a)
        every = a.every or route["every"]
        if every > 1 and route["engine"] == "ebsynth" \
                and all((d / "raw" / f"{k:03d}.png").exists() for k in range(0, n, every)):
            print(f"keys only (every {every} frames): propagating them with EbSynth, as planned")
            a.every = every
            return cmd_sequence_propagate(a)
    if missing:
        raise SystemExit(f"frames missing from raw/: {missing[:10]} — run `sequence ... render` again, "
                         "or `sequence ... propagate` if you rendered keys only (--every)")
    ref = _build_opts(d, meta, a)
    static = _seq_static_mask(d, meta)
    frames = [_seq_moving(d, meta, a, i, p, static) for i, p in enumerate(raws)]
    if ref is not None:
        # per frame, not one transform for the loop: every frame is an independent render with its own exposure, and
        # pulling each onto the approved part evens that out (measured shimmer 9.4 vs 10.5 for one shared transform)
        frames = [_match_colors([f], ref)[0] for f in frames]
    meta.pop("propagated", None)
    _seq_meta(d, propagated=None)
    _seq_finish(d, meta, a, frames)


def _build_opts(d, meta, a):
    """--ref/--below/--quality/--preview-bg for this build: what was passed, else what the last build or propagate
    used (kept in sequence.json, so a reroll + build does not silently change the output). Returns the --ref image."""
    last = meta.get("build_opts") or {}
    opts, reused = {}, []
    for k, default in (("ref", None), ("below", None), ("quality", 80), ("preview_bg", "#a0aabe")):
        v = getattr(a, k, None)
        if v is None and last.get(k) is not None:
            v = last[k]
            if v != default:
                reused.append(f"--{k.replace('_', '-')} {v}")
        opts[k] = default if v is None else v
        setattr(a, k, opts[k])
    if reused:
        print(f"  using {' '.join(reused)} from the last build")
    ref = None
    if opts["ref"]:
        src = _seq_path(d, opts["ref"])
        if getattr(a, "ref", None) and not str(opts["ref"]).startswith("inputs/"):
            ref = _open_image(src, "--ref")
            shutil.copyfile(src, d / "inputs" / "ref.png")
            opts["ref"] = "inputs/ref.png"
        else:
            ref = _open_image(src, "--ref", "the copy made at the last build is gone; pass --ref again")
        if _opaque_share(ref)[0] > 0.9:
            print("  ! --ref has no transparent background: its backdrop takes part in the colour statistics; crop or "
                  "cut it out for a truer match")
    if opts["below"] is not None and not 0 < opts["below"] <= meta["canvas"]:
        sys.exit(f"--below {opts['below']} must be a canvas row between 1 and {meta['canvas']}")
    meta["build_opts"] = opts
    _seq_meta(d, build_opts=opts)
    return ref


def _seq_finish(d, meta, a, frames):
    """Crop the loop to its union box, encode the WebP, measure it and write the preview and contact sheet."""
    n = len(frames)
    union = np.zeros(frames[0].size[::-1], bool)
    for f in frames:
        union |= np.asarray(f)[..., 3] > 3
    if not union.any():
        sys.exit("sequence build: every frame is empty after cutting the locked art: the renders in raw/ came back "
                 "blank or entirely inside --base; check raw/ and --below")
    for old in (d / "frames").glob("*.png"):
        old.unlink()
    for i, f in enumerate(frames):
        f.save(d / "frames" / f"{i:03d}.png")
    ys, xs = np.nonzero(union)
    box = (max(0, xs.min() - 4), max(0, ys.min() - 4), min(union.shape[1], xs.max() + 5), min(union.shape[0], ys.max() + 5))
    crops = [f.crop(box) for f in frames]
    webp = d / f"{a.name}.webp"
    crops[0].save(webp, save_all=True, append_images=crops[1:], duration=_durations(n, meta["fps"]), loop=0,
                  format="WEBP", lossless=False, quality=a.quality, method=6, exact=True)
    static = _seq_static_mask(d, meta)
    layout = []
    for i in range(n):
        part = _seq_layout_part(d, meta, i, static)
        if static is not None and meta.get("plume"):    # the renders lose what the locked art covers; so must the layout
            arr = np.array(part)
            arr[static] = 0
            part = Image.fromarray(arr)
        layout.append(part.crop(box))
    if a.below is not None and int(a.below) - box[1] < box[3] - box[1]:
        for f in layout:
            ImageDraw.Draw(f).rectangle([0, max(0, int(a.below) - box[1]), f.width, f.height], fill=(0, 0, 0, 0))
    qa = _seq_qa(crops, layout)
    if meta.get("plume"):
        try:
            qa["follows"] = _seq_follows(d, frames, meta, box)
        except ImportError:
            print("  (OpenCV is not installed: the `follows` check is skipped; pip install opencv-python-headless)")
    rect = [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
    _seq_meta(d, rect=rect, qa=qa)
    (d / "report.json").write_text(json.dumps({"rect": rect, **qa}, indent=1))
    base = _rgba(_seq_path(d, meta["base"])).resize((meta["canvas"], meta["canvas"])) if meta.get("base") else None
    comps = []
    for f in frames:
        c = Image.new("RGBA", f.size, a.preview_bg)
        c.alpha_composite(f)
        if base is not None:
            c.alpha_composite(base)
        comps.append(c.convert("RGB"))
    comps[0].save(d / "preview.gif", save_all=True, append_images=comps[1:], duration=_durations(n, meta["fps"], 10),
                  loop=0)
    picks = list(range(0, n, max(1, n // 12)))[:12]
    sheet([[(f"f{i}{' !' if i in qa['outliers'] else ''}", comps[i]) for i in picks[r:r + 6]] for r in (0, 6)],
          d / "contact.png", T=200)
    prop = meta.get("propagated")
    if prop and qa["outliers"]:
        near = sorted({(i // prop["every"]) * prop["every"] for i in qa["outliers"]}
                      | {((i // prop["every"]) * prop["every"] + prop["every"]) % n for i in qa["outliers"]})
        fix = (f"they were synthesised from keys {near}: delete the nearest key's raw/NNN.png, run `sequence ... "
               "render` (it renders only the missing keys), then `build`")
    else:
        fix = "run `sequence ... reroll` on them (2 candidates each, the best neighbour fit wins), then `build`"
    direction = (f"{qa['backwards']} steps backwards, seam {qa['seam_step']} vs median {qa['median_step']}"
                 if qa.get("directional", True) else "no direction QA (the layout has no dominant heading)")
    print(f"{n} frames @ {meta['fps']:g} fps -> {webp} ({webp.stat().st_size // 1024} KB), rect {rect} on the "
          f"{meta['canvas']} canvas\n"
          f"  QA: {direction}, shimmer {qa['shimmer_mean']} (max {qa['shimmer_max']})"
          + (f", follows the layout {qa['follows']}%" if "follows" in qa else "")
          + (f"\n  ! outliers {qa['outliers']}: {fix}" if qa["outliers"] else "")
          + ("\n  ! the texture barely moves with the layout: GPT likely painted the same plume in every key. Make the "
             "elements distinct (fewer, rounder, spaced by --chuff) and render the keys again"
             if qa.get("follows", 100) < 5 else ""))


def _seq_follows(d, frames, meta, box):
    """Share of the frame-to-frame change that disappears when each frame is carried along the layout's exact motion.
    A texture that travels with its plume scores about 15%+; one that boils in place scores about 0, which the other
    checks miss (a dense exhaust whose keys were all the same generic plume scored 0.8%)."""
    import cv2
    p = meta["plume"]
    sprites = _seq_sprites(d, meta)
    x0, y0, x1, y1 = box
    fps, n = meta["fps"], len(frames)
    k = min(1.0, 384 / max(x1 - x0, y1 - y0))          # flow-free, but the remaps scale with area: cap at 384 px
    size = (max(1, round((x1 - x0) * k)), max(1, round((y1 - y0) * k)))
    res, still = [], []
    for i in range(n):
        mx, my = _plume_source_coords(i / fps, (i + 1) / fps, p, sprites, meta["canvas"])
        mx, my = (mx[y0:y1, x0:x1] - x0) * k, (my[y0:y1, x0:x1] - y0) * k
        if k < 1:
            mx, my = cv2.resize(mx, size, interpolation=cv2.INTER_LINEAR), cv2.resize(my, size, interpolation=cv2.INTER_LINEAR)
        a = np.asarray(frames[i].crop(box).resize(size, Image.LANCZOS), dtype=np.float32)
        b = np.asarray(frames[(i + 1) % n].crop(box).resize(size, Image.LANCZOS), dtype=np.float32)
        pa, pb = a[..., :3] * a[..., 3:4] / 255, b[..., :3] * b[..., 3:4] / 255
        carried = cv2.remap(pa, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        m = b[..., 3] > 60
        if m.any():
            res.append(np.abs(carried - pb)[m].mean())
            still.append(np.abs(pa - pb)[m].mean())
    return round(float(100 * (1 - np.mean(res) / max(1e-6, np.mean(still)))), 1) if still else 0.0


# -- propagate: GPT paints keys, EbSynth carries their texture along the layout to every frame

def _find_ebsynth(a):
    """--ebsynth, $GPTKIT_EBSYNTH, PATH, then ~/.cache/gptkit; a path named explicitly must exist (no silent fallback)."""
    for c, what in ((getattr(a, "ebsynth", None), "--ebsynth"), (os.environ.get("GPTKIT_EBSYNTH"), "GPTKIT_EBSYNTH")):
        if c and not Path(c).exists():
            sys.exit(f"gptkit: {what} {c} not found")
        if c:
            return c
    for c in (shutil.which("ebsynth"), str(Path.home() / ".cache" / "gptkit" / "ebsynth" / "bin" / "ebsynth")):
        if c and Path(c).exists():
            return c
    return None


EBSYNTH_BUILD = ("  git clone https://github.com/jamriska/ebsynth ~/.cache/gptkit/ebsynth\n"
                 f"  cd ~/.cache/gptkit/ebsynth && sh build-{'macos' if sys.platform == 'darwin' else 'linux'}-cpu_only.sh"
                 f"   # build-{'linux' if sys.platform == 'darwin' else 'macos'}-cpu_only.sh on "
                 f"{'Linux' if sys.platform == 'darwin' else 'macOS'}\n"
                 "or pass --ebsynth PATH / set GPTKIT_EBSYNTH.")


def _ebsynth_bin(a):
    eb = _find_ebsynth(a)
    if not eb:
        raise SystemExit("EbSynth not found. Build it once (public-domain code; the PatchMatch algorithm is patented, "
                         "check that before redistributing commercially; CPU is enough):\n" + EBSYNTH_BUILD)
    return eb


# -- plan: decide whether EbSynth carries the keys (and how far apart they sit) or GPT renders every frame

SEQ_KEY_SECONDS = 0.25   # measured: keys every 0.25 s never stepped backwards, every 0.5 s stalled mid-interval
SEQ_MIN_REACH = 0.9      # share of each in-between's layout that a neighbouring key must show for EbSynth to carry it
SEQ_PATCH_MATCH = 0.08   # RMS patch distance below which a key shows the same thing (measured: shifts and growth match, a new shape or colour does not)
SEQ_DENSE = 3.0          # mean elements stacked on a plume pixel above which GPT paints one generic plume


def _seq_plume_reach(meta, every):
    """Worst share of an in-between's plume (by element area) that exists in one of its two keys. An element born and
    gone between two keys has nothing to carry it; that happens when the keys sit further apart than --life."""
    p, fps, n = meta["plume"], meta["fps"], meta["frames"]
    worst = 1.0
    for i in range(n):
        ka = (i // every) * every
        if i == ka:
            continue
        have = set(plume_layout(ka / fps, p)) | set(plume_layout((ka + every) / fps, p))
        live = plume_layout(i / fps, p)
        total = sum(st[2] ** 2 * st[3] for st in live.values())
        if total > 0:
            worst = min(worst, sum(st[2] ** 2 * st[3] for k, st in live.items() if k in have) / total)
    return worst


def _seq_patches(img, cell=8):
    """3x3-cell patches (premultiplied RGBA, lightly blurred) of a layout part, with each patch's coverage."""
    n = img.width // cell
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(3)).resize((n, n), Image.BOX), dtype=np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)))
    return np.stack([pad[y:y + n, x:x + n] for y in range(3) for x in range(3)], 2).reshape(n * n, -1), arr[..., 3].reshape(-1)


def _seq_key_patches(img, cell=8):
    """The patches a key offers EbSynth: at sub-cell offsets and a little smaller or larger, since elements travel and
    grow between keys."""
    out = []
    for s in (0.85, 1.0, 1.18):
        big = img.resize((round(img.width * s),) * 2, Image.LANCZOS)
        for dy in range(0, cell, 2):
            for dx in range(0, cell, 2):
                c = Image.new("RGBA", big.size)
                c.alpha_composite(big, (dx, dy))
                pt, w = _seq_patches(c, cell)
                out.append(pt[w > 0.02])
    return np.concatenate(out)


def _seq_guide_reach(d, meta, every, static, cache=None):
    """Worst share of an in-between's content that looks like something in one of its two keys. EbSynth matches
    patches by appearance wherever they sit, so travel, jumps and growth carry; a splash, a blink or a colour that
    neither key shows has no paint to carry and would be invented."""
    n = meta["frames"]
    cache = {} if cache is None else cache

    def part(i):
        if ("part", i) not in cache:
            cache[("part", i)] = _seq_layout_part(d, meta, i, static).resize((256, 256), Image.LANCZOS)
        return cache[("part", i)]

    def keyp(k):
        if ("key", k) not in cache:
            cache[("key", k)] = _seq_key_patches(part(k))
        return cache[("key", k)]
    worst = 1.0
    for i in range(n):
        ka = (i // every) * every
        if i == ka:
            continue
        pt, w = _seq_patches(part(i))
        side = math.isqrt(len(w))
        edge = np.ones((side, side), bool)
        edge[2:-2, 2:-2] = False                    # content sliced by the canvas edge is entering or leaving, not new
        m = (w > 0.02) & ~edge.reshape(-1)
        if not m.any():
            continue
        keys = np.concatenate([keyp(ka), keyp((ka + every) % n)])
        if not len(keys):
            return 0.0
        q = pt[m]
        k2 = (keys ** 2).sum(1)[None]
        best = np.concatenate([((q[j:j + 256] ** 2).sum(1)[:, None] + k2 - 2 * q[j:j + 256] @ keys.T).min(1)
                               for j in range(0, len(q), 256)])   # 256-row blocks: the full matrix is 1k x 100k
        dist = np.sqrt(np.maximum(best, 0) / pt.shape[1])
        worst = min(worst, float(w[m][dist < SEQ_PATCH_MATCH].sum() / w[m].sum()))
    return worst


def _seq_density(d, meta):
    """Mean number of plume elements stacked on a covered pixel, over a few frames of the loop."""
    p = meta["plume"]
    sprites = _seq_sprites(d, meta)
    canvas = meta["canvas"]
    depths = []
    for t in np.linspace(0, meta["seconds"], 4, endpoint=False):
        stack = np.zeros((canvas, canvas), np.float32)
        for x, y, s, al, v in plume_layout(t, p).values():
            sp = sprites[v]
            w = p["size"] * s
            h = w * sp.height / sp.width
            foot = Image.new("L", (canvas, canvas))
            foot.paste(sp.getchannel("A").resize((max(1, round(w)), max(1, round(h)))), (round(x - w / 2), round(y - h / 2)))
            stack += np.asarray(foot, dtype=np.float32) > 60
        if (stack > 0).any():
            depths.append(float(stack[stack > 0].mean()))
    return round(float(np.mean(depths)), 2) if depths else 0.0


def seq_plan(d, meta, a, announce=True):
    """Decide the engine for a laid-out sequence and record it in sequence.json as `route`.

    EbSynth (`propagate`) is the default: GPT paints keys a quarter-second apart and EbSynth carries their texture to
    every frame, which costs a fraction of the image calls and shimmers less. It is skipped, and GPT renders every
    frame (`build`), when the binary is missing, when the loop is too short or too slow for keys to save anything, or
    when the layout changes between keys in ways no key shows (EbSynth can only carry paint that a key has).
    `--engine` is remembered in sequence.json; the plan is cached there too, keyed by the guides and the EbSynth
    install, and only recomputed when they change (a 72-guide plan at 1024 px takes 40 s)."""
    n, fps = meta["frames"], meta["fps"]
    if getattr(a, "engine", None):
        meta["engine"] = a.engine
        _seq_meta(d, engine=a.engine)
    engine = meta.get("engine") or "auto"
    eb = _find_ebsynth(a)
    key = _sha1(json.dumps([meta.get("guide_sha1"), n, fps, engine, eb, meta.get("base"), meta.get("plume")],
                           sort_keys=True, default=str).encode())[:12]
    cached = meta.get("route")
    if cached and cached.get("key") == key:
        if announce:
            _seq_plan_print(cached, n)
        return cached
    reasons, warnings = [], []
    every = 1
    if engine == "gpt":
        reasons.append("--engine gpt: GPT renders every frame")
    elif engine == "ebsynth" and not eb:
        _ebsynth_bin(a)
    elif not eb:
        reasons.append("EbSynth is not installed, so GPT renders every frame; to save calls, build it once:\n"
                       + EBSYNTH_BUILD)
    else:
        target = int(round(fps * SEQ_KEY_SECONDS))
        options = [k for k in range(target, 1, -1) if n % k == 0]
        if target < 2:
            reasons.append(f"at {fps:g} fps a quarter-second key is every frame, so there is nothing to carry")
        elif not options:
            reasons.append(f"no key interval from 2 to {target} frames divides the {n} frames, so keys cannot repeat "
                           "with the loop")
        else:
            static = None if meta.get("plume") else _seq_static_mask(d, meta)
            cache = {}
            for k in options:
                reach = _seq_plume_reach(meta, k) if meta.get("plume") else _seq_guide_reach(d, meta, k, static, cache)
                if reach >= SEQ_MIN_REACH:
                    every = k
                    if k != target:
                        reasons.append(f"keys every {k} frames: at {target} something between the keys "
                                       "was in neither of them")
                    break
            else:
                reasons.append(f"the layout shows something between neighbouring frames that no key shows (at "
                               f"most {reach:.0%} of an in-between is carried at {options[-1]} frames apart): a splash, "
                               "a blink or a change of colour or shape, so GPT renders every frame")
            if every > 1:
                if engine == "ebsynth":
                    reasons.insert(0, "--engine ebsynth")
                reasons.insert(0, f"EbSynth carries keys every {every} frames ({every / fps:g} s) to all {n} frames")
    if engine == "ebsynth" and every == 1:
        raise SystemExit("--engine ebsynth, but " + "; ".join(reasons))
    if meta.get("plume"):
        dense = _seq_density(d, meta)
        if dense > SEQ_DENSE:
            warnings.append(f"about {dense:g} elements overlap on each plume pixel: GPT reads a dense stream as one "
                            "generic plume and paints it the same in every frame or key, so the texture will not "
                            "travel with either engine. Space the elements (a longer --chuff, a smaller --size)")
    keys = len(range(0, n, every))
    route = {"engine": "ebsynth" if every > 1 else "gpt", "every": every, "image_calls": keys,
             "ebsynth_runs": (2 * n - keys) if every > 1 else 0, "reasons": reasons, "warnings": warnings, "key": key}
    meta["route"] = route
    _seq_meta(d, route=route)
    _seq_plan_print(route, n)
    return route


def _seq_plan_print(route, n):
    every = route["every"]
    print(f"plan: {'propagate (EbSynth)' if every > 1 else 'build (GPT renders every frame)'}, {route['image_calls']} "
          f"image calls for {n} frames\n" + "".join(f"  - {r}\n" for r in route["reasons"])
          + "".join(f"  ! {w}\n" for w in route["warnings"]), end="")


def _seq_route(d, meta, a):
    """The plan this run follows: the recorded one while the guides, the engine choice and the EbSynth install are
    unchanged (silently), else a new one (printed)."""
    return seq_plan(d, meta, a, announce=False)


def cmd_sequence_plan(a):
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    seq_plan(d, meta, a)


def _seq_layout_part(d, meta, i, static):
    """The moving part of frame i's guide on the base canvas: the plume alone, or the guide with the locked art cut."""
    canvas = meta["canvas"]
    if meta.get("plume"):
        return plume_guide(i / meta["fps"], meta["plume"], None, _seq_sprites(d, meta), canvas)
    g = np.array(_rgba(d / "guides" / f"{i % meta['frames']:03d}.png").resize((canvas, canvas), Image.LANCZOS))
    if static is not None:
        g[static] = 0
    return Image.fromarray(g)


def _plume_source_coords(t_key, t, p, sprites, canvas):
    """For each pixel at time t, where that point of its element was at t_key (identity outside the plume).

    Positions are layered like the elements are drawn, oldest first, so the front-most element owns each pixel. Averaging
    over overlapping elements instead points into none of them: in a dense exhaust plume (about 9 puffs overlapping) it
    made the synthesised texture creep at half the layout's speed with backward jitter."""
    src, dst = plume_layout(t_key, p), plume_layout(t, p)
    gy, gx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    mx, my = gx.copy(), gy.copy()
    for k, (x1, y1, s1, al, v) in sorted(dst.items()):
        if k not in src:
            continue
        x0, y0, s0 = src[k][0], src[k][1], src[k][2]
        sp = sprites[v]
        w_ = p["size"] * s1
        h_ = w_ * sp.height / sp.width
        foot = Image.new("L", (canvas, canvas))
        foot.paste(sp.getchannel("A").resize((max(1, round(w_)), max(1, round(h_)))), (round(x1 - w_ / 2), round(y1 - h_ / 2)))
        a = np.clip(np.asarray(foot, dtype=np.float32) / 255 * 2.5, 0, 1)   # opaque inside, soft only at the rim
        mx = mx * (1 - a) + (x0 + (gx - x1) * s0 / s1) * a
        my = my * (1 - a) + (y0 + (gy - y1) * s0 / s1) * a
    return mx, my


def _position_image(mx, my, box):
    x = np.clip((mx - box[0]) / max(1, box[2] - box[0] - 1), 0, 1)
    y = np.clip((my - box[1]) / max(1, box[3] - box[1] - 1), 0, 1)
    b = (np.floor(x * 15) + 16 * np.floor(y * 15)) / 255
    return Image.fromarray((np.dstack([x, y, b]) * 255).astype(np.uint8)[box[1]:box[3], box[0]:box[2]])


def _premul_rgb(img):
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    return Image.fromarray((arr[..., :3] * arr[..., 3:4] / 255).astype(np.uint8))


def cmd_sequence_propagate(a):
    """Every frame is synthesised from the two nearest GPT keys by EbSynth: patches of the real renders, placed where
    the layout says each part has moved (appearance guide, plus a positional guide for plumes). Keys go through the
    same synthesis, so they never pop against their in-betweens; frames are cross-faded between the two keys.
    Resumable: each synthesis is named by a hash of its inputs and reused unless --force."""
    import shlex
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    if a.every is None:
        route = _seq_route(d, meta, a)
        if route["every"] < 2:
            raise SystemExit("the plan renders every frame with GPT (" + "; ".join(route["reasons"]) + ") — run "
                             "`sequence ... build`, or pass --every N / --engine ebsynth to propagate anyway")
        a.every = route["every"]
    n, every = meta["frames"], a.every
    if n % every:
        raise SystemExit(f"--every {every} must divide the {n} frames so the keys repeat with the loop")
    eb = _ebsynth_bin(a)
    timeout = a.ebsynth_timeout or 600
    keys = list(range(0, n, every))
    _seq_stale(d, meta, keys)
    missing = [k for k in keys if not (d / "raw" / f"{k:03d}.png").exists()]
    if missing:
        raise SystemExit(f"keys missing from raw/: {missing} — run `sequence ... render --every {every}`")
    ref = _build_opts(d, meta, a)
    static = _seq_static_mask(d, meta)
    work = d / "propagate"
    work.mkdir(exist_ok=True)
    if a.force:
        for old in work.glob("syn_*.png"):
            old.unlink()
    parts = [_seq_layout_part(d, meta, i, static) for i in range(n)]
    union = np.zeros(parts[0].size[::-1], bool)
    for pt in parts:
        union |= np.asarray(pt)[..., 3] > 3
    if not union.any():
        sys.exit("sequence propagate: the layout has no moving content (every guide is empty after cutting --base): "
                 "run `sequence ... layout` again")
    ys, xs = np.nonzero(union)
    pad = 24
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(union.shape[1], xs.max() + pad), min(union.shape[0], ys.max() + pad))
    raw_sha = {}
    for k in keys:
        raw_sha[k] = _sha1((d / "raw" / f"{k:03d}.png").read_bytes())
        key = _seq_moving(d, meta, a, k, d / "raw" / f"{k:03d}.png", static)
        if ref is not None:
            key = _match_colors([key], ref)[0]
        key.crop(box).save(work / f"key_{k:03d}.png")
    for i in range(n):
        _premul_rgb(parts[i]).crop(box).save(work / f"app_{i:03d}.png")
    plume = meta.get("plume")
    sprites = _seq_sprites(d, meta) if plume else None
    canvas = meta["canvas"]
    if plume:
        gy, gx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
        ident = _position_image(gx, gy, box)
        for k in keys:
            ident.save(work / f"pos_{k:03d}.png")
    guide_sha = meta.get("guide_sha1") or [""] * n
    ref_sha = _sha1(_png_bytes(ref)) if ref is not None else ""

    def stamp(kk, i):
        return _sha1(f"{raw_sha[kk]}|{guide_sha[kk]}|{guide_sha[i]}|{a.guide_weight}|{a.position_weight}|{a.below}|"
                     f"{ref_sha}|{box}".encode())[:10]

    def synth(job):
        k, i, t_key = job                     # key frame, target frame, key's absolute time (k == n at the wrap)
        kk = k % n
        out = work / f"syn_{i:03d}_from_{k:03d}_{stamp(kk, i)}.png"
        if out.exists():
            return out, False
        cmd = [eb, "-style", str(work / f"key_{kk:03d}.png"),
               "-guide", str(work / f"app_{kk:03d}.png"), str(work / f"app_{i:03d}.png"), "-weight", str(a.guide_weight)]
        if plume:
            pos = work / f"pos_{i:03d}_from_{k:03d}.png"
            _position_image(*_plume_source_coords(t_key, i / meta["fps"], plume, sprites, canvas), box).save(pos)
            cmd += ["-guide", str(work / f"pos_{kk:03d}.png"), str(pos), "-weight", str(a.position_weight)]
        cmd += ["-output", str(out)]
        where = f"frame {i} from key {kk}"
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"EbSynth timed out after {timeout} s on {where} (raise --ebsynth-timeout, or fewer --jobs)"
                             f"\n  command: {shlex.join(cmd)}")
        except OSError as e:
            raise SystemExit(f"EbSynth could not be run ({e}): pass --ebsynth PATH\n  command: {shlex.join(cmd)}")
        if r.returncode or not out.exists():
            log = (r.stdout + r.stderr).strip()[-300:] or "no output"
            raise SystemExit(f"EbSynth failed on {where} (exit {r.returncode}"
                             f"{', wrote no output file' if not r.returncode else ''}): {log}\n  command: {shlex.join(cmd)}")
        try:
            _rgba(out)
        except (OSError, ValueError) as e:
            out.unlink(missing_ok=True)
            raise SystemExit(f"EbSynth wrote an unreadable file for {where} ({e})\n  command: {shlex.join(cmd)}")
        return out, True
    jobs = []
    for i in range(n):
        ka = (i // every) * every
        jobs.append((ka, i, ka / meta["fps"]))
        if i != ka:
            jobs.append((ka + every, i, (ka + every) / meta["fps"]))
    print(f"propagating {len(keys)} keys to {n} frames: {len(jobs)} EbSynth runs, {a.jobs} at a time"
          + (" (finished syntheses from the last run are reused)" if list(work.glob("syn_*.png")) else ""))
    ex = cf.ThreadPoolExecutor(max(1, a.jobs))
    futs = [ex.submit(synth, j) for j in jobs]
    syn, runs = {}, 0
    try:
        for done, f in enumerate(cf.as_completed(futs), 1):
            out, ran = f.result()
            runs += ran
            if ran and runs % 10 == 0:
                print(f"  {done}/{len(jobs)} syntheses ({runs} run now)", flush=True)
    except KeyboardInterrupt:                   # stop queuing work; the finished syntheses stay for a resume
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:                       # a failed run: let the in-flight ones end (bounded by the timeout)
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown()
    print(f"  {runs} EbSynth runs, {len(jobs) - runs} reused")
    frames = []
    for i in range(n):
        ka = (i // every) * every
        pm = None
        for k, w in ((ka, 1.0),) if i == ka else ((ka, None), (ka + every, None)):
            arr = np.asarray(_rgba(work / f"syn_{i:03d}_from_{k:03d}_{stamp(k % n, i)}.png"), dtype=np.float32).copy()
            arr[..., :3] *= arr[..., 3:4] / 255
            if w is None:
                f = (i - ka) / every
                g = f * f * (3 - 2 * f)       # smoothstep: each key's texture eases in and out
                w = 1 - g if k == ka else g
            pm = arr * w if pm is None else pm + arr * w
        a_ = pm[..., 3:4]
        rgb = np.where(a_ > 0.5, pm[..., :3] * 255 / np.maximum(a_, 1e-3), 0)
        crop = np.dstack([rgb, a_])
        # keep content only where the layout has it (soft margin); patch synthesis fills its whole box otherwise
        m = parts[i].getchannel("A").point(lambda v: 255 if v > 10 else 0).filter(ImageFilter.MaxFilter(21))
        m = np.asarray(m.filter(ImageFilter.GaussianBlur(6)).crop(box), dtype=np.float32) / 255
        crop[..., 3] *= m
        full = Image.new("RGBA", (canvas, canvas))
        full.paste(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)), (box[0], box[1]))
        frames.append(full)
    meta["propagated"] = {"every": every, "keys": keys}
    _seq_meta(d, propagated=meta["propagated"])
    _seq_finish(d, meta, a, frames)
    if not a.keep_work:
        for old in list(work.glob("syn_*.png")) + list(work.glob("pos_*_from_*.png")):
            old.unlink()
    else:
        print(f"  work files kept in {work}")


def cmd_sequence_reroll(a):
    """Render candidates for the worst frames and keep whichever fits its neighbours best (the current one included)."""
    d = _seq_dir(a)
    meta = _seq_meta(d)
    if not meta.get("frames"):
        sys.exit(_seq_no_layout(a, d))
    n = meta["frames"]
    if a.frames:
        try:
            targets = sorted({int(x) for x in a.frames.split(",") if x.strip()})
        except ValueError:
            sys.exit(f"sequence reroll: --frames takes a comma list of frame numbers (0-{n - 1}), got {a.frames!r}")
        bad = [i for i in targets if not 0 <= i < n]
        if bad or not targets:
            sys.exit(f"sequence reroll: frames {bad} are outside 0-{n - 1} ({n} frames)")
    else:
        targets = (meta.get("qa") or {}).get("outliers", [])
    if not targets:
        sys.exit("no outliers recorded: run `sequence ... build` first, or pass --frames 0,6,30")
    prop = meta.get("propagated")
    if prop:
        every, keys = prop["every"], prop["keys"]
        for i in targets:
            if i not in keys:
                ka = (i // every) * every
                sys.exit(f"sequence reroll: frame {i} was synthesised from keys {ka} and {(ka + every) % n}: to change it "
                         f"delete raw/{ka:03d}.png or raw/{(ka + every) % n:03d}.png, run `sequence ... render` (it renders "
                         "only the missing keys), then `build`")
    _seq_stale(d, meta, [i for i in targets for i in ((i - 1) % n, i, (i + 1) % n)])
    for i in targets:
        for j in ((i - 1) % n, i, (i + 1) % n):
            if not (d / "raw" / f"{j:03d}.png").exists():
                sys.exit(f"sequence reroll: raw/{j:03d}.png is missing (frame {i} is judged against its neighbours): run "
                         "`sequence ... render` first")
    static = _seq_static_mask(d, meta)
    jobs = [(i, c) for i in targets for c in range(a.candidates)]
    printed = set()

    def job(ic):
        i, c = ic
        out = d / "raw" / f"{i:03d}_cand{c}.png"
        for attempt in range(3):
            try:
                return _seq_render_one(d, meta, i, out, a.model)
            except (ApiError, RuntimeError) as e:
                again = _retryable(e) and attempt < 2
                _fail_note(f"frame {i} candidate {c}" + (" (retrying)" if again else ""), e, printed, _seq_prompt(meta, i))
                if not again:
                    return None
                time.sleep(4 * (attempt + 1))
        return None
    rendered = _pool(a.jobs, job, jobs)
    if not rendered:
        sys.exit("sequence reroll: no candidate was rendered (the errors are above); the frames are unchanged")
    replaced = 0
    for i in targets:
        prev = _seq_extract(d, meta, d / "raw" / f"{(i - 1) % n:03d}.png", (i - 1) % n, static)
        nxt = _seq_extract(d, meta, d / "raw" / f"{(i + 1) % n:03d}.png", (i + 1) % n, static)
        qa = meta.get("qa") or {}
        speed, heading = qa.get("median_step", 0.0), np.array(qa.get("heading", [1.0, 0.0]))
        options = [d / "raw" / f"{i:03d}.png"] + sorted((d / "raw").glob(f"{i:03d}_cand*.png"))
        scored = []
        for path in options:
            f = _seq_extract(d, meta, path, i, static)
            (v1, r1), (v2, r2) = _seq_step(prev, f), _seq_step(f, nxt)
            drift = abs(float(np.dot(v1, heading)) - speed) + abs(float(np.dot(v2, heading)) - speed)
            scored.append((r1 + r2 + 4 * drift, path))
        scored.sort(key=lambda s: s[0])
        best = scored[0][1]
        print(f"  frame {i}: " + ", ".join(f"{p.stem.split('_')[-1] if '_' in p.stem else 'current'} {s:.1f}" for s, p in scored))
        if best != options[0]:
            shutil.copyfile(options[0], options[0].with_name(f"{i:03d}.prev.png"))   # the replaced render is kept
            shutil.copyfile(best, options[0])
            if best.with_suffix(".json").exists():
                shutil.copyfile(best.with_suffix(".json"), options[0].with_suffix(".json"))
            replaced += 1
    print(f"  {replaced} of {len(targets)} frames replaced (previous renders kept as raw/NNN.prev.png); now run "
          "`sequence ... build` again")


def cmd_sequence(a):
    {"layout": cmd_sequence_layout, "plan": cmd_sequence_plan, "render": cmd_sequence_render, "build": cmd_sequence_build,
     "propagate": cmd_sequence_propagate, "reroll": cmd_sequence_reroll}[a.stage](a)
