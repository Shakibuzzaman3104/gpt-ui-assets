# gptkit part 11_export: export to Android/iOS/web.
# Executed into gptkit.py's namespace in file order (see gptkit.py): names from earlier parts are
# plain globals here, and the tests' monkeypatching of gptkit.<name> reaches every part.

# ---- export

DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}


def _fit(img, w, h):
    im = img.copy()
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def _px(v):
    """Android rounds density sizes as (int)(px + 0.5), not banker's rounding (round(23 * 1.5) is 34, Android says 35)."""
    return max(1, int(v + 0.5))


def _flat_art(img, max_colors=256):
    """True when the image has at most max_colors distinct RGBA values: lossless WebP is then small and exact."""
    return img.getcolors(max_colors) is not None


def _save_webp(im, path, lossless, quality):
    if lossless:
        im.save(path, "WEBP", lossless=True, method=6)
    else:
        im.save(path, "WEBP", quality=quality, method=6)


def cmd_export(a):
    img = _open_image(a.file, "source image")
    name = a.name or Path(a.file).stem
    fmt = a.format or "webp"
    if a.platform == "ios" and (a.format or a.quality is not None):
        sys.exit("export: --format/--quality do not apply to --platform ios: imagesets are PNG")
    if a.quality is not None and not 0 <= a.quality <= 100:
        sys.exit(f"export: --quality {a.quality} must be between 0 and 100 (100 = lossless)")
    if a.platform == "android":
        rn = _res_name(name)
        if rn != name:
            print(f'  (renamed "{name}" -> {rn} for Android: resource names are [a-z0-9_] and start with a letter)')
        name = rn
    elif a.platform == "web":  # a URL in src/srcset: no spaces (srcset splits on them), slashes or other URL-hostile characters
        wn = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "img"
        if wn != name:
            print(f'  (renamed "{name}" -> {wn} for the web: file names go into src/srcset URLs, [A-Za-z0-9._-] only)')
        name = wn
    elif "/" in name or "\\" in name or ":" in name:  # an imageset is a folder: path separators are not asset names
        nn = re.sub(r"[/\\:]+", "-", name).strip("-") or "img"
        print(f'  (renamed "{name}" -> {nn} for Xcode: asset names cannot contain path separators)')
        name = nn
    wdp = a.dp
    hdp = a.hdp or round(wdp * img.height / img.width, 2)
    res = Path(a.res)
    lossless = a.quality == 100 or (a.quality is None and _flat_art(img))
    quality = 90 if a.quality is None else a.quality
    biggest = {"android": max(DENSITIES.values()), "ios": 3, "web": 2}[a.platform]
    tw, th = _px(wdp * biggest), _px(hdp * biggest)
    if max(tw, th) > 8192:
        sys.exit(f"export: --dp {wdp:g}" + (f" --hdp {hdp:g}" if a.hdp else "") + f" makes a {tw}x{th} px file at "
                 f"{biggest}x (the limit is 8192 px): --dp/--hdp are dp/pt/css px, not device pixels")
    if res.exists() and not res.is_dir():
        sys.exit(f"export: --res {res} is a file; pass the res/ directory (android), the .xcassets folder (ios) or an "
                 "output directory (web)")
    if a.platform != "android" and (not name or name in (".", "..") or re.search(r"[/\\]", name)):
        sys.exit(f"export: --name {name!r} must be a plain file name (no / or \\): the files go inside --res")
    if img.width < tw or img.height < th:
        print(f"  ! source {img.width}x{img.height} px is smaller than the largest target {tw}x{th} px; it is upscaled and will "
              f"look soft (use a source at least {tw} px wide)")
    if img.getchannel("A").getextrema()[1] == 0:
        print("  ! source is fully transparent: every exported file is an empty image")
    if a.platform == "android":
        if res.name.lower().startswith(("drawable", "mipmap")):
            print(f"  ! --res {res} is itself a resource folder; density folders go under res/ (did you mean --res {res.parent}?)")
        for dens, sc in DENSITIES.items():
            out = res / f"drawable-{dens}"
            out.mkdir(parents=True, exist_ok=True)
            im = _fit(img, _px(wdp * sc), _px(hdp * sc))
            p = out / f"{name}.{fmt}"
            _save_webp(im, p, lossless, quality) if fmt == "webp" else im.save(p)
            print(f"  {p} {im.size} {p.stat().st_size} B")
        if fmt == "webp":
            print(f"  WebP {'lossless' if lossless else f'q{quality}'} (transparent WebP needs API 18+; --quality 100 forces lossless)")
    elif a.platform == "ios":
        if res.suffix == ".xcassets":
            res.mkdir(parents=True, exist_ok=True)
            if not (res / "Contents.json").exists():  # a new catalog: Xcode expects its root Contents.json
                (res / "Contents.json").write_text(json.dumps({"info": {"version": 1, "author": "xcode"}}, indent=2))
        out = res / f"{name}.imageset"
        out.mkdir(parents=True, exist_ok=True)
        images = []
        for sc in (1, 2, 3):
            fn = f"{name}@{sc}x.png"
            _fit(img, _px(wdp * sc), _px(hdp * sc)).save(out / fn)
            images.append({"idiom": "universal", "filename": fn, "scale": f"{sc}x"})
        (out / "Contents.json").write_text(json.dumps({"images": images, "info": {"version": 1, "author": "xcode"}}, indent=2))
        print(f"  {out} ({_px(wdp)}x{_px(hdp)} pt @1x/@2x/@3x)")
    else:
        res.mkdir(parents=True, exist_ok=True)
        for sc, suffix in ((1, ""), (2, "@2x")):
            p = res / f"{name}{suffix}.{fmt}"
            im = _fit(img, _px(wdp * sc), _px(hdp * sc))
            _save_webp(im, p, lossless, quality) if fmt == "webp" else im.save(p)
            print(f"  {p} {im.size} {p.stat().st_size} B")
        print(f'  <img src="{name}.{fmt}" srcset="{name}.{fmt} 1x, {name}@2x.{fmt} 2x" width="{_px(wdp)}" height="{_px(hdp)}" alt="">')
