"""The asset gallery: every image the kit makes under a project's ui-assets/, with picks, discards and notes, as one
offline page (ui-assets/index.html) and an optional local server that writes the choices back."""
import concurrent.futures as cf
import contextlib
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

GALLERY_DIR = "ui-assets"
IMG_EXT = {".png", ".webp", ".jpg", ".jpeg", ".gif", ".svg"}
THUMB_BOX = (900, 900)
STATES = ("new", "picked", "discarded")
SKIP_DIRS = {".thumbs", ".history", ".git", "node_modules", "review", "preview", "propagate", "work", "bare",
             "findings", "captures", "tokens", "fonts", "redline", "compare", "traced", "rig", "values"}
SKIP_NAME = re.compile(
    r"(_critique|_raw|_raw\.opaque|_review|_existing|_aligned|_api_raw|_cm)\.\w+$"
    r"|^(review|sheet|sheet_raw|static|static_raw|contact|search_row|manifest_overlay|motion_sheet|heatmap|"
    r"layout_review|frames_review|keys_review|fonts_review|_glyph_probe|foreground_concept_aligned|mockup)\.\w+$"
    r"|^(frame_\d+|moving_f\d+|key_\d+|pos_\d+|syn_\d+|app_\d+|competitors_\d+|\d{3})[._]", re.I)
VARIANT = re.compile(r"^(?P<base>.+?)_(?P<label>v\d+|c\d+|t\d+|fix\d+)$")
APPICON_FINALS = {"play_store_512.png", "AppIcon-1024.png"}
CMD_KINDS = {"mockup": {"mockup"}, "critique": {"mockup"}, "forge": {"asset"}, "fix": {"asset"}, "pick": {"asset"},
             "svg": {"icon"}, "icons": {"icon"}, "appicon concepts": {"appicon"}, "appicon build": {"appicon"},
             "material": {"hybrid"}, "state": {"hybrid", "other"}, "stage": {"hybrid", "other"},
             "frames": {"animation", "other"}, "vectorize": {"animation", "other"}}
QUIET_CMDS = {"status", "cost", "doctor", "setup", "gallery", "login", "logout"}


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gallery_dir_of(path):
    """The ui-assets folder a path lies in (the path itself, or an ancestor), else None."""
    try:
        p = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for q in (p, *p.parents):
        if q.name == GALLERY_DIR:
            return q
    return None


def _gallery_decisions_file():
    cfg = os.environ.get("GPTKIT_CONFIG") or str(Path.home() / ".config/gptkit/env")
    return Path(cfg).expanduser().parent / "gallery.json"


def gallery_decision(gdir):
    """'on', 'off' or None (undecided, which behaves as on) for this project's gallery."""
    env = os.environ.get("GPTKIT_GALLERY", "").strip().lower()
    if env in ("on", "off"):
        return env
    try:
        value = json.loads(_gallery_decisions_file().read_text()).get(str(Path(gdir).resolve()))
    except (OSError, ValueError, AttributeError):
        return None
    return value if value in ("on", "off") else None


def _set_gallery_decision(gdir, value):
    f = _gallery_decisions_file()
    try:
        data = json.loads(f.read_text())
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    data[str(Path(gdir).resolve())] = value
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=1, sort_keys=True))


@contextlib.contextmanager
def _gallery_locked(gdir):
    """Cross-process lock: forge and svg often run in parallel on the same project."""
    gdir.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows: no flock; the atomic manifest write still keeps the file whole
        yield
        return
    with open(gdir / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_gallery(gdir):
    f = Path(gdir) / "manifest.json"
    try:
        m = json.loads(f.read_text())
    except FileNotFoundError:
        m = None
    except (OSError, ValueError, UnicodeError):
        m = False
    if m is not None and (not isinstance(m, dict) or not isinstance(m.get("entries"), dict)):
        keep = f.with_name(f"manifest.damaged-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
        with contextlib.suppress(OSError):
            f.replace(keep)
            print(f"gallery: {f} was damaged; moved to {keep.name} and rebuilt (picks and notes in it are not "
                  "carried over)", file=sys.stderr)
        m = None
    if m is None:
        m = {"version": 1, "entries": {}, "sources": []}
    m.setdefault("sources", [])
    return m


def _write_gallery_file(path, text):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _root_rel(p, root):
    return Path(p).resolve().relative_to(root).as_posix()


def _cached_json(path, cache):
    key = str(path)
    if key not in cache:
        try:
            cache[key] = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            cache[key] = None
    return cache[key]


def _dir_names(d, cache):
    key = ("names", str(d))
    if key not in cache:
        try:
            cache[key] = {x.name for x in d.iterdir()}
        except OSError:
            cache[key] = set()
    return cache[key]


def _ancestor_with_file(d, stop, marker, cache):
    for q in (d, *d.parents):
        if marker in _dir_names(q, cache):
            return q
        if q == stop:
            return None
    return None


def _variant_split(stem):
    m = VARIANT.match(stem)
    return (m.group("base"), m.group("label")) if m else (stem, None)


def classify_output(p, root, top, cache):
    """What a file under `top` is in the gallery: kind, unit dir, group, label, choice and final; None to leave it out."""
    p = Path(p)
    if p.suffix.lower() not in IMG_EXT or SKIP_NAME.search(p.name):
        return None
    d, stem = p.parent, p.stem
    base, label = _variant_split(stem)
    out = {"unit": d, "group": base, "label": label or stem, "choice": bool(label), "final": False, "kind": "other"}
    icon_root = _ancestor_with_file(d, top, "appicon.json", cache)
    if icon_root:
        if p.name not in APPICON_FINALS:
            return None
        return dict(out, kind="appicon", unit=icon_root, group="launcher icon", label=p.stem, choice=False, final=True)
    build = _ancestor_with_file(d, top, "assets.json", cache)
    if build:
        sub = d.relative_to(build).parts
        if sub == ("raw",):
            if not re.search(r"_(c|fix)\d+$", stem):
                return None
            return dict(out, kind="asset", unit=build, choice=True)
        if sub == ("assets",):
            raw = build / "raw"
            if raw.is_dir() and any(raw.glob(f"{stem}_c*.png")):
                return None  # the final file is a copy of the picked candidate, which is shown instead
            return dict(out, kind="asset", unit=build, group=stem, label="final", choice=False, final=True)
        if sub in (("svg",), ("icons",)):
            if p.suffix.lower() != ".svg":
                return None
            return dict(out, kind="icon", unit=build, group="icons" if sub == ("svg",) else "new icons",
                        label=stem, choice=False)
        if not sub:
            return None
        if sub[0] in ("frames", "vector", "anim") and len(sub) > 1:
            out["unit"] = build
    mat = d if "material.json" in _dir_names(d, cache) else \
        d.parent if d.name == "raw" and "material.json" in _dir_names(d.parent, cache) else None
    if mat:
        if d != mat:
            return dict(out, kind="hybrid", unit=mat, choice=True) if label else None
        if not label and any((d / "raw").glob(f"{stem}_c*.png")):
            return None  # the final file is the best candidate, trimmed: the candidates are shown instead
        return dict(out, kind="hybrid")
    if d.name == "raw":
        return None  # every other raw/ holds intermediate renders and retries
    if "stage.json" in _dir_names(d, cache):
        if stem == "motion":
            return dict(out, kind="animation", group="stage", label="motion", choice=False)
        return dict(out, kind="hybrid", group="stage", label=stem, choice=False)
    if "sequence.json" in _dir_names(d, cache) or "frames_review.png" in _dir_names(d, cache) \
            or "animation_report.json" in _dir_names(d, cache):
        if p.suffix.lower() == ".webp" or (p.suffix.lower() == ".gif" and not any(d.glob("*.webp"))):
            return dict(out, kind="animation", group=d.name, label=stem, choice=False)
        return None
    if "design.json" in _dir_names(d, cache):
        qa = re.match(r"^(?P<src>.+)_qa(?P<n>\d+)$", stem)
        if qa:  # a `critique --apply` revision of a variant: shown in that variant's group
            sbase, slabel = _variant_split(qa.group("src"))
            return dict(out, kind="mockup", group=sbase, label=f"{slabel or qa.group('src')} fix {qa.group('n')}",
                        choice=True, revision_of=qa.group("src"))
        return dict(out, kind="mockup")
    if "review" in _dir_names(d, cache) and any((d / "review").glob("concepts_*")):
        return dict(out, kind="appicon")
    store = _ancestor_with_file(d, top, "captures.json", cache)
    if store:
        out.update(kind="store", unit=store)
        if d.name == "_replaced" and "scenes.json" in _dir_names(d.parent, cache):
            return dict(out, group=f"{base} scene" + ("" if d.parent.name == "main" else f" ({d.parent.name})"),
                        label=f"{label or stem} · replaced", choice=True, replaced=True)
        if "scenes.json" in _dir_names(d, cache):
            return dict(out, group=f"{base} scene" + ("" if d.name == "main" else f" ({d.name})"), scene_set=d.name)
        if d.name == "art":
            if not label and any(d.glob(f"{stem}_t*.png")):
                return None
            return dict(out, group=f"{base} art")
        if re.match(r"^\d\d_", stem) or stem == "feature":  # a finished shot: grouped by store target
            where = d.relative_to(store).parts
            painted = "scenes" in where
            return dict(out, group=d.name + (" painted" if painted else ""), label=stem, choice=False)
        return out
    return out


def _gallery_info(p, c, root, cache):
    """Facts the kit already recorded about a file: critique issues, forge scores, material flags."""
    d = p.parent
    if c["kind"] == "mockup" and c.get("revision_of"):
        crit = _cached_json(d / f"{c['revision_of']}_critique.json", cache)
        for r in (crit or {}).get("rounds", []) if isinstance(crit, dict) else []:
            if isinstance(r, dict) and r.get("image") and Path(r["image"]).name == p.name:
                return {"kept": bool(r.get("kept")), "weight": [r.get("weight_before"), r.get("weight_after")],
                        "flags": [] if r.get("kept") else ["NOT KEPT"]}
        return {}
    if c["kind"] == "mockup":
        crit = _cached_json(d / f"{p.stem}_critique.json", cache)
        if isinstance(crit, dict):
            rounds = [r for r in crit.get("rounds", []) if isinstance(r, dict)] or [crit]
            issues = [x for x in (rounds[0].get("issues") or []) if isinstance(x, dict)]
            sev = {}
            for x in issues:
                sev[x.get("severity", "low")] = sev.get(x.get("severity", "low"), 0) + 1
            out = {"issues": sev}
            sheet = d / f"{p.stem}_critique.png"
            if sheet.exists():
                out["critique"] = _root_rel(sheet, root)
            return out
    if c["kind"] == "asset" and c["choice"]:
        rep = _cached_json(c["unit"] / "forge_report.json", cache)
        entry = rep.get(c["group"]) if isinstance(rep, dict) else None
        if isinstance(entry, dict):
            relp = p.relative_to(c["unit"]).as_posix()
            for cand in entry.get("candidates") or []:
                if isinstance(cand, dict) and cand.get("path") == relp and isinstance(cand.get("score"), dict):
                    sc = cand["score"]
                    return {"fid": sc.get("fid"), "flags": sc.get("flags") or []}
    if c.get("scene_set"):
        sc = _cached_json(d / "scenes.json", cache)
        slot = ((sc or {}).get("slots") or {}).get(_variant_split(p.stem)[0]) if isinstance(sc, dict) else None
        for v in (slot or {}).get("variants") or []:
            if isinstance(v, dict) and v.get("file") == p.name:
                return {"flags": v.get("flags") or [], **({"fit": v["fit"]} if v.get("fit") is not None else {})}
    if c["kind"] == "hybrid" and c["choice"]:
        mat = _cached_json(c["unit"] / "material.json", cache)
        cand = ((mat or {}).get(c["group"]) or {}).get("candidates", {}).get(c["label"]) if isinstance(mat, dict) else None
        if isinstance(cand, dict):
            return {"flags": cand.get("flags") or []}
    return {}


def _kit_chosen(p, c, cache):
    """True/False when the kit itself records whether this file is the chosen one; None when it doesn't."""
    if c["kind"] == "asset" and c["choice"]:
        rep = _cached_json(c["unit"] / "forge_report.json", cache)
        entry = rep.get(c["group"]) if isinstance(rep, dict) else None
        chosen = entry.get("chosen") if isinstance(entry, dict) else None
        if isinstance(chosen, dict):
            return chosen.get("path") == p.relative_to(c["unit"]).as_posix()
    if c.get("scene_set"):
        sc = _cached_json(p.parent / "scenes.json", cache)
        slot = ((sc or {}).get("slots") or {}).get(_variant_split(p.stem)[0]) if isinstance(sc, dict) else None
        if isinstance(slot, dict) and slot.get("pick"):
            return slot["pick"] == p.name
    if c["kind"] == "hybrid" and c["choice"]:
        mat = _cached_json(c["unit"] / "material.json", cache)
        entry = mat.get(c["group"]) if isinstance(mat, dict) else None
        if isinstance(entry, dict) and entry.get("best"):
            return entry["best"] == c["label"]
    return None


def _gallery_walk(top):
    for here, dirs, files in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith(("mipmap-", "drawable")))
        for f in sorted(files):
            yield Path(here) / f


def _gallery_thumb(gdir, p, rel, mtime):
    """A small WebP for the grid; SVG and animated files are shown as they are."""
    if p.suffix.lower() in (".svg", ".gif"):
        return rel
    t = gdir / ".thumbs" / (hashlib.sha1(rel.encode()).hexdigest()[:16] + ".webp")
    if t.exists() and t.stat().st_mtime >= mtime:
        return t.relative_to(gdir.parent).as_posix()
    try:
        with Image.open(p) as im:
            if getattr(im, "is_animated", False):
                return rel
            im.thumbnail(THUMB_BOX, Image.LANCZOS)
            im = im.convert("RGBA") if im.mode in ("P", "LA", "RGBA", "PA") else im.convert("RGB")
            t.parent.mkdir(exist_ok=True)
            im.save(t, "WEBP", quality=82, method=4)
    except (OSError, ValueError, Image.DecompressionBombError):
        return rel
    return t.relative_to(gdir.parent).as_posix()


def _gallery_size(p):
    if p.suffix.lower() == ".svg":
        m = re.search(r'viewBox="[\d.\-]+[ ,]+[\d.\-]+[ ,]+([\d.]+)[ ,]+([\d.]+)"', p.read_text(errors="ignore")[:4000])
        return [round(float(m.group(1))), round(float(m.group(2)))] if m else [24, 24]
    try:
        with Image.open(p) as im:
            return list(im.size)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def gallery_index(gdir, run=None):
    """Rescan the project into ui-assets/manifest.json, then rewrite index.html and .gitignore. Returns (manifest, new)."""
    gdir = Path(gdir).resolve()
    root = gdir.parent
    with _gallery_locked(gdir):
        m = load_gallery(gdir)
        old, entries, cache, new, thumbs = m["entries"], {}, {}, 0, {}
        pool = cf.ThreadPoolExecutor(max(2, min(8, os.cpu_count() or 2)))  # resizing and WebP encoding release the GIL
        tops = [gdir] + [root / s for s in m["sources"] if (root / s).is_dir()]
        for top in tops:
            for p in _gallery_walk(top):
                c = classify_output(p, root, top, cache)
                if c is None:
                    continue
                rel = _root_rel(p, root)
                if rel in entries:
                    continue
                st = p.stat()
                e = dict(old.get(rel) or {})
                if not e:
                    new += 1
                    e = {"state": "discarded" if c.get("revision_of") and _gallery_info(p, c, root, cache).get("kept") is False
                         else "new", "created": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
                unit = c["unit"].relative_to(top) if c["unit"] != top else Path(".")
                unit = (Path(top.name) / unit if top != gdir else unit).as_posix()
                e.update(path=rel, kind=c["kind"], unit=unit, unit_path=_root_rel(c["unit"], root), group=c["group"], label=c["label"], choice=c["choice"],
                         final=c["final"], replaced=bool(c.get("replaced")), mtime=round(st.st_mtime, 3),
                         bytes=st.st_size)
                if e.get("_seen_mtime") != e["mtime"]:
                    e["size"] = _gallery_size(p)
                    e["_seen_mtime"] = e["mtime"]
                thumbs[rel] = pool.submit(_gallery_thumb, gdir, p, rel, st.st_mtime)
                e["info"] = _gallery_info(p, c, root, cache)
                kit = _kit_chosen(p, c, cache)
                e["kit"] = (c["kind"] == "asset" and c["choice"]) or bool(c.get("scene_set"))
                if c.get("scene_set"):
                    e["scene_set"] = c["scene_set"]
                if kit is not None and (e["kit"] or not e.get("decided")):
                    if kit:
                        e["state"] = "picked"
                    elif e["state"] == "picked":
                        e["state"] = "new"
                if run and c["kind"] in CMD_KINDS.get(run["cmd"], {"other"}) \
                        and run.get("before", {}).get(str(p)) != st.st_mtime \
                        and any(Path(sc) == p or Path(sc) in p.parents for sc in run.get("scope") or []):
                    e.update({k: v for k, v in run.items() if k in ("cmd", "prompt", "parent") and v})
                entries[rel] = e
        for rel, fut in thumbs.items():
            entries[rel]["thumb"] = fut.result()
        pool.shutdown()
        for rel, e in old.items():  # earlier versions the kit overwrote, kept under .history/
            if e.get("history") and (root / rel).exists():
                entries[rel] = e
        tpl = Path(__file__).parent / "gallery.html"
        sig = hashlib.sha1((json.dumps(entries, sort_keys=True) + str(tpl.stat().st_mtime)).encode()).hexdigest()
        if sig == m.get("page_sig") and (gdir / "index.html").exists():
            return m, 0
        m["entries"], m["page_sig"] = entries, sig
        m["updated"] = _utc_now()
        _write_gallery_file(gdir / "manifest.json", json.dumps(m, indent=1))
        write_gallery_page(gdir, m)
        _write_gitignore(gdir, m)
    return m, new


def gallery_archive(gdir, paths):
    """Copy files a run is about to overwrite into .history/ so earlier variants stay in the gallery."""
    gdir = Path(gdir).resolve()
    root = gdir.parent
    with _gallery_locked(gdir):
        m = load_gallery(gdir)
        changed = False
        for p in paths:
            p = Path(p).resolve()
            try:
                rel = _root_rel(p, root)
            except ValueError:
                continue
            e = m["entries"].get(rel)
            if not e or not p.is_file() or e.get("history"):
                continue
            stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
            sub = p.parent.relative_to(gdir) if gdir in p.parents else Path(rel).parent
            dest = gdir / ".history" / sub / f"{p.stem}@{stamp}{p.suffix}"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            h = dict(e, path=_root_rel(dest, root), history=True, replaced=True, label=f"{e['label']} · {stamp[4:8]} {stamp[9:13]}",
                     thumb=_gallery_thumb(gdir, dest, _root_rel(dest, root), dest.stat().st_mtime), kit=False)
            if h["state"] == "picked":
                h["state"] = "new"
            m["entries"][h["path"]] = h
            changed = True
        if changed:
            _write_gallery_file(gdir / "manifest.json", json.dumps(m, indent=1))


def _write_gitignore(gdir, m):
    """Commit the manifest and the picked and final files; leave out thumbnails, history, discards and unpicked variants."""
    ignored = [".thumbs/", ".history/", ".lock", "**/review/", "**/*_critique.png", "**/*_raw.png"]
    for rel, e in sorted(m["entries"].items()):
        if e.get("history") or not rel.startswith(GALLERY_DIR + "/"):
            continue
        if e["state"] == "discarded" or (e.get("choice") and e["state"] != "picked"):
            ignored.append("/" + rel[len(GALLERY_DIR) + 1:])
    begin, end = "# gptkit gallery: generated, edits between these lines are replaced", "# end gptkit gallery"
    f = gdir / ".gitignore"
    text = f.read_text() if f.exists() else ""
    if begin in text and end in text:
        head, _, rest = text.partition(begin)
        tail = rest.partition(end)[2]
    else:
        head, tail = text, ""
    block = "\n".join([begin, *ignored, end])
    _write_gallery_file(f, (head.rstrip("\n") + "\n\n" if head.strip() else "") + block + "\n" + tail.lstrip("\n"))


def write_gallery_page(gdir, m):
    tpl = (Path(__file__).parent / "gallery.html").read_text()
    data = json.dumps({"project": gdir.parent.name, "updated": m.get("updated"),
                       "entries": [{k: v for k, v in e.items() if not k.startswith("_")} for e in m["entries"].values()]},
                      separators=(",", ":"))
    data = data.replace("<", "\\u003c")
    page = tpl.replace("__TITLE__", html.escape(gdir.parent.name)).replace("/*__DATA__*/null", data)
    _write_gallery_file(gdir / "index.html", page)


def gallery_set(gdir, path, state=None, note=None):
    """Record a pick, discard or note. A forge candidate is picked through `pick`, so the kit's own record agrees."""
    gdir = Path(gdir).resolve()
    root = gdir.parent
    p = Path(path)
    p = (p if p.is_absolute() else (Path.cwd() / p)).resolve()
    if not p.exists():
        p = (root / path).resolve()
    try:
        rel = _root_rel(p, root)
    except ValueError:
        raise ValueError(f"{path} is outside the project {root}")
    m = load_gallery(gdir)
    e = m["entries"].get(rel)
    if e is None:
        gallery_index(gdir)
        m = load_gallery(gdir)
        e = m["entries"].get(rel)
        if e is None:
            raise ValueError(f"{rel} is not in the gallery (not an image the kit made, or left out as a review sheet)")
    if state not in (None, *STATES):
        raise ValueError(f"state must be one of {', '.join(STATES)}")
    if e.get("kit") and state is not None:
        if state == "picked":
            from .cli import main
            if e.get("scene_set"):
                slot, n = re.match(r"^(.+)_v(\d+)$", p.stem).groups()
                main(argv=["store", "scene", str(root / e["unit_path"]), "--set", e["scene_set"], "--pick", f"{slot}={n}"],
                     gallery=False)
            else:
                main(argv=["pick", str(root / e["unit_path"]), e["group"], str(p)], gallery=False)
        elif e["state"] == "picked":
            raise ValueError(f"{rel} is the kit's chosen one for {e['group']}: pick another first")
    with _gallery_locked(gdir):
        m = load_gallery(gdir)
        e = m["entries"][rel]
        if state is not None:
            if state == "picked" and e.get("choice"):
                for other in m["entries"].values():
                    if other is not e and other.get("unit") == e["unit"] and other.get("group") == e["group"] \
                            and other.get("state") == "picked":
                        other["state"], other["decided"] = "new", _utc_now()
            e["state"], e["decided"] = state, _utc_now()
        if note is not None:
            e["note"] = note.strip()
            e["decided"] = e.get("decided") or _utc_now()
        _write_gallery_file(gdir / "manifest.json", json.dumps(m, indent=1))
    return gallery_index(gdir)[0]["entries"][rel]


# ---- the command hook -----------------------------------------------------------------------------------------------

def _gallery_run_paths(a):
    vals = []
    for k in ("out", "dir", "mockup", "file", "concept", "edit", "style_ref", "anchor"):
        v = getattr(a, k, None)
        if isinstance(v, str):
            vals.append(v)
    vals += [v for v in (getattr(a, "svgs", None) or []) if isinstance(v, str)]
    return vals


def _gallery_overwrites(a, cmd):
    """Files this run will write over: kept in .history/ first."""
    out = Path(getattr(a, "out", None) or getattr(a, "dir", None) or ".")
    if cmd == "mockup" and not getattr(a, "set", None):
        if getattr(a, "exact_name", None):
            return [out / f"{a.exact_name}.png"]
        return [out / f"{a.name}_v{i + 1}.png" for i in range(a.variants)]
    if cmd == "appicon concepts":
        return [out / f"{a.name}_v{i + 1}.png" for i in range(a.variants)]
    if cmd == "forge" and getattr(a, "force", False):
        return sorted((out / "raw").glob("*_c*.png"))
    if cmd == "critique" and getattr(a, "apply", False):
        src = Path(a.mockup)
        qa_dir = Path(a.out) if getattr(a, "out", None) else src.parent
        return sorted(qa_dir.glob(f"{src.stem}_qa*.png"))
    if cmd == "material":
        return sorted(out.glob(f"{a.name}_c*.png"))
    return []


def before_command(a, cmd):
    """Find the run's gallery and keep the files it will overwrite. Never raises: the gallery must not fail a command."""
    if cmd.split()[0] in QUIET_CMDS:
        return None
    try:
        gdir = next((g for g in map(gallery_dir_of, _gallery_run_paths(a)) if g), None)
        if gdir is None or gallery_decision(gdir) == "off":
            return None
        if (gdir / "manifest.json").exists():
            gallery_archive(gdir, _gallery_overwrites(a, cmd))
        root = gdir.parent
        parent = next((getattr(a, k) for k in ("edit", "style_ref", "anchor", "mockup", "concept")
                       if isinstance(getattr(a, k, None), str)), None)
        try:
            parent = _root_rel(parent, root) if parent else None
        except ValueError:
            parent = str(parent)
        prompt = next((getattr(a, k) for k in ("prompt", "brief", "concepts", "note", "moving")
                       if isinstance(getattr(a, k, None), str)), None)
        scope = [str(Path(v).expanduser().resolve()) for v in (getattr(a, "out", None), getattr(a, "dir", None))
                 if isinstance(v, str)]
        if cmd == "critique" and not scope and isinstance(getattr(a, "mockup", None), str):
            scope = [str(Path(a.mockup).expanduser().resolve().parent)]
        before = {}
        for sc in scope:  # what the run's folder held before it: only files it writes or changes get its prompt
            if Path(sc).is_dir():
                for f in _gallery_walk(Path(sc)):
                    with contextlib.suppress(OSError):
                        before[str(f)] = f.stat().st_mtime
        return {"gdir": gdir, "cmd": cmd, "started": time.time(), "prompt": prompt, "parent": parent, "scope": scope,
                "before": before,
                "shipped": [(p, getattr(a, "res", None) or getattr(a, "out", None))
                            for p in ([a.file] if cmd == "export" else list(getattr(a, "svgs", []) or []) if cmd == "vd" else [])]}
    except Exception as e:  # noqa: BLE001
        print(f"gallery: skipped ({type(e).__name__}: {e})", file=sys.stderr)
        return None


def after_command(run):
    if not run:
        return
    try:
        gdir = run["gdir"]
        m, new = gallery_index(gdir, {k: run[k] for k in ("cmd", "started", "prompt", "parent", "scope", "before")})
        if run["shipped"]:
            root = gdir.parent
            with _gallery_locked(gdir):
                m = load_gallery(gdir)
                for src, dest in run["shipped"]:
                    srcs = sorted(Path(src).glob("*.svg")) if Path(src).is_dir() else [Path(src)]
                    for s in srcs:
                        with contextlib.suppress(ValueError, OSError):
                            e = m["entries"].get(_root_rel(s, root))
                            if e is None and s.resolve().parent.name == "assets":
                                unit = _root_rel(s.resolve().parent.parent, root)
                                e = next((x for x in m["entries"].values() if x.get("unit_path") == unit
                                          and x.get("group") == s.stem and x.get("state") == "picked"
                                          and not x.get("history")), None)
                            if e is not None and dest and str(dest) not in e.setdefault("shipped", []):
                                e["shipped"].append(str(dest))
                _write_gallery_file(gdir / "manifest.json", json.dumps(m, indent=1))
                write_gallery_page(gdir, m)
        page = gdir / "index.html"
        shown = os.path.relpath(page) if not os.path.relpath(page).startswith("..") else str(page)
        print(f"gallery: {shown} ({len(m['entries'])} images, {new} new)", file=sys.stderr)
        if gallery_decision(gdir) is None:
            print("gallery: undecided for this project. Ask the user whether to keep the asset dashboard "
                  "(recommended), then run `gallery on` or `gallery off`.", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"gallery: not updated ({type(e).__name__}: {e})", file=sys.stderr)


# ---- the gallery command --------------------------------------------------------------------------------------------

def _gallery_dir_arg(dir_arg):
    p = Path(dir_arg or ".").expanduser().resolve()
    return gallery_dir_of(p) or p / GALLERY_DIR


def cmd_gallery(a):
    gdir = _gallery_dir_arg(getattr(a, "dir", None))
    g = a.gcmd
    if g == "off":
        _set_gallery_decision(gdir, "off")
        for x in (gdir / "index.html", gdir / ".lock"):
            x.unlink(missing_ok=True)
        shutil.rmtree(gdir / ".thumbs", ignore_errors=True)
        print(f"gallery off for {gdir.parent} (outputs and picks in manifest.json are kept; `gallery on` restores the page)")
        return
    if g == "on":
        _set_gallery_decision(gdir, "on")
    if g == "add":
        root = gdir.parent
        src = Path(a.source).expanduser().resolve()
        try:
            rel = _root_rel(src, root)
        except ValueError:
            sys.exit(f"gptkit gallery: {src} is outside the project {root}")
        with _gallery_locked(gdir):
            m = load_gallery(gdir)
            if rel not in m["sources"] and not rel.startswith(GALLERY_DIR + "/"):
                m["sources"].append(rel)
            _write_gallery_file(gdir / "manifest.json", json.dumps(m, indent=1))
    if g in ("on", "index", "add"):
        if gallery_decision(gdir) == "off":
            sys.exit("gptkit gallery: the gallery is off for this project; `gallery on` turns it back on")
        m, new = gallery_index(gdir)
        print(f"gallery: {gdir / 'index.html'} ({len(m['entries'])} images, {new} new)")
        return
    if g == "set":
        e = gallery_set(gdir, a.path, a.state, a.note)
        print(f"{e['path']}: {e['state']}" + (f"  note: {e['note']}" if e.get("note") else ""))
        return
    if g == "picks":
        m = load_gallery(gdir)
        rows = sorted(m["entries"].values(), key=lambda e: e.get("decided") or "", reverse=True)
        if a.json:
            print(json.dumps([{k: e.get(k) for k in ("path", "state", "note", "kind", "unit", "group", "decided")}
                              for e in rows if e.get("decided")], indent=1))
            return
        for state, title in (("picked", "picked"), ("discarded", "discarded")):
            sel = [e for e in rows if e["state"] == state and e.get("decided")]
            if sel:
                print(title)
                for e in sel:
                    print(f"  {e['path']}  ({e['kind']} · {e['unit']} · {e['group']})"
                          + (f"  note: {e['note']}" if e.get("note") else ""))
        notes = [e for e in rows if e.get("note") and e["state"] == "new"]
        if notes:
            print("notes")
            for e in notes:
                print(f"  {e['path']}: {e['note']}")
        if not any(e.get("decided") for e in rows):
            print("no picks, discards or notes yet")
        return
    if g == "serve":
        from .gallery_server import serve
        if gallery_decision(gdir) == "off":
            sys.exit("gptkit gallery: the gallery is off for this project; `gallery on` turns it back on")
        gallery_index(gdir)
        serve(gdir, a.port, a.open)


def _gallery_parsers(sub):
    p = sub.add_parser("gallery", help="the ui-assets dashboard: every generated image with picks, discards and notes")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    for name, text in (("on", "turn the dashboard on for this project and index it"),
                       ("off", "turn it off (removes index.html and thumbnails; outputs and picks stay)"),
                       ("index", "rescan the project and rewrite ui-assets/index.html")):
        q = gsub.add_parser(name, help=text)
        q.add_argument("dir", nargs="?", help="project root or its ui-assets folder (default: the current directory)")
    q = gsub.add_parser("add", help="also index an output folder outside ui-assets/ (an older project layout)")
    q.add_argument("source"); q.add_argument("--dir", help="project root (default: the current directory)")
    q = gsub.add_parser("set", help="pick, discard or annotate an image (a forge candidate is picked through `pick`)")
    q.add_argument("path"); q.add_argument("state", nargs="?", choices=STATES)
    q.add_argument("--note", help="a note to keep with the image"); q.add_argument("--dir", help="project root")
    q = gsub.add_parser("picks", help="list the user's picks, discards and notes, newest first")
    q.add_argument("dir", nargs="?"); q.add_argument("--json", action="store_true")
    q = gsub.add_parser("serve", help="serve the dashboard on 127.0.0.1 so picks and notes made on the page are saved")
    q.add_argument("dir", nargs="?"); q.add_argument("--port", type=int, default=8765)
    q.add_argument("--open", action="store_true", help="open the page in the default browser")
    p.set_defaults(fn=cmd_gallery)
