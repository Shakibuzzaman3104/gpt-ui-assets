"""Looking at designs: labelled contact sheets and zoomed regions (sheet)."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .core import _need
from .imaging import checker

# ---- sheet: several images side by side, labelled, at a size one look can take in

SHEET_MAX = 2000          # long side an image viewer shows without downscaling again
SHEET_GAP, SHEET_LABEL = 16, 34
IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
SHEET_SKIP = re.compile(r"(_critique|_raw|_review|_qa\d+)\.\w+$|^(sheet|review|manifest_overlay)[._]")
REGIONS = {"top": (0, 0, 1, 1 / 3), "middle": (0, 1 / 3, 1, 1 / 3), "bottom": (0, 2 / 3, 1, 1 / 3),
           "upper": (0, 0, 1, 0.5), "lower": (0, 0.5, 1, 0.5)}


def _sheet_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _sheet_inputs(paths):
    """Image files of the given files and folders (a folder gives its own images, review files left out), in order."""
    out = []
    for raw in paths:
        p = Path(_need(raw, "image or folder"))
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_SUFFIXES and not SHEET_SKIP.search(f.name))
            if not found:
                raise ValueError(f"sheet: no images in {p}")
            out += found
        else:
            out.append(p)
    return out


def _sheet_region(spec):
    """--crop as fractions (x, y, w, h): a named band, or four numbers (fractions up to 1, else pixels, resolved later)."""
    if not spec:
        return None
    if spec in REGIONS:
        return REGIONS[spec]
    try:
        box = tuple(float(v) for v in spec.split(","))
    except ValueError:
        box = ()
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0 or min(box) < 0:
        raise ValueError(f"sheet: --crop is one of {', '.join(REGIONS)} or x,y,w,h (fractions of the image, or pixels)")
    return box


def _sheet_crop(img, box):
    if box is None:
        return img
    x, y, w, h = box
    if max(box) <= 1:
        x, y, w, h = x * img.width, y * img.height, w * img.width, h * img.height
    x, y = min(int(x), img.width - 1), min(int(y), img.height - 1)
    return img.crop((x, y, min(img.width, max(x + 1, int(round(x + w)))), min(img.height, max(y + 1, int(round(y + h))))))


def _sheet_note(path):
    """The design QA result the kit recorded for an image (issue count and weight), when there is one."""
    try:
        first = json.loads((path.parent / f"{path.stem}_critique.json").read_text())["rounds"][0]
        return f"{len(first['issues'])} issues w{first['weight']}"
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return ""


def _sheet_labels(files, given):
    if given:
        names = [s.strip() for s in given.split(",")]
        if len(names) != len(files):
            raise ValueError(f"sheet: --labels has {len(names)} names for {len(files)} images")
        return names
    stems = [f.stem for f in files]
    return [f"{f.parent.name}/{f.stem}" if stems.count(f.stem) > 1 or len({x.parent for x in files}) > 1 else f.stem
            for f in files]


def make_sheet(files, out, labels=None, cols=None, crop=None, bg="light", max_side=SHEET_MAX):
    """Write one labelled contact sheet of `files`; returns (path, [(letter, file)])."""
    files = [Path(f) for f in files]
    box = _sheet_region(crop) if isinstance(crop, str) else crop
    images = []
    for f in files:
        try:
            with Image.open(f) as im:
                images.append(_sheet_crop(im.convert("RGBA"), box))
        except (OSError, ValueError) as e:
            raise ValueError(f"sheet: cannot read {f}: {e}") from e
    names = _sheet_labels(files, labels)
    n = len(images)
    wide = sum(im.width >= im.height for im in images) > n / 2
    cols = max(1, min(n, cols or (2 if wide else 4 if n <= 8 else 5)))
    tile_w = (max_side - SHEET_GAP * (cols + 1)) // cols
    tile_w = min(tile_w, max(im.width for im in images))  # never upscale past the sharpest source
    scaled = [im.resize((tile_w, max(1, round(im.height * tile_w / im.width))), Image.LANCZOS) for im in images]
    rows = [scaled[i:i + cols] for i in range(0, n, cols)]
    heights = [max(im.height for im in row) for row in rows]
    W = SHEET_GAP + cols * (tile_w + SHEET_GAP)
    H = SHEET_GAP + sum(h + SHEET_LABEL + SHEET_GAP for h in heights)
    page, ink = {"dark": ("#16171a", "#f2f2f2"), "light": ("#f4f4f5", "#18181b"), "checker": ("#f4f4f5", "#18181b")}[bg]
    sheet = Image.new("RGB", (W, H), page)
    draw = ImageDraw.Draw(sheet)
    font = _sheet_font(20)
    index, y = [], SHEET_GAP
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            i = r * cols + c
            letter = chr(65 + i) if n <= 26 else str(i + 1)
            x = SHEET_GAP + c * (tile_w + SHEET_GAP)
            note = _sheet_note(files[i])
            text = f"{letter}  {names[i]}" + (f"  ({note})" if note else "")
            while len(text) > 4 and draw.textlength(text, font=font) > tile_w:
                text = text[:-2].rstrip() + "…"
            draw.text((x, y + 4), text, fill=ink, font=font)
            if bg == "checker":
                sheet.paste(checker(im.width, im.height), (x, y + SHEET_LABEL))
            sheet.paste(im, (x, y + SHEET_LABEL), im)
            index.append((letter, files[i]))
        y += heights[r] + SHEET_LABEL + SHEET_GAP
    if max(sheet.size) > max_side * 3:  # a very long sheet: keep it readable rather than unbounded
        sheet.thumbnail((max_side * 3, max_side * 3), Image.LANCZOS)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88) if out.suffix.lower() in (".jpg", ".jpeg") else sheet.save(out)
    return out, index


def cmd_sheet(a):
    files = _sheet_inputs(a.images)
    if a.out:
        out = Path(a.out)
    else:
        base = files[0].parent
        tag = re.sub(r"[^A-Za-z0-9]+", "-", a.crop).strip("-") if a.crop else "all"
        out = base / "review" / f"sheet_{files[0].stem}_{tag}.jpg"
    out, index = make_sheet(files, out, a.labels, a.cols, a.crop, a.bg, a.max)
    for letter, f in index:
        print(f"  {letter}  {f}")
    print(f"sheet -> {out}  ({len(index)} image{'s' if len(index) != 1 else ''}; Read it)")
