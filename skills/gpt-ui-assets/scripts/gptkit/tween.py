"""In-between frames for raster loops: built-in optical flow (OpenCV), RIFE (rife-ncnn-vulkan) or a crossfade."""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

TWEEN_ENGINES = ("flow", "rife", "blend")
RIFE_MODEL = "rife-v4.6"
RIFE_URL = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-{}.zip"
_RIFE_OS = "macos" if sys.platform == "darwin" else "windows" if os.name == "nt" else "ubuntu"
RIFE_INSTALL = (f"  mkdir -p ~/.cache/gptkit/rife && cd ~/.cache/gptkit/rife && curl -LO {RIFE_URL.format(_RIFE_OS)}\n"
                f"  unzip -j rife-ncnn-vulkan-20221029-{_RIFE_OS}.zip '*/rife-ncnn-vulkan*' && "
                f"unzip rife-ncnn-vulkan-20221029-{_RIFE_OS}.zip '*/{RIFE_MODEL}/*' && "
                f"mv rife-ncnn-vulkan-20221029-{_RIFE_OS}/{RIFE_MODEL} . && "
                f"rm -r rife-ncnn-vulkan-20221029-{_RIFE_OS} rife-ncnn-vulkan-20221029-{_RIFE_OS}.zip\n"
                "(MIT; the 436 MB archive holds every model, only the binary and rife-v4.6 (~35 MB) are kept)\n"
                "or pass --rife PATH / set GPTKIT_RIFE.")


def _premul(img):
    p = np.asarray(img.convert("RGBA"), np.float32) / 255
    p = p.copy()
    p[..., :3] *= p[..., 3:]
    return p


def _unpremul(p):
    p = np.clip(p, 0, 1)
    a = p[..., 3:]
    rgb = np.where(a > 1e-4, p[..., :3] / np.maximum(a, 1e-4), 0)
    return Image.fromarray(np.round(np.concatenate([np.clip(rgb, 0, 1), a], -1) * 255).astype(np.uint8), "RGBA")


def find_rife(path=None):
    """--rife, $GPTKIT_RIFE, PATH, then ~/.cache/gptkit/rife; a path named explicitly must exist (no silent fallback)."""
    for c, what in ((path, "--rife"), (os.environ.get("GPTKIT_RIFE"), "GPTKIT_RIFE")):
        if c and not Path(c).exists():
            sys.exit(f"gptkit: {what} {c} not found")
        if c:
            return c
    exe = "rife-ncnn-vulkan" + (".exe" if os.name == "nt" else "")
    for c in (shutil.which("rife-ncnn-vulkan"), str(Path.home() / ".cache" / "gptkit" / "rife" / exe)):
        if c and Path(c).exists():
            return c
    return None


def _rife_bin(path=None):
    rife = find_rife(path)
    if not rife:
        sys.exit("RIFE not found. Install it once (or use --tween-engine flow, the default):\n" + RIFE_INSTALL)
    model = Path(rife).resolve().parent / RIFE_MODEL
    if not (model / "flownet.param").exists():
        sys.exit(f"RIFE: {model} is missing; keep the {RIFE_MODEL} folder next to the binary:\n" + RIFE_INSTALL)
    return rife, model


def _feat(p):
    lum = p[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.clip((0.6 * lum + 0.4 * p[..., 3]) * 255, 0, 255).astype(np.uint8)


def _flow_pair(cv2, p0, p1, ts):
    """Bidirectional DIS flow, both keys warped to each t, blended by forward-backward consistency; where the two
    warps still disagree the nearer key wins (a soft switch), which keeps edges single instead of doubled."""
    h, w = p0.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    def warp(x, fl, mode):
        return cv2.remap(x, gx + fl[..., 0], gy + fl[..., 1], mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    a, b = _feat(p0), _feat(p1)
    f01, f10 = dis.calc(a, b, None), dis.calc(b, a, None)
    e0 = np.linalg.norm(f01 + warp(f10, f01, cv2.INTER_LINEAR), axis=-1)
    e1 = np.linalg.norm(f10 + warp(f01, f10, cv2.INTER_LINEAR), axis=-1)
    out = []
    for t in ts:
        ft0 = -(1 - t) * t * f01 + t * t * f10
        ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
        w0 = np.clip(warp(p0, ft0, cv2.INTER_CUBIC), 0, 1)
        w1 = np.clip(warp(p1, ft1, cv2.INTER_CUBIC), 0, 1)
        v0 = (1 - t) * (np.exp(-warp(e0, ft0, cv2.INTER_LINEAR) ** 2 / 4) + 1e-3)
        v1 = t * (np.exp(-warp(e1, ft1, cv2.INTER_LINEAR) ** 2 / 4) + 1e-3)
        s = v0 / (v0 + v1)
        k = np.clip((np.abs(w0 - w1).max(-1) - 0.08) / 0.25, 0, 1)
        s = ((1 - k) * s + k * (s > 0.5))[..., None]
        out.append(_unpremul(s * w0 + (1 - s) * w1))
    return out


def _rife_run(rife, model, jobs, workers=4):
    """jobs: (rgb0, rgb1, t) float images in 0..1 -> RGB results. One process per job (~0.4 s each on a GPU)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def one(i):
            a, b, t = jobs[i]
            for tag, x in (("a", a), ("b", b)):
                Image.fromarray(np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)).save(d / f"{tag}{i}.png")
            r = subprocess.run([rife, "-0", str(d / f"a{i}.png"), "-1", str(d / f"b{i}.png"), "-o", str(d / f"o{i}.png"),
                                "-m", str(model), "-s", f"{t:.6f}"], capture_output=True, text=True)
            if r.returncode or not (d / f"o{i}.png").exists():
                tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
                raise RuntimeError(f"rife-ncnn-vulkan failed: {tail[0]}")
            return np.asarray(Image.open(d / f"o{i}.png").convert("RGB"), np.float32) / 255
        with cf.ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, range(len(jobs))))


def _rife_pairs(rife, model, pairs, ts):
    """RIFE drops alpha, so each pair runs over black and over white and alpha is recovered from the difference."""
    jobs = []
    for p0, p1 in pairs:
        for t in ts:
            jobs += [(p0[..., :3], p1[..., :3], t), (p0[..., :3] + 1 - p0[..., 3:], p1[..., :3] + 1 - p1[..., 3:], t)]
    res = _rife_run(rife, model, jobs)
    out = []
    for j in range(0, len(res), 2):
        blk, wht = res[j], res[j + 1]
        a = np.clip(1 - (wht - blk).mean(-1, keepdims=True), 0, 1)
        out.append(_unpremul(np.concatenate([np.minimum(blk, a), a], -1)))
    return [out[i:i + len(ts)] for i in range(0, len(out), len(ts))]


def tween(frames, factor, engine="flow", rife=None):
    """A loop of n frames -> n * factor frames: the originals untouched, factor - 1 in-betweens after each, the last
    pair wrapping to the first. Returns (frames, engine used)."""
    if factor <= 1 or len(frames) < 2:
        return list(frames), None
    if len({f.size for f in frames}) > 1:
        raise ValueError("tween: frames differ in size")
    if engine == "flow":
        try:
            import cv2
        except ImportError:
            print("  ! OpenCV is not installed: in-betweens are crossfades, which double moving edges "
                  "(pip install opencv-python-headless for the flow engine)")
            engine = "blend"
    ts = [j / factor for j in range(1, factor)]
    n = len(frames)
    p = [_premul(f) for f in frames]
    pairs = [(p[i], p[(i + 1) % n]) for i in range(n)]
    if engine == "rife":
        mids = _rife_pairs(*_rife_bin(rife), pairs, ts)
    elif engine == "flow":
        mids = [_flow_pair(cv2, a, b, ts) for a, b in pairs]
    else:
        mids = [[_unpremul((1 - t) * a + t * b) for t in ts] for a, b in pairs]
    out = []
    for i in range(n):
        out += [frames[i]] + mids[i]
    return out, engine
