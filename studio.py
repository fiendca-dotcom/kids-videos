"""Nuni Studio v3 - picture-book renderer for the Nuni kids channel.

bot.py writes the script and the voice; this module turns them into a 1080p video:
  * painted backgrounds and objects in the Nuni picture-book style (Gemini image -> Pollinations -> emoji),
    cached in assets/lib so the library grows with every episode
  * Nuni from the book: blinks, talks, hops, waves
  * riddles / sound games as a silhouette game: clue icons pop in, big countdown, colour reveal with confetti
  * slow camera moves, crossfades, bouncing letters, original music with drums, sound effects
"""
import base64
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None

W, H, FPS = 1920, 1080, 30
SR = 44100
XFADE = 0.4  # crossfade into every scene (s)
VOICE_AT = 0.45  # the voice starts this long after a scene starts (s)
BGW, BGH = int(W * 1.12), int(H * 1.12)  # stored background size (room for camera moves)

STYLE = ("Soft painted 3D children's picture-book illustration, warm golden daylight, gentle pastel colours, "
         "cute rounded shapes, fluffy textures, rich detail, cosy and friendly mood")
PALETTE = [(255, 94, 98), (255, 160, 40), (255, 205, 50), (80, 195, 110), (60, 165, 255), (165, 110, 255)]
DEFAULT_PLACE = {"meadow": "sunny green meadow with flowers and soft rolling hills",
                 "forest": "friendly sunny forest clearing with big round trees",
                 "sea": "sunny sandy beach by a calm blue sea",
                 "night": "calm starry night sky over soft hills with a big moon"}
TEXTS = {
    "who": {"en": "Who am I?", "he": "מי אני?", "ar": "من أنا؟"},
    "bye": {"en": "Bye bye!", "he": "!להתראות", "ar": "مع السلامة!"},
}
FONT_CHAIN = {
    "en": ["/usr/share/fonts/truetype/noto/NotoSans-Black.ttf", "/usr/share/fonts/truetype/noto/NotoSans-ExtraBold.ttf",
           "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"],
    "he": ["/usr/share/fonts/truetype/noto/NotoSansHebrew-Black.ttf", "/usr/share/fonts/truetype/noto/NotoSansHebrew-Bold.ttf"],
    "ar": ["/usr/share/fonts/truetype/noto/NotoSansArabic-Black.ttf", "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"],
    "any": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


def log(*a):
    print(*a, flush=True)


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:60] or "x"


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_back(x):
    x = clamp(x)
    c = 1.7
    return max(0.0, 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def find_font(lang):
    for p in FONT_CHAIN.get(lang, []) + FONT_CHAIN["en"] + FONT_CHAIN["any"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no font")


# ======================================================================= pictures
def cover(img, w, h):
    r = max(w / img.width, h / img.height)
    im = img.resize((max(w, int(img.width * r + 0.5)), max(h, int(img.height * r + 0.5))), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def cutout(img):
    """Object painted on a white background -> RGBA cut-out (None if it does not look clean)."""
    if ndimage is None:
        return None
    im = img.convert("RGB")
    im.thumbnail((900, 900), Image.LANCZOS)
    a = np.asarray(im).astype(np.int16)
    mn, mx = a.min(2), a.max(2)
    bgish = (mn > 226) & ((mx - mn) < 24)
    lab, n = ndimage.label(bgish)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, border[border > 0])
    fg = ~bg
    lab2, n2 = ndimage.label(fg)
    if n2 == 0:
        return None
    sizes = ndimage.sum(fg, lab2, range(1, n2 + 1))
    keep = np.isin(lab2, 1 + np.where(sizes >= sizes.max() * 0.04)[0])
    keep = ndimage.binary_opening(keep, iterations=1)
    frac = keep.mean()
    if frac < 0.03 or frac > 0.88:
        return None
    alpha = ndimage.gaussian_filter(ndimage.binary_erosion(keep, iterations=1).astype(np.float32), 0.7)
    out = np.dstack([a.astype(np.uint8), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
    res = Image.fromarray(out, "RGBA")
    bb = res.getbbox()
    return res.crop(bb) if bb else None


_EMOJI = {}


def emoji_img(ch, size=None):
    """Noto colour emoji as RGBA (native ~136px; scaled if size given)."""
    if ch not in _EMOJI:
        f = ImageFont.truetype(EMOJI_FONT, 109)
        im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
        bb = im.getbbox()
        if not bb or bb[2] - bb[0] < 40:
            im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((10, 10), "⭐", font=f, embedded_color=True)
            bb = im.getbbox()
        _EMOJI[ch] = im.crop(bb)
    e = _EMOJI[ch]
    if size:
        r = size / max(e.size)
        e = e.resize((max(1, int(e.width * r)), max(1, int(e.height * r))), Image.LANCZOS)
        if r > 1.5:
            from PIL import ImageFilter
            e = e.filter(ImageFilter.UnsharpMask(radius=max(2, int(r)), percent=70, threshold=2))
    return e


def sticker(rgba, size, silhouette=False):
    """Cut-out -> sticker: thick white outline + soft shadow. size = longest side of the object."""
    im = rgba.copy()
    r = size / max(im.size)
    im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    b = max(8, int(size * 0.035))
    pad = b * 3
    w, h = im.size
    A = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    A[pad:pad + h, pad:pad + w] = np.asarray(im.split()[3]) / 255.0
    if ndimage is not None:
        dil = (ndimage.distance_transform_edt(A < 0.5) <= b).astype(np.float32)
        outline = ndimage.gaussian_filter(dil, 1.0)
        shadow = np.roll(ndimage.gaussian_filter(dil, b * 0.9), int(b * 0.8), axis=0) * 0.30
    else:
        outline, shadow = A, A * 0
    layer = np.zeros(A.shape + (4,), np.float32)
    layer[..., 3] = shadow
    # white outline over the shadow
    oa = outline
    layer[..., :3] = layer[..., :3] * (1 - oa[..., None]) + 255 * oa[..., None]
    layer[..., 3] = oa + layer[..., 3] * (1 - oa)
    layer[..., 3] *= 255
    base = Image.fromarray(np.clip(layer, 0, 255).astype(np.uint8), "RGBA")
    if silhouette:
        fill = Image.new("RGBA", im.size, (58, 48, 128, 255))
        fill.putalpha(im.split()[3])
        im = fill
    base.alpha_composite(im, (pad, pad))
    return base


class ArtLib:
    """Generates and caches painted pictures. Never raises: returns None when nothing works."""
    MODELS = ["gemini-3.1-flash-image", "gemini-2.5-flash-image", "gemini-3.1-flash-image-preview",
              "gemini-3.1-flash-lite-image"]

    def __init__(self, root, key, budget_s=1500):
        self.dir = Path(root) / "assets" / "lib"
        (self.dir / "bg").mkdir(parents=True, exist_ok=True)
        (self.dir / "items").mkdir(parents=True, exist_ok=True)
        self.key = key
        self.models = list(self.MODELS) if key else []
        self.fails = {}
        self.no_cfg = False
        self.poll_on = os.environ.get("NO_POLLINATIONS", "") == ""
        self.poll_fails = 0
        self.t0 = time.time()
        self.budget = budget_s
        self.last = 0.0
        self.stats = {"cache": 0, "gemini": 0, "pollinations": 0, "none": 0}

    def left(self):
        return self.budget - (time.time() - self.t0)

    def _gemini(self, prompt, aspect):
        while self.models and self.left() > 40:
            model = self.models[0]
            gap = (6.5 if self.stats["gemini"] else 0.5) - (time.time() - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.time()
            gen = {"responseModalities": ["TEXT", "IMAGE"]}
            if not self.no_cfg:
                gen["imageConfig"] = {"aspectRatio": aspect}
            body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.key}"
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read())
                for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    if "inlineData" in p:
                        im = Image.open(io.BytesIO(base64.b64decode(p["inlineData"]["data"])))
                        im.load()
                        return im.convert("RGB"), model
                log("  image: no picture from", model)
                return None
            except urllib.error.HTTPError as e:
                txt = e.read().decode(errors="ignore")
                log(f"  image {model} HTTP {e.code}: {txt[:150]}")
                k = self.fails.get(model, 0) + 1
                self.fails[model] = k
                if e.code == 400 and not self.no_cfg and ("imageConfig" in txt or "aspect" in txt.lower()):
                    self.no_cfg = True
                    continue
                quota_gone = ("PerDay" in txt or "limit: 0" in txt or "exceeded your current quota" in txt
                              or e.code in (400, 402, 403, 404))
                if quota_gone or k >= 3:
                    self.models.pop(0)
                else:
                    time.sleep(25 * k)
            except Exception as e:  # noqa: BLE001
                log(f"  image {model} error: {str(e)[:120]}")
                k = self.fails.get(model, 0) + 1
                self.fails[model] = k
                if k >= 2:
                    self.models.pop(0)
        return None

    def _pollinations(self, prompt, w, h, seed):
        if not self.poll_on or self.left() < 40:
            return None
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:900]) +
               f"?width={w}&height={h}&seed={seed}&nologo=true&model=flux")
        for _ in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nuni-kids-bot)"})
                with urllib.request.urlopen(req, timeout=150) as r:
                    data = r.read()
                im = Image.open(io.BytesIO(data))
                im.load()
                return im.convert("RGB")
            except urllib.error.HTTPError as e:
                log("  pollinations: HTTP", e.code)
                if e.code in (401, 402, 403):
                    self.poll_on = False
                    return None
                time.sleep(6)
            except Exception as e:  # noqa: BLE001
                log("  pollinations:", str(e)[:120])
                time.sleep(6)
        self.poll_fails += 1
        if self.poll_fails >= 3:
            self.poll_on = False
        return None

    def _make(self, prompt, aspect, w, h, seed):
        g = self._gemini(prompt, aspect)
        if g:
            self.stats["gemini"] += 1
            return g[0], g[1]
        p = self._pollinations(prompt, w, h, seed)
        if p is not None:
            self.stats["pollinations"] += 1
            return p, "pollinations"
        return None, None

    def background(self, place):
        path = self.dir / "bg" / f"{slug(place)}.jpg"
        if path.exists():
            self.stats["cache"] += 1
            return Image.open(path).convert("RGB")
        prompt = (f"{STYLE}. Wide landscape background scene: {place}. Scenery only - no animals, no people, no "
                  "characters, no text or letters. Keep the centre and lower part open and uncluttered.")
        im, src = self._make(prompt, "16:9", 1344, 768, abs(hash(place)) % 100000)
        if im is None:
            self.stats["none"] += 1
            return None
        im = cover(im, BGW, BGH)
        im.save(path, quality=88)
        log(f"  + background '{place}' ({src})")
        return im

    def item(self, name):
        path = self.dir / "items" / f"{slug(name)}.webp"
        if path.exists():
            self.stats["cache"] += 1
            return Image.open(path).convert("RGBA")
        prompt = (f"{STYLE}. One single {name}, cute and friendly, the whole thing fully visible and centred, "
                  "isolated on a plain pure white background. No shadow, no ground, no frame, no text, nothing else.")
        for attempt in range(2):
            im, src = self._make(prompt, "1:1", 768, 768, abs(hash(name)) % 100000 + attempt)
            if im is None:
                break
            cut = cutout(im)
            if cut is not None:
                cut.thumbnail((720, 720), Image.LANCZOS)
                cut.save(path, "WEBP", quality=90)
                log(f"  + item '{name}' ({src})")
                return cut
            log(f"  item '{name}': background not clean, retrying")
        self.stats["none"] += 1
        return None


# ======================================================================= procedural fallback backgrounds
def painted_fallback(kind, seed=0):
    rng = random.Random(f"{kind}{seed}")
    w, h = BGW, BGH
    g = np.linspace(0, 1, h)[:, None, None]
    if kind == "night":
        top, bot = np.array([24, 28, 78]), np.array([88, 70, 150])
    elif kind == "sea":
        top, bot = np.array([105, 185, 250]), np.array([215, 238, 255])
    else:
        top, bot = np.array([100, 185, 250]), np.array([225, 243, 255])
    arr = (top * (1 - g) + bot * g).repeat(w, axis=1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img, "RGBA")
    if kind == "night":
        for _ in range(160):
            x, y, s = rng.uniform(0, w), rng.uniform(0, h * 0.6), rng.uniform(1.5, 4)
            d.ellipse([x - s, y - s, x + s, y + s], fill=(255, 255, 220, rng.randint(120, 255)))
        d.ellipse([w * 0.78, h * 0.1, w * 0.78 + 150, h * 0.1 + 150], fill=(255, 248, 210))
        hills = [(40, 70, 90), (30, 55, 75)]
    else:
        sx, sy = w * 0.82, h * 0.16
        for rr, a in ((190, 40), (140, 70), (100, 255)):
            d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=(255, 225, 90, a))
        for _ in range(6):
            cx, cy, sc = rng.uniform(0, w), rng.uniform(h * 0.08, h * 0.35), rng.uniform(0.8, 1.6)
            for dx, dy, r0 in [(0, 0, 50), (55, -22, 62), (115, 0, 48), (55, 18, 55)]:
                r1 = r0 * sc
                d.ellipse([cx + dx * sc - r1, cy + dy * sc - r1, cx + dx * sc + r1, cy + dy * sc + r1], fill=(255, 255, 255, 230))
        hills = [(130, 210, 100), (100, 190, 85)]
    if kind == "sea":
        d.rectangle([0, h * 0.58, w, h], fill=(40, 150, 220))
        for row in range(6):
            y0 = h * 0.6 + row * 30
            d.line([(x, y0 + 6 * math.sin(x / 45 + row)) for x in range(0, w + 20, 20)], fill=(255, 255, 255, 110), width=4)
        d.ellipse([-200, h * 0.8, w * 0.65, h * 1.35], fill=(250, 222, 160))
    else:
        d.ellipse([-300, h * 0.6, w * 0.7, h * 1.6], fill=hills[0])
        d.ellipse([w * 0.3, h * 0.66, w + 350, h * 1.6], fill=hills[1])
        if kind != "night":
            for _ in range(70):
                x, y = rng.uniform(0, w), rng.uniform(h * 0.75, h * 0.99)
                c = rng.choice([(255, 120, 160), (255, 225, 80), (255, 255, 255), (190, 140, 255)])
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=c)
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 200, 40))
        if kind == "forest":
            for x in [90, 300, w - 300, w - 100]:
                d.rectangle([x - 22, h * 0.42, x + 22, h * 0.8], fill=(130, 85, 45))
                for dx, dy, rr in [(0, -50, 120), (-70, 10, 95), (70, 10, 95)]:
                    d.ellipse([x + dx - rr, h * 0.4 + dy - rr, x + dx + rr, h * 0.4 + dy + rr], fill=(60, 155, 75))
    return img


def vignette(img):
    v = getattr(vignette, "mask", None)
    if v is None:
        yy, xx = np.mgrid[0:BGH, 0:BGW]
        r = np.sqrt(((xx - BGW / 2) / (BGW / 2)) ** 2 + ((yy - BGH / 2) / (BGH / 2)) ** 2)
        v = vignette.mask = np.clip(1 - 0.22 * np.clip(r - 0.55, 0, 1) ** 1.5, 0, 1)[..., None]
    return Image.fromarray((np.asarray(img).astype(np.float32) * v).astype(np.uint8))


# ======================================================================= layered painted scenery (no API needed)
LW = int(W * 1.32)  # layer width (room for parallax)
SPEEDS = (0.12, 0.3, 0.6, 1.0)  # sky, far, mid, near


def place_kind(place, bg):
    p = (place or "").lower()
    rules = [("underwater", ("underwater", "under the sea", "coral", "ocean floor", "sea bed", "seabed", "reef")),
             ("snow", ("snow", "winter", "ice", "arctic", "north pole", "frozen", "igloo")),
             ("sunset", ("sunset", "evening", "dusk")),
             ("night", ("night", "moon", "star", "bedtime", "space", "planet", "dark")),
             ("sea", ("beach", "sea", "ocean", "shore", "lake", "pond", "river", "island", "harbor", "harbour")),
             ("forest", ("forest", "woods", "jungle", "tree", "park", "garden", "orchard"))]
    for kind, keys in rules:
        if any(k in p for k in keys):
            return kind
    return bg if bg in ("forest", "sea", "night") else "meadow"


def _grad(h, w, top, bot, y0=0, y1=None):
    y1 = h if y1 is None else y1
    t = np.clip((np.arange(h) - y0) / max(1, y1 - y0), 0, 1)[:, None, None]
    return (np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t).repeat(w, axis=1)


def _curve(w, base, amps, rng):
    x = np.arange(w, dtype=np.float32)
    y = np.full(w, base, np.float32)
    for amp, per in amps:
        y += amp * np.sin(2 * np.pi * x / per + rng.uniform(0, 6.28))
    return y


def _band(w, h, curve, top, bot, rim=None, rimw=7, blur=0.0):
    Y = np.arange(h, dtype=np.float32)[:, None]
    mask = Y >= curve[None, :]
    t = np.clip((Y - curve[None, :]) / np.maximum(1, h - curve[None, :]), 0, 1)[..., None]
    rgb = np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t
    if rim is not None:
        edge = np.clip(1 - (Y - curve[None, :]) / rimw, 0, 1)[..., None] * mask[..., None]
        rgb = rgb * (1 - edge * 0.6) + np.array(rim, np.float32) * edge * 0.6
    a = mask.astype(np.float32)
    if ndimage is not None:
        a = ndimage.gaussian_filter(a, 0.8 + blur)
    out = np.dstack([np.clip(rgb, 0, 255), a[..., None] * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _tree(d, x, y, s, leaf, dark, light, trunk=(120, 80, 45)):
    d.rounded_rectangle([x - 9 * s, y - 70 * s, x + 9 * s, y + 4], int(5 * s), fill=trunk)
    for dx, dy, r in ((0, -120, 62), (-45, -88, 48), (45, -88, 48), (0, -80, 50)):
        d.ellipse([x + dx * s - r * s + 6 * s, y + dy * s - r * s + 8 * s, x + dx * s + r * s + 6 * s, y + dy * s + r * s + 8 * s], fill=dark)
    for dx, dy, r in ((0, -120, 62), (-45, -88, 48), (45, -88, 48), (0, -80, 50)):
        d.ellipse([x + dx * s - r * s, y + dy * s - r * s, x + dx * s + r * s, y + dy * s + r * s], fill=leaf)
    for dx, dy, r in ((-18, -140, 26), (-55, -100, 18), (28, -105, 16)):
        d.ellipse([x + dx * s - r * s, y + dy * s - r * s, x + dx * s + r * s, y + dy * s + r * s], fill=light)


def _pine(d, x, y, s, leaf, dark, snow=False):
    d.rectangle([x - 7 * s, y - 30 * s, x + 7 * s, y + 2], fill=(110, 75, 50))
    for k in range(4):
        top = y - 40 * s - k * 38 * s
        wd = (70 - k * 13) * s
        d.polygon([(x, top - 60 * s), (x - wd, top + 20 * s), (x + wd, top + 20 * s)], fill=dark if k % 2 else leaf)
        if snow:
            d.polygon([(x, top - 60 * s), (x - wd * 0.45, top - 20 * s), (x + wd * 0.45, top - 20 * s)], fill=(250, 252, 255))


def _flower(d, x, y, r, col):
    for k in range(5):
        a = k * 2 * math.pi / 5
        d.ellipse([x + math.cos(a) * r - r * 0.7, y + math.sin(a) * r - r * 0.7, x + math.cos(a) * r + r * 0.7, y + math.sin(a) * r + r * 0.7], fill=col)
    d.ellipse([x - r * 0.6, y - r * 0.6, x + r * 0.6, y + r * 0.6], fill=(255, 205, 60))


def _soft(img, r):
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(r))


def _clouds(w, h, rng, n=7, alpha=235, ymax=0.4):
    L = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(L)
    for _ in range(n):
        cx, cy, sc = rng.uniform(0, w), rng.uniform(h * 0.06, h * ymax), rng.uniform(0.9, 1.9)
        for dx, dy, r0 in [(0, 0, 50), (55, -24, 64), (118, 0, 50), (58, 16, 58), (-40, 12, 40)]:
            r1 = r0 * sc
            d.ellipse([cx + dx * sc - r1, cy + dy * sc - r1, cx + dx * sc + r1, cy + dy * sc + r1], fill=(255, 255, 255, alpha))
    return _soft(L, 3)


def _glow(arr, cx, cy, r, col, strength=1.0):
    h, w = arr.shape[:2]
    yy, xx = np.ogrid[0:h, 0:w]
    dd = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    g = np.clip(1 - dd, 0, 1) ** 2 * strength
    arr[...] = arr * (1 - g[..., None]) + np.array(col, np.float32) * g[..., None]


_SCENERY = {}


def scenery(kind):
    if kind in _SCENERY:
        return _SCENERY[kind]
    rng = random.Random(kind)
    w, h = LW, H
    skies = {"meadow": ((90, 170, 245), (205, 235, 255)), "forest": ((100, 180, 240), (215, 240, 250)),
             "sea": ((80, 165, 245), (215, 238, 255)), "night": ((14, 18, 60), (70, 60, 140)),
             "snow": ((150, 195, 240), (235, 245, 255)), "underwater": ((40, 150, 200), (10, 60, 120)),
             "sunset": ((90, 80, 170), (255, 170, 110))}
    top, bot = skies.get(kind, skies["meadow"])
    sky = _grad(h, w, top, bot, 0, int(h * (0.95 if kind != "underwater" else 1.0)))
    if kind in ("meadow", "forest", "sea", "snow"):
        _glow(sky, w * 0.78, h * 0.16, 420, (255, 250, 220), 0.8)
        _glow(sky, w * 0.78, h * 0.16, 95, (255, 236, 150), 1.0)
    elif kind == "sunset":
        _glow(sky, w * 0.5, h * 0.62, 700, (255, 210, 150), 0.9)
        _glow(sky, w * 0.5, h * 0.62, 130, (255, 225, 120), 1.0)
    elif kind == "night":
        _glow(sky, w * 0.8, h * 0.17, 300, (120, 120, 190), 0.6)
        _glow(sky, w * 0.8, h * 0.17, 70, (255, 250, 220), 1.0)
    skyimg = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(skyimg, "RGBA")
    if kind == "night":
        mx, my = w * 0.8, h * 0.17
        d.ellipse([mx - 55, my - 55, mx + 55, my + 55], fill=(255, 248, 215))
        d.ellipse([mx - 30, my - 70, mx + 75, my + 35], fill=tuple(int(v) for v in sky[int(my - 20), int(mx + 30)]))
        for _ in range(260):
            x, y, s = rng.uniform(0, w), rng.uniform(0, h * 0.62), rng.uniform(1, 3.2)
            d.ellipse([x - s, y - s, x + s, y + s], fill=(255, 255, 230, rng.randint(110, 255)))
    elif kind == "underwater":
        rays = Image.new("RGBA", (w, h), (200, 240, 255, 0))
        dr = ImageDraw.Draw(rays)
        for _ in range(9):
            x = rng.uniform(0, w)
            dr.polygon([(x, 0), (x + rng.uniform(60, 140), 0), (x + rng.uniform(200, 420), h), (x + rng.uniform(-80, 120), h)],
                       fill=(200, 240, 255, rng.randint(18, 40)))
        skyimg.paste(_soft(rays, 14), (0, 0), _soft(rays, 14))
    else:
        cl = _clouds(w, h, rng, 8, 225 if kind != "sunset" else 150)
        skyimg.paste(cl, (0, 0), cl)
    # far / mid / near
    pal = {
        "meadow": [((150, 200, 170), (175, 215, 185)), ((110, 195, 90), (70, 160, 70)), ((95, 180, 70), (55, 135, 50))],
        "forest": [((95, 150, 120), (120, 170, 140)), ((75, 160, 80), (45, 120, 55)), ((80, 165, 65), (45, 120, 45))],
        "sea": [((70, 150, 220), (90, 175, 230)), ((40, 140, 210), (30, 110, 190)), ((245, 220, 160), (225, 195, 135))],
        "night": [((45, 55, 105), (55, 60, 115)), ((30, 70, 85), (20, 50, 65)), ((25, 60, 60), (15, 40, 45))],
        "snow": [((190, 210, 240), (215, 228, 250)), ((235, 242, 252), (200, 215, 240)), ((248, 250, 255), (215, 228, 245))],
        "underwater": [((25, 90, 140), (20, 70, 120)), ((30, 110, 130), (20, 80, 105)), ((230, 205, 150), (200, 170, 115))],
        "sunset": [((130, 90, 160), (110, 80, 150)), ((95, 70, 130), (70, 55, 110)), ((70, 120, 80), (45, 90, 60))],
    }[kind]
    if kind == "sea":
        far = _band(w, h, np.full(w, h * 0.56, np.float32), *pal[0], rim=(235, 248, 255), rimw=5)
        fd = ImageDraw.Draw(far, "RGBA")
        for _ in range(140):
            x, y = rng.uniform(0, w), rng.uniform(h * 0.58, h * 0.8)
            fd.line([(x, y), (x + rng.uniform(15, 50), y)], fill=(255, 255, 255, rng.randint(60, 140)), width=3)
    else:
        far = _band(w, h, _curve(w, h * 0.55, [(38, 900), (18, 380)], rng), *pal[0], rim=(240, 245, 255), blur=1.5)
        if kind == "forest":
            fd = ImageDraw.Draw(far)
            for x in range(0, w, 70):
                y = h * 0.56 + rng.uniform(-10, 10)
                r = rng.uniform(45, 70)
                fd.ellipse([x - r, y - r, x + r, y + r], fill=pal[0][0])
            far = _soft(far, 1.5)
    mid_curve = _curve(w, h * (0.68 if kind != "sea" else 0.9), [(45, 1100), (22, 450)], rng)
    if kind == "sea":
        mid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    else:
        mid = _band(w, h, mid_curve, *pal[1], rim=(255, 255, 230) if kind not in ("night", "underwater") else None)
    md = ImageDraw.Draw(mid)
    if kind in ("meadow", "forest", "sunset"):
        n = 9 if kind == "forest" else 4
        xs = sorted(rng.uniform(0, w) for _ in range(n))
        for x in xs:
            s = rng.uniform(0.9, 1.5) if kind == "forest" else rng.uniform(0.8, 1.2)
            leaf = (70, 165, 75) if kind != "sunset" else (70, 90, 110)
            _tree(md, x, mid_curve[int(min(w - 1, x))] + 10, s, leaf, (45, 120, 55) if kind != "sunset" else (50, 65, 90),
                  (130, 210, 110) if kind != "sunset" else (110, 120, 150))
    elif kind == "snow":
        for x in sorted(rng.uniform(0, w) for _ in range(8)):
            _pine(md, x, mid_curve[int(min(w - 1, x))] + 10, rng.uniform(0.8, 1.3), (50, 120, 90), (35, 95, 70), snow=True)
    elif kind == "night":
        for x in sorted(rng.uniform(0, w) for _ in range(4)):
            _tree(md, x, mid_curve[int(min(w - 1, x))] + 10, rng.uniform(0.9, 1.2), (30, 75, 80), (20, 55, 60), (45, 95, 100))
    elif kind == "underwater":
        for _ in range(16):
            x = rng.uniform(0, w)
            y0 = mid_curve[int(min(w - 1, x))] + 20
            ln, col = rng.uniform(140, 320), rng.choice([(60, 170, 90), (40, 140, 80), (90, 190, 110)])
            pts = [(x + 14 * math.sin(k / 3), y0 - k * ln / 12) for k in range(13)]
            md.line(pts, fill=col, width=int(rng.uniform(10, 18)), joint="curve")
        for _ in range(10):
            x = rng.uniform(0, w)
            y0 = mid_curve[int(min(w - 1, x))] + 10
            col = rng.choice([(255, 120, 140), (255, 170, 90), (240, 110, 200)])
            for k in range(5):
                r = rng.uniform(14, 26)
                xx, yy = x + rng.uniform(-35, 35), y0 - rng.uniform(0, 50)
                md.ellipse([xx - r, yy - r, xx + r, yy + r], fill=col)
    near_curve = _curve(w, h * 0.86, [(16, 800), (8, 260)], rng)
    if kind == "sea":
        near_curve = _curve(w, h * 0.8, [(30, 1400), (10, 400)], rng)
    near = _band(w, h, near_curve, *pal[2], rim=(255, 255, 240) if kind not in ("night", "underwater") else None, rimw=10)
    nd = ImageDraw.Draw(near)
    if kind in ("meadow", "forest", "sunset", "night"):
        gcol = [(70, 160, 55), (100, 190, 70), (50, 130, 45)] if kind != "night" else [(30, 75, 70), (40, 90, 85)]
        for x in range(0, w, 7):
            y = near_curve[x] + 4
            hgt = rng.uniform(14, 34)
            nd.polygon([(x - 4, y + 6), (x + rng.uniform(-6, 6), y - hgt), (x + 4, y + 6)], fill=rng.choice(gcol))
        if kind != "night":
            for _ in range(90):
                x = rng.uniform(0, w)
                y = rng.uniform(near_curve[int(min(w - 1, x))] + 18, h - 10)
                _flower(nd, x, y, rng.uniform(5, 9), rng.choice([(255, 120, 160), (255, 255, 255), (190, 140, 255), (255, 200, 80)]))
    elif kind == "sea":
        for _ in range(40):
            x = rng.uniform(0, w)
            y = rng.uniform(near_curve[int(min(w - 1, x))] + 25, h - 10)
            r = rng.uniform(4, 9)
            nd.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=rng.choice([(255, 240, 220), (250, 200, 190), (235, 215, 180)]))
        for _ in range(10):
            x = rng.uniform(0, w)
            y = near_curve[int(min(w - 1, x))] + 6
            for k in range(7):
                nd.line([(x, y), (x + rng.uniform(-30, 30), y - rng.uniform(25, 55))], fill=(110, 170, 80), width=4)
    elif kind == "underwater":
        for _ in range(30):
            x = rng.uniform(0, w)
            y = rng.uniform(near_curve[int(min(w - 1, x))] + 20, h - 10)
            r = rng.uniform(5, 12)
            nd.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=rng.choice([(255, 230, 210), (240, 200, 160), (255, 190, 200)]))
    elif kind == "snow":
        for _ in range(6):
            x = rng.uniform(0, w)
            _pine(nd, x, near_curve[int(min(w - 1, x))] + 60, rng.uniform(1.2, 1.6), (45, 115, 85), (30, 90, 65), snow=True)
    out = {"kind": kind, "sky": skyimg, "far": far, "mid": mid, "near": near}
    _SCENERY[kind] = out
    return out


def particles(img, kind, gt, seed=0):
    if kind not in ("night", "snow", "underwater", "meadow", "sunset"):
        return
    d = ImageDraw.Draw(img, "RGBA")
    rng = random.Random(f"p{kind}")
    for k in range(40 if kind == "snow" else 22):
        x0, y0, sp, ph = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.4, 1.0), rng.uniform(0, 6.28)
        if kind == "snow":
            y = (y0 + gt * 60 * sp) % (H + 20) - 10
            x = x0 + math.sin(gt * 0.8 + ph) * 25
            r = 3 + 4 * sp
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 220))
        elif kind == "underwater":
            y = H - (y0 + gt * 70 * sp) % (H + 30)
            x = x0 + math.sin(gt * 1.5 + ph) * 12
            r = 5 + 9 * sp
            d.ellipse([x - r, y - r, x + r, y + r], outline=(220, 245, 255, 170), width=3)
            d.ellipse([x - r * 0.45, y - r * 0.6, x - r * 0.1, y - r * 0.25], fill=(255, 255, 255, 170))
        elif kind == "night":
            x = x0 + math.sin(gt * 0.5 + ph) * 40
            y = H * 0.55 + (y0 % (H * 0.4)) + math.cos(gt * 0.7 + ph) * 25
            a = int(120 + 120 * math.sin(gt * 3 + ph))
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 240, 120, max(0, a // 3)))
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 250, 180, max(0, a)))
        else:
            x = (x0 + gt * 18 * sp) % (W + 20) - 10
            y = y0 * 0.7 + math.sin(gt * 1.2 + ph) * 20
            a = int(90 + 80 * math.sin(gt * 2 + ph))
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 230, max(0, a)))


def layers_frame(S, t):
    L = S["layers"]
    cam = S["cam"]
    u = clamp((S["start"] + t - cam["t0"]) / max(0.1, cam["len"]))
    span = (LW - W)
    pos = cam["p0"] + (cam["p1"] - cam["p0"]) * u  # 0..1 across the layer
    img = None
    for name, sp in zip(("sky", "far", "mid", "near"), SPEEDS):
        ox = int(span * (0.5 + (pos - 0.5) * sp))
        layer = L[name]
        crop = layer.crop((ox, 0, ox + W, H))
        if img is None:
            img = crop.convert("RGB") if crop.mode != "RGB" else crop.copy()
        else:
            img.paste(crop, (0, 0), crop)
    particles(img, L["kind"], S["start"] + t)
    return img


# ======================================================================= Nuni
class Nuni:
    def __init__(self, root):
        p = Path(root) / "assets"
        self.src = {}
        for k in ("idle", "talk", "happy"):
            for ext in ("png", "webp"):
                f = p / f"nuni_{k}.{ext}"
                if f.exists():
                    self.src[k] = Image.open(f).convert("RGBA")
                    break
        if "idle" not in self.src:
            owl = emoji_img("🦉", 400)
            self.src = {"idle": owl, "talk": owl, "happy": owl}
        for k in ("talk", "happy"):
            self.src.setdefault(k, self.src["idle"])
        self._face()
        self.cache = {}

    def _face(self):
        im = self.src["idle"]
        self.levels = 6
        for k in range(self.levels):
            self.src[f"m{k}"] = im
            self.src[f"b{k}"] = im
        if ndimage is None:
            return
        a = np.asarray(im).astype(np.int16)
        h, w = a.shape[:2]
        r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        beak = (r > 185) & (g > 105) & (g < 205) & (b < 120) & (r - b > 90) & (al > 200)
        beak[int(h * 0.55):] = False
        beak[: int(h * 0.15)] = False
        if beak.sum() < 30:
            return
        lab, n = ndimage.label(beak)
        big = 1 + int(np.argmax(ndimage.sum(beak, lab, range(1, n + 1))))
        beak = ndimage.binary_dilation(ndimage.binary_closing(lab == big, iterations=2), iterations=1)
        ys, xs = np.nonzero(beak)
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh, bcx = bx1 - bx0, by1 - by0, (bx0 + bx1) / 2
        bk = a[beak][:, :3]
        light = tuple(int(v) for v in np.percentile(bk, 80, axis=0))
        darkc = tuple(int(v) for v in np.percentile(bk, 15, axis=0))
        lay = a.copy().astype(np.uint8)
        lay[..., 3] = (np.minimum(ndimage.gaussian_filter(beak.astype(np.float32), 0.6), 1) * a[..., 3]).astype(np.uint8)
        beak_layer = Image.fromarray(lay, "RGBA")
        from PIL import ImageFilter

        def mouth(base, o):
            if o <= 0.02:
                return base
            out = base.copy()
            L = Image.new("RGBA", base.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(L)
            cw = bw * (0.3 + 0.12 * o)
            top, bot = by1 - bh * 0.5, by1 + o * bh * 0.88
            d.ellipse([bcx - cw, top, bcx + cw, bot], fill=(88, 26, 36, 255))
            if o > 0.3:
                tw, th = cw * 0.62, (bot - top) * 0.38
                d.ellipse([bcx - tw, bot - th - bh * 0.04, bcx + tw, bot + bh * 0.02], fill=(228, 108, 124, 255))
            jw, jh = cw * 0.95, bh * 0.2
            d.chord([bcx - jw, bot - jh * 1.6, bcx + jw, bot + jh * 0.6], 0, 180, fill=darkc + (255,))
            d.chord([bcx - jw * 0.9, bot - jh * 1.6, bcx + jw * 0.9, bot + jh * 0.35], 0, 180, fill=light + (255,))
            out.alpha_composite(L.filter(ImageFilter.GaussianBlur(0.7)))
            out.alpha_composite(beak_layer)
            return out
        for k in range(self.levels):
            self.src[f"m{k}"] = mouth(im, k / (self.levels - 1))
        # eyes = the two biggest very dark blobs above the beak
        dark = (r < 75) & (g < 65) & (b < 65) & (al > 200)
        dark[int(by1):] = False
        lab, n = ndimage.label(dark)
        if n < 2:
            return
        sizes = ndimage.sum(dark, lab, range(1, n + 1))
        eyes = []
        for o in np.argsort(sizes)[::-1][:2]:
            yy, xx = np.nonzero(lab == o + 1)
            eyes.append((xx.mean(), yy.mean(), math.sqrt(len(xx) / math.pi)))
        yy, xx = np.mgrid[0:h, 0:w]
        lid = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lid)
        for cx, cy, rad in eyes:
            R = rad * 2.5
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            ring = (rr > R * 1.15) & (rr < R * 1.45) & (al > 200) & (r + g + b > 520)
            col = tuple(int(v) for v in np.median(a[ring][:, :3], axis=0)) if ring.sum() > 20 else (236, 222, 200)
            d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=col + (255,))
            d.arc([cx - R * 0.8, cy - R * 0.7, cx + R * 0.8, cy + R * 0.45], 20, 160, fill=(80, 52, 45, 255),
                  width=max(3, int(R * 0.14)))
        lid = lid.filter(ImageFilter.GaussianBlur(1.2))
        for k in range(self.levels):
            m = self.src[f"m{k}"].copy()
            m.alpha_composite(lid)
            self.src[f"b{k}"] = m

    def sprite(self, key, height, tilt=0):
        q = max(8, int(height / 4) * 4)
        tq = int(round(tilt))
        k = (key, q, tq)
        sp = self.cache.get(k)
        if sp is None:
            s = self.src[key]
            r = q / s.height
            sp = s.resize((max(1, int(s.width * r)), q), Image.LANCZOS)
            if tq:
                sp = sp.rotate(tq, resample=Image.BICUBIC, expand=True)
            if len(self.cache) > 600:
                self.cache.clear()
            self.cache[k] = sp
        return sp

    def draw(self, img, cx, foot, height, t, pose="idle", mouth=0, hop=0.0, tilt=0.0):
        if pose == "idle":
            blink = (t % 3.9) < 0.12 or (t % 7.3) < 0.1
            key = f"{'b' if blink else 'm'}{int(mouth)}"
        else:
            key = pose
        breathe = 1 + 0.012 * math.sin(t * 2.3)
        sp = self.sprite(key, height * breathe, tilt)
        d = ImageDraw.Draw(img, "RGBA")
        sw = sp.width * 0.42 * (1 - 0.35 * clamp(hop / 120))
        d.ellipse([cx - sw, foot - sw * 0.13, cx + sw, foot + sw * 0.13], fill=(30, 40, 20, 60))
        img.paste(sp, (int(cx - sp.width / 2), int(foot - sp.height + sp.height * 0.02 - hop)), sp)


# ======================================================================= letters
class Letters:
    def __init__(self, lang):
        self.lang = lang
        self.path = find_font(lang)
        self.fonts = {}
        self.split_letters = lang not in ("he", "ar")

    def font(self, size):
        size = int(size)
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.path, size)
        return self.fonts[size]

    def glyph(self, txt, size, fill):
        f = self.font(size)
        sw = max(4, int(size * 0.09))
        l, t, r, b = f.getbbox(txt, stroke_width=sw)
        pad = int(size * 0.12)
        im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        off = int(size * 0.05)
        d.text((pad - l + off, pad - t + off), txt, font=f, fill=(45, 30, 90, 150), stroke_width=sw, stroke_fill=(45, 30, 90, 150))
        d.text((pad - l, pad - t), txt, font=f, fill=fill, stroke_width=sw, stroke_fill=(255, 255, 255))
        d.text((pad - l, pad - t), txt, font=f, fill=fill)
        return im, -l + pad, -t + pad

    def line(self, text, size, maxw, palette=True, color=(255, 255, 255)):
        """-> list of (image, cx_offset_from_line_centre) and the final size."""
        text = text.strip()
        while size > 30 and self.font(size).getlength(text) > maxw:
            size *= 0.92
        size = int(size)
        f = self.font(size)
        total = f.getlength(text)
        out = []
        if not self.split_letters:
            im, ox, _ = self.glyph(text, size, color)
            return [(im, 0.0)], size
        ci = 0
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            x0 = f.getlength(text[:i])
            cw = f.getlength(ch)
            col = PALETTE[ci % len(PALETTE)] if palette else color
            ci += 1
            im, ox, _ = self.glyph(ch, size, col)
            out.append((im, x0 + cw / 2 - total / 2))
        return out, size


def paste_center(img, sp, cx, cy):
    img.paste(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)), sp)


_SCALED = {}


def scaled(key, im, s):
    q = round(s * 50) / 50
    if q <= 0.02:
        return None
    k = (key, q)
    r = _SCALED.get(k)
    if r is None:
        if len(_SCALED) > 3000:
            _SCALED.clear()
        r = im if q == 1 else im.resize((max(1, int(im.width * q)), max(1, int(im.height * q))), Image.BILINEAR)
        _SCALED[k] = r
    return r


def draw_letters(img, key, letters, cx, cy, t, start=0.0, stagger=0.045, wave=7):
    for i, (im, dx) in enumerate(letters):
        s = ease_back((t - start - i * stagger) / 0.35)
        sp = scaled((key, i), im, s)
        if sp is None:
            continue
        y = cy + math.sin(t * 3.2 + i * 0.7) * wave
        paste_center(img, sp, cx + dx, y)


def star(d, x, y, s, fill):
    d.polygon([(x, y - s), (x + s * 0.28, y - s * 0.28), (x + s, y), (x + s * 0.28, y + s * 0.28), (x, y + s),
               (x - s * 0.28, y + s * 0.28), (x - s, y), (x - s * 0.28, y - s * 0.28)], fill=fill)


def badge(content, r=82, ring=(255, 160, 40)):
    size = r * 2 + 30
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size / 2
    d.ellipse([c - r + 4, c - r + 10, c + r + 4, c + r + 10], fill=(40, 30, 80, 70))
    d.ellipse([c - r, c - r, c + r, c + r], fill=(255, 255, 255), outline=ring, width=9)
    e = content
    paste_center(im, e, c, c)
    return im


# ======================================================================= audio
def _add(buf, start, sig, vol=1.0):
    a = int(start * SR)
    if a >= len(buf) or a < 0:
        return
    seg = sig[: len(buf) - a]
    buf[a:a + len(seg)] += seg * vol


def tone(f, dur, decay=6.0, harm=(1.0, 0.0, 0.25)):
    t = np.arange(int(dur * SR)) / SR
    s = sum(h * np.sin(2 * np.pi * f * (k + 1) * t) for k, h in enumerate(harm) if h)
    return (s * np.exp(-t * decay)).astype(np.float32)


def sfx_pop():
    t = np.arange(int(0.14 * SR)) / SR
    f = 380 + 1100 * t / 0.14
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 26) * 0.35).astype(np.float32)


def sfx_tick():
    return tone(1760, 0.08, 45, (1, 0, 0.3)) * 0.35


def sfx_whoosh(seed=0):
    n = int(0.45 * SR)
    noise = np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)
    out = np.zeros(n, np.float32)
    for k, win in enumerate((40, 20, 10, 6)):
        part = np.convolve(noise, np.ones(win) / win, mode="same")
        out += part * (k + 1) / 10
    env = np.sin(np.linspace(0, np.pi, n)) ** 2
    return out * env * 0.5


def sfx_tada():
    out = np.zeros(int(1.6 * SR), np.float32)
    for i, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
        _add(out, i * 0.09, tone(f, 1.2, 3.5, (1, 0.35, 0.2)) * 0.22)
    rng = np.random.default_rng(7)
    for i in range(14):
        _add(out, 0.35 + i * 0.06, tone(rng.uniform(1800, 3600), 0.3, 14, (1, 0, 0)) * 0.06)
    return out


def sfx_sparkle():
    out = np.zeros(int(1.0 * SR), np.float32)
    rng = np.random.default_rng(3)
    for i in range(10):
        _add(out, i * 0.07, tone(rng.uniform(1500, 3200), 0.35, 10, (1, 0, 0)) * 0.07)
    return out


def music(total, seed):
    rng = np.random.default_rng(seed)
    n = int(total * SR) + SR
    out = np.zeros(n, np.float32)
    bpm = float(rng.choice([100, 104, 108]))
    beat = 60 / bpm
    root = 196.0 * 2 ** (int(rng.choice([0, 2, 3, 5, 7])) / 12)  # around G3
    progs = [[0, 7, 9, 5], [0, 5, 7, 0], [0, 9, 5, 7], [5, 7, 0, 0]]
    prog = progs[int(rng.integers(len(progs)))]
    minor = {2, 4, 9}
    scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    motif_a = [int(rng.choice(scale)) if rng.random() > 0.25 else None for _ in range(8)]
    motif_b = [int(rng.choice(scale)) if rng.random() > 0.3 else None for _ in range(8)]
    bars = int(total / (4 * beat)) + 2
    kick = tone(1, 0.01)  # placeholder
    tk = np.arange(int(0.22 * SR)) / SR
    kick = (np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-tk * 30)) / SR) * np.exp(-tk * 12)).astype(np.float32)
    nz = np.random.default_rng(1).normal(0, 1, int(0.18 * SR)).astype(np.float32)
    clap = (np.diff(nz, prepend=0) * np.exp(-np.arange(len(nz)) / SR * 22)).astype(np.float32)
    shk = (np.diff(np.diff(nz[: int(0.06 * SR)], prepend=0), prepend=0) * np.exp(-np.arange(int(0.06 * SR)) / SR * 60)).astype(np.float32)
    for bar in range(bars):
        t0 = bar * 4 * beat
        if t0 > total:
            break
        rs = prog[bar % 4]
        fr = root * 2 ** (rs / 12)
        third = 3 if rs % 12 in minor else 4
        # pad: soft chord, slow swell
        L = 4 * beat
        tt = np.arange(int(L * SR)) / SR
        env = np.minimum(1, tt / 0.4) * np.minimum(1, (L - tt) / 0.5)
        pad = sum(np.sin(2 * np.pi * fr * 2 ** (iv / 12) * tt * (1 + det)) for iv in (0, third, 7, 12) for det in (-0.002, 0.002))
        _add(out, t0, (pad * env).astype(np.float32), 0.012)
        # bass on beats 1 and 3
        for bt in (0, 2):
            _add(out, t0 + bt * beat, tone(fr / 2, beat * 1.6, 3.0, (1, 0.4, 0.1)), 0.16)
        drums = bar >= 2
        for bt in range(4):
            if drums and bt in (0, 2):
                _add(out, t0 + bt * beat, kick, 0.32)
            if drums and bt in (1, 3):
                _add(out, t0 + bt * beat, clap, 0.05)
            if drums:
                _add(out, t0 + bt * beat + beat / 2, shk, 0.04)
        # marimba melody (eighths), motif A/B, rests every 4th bar for air
        motif = motif_a if bar % 4 < 2 else motif_b
        if bar % 8 == 7:
            continue
        for k, nt in enumerate(motif):
            if nt is None:
                continue
            f = fr * 2 * 2 ** (nt / 12)
            while f > 1100:
                f /= 2
            _add(out, t0 + k * beat / 2, tone(f, 0.5, 7.0, (1, 0, 0, 0.25)), 0.09)
    fade = int(2.5 * SR)
    end = int(total * SR)
    out[max(0, end - fade):end] *= np.linspace(1, 0, min(fade, end))
    out[end:] = 0
    return out[:n]


def load_wav_mono(path):
    tmpw = path + ".m44.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", str(SR), "-ac", "1", tmpw], check=True)
    with wave.open(tmpw) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return a


# ======================================================================= planning
SCN = []  # filled by render(); read by worker processes (fork)
ART = {}


def scene_kind(sd):
    if sd.get("intro"):
        return "intro"
    if sd.get("outro"):
        return "outro"
    if sd.get("hide"):
        return "hide"
    if sd.get("reveal"):
        return "reveal"
    return "normal"


def plan(episode, lang):
    raw = episode["scenes"]
    out = []
    prev_place = None
    for i, sd in enumerate(raw):
        kind = scene_kind(sd)
        bg = sd.get("bg") if sd.get("bg") in DEFAULT_PLACE else "meadow"
        place = (sd.get("place") or "").strip()
        if not place:
            place = prev_place or DEFAULT_PLACE[bg]
        if kind == "outro" and not sd.get("place"):
            place = "warm golden sunset over soft green hills"
            bg = "meadow"
        prev_place = place
        show = [e for e in (sd.get("show") or []) if isinstance(e, str) and e.strip()][:3]
        art = [re.sub(r"\b(silhouette|shadow|mystery|hidden)\b", "", a, flags=re.I).strip(" ,-")
               for a in (sd.get("art") or []) if isinstance(a, str) and a.strip()][:3]
        if kind == "hide":
            nxt = raw[i + 1] if i + 1 < len(raw) else {}
            if not art:
                art = [a for a in (nxt.get("art") or []) if isinstance(a, str)][:1]
            if not show or show[0] in ("❓", "?"):
                show = [e for e in (nxt.get("show") or []) if isinstance(e, str)][:1] or show
            show, art = show[:1], art[:1]
        if kind in ("intro", "outro"):
            show = [e for e in show if e not in ("🦉",)][:2]
            art = []
        items = []
        for k, e in enumerate(show or ([] if kind in ("intro", "outro") else ["⭐"])):
            items.append({"emoji": e, "name": art[k] if k < len(art) else ""})
        clues = [c for c in (sd.get("clues") or []) if isinstance(c, str) and c.strip()][:4]
        cap = (sd.get("caption") or "").strip()
        if kind == "hide" and (not cap or cap == "?"):
            cap = TEXTS["who"].get(lang, "?")
        if kind == "outro" and not cap:
            cap = TEXTS["bye"].get(lang, "")
        out.append({"i": i, "kind": kind, "bg": bg, "place": place, "items": items, "clues": clues, "caption": cap,
                    "big": "" if (sd.get("big") or "").strip() in ("?", "❓") else (sd.get("big") or "").strip()[:2], "say": sd.get("say", ""), "voice": sd.get("voice")})
    return out


def fetch_art(scenes, lib, max_bg=10, max_items=26):
    places, names = [], []
    for s in scenes:
        if s["place"] not in places:
            places.append(s["place"])
        for it in s["items"]:
            if it["name"] and it["name"] not in names:
                names.append(it["name"])
    bgs, items = {}, {}
    # answers of riddles first, then backgrounds, then the rest
    first = [it["name"] for s in scenes if s["kind"] in ("hide", "reveal") for it in s["items"] if it["name"]]
    order = list(dict.fromkeys(first + names))[:max_items]
    for nm in order[:12]:
        items[nm] = lib.item(nm)
    for pl in places[:max_bg]:
        bgs[pl] = lib.background(pl)
    for nm in order[12:]:
        items[nm] = lib.item(nm)
    log("Art:", json.dumps(lib.stats))
    return bgs, items


# ======================================================================= frames
def bg_frame(S, t):
    if S.get("layers"):
        return layers_frame(S, t)
    base = S["bgimg"]
    cam = S["cam"]
    u = clamp((S["start"] + t - cam["t0"]) / max(0.1, cam["len"]))
    z = cam["z0"] + (cam["z1"] - cam["z0"]) * u
    cw, ch = BGW / z, BGH / z
    cx = BGW / 2 + (cam["x1"] - cam["x0"]) * u * (BGW - cw) / 2 + cam["x0"] * (BGW - cw) / 2
    cy = BGH / 2 + cam["y"] * (BGH - ch) / 2
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    return base.resize((W, H), Image.BILINEAR, box=box)


def item_slots(n, big):
    if big:
        xs = {1: [0.70], 2: [0.64, 0.84], 3: [0.60, 0.74, 0.88]}[n]
        return [(x * W, 0.5 * H) for x in xs], {1: 0.44, 2: 0.32, 3: 0.25}[n] * H
    xs = {1: [0.6], 2: [0.48, 0.76], 3: [0.42, 0.62, 0.82]}[n]
    return [(x * W, 0.52 * H) for x in xs], {1: 0.56, 2: 0.42, 3: 0.33}[n] * H


def frame(si, f):
    S = SCN[si]
    t = f / FPS
    gt = S["start"] + t
    img = bg_frame(S, t)
    d = ImageDraw.Draw(img, "RGBA")
    kind = S["kind"]
    mo = int(S["mouth"][min(f, len(S["mouth"]) - 1)])
    nuni = ART["nuni"]
    if kind == "intro":
        k = ease_back((t - 0.1) / 0.8)
        pose = "talk" if t < 1.3 else ("happy" if 4.0 < (t % 7) < 4.7 else "idle")
        hop = abs(math.sin(t * 4)) * 30 if pose == "happy" else 0
        nuni.draw(img, W / 2, H * 0.97, H * 0.5 * k + 1, gt, pose, mo, hop, math.sin(t * 2) * 2 + mo * 0.4)
        for j, (letters, y) in enumerate(S["title"]):
            draw_letters(img, ("title", j), letters, W / 2, y, t, start=0.5 + j * 0.4)
        tz = S.get("teaser") or []
        spots = [(0.2, 0.52), (0.8, 0.52), (0.1, 0.78), (0.9, 0.78), (0.3, 0.8), (0.7, 0.8)]
        for j, sil in enumerate(tz[:6]):
            k = ease_back((t - 1.2 - j * 0.35) / 0.45)
            sp = scaled(("teaser", j), sil, k)
            if sp is not None:
                x, y = spots[j]
                paste_center(img, sp, x * W, y * H + math.sin(t * 2.4 + j) * 10)
        for j in range(10):
            a = j / 10 * 2 * math.pi + t * 0.5
            x, y = W / 2 + math.cos(a) * 520, H * 0.62 + math.sin(a) * 260
            star(d, x, y, 10 + 8 * abs(math.sin(t * 3 + j)), (255, 240, 140, 200))
    elif kind == "outro":
        pose = "happy" if (t % 3.2) < 1.0 else ("talk" if (t % 3.2) < 1.6 else "idle")
        hop = abs(math.sin(t * math.pi / 1.3)) * 45 if pose == "happy" else 0
        nuni.draw(img, W / 2, H * 0.97, H * 0.52, gt, pose, mo, hop)
        if S["cap"]:
            draw_letters(img, (si, "cap"), S["cap"], W / 2, H * 0.16, t, start=0.3)
        for j in range(14):
            sp = S["hearts"][j % len(S["hearts"])]
            ph = (t * 0.22 + j / 14) % 1
            x = (j * 137) % W
            y = H * (1.05 - ph * 1.2)
            s = 0.6 + 0.4 * math.sin(j)
            spr = scaled(("heart", j % len(S["hearts"])), sp, s)
            if spr:
                paste_center(img, spr, x + math.sin(t * 2 + j) * 30, y)
    else:
        hopping = kind == "reveal" and t < 1.8
        pose = "happy" if hopping else "idle"
        hop = abs(math.sin(t * 7)) * 40 if hopping else 0
        nuni.draw(img, W * 0.16, H * 0.97, H * 0.44, gt, pose, 0 if hopping else mo, hop, math.sin(gt * 1.7) * 2.5 + mo * 0.5)
        slots, size = item_slots(len(S["stk"]), bool(S["bigtile"])) if S["stk"] else ([], 0)
        if S["bigtile"] is not None:
            k = ease_back((t - 0.1) / 0.45)
            sp = scaled((si, "big"), S["bigtile"], k)
            if sp:
                paste_center(img, sp, W * (0.43 if S["stk"] else 0.58), H * 0.5 + math.sin(t * 2) * 8)
        for j, ((x, y), stk) in enumerate(zip(slots, S["stk"])):
            if kind == "reveal":
                k = 1 + 0.12 * math.sin(clamp(t / 0.6) * math.pi)
            else:
                k = ease_back((t - 0.12 - j * 0.18) / 0.5)
            sp = scaled((si, "stk", j), stk, k)
            if sp is None:
                continue
            yy = y + math.sin(gt * 2.2 + j) * 12
            paste_center(img, sp, x, yy)
            if kind == "hide":
                qb = S["qbadge"]
                bob = math.sin(t * 4) * 10
                paste_center(img, qb, x + sp.width * 0.36, yy - sp.height * 0.36 + bob)
        if kind == "reveal" and t < 2.6:
            cx, cy = slots[0] if slots else (W * 0.6, H * 0.5)
            for (x0, y0, vx, vy, col, sz, rs) in S["confetti"]:
                tt = t
                x = cx + x0 + vx * tt
                y = cy + y0 + vy * tt + 0.5 * 900 * tt * tt
                if y > H + 20:
                    continue
                a = rs * tt
                ca, sa = math.cos(a) * sz, math.sin(a) * sz * 0.5
                d.polygon([(x - ca, y - sa), (x + sa, y - ca * 0.5), (x + ca, y + sa), (x - sa, y + ca * 0.5)], fill=col)
            if t < 0.9:
                rr = 80 + t * 700
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 255, 255, int(220 * (1 - t / 0.9))), width=14)
        if kind in ("reveal", "normal") and S["stk"]:
            for j in range(6):
                a = j * 1.05 + t * 0.8
                x, y = slots[0][0] + math.cos(a) * size * 0.62, slots[0][1] + math.sin(a) * size * 0.55
                star(d, x, y, 6 + 10 * abs(math.sin(t * 3 + j * 1.3)), (255, 245, 170, 210))
        # clue badges
        for j, (bimg, at) in enumerate(S["clues"]):
            k = ease_back((t - at) / 0.4)
            sp = scaled((si, "clue", j), bimg, k)
            if sp is None:
                continue
            n = len(S["clues"])
            x = W * 0.6 + (j - (n - 1) / 2) * 205
            paste_center(img, sp, x, H * 0.87 + math.sin(t * 3 + j) * 5)
        # countdown
        if kind == "hide" and S["cd"] is not None:
            c0 = S["cd"]
            if c0 <= t < c0 + 3.0:
                num = 3 - int(t - c0)
                fr = (t - c0) % 1
                cx, cy, r = W * 0.88, H * 0.2, 92 * (1.15 - 0.15 * ease_out(fr / 0.3))
                d.ellipse([cx - r - 6, cy - r + 4, cx + r + 6, cy + r + 14], fill=(40, 30, 80, 70))
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 245))
                d.arc([cx - r + 6, cy - r + 6, cx + r - 6, cy + r - 6], -90, -90 + 360 * (1 - fr), fill=(255, 150, 40), width=16)
                sp = S["nums"][num - 1]
                paste_center(img, sp, cx, cy)
        if S["cap"]:
            draw_letters(img, (si, "cap"), S["cap"], W * 0.58, H * 0.12, t, start=0.2 if kind != "reveal" else 0.05)
    if si > 0 and t < XFADE:
        prev = ART["prevlast"].get(si)
        if prev is None:
            P = SCN[si - 1]
            prev = frame(si - 1, P["nf"] - 1)
            ART["prevlast"][si] = prev
        img = Image.blend(prev, img, ease_out(t / XFADE) * 0.85 + 0.15 * (t / XFADE))
    return img


def render_scene(si):
    S = SCN[si]
    final = os.path.join(ART["tmp"], f"scene_{si:03d}.mp4")
    part = os.path.join(ART["tmp"], f"scene_{si:03d}.part.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-g", str(FPS * 2), "-threads", "2", part], stdin=subprocess.PIPE)
    for f in range(S["nf"]):
        ff.stdin.write(frame(si, f).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"encode failed for scene {si}")
    os.replace(part, final)
    return final


# ======================================================================= main entry
def prepare(scenes, lang, root, seed, tmp, key, budget):
    lib = ArtLib(root, key, budget)
    bgs, items = fetch_art(scenes, lib)
    letters = Letters(lang)
    nuni = Nuni(root)
    # durations from the voice
    t = 0.0
    fstart = 0
    for s in scenes:
        a = load_wav_mono(s["voice"]) if s.get("voice") else np.zeros(SR // 2, np.float32)
        nz = np.where(np.abs(a) > 0.01)[0]
        a = a[max(0, nz[0] - 200):nz[-1] + 2000] if len(nz) else a[:SR // 4]
        rms = np.sqrt(np.mean(a ** 2)) if len(a) else 0
        if rms > 1e-4:
            a = np.clip(a * (0.085 / rms), -1, 1)
        s["audio"] = a
        sp = len(a) / SR
        s["speech"] = sp
        tail = {"intro": 1.0, "outro": 2.2, "hide": 3.9, "reveal": 1.6, "normal": 0.9}[s["kind"]]
        dur = max(2.6, VOICE_AT + sp + tail)
        f0 = int(round(t * FPS))
        t += dur
        s["nf"] = int(round(t * FPS)) - f0
        s["start"] = f0 / FPS
        hop = SR // FPS
        start = int(VOICE_AT * SR)
        env = np.zeros(s["nf"])
        for f in range(s["nf"]):
            a0 = f * hop - start
            if 0 <= a0 < len(a):
                env[f] = np.sqrt(np.mean(a[a0:a0 + hop] ** 2))
        if env.max() > 0:
            env = env / max(np.percentile(env[env > 0], 90), 1e-4)
        mo = np.zeros(s["nf"])
        cur = 0.0
        for f in range(s["nf"]):
            target = clamp((env[f] - 0.12) / 0.75)
            cur += (target - cur) * (0.65 if target > cur else 0.45)
            mo[f] = round(cur * 5)
        s["env"] = env
        s["mouth"] = mo
    total = t
    # camera: one continuous slow move per run of scenes in the same place
    runs = []
    for s in scenes:
        if runs and runs[-1]["place"] == s["place"]:
            runs[-1]["scenes"].append(s)
        else:
            runs.append({"place": s["place"], "scenes": [s]})
    rng = random.Random(seed)
    for k, r in enumerate(runs):
        t0 = r["scenes"][0]["start"]
        last = r["scenes"][-1]
        ln = last["start"] + last["nf"] / FPS - t0
        z0, z1 = (1.0, 1.07) if k % 2 == 0 else (1.07, 1.0)
        x0 = rng.uniform(-0.6, 0.6)
        cam = {"t0": t0, "len": ln, "z0": z0, "z1": z1, "x0": x0, "x1": -x0 * 0.5, "y": rng.uniform(-0.3, 0.3)}
        cam["p0"], cam["p1"] = (0.15, 0.85) if k % 2 == 0 else (0.85, 0.15)
        bg = bgs.get(r["place"])
        layers = None
        if bg is None:
            first = r["scenes"][0]
            kind = "sunset" if first["kind"] == "outro" and not first.get("place_set") else place_kind(r["place"], first["bg"])
            layers = scenery(kind)
        else:
            bg = vignette(cover(bg, BGW, BGH))
        for s in r["scenes"]:
            s["cam"] = cam
            s["bgimg"] = bg
            s["layers"] = layers
    hearts = [emoji_img(e, 90) for e in ("💖", "⭐", "💛", "🌟")]
    qbadge = badge(letters.glyph("?", 120, (255, 140, 40))[0], 70, (255, 140, 40))
    nums = [letters.glyph(str(n), 104, (255, 120, 40))[0] for n in (1, 2, 3)]
    for s in scenes:
        stk = []
        for it in s["items"]:
            cut = items.get(it["name"]) if it["name"] else None
            if cut is None:
                cut = emoji_img(it["emoji"], 420)
            n = len(s["items"])
            sz = {1: 0.5, 2: 0.36, 3: 0.28}[min(n, 3)] * H * (0.8 if s["big"] else 1.0)
            stk.append(sticker(cut, sz, silhouette=s["kind"] == "hide"))
            s.setdefault("stk_src", cut)
        s["stk"] = stk
        s["bigtile"] = None
        if s["big"]:
            tile = Image.new("RGBA", (380, 380), (0, 0, 0, 0))
            dd = ImageDraw.Draw(tile)
            col = PALETTE[s["i"] % len(PALETTE)]
            dd.rounded_rectangle([14, 22, 374, 374], 60, fill=(40, 30, 80, 70))
            dd.rounded_rectangle([6, 6, 362, 362], 60, fill=col, outline=(255, 255, 255), width=12)
            g = letters.glyph(s["big"], 240, (255, 255, 255))[0]
            paste_center(tile, g, 184, 184)
            s["bigtile"] = tile
        raw_clues = s["clues"]
        cl = []
        for j, c in enumerate(raw_clues):
            at = VOICE_AT + 0.25 + s["speech"] * (j / max(1, len(raw_clues))) * 0.9
            cl.append((badge(emoji_img(c, 104), 80, PALETTE[(j + 1) % len(PALETTE)]), at))
        s["clues"] = cl
        s["cd"] = VOICE_AT + s["speech"] + 0.35 if s["kind"] == "hide" else None
        s["nums"] = nums
        s["qbadge"] = qbadge
        s["hearts"] = hearts
        cap = s["caption"]
        s["cap"] = letters.line(cap, 120 if s["kind"] != "reveal" else 140, W * 0.62)[0] if cap and s["kind"] != "intro" else None
        if s["kind"] == "outro" and cap:
            s["cap"] = letters.line(cap, 130, W * 0.85)[0]
        crng = random.Random(seed * 100 + s["i"])
        s["confetti"] = [(crng.uniform(-60, 60), crng.uniform(-60, 60), crng.uniform(-700, 700), crng.uniform(-1100, -300),
                          PALETTE[crng.randrange(len(PALETTE))] + (255,), crng.uniform(8, 16), crng.uniform(-8, 8))
                         for _ in range(80)] if s["kind"] == "reveal" else []
    return total, nuni, lib


def build_audio(scenes, total, seed, path):
    n = int(total * SR) + SR
    voice = np.zeros(n, np.float32)
    fx = np.zeros(n, np.float32)
    pop, tick, tada, whoosh, sparkle = sfx_pop(), sfx_tick(), sfx_tada(), sfx_whoosh(seed), sfx_sparkle()
    for k, s in enumerate(scenes):
        st = s["start"]
        _add(voice, st + VOICE_AT, s["audio"])
        if k > 0 and s["kind"] != "reveal":
            _add(fx, st, whoosh, 0.25)
        if s["kind"] == "intro":
            _add(fx, st + 0.4, sparkle, 1.0)
        if s["kind"] == "reveal":
            _add(fx, st + 0.02, tada, 1.0)
        elif s["kind"] in ("normal", "hide"):
            for j in range(len(s["stk"])):
                _add(fx, st + 0.2 + j * 0.18, pop, 0.8)
        for _, at in s["clues"]:
            _add(fx, st + at + 0.05, pop, 0.9)
        if s["cd"] is not None:
            for j in range(3):
                _add(fx, st + s["cd"] + j, tick, 1.0)
    mus = music(total, seed)[:n]
    talk = (np.abs(voice) > 0.015).astype(np.float32)
    win = int(0.35 * SR)
    talk = np.convolve(talk, np.ones(win) / win, mode="same")
    duck = 1.0 - 0.62 * np.clip(talk * 4, 0, 1)
    mix = voice * 1.0 + fx * 0.9 + mus * 0.55 * duck
    mix = np.clip(mix / max(1.0, np.abs(mix).max() / 0.97), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())


CHAPTER_WORDS = {"intro": {"en": "Hello!", "he": "שלום!", "ar": "مرحبا!"},
                 "riddle": {"en": "Riddle", "he": "חידה", "ar": "لغز"},
                 "letter": {"en": "Letter", "he": "האות", "ar": "حرف"},
                 "bye": {"en": "Goodbye", "he": "להתראות", "ar": "مع السلامة"},
                 "guess": {"en": "Can you guess?", "he": "?תצליחו לנחש", "ar": "هل تستطيع التخمين؟"},
                 "full": {"en": "Full video on the channel!", "he": "!הסרטון המלא בערוץ", "ar": "الفيديو الكامل في القناة!"}}
LAST = {}


def chapters(scenes, lang):
    out, n = [], 0
    for s in scenes:
        if s["kind"] == "intro":
            out.append((s["start"], CHAPTER_WORDS["intro"].get(lang, "Hello!")))
        elif s["kind"] == "hide":
            n += 1
            out.append((s["start"], f'{CHAPTER_WORDS["riddle"].get(lang, "Riddle")} {n}'))
        elif s["big"] and s["kind"] == "normal" and (not out or not out[-1][1].endswith(" " + s["big"])):
            out.append((s["start"], f'{CHAPTER_WORDS["letter"].get(lang, "Letter")} {s["big"]}'))
        elif s["kind"] == "outro":
            out.append((s["start"], CHAPTER_WORDS["bye"].get(lang, "Bye")))
    keep = []
    for t, lab in out:
        if not keep or t - keep[-1][0] >= 10:
            keep.append((t, lab))
        elif len(keep) == 1:
            keep[0] = (t, lab)
    if keep:
        keep[0] = (0.0, keep[0][1])
    return keep if len(keep) >= 3 else []


def thumbnail(scenes, text, path, lang):
    nuni = ART["nuni"]
    pick = next((s for s in scenes if s["kind"] == "reveal" and s["stk"]), None) or next((s for s in scenes if s["stk"]), scenes[0])
    img = bg_frame(pick, 0.0)
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.ogrid[0:H, 0:W]
    g = np.clip(1 - np.sqrt(((xx - W * 0.68) / (W * 0.4)) ** 2 + ((yy - H * 0.58) / (H * 0.55)) ** 2), 0, 1)[..., None]
    arr = arr * (1 - 0.45 * g) + 255 * 0.45 * g
    img = Image.fromarray(arr.astype(np.uint8))
    if pick["stk"]:
        st = pick["stk_src"]
        big = sticker(st, H * 0.62)
        paste_center(img, big, W * 0.68, H * 0.6)
    pi = scenes.index(pick)
    hide = next((s for i, s in enumerate(scenes) if s["kind"] == "hide" and s.get("stk_src") is not None and i != pi - 1), None)
    if hide is not None:
        sil = sticker(hide["stk_src"], H * 0.3, silhouette=True)
        paste_center(img, sil, W * 0.9, H * 0.82)
        qb = scenes[0]["qbadge"]
        paste_center(img, qb, W * 0.9 + sil.width * 0.3, H * 0.82 - sil.height * 0.35)
    nuni.draw(img, W * 0.2, H * 1.02, H * 0.78, 0.0, "happy", 0, 0, -4)
    L = Letters(lang)
    line, _ = L.line(text.upper() if lang == "en" else text, 190, W * 0.9)
    for i, (im, dx) in enumerate(line):
        paste_center(img, im, W / 2 + dx, H * 0.16 + (math.sin(i * 1.3) * 10))
    img.resize((1280, 720), Image.LANCZOS).save(path, quality=92)
    return path


def make_short(video, scenes, path, lang, tmp):
    hides = [i for i, s in enumerate(scenes) if s["kind"] == "hide"]
    if not hides:
        return None
    start = scenes[hides[0]]["start"]
    end = start + 57
    for i in hides[:3]:
        j = i + 1
        if j < len(scenes) and scenes[j]["kind"] == "reveal":
            e = scenes[j]["start"] + scenes[j]["nf"] / FPS
            if e - start <= 58:
                end = e
    dur = max(10.0, min(58.0, end - start))
    ov = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    L = Letters(lang)
    for text, y, size in ((CHAPTER_WORDS["guess"].get(lang, "?"), 300, 120), (CHAPTER_WORDS["full"].get(lang, ""), 1640, 70)):
        line, _ = L.line(text, size, 1000)
        for im, dx in line:
            paste_center(ov, im, 540 + dx, y)
    ovp = os.path.join(tmp, "short_overlay.png")
    ov.save(ovp)
    fc = ("[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2,"
          "eq=brightness=-0.06[bg];[b]crop=1536:1080:96:0,scale=1080:-2[fg];[bg][fg]overlay=0:(H-h)/2[v];"
          f"[v][1:v]overlay=0:0,fade=t=out:st={dur - 0.6:.2f}:d=0.6[vo];[0:a]afade=t=out:st={dur - 0.6:.2f}:d=0.6[ao]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", video, "-i", ovp,
                    "-filter_complex", fc, "-map", "[vo]", "-map", "[ao]", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", path], check=True)
    return path


def render(episode, lang, out_path, seed, tmp, bot=None, root=None, key=None, budget=None):
    """Main entry. episode['scenes'][i]['voice'] must already point to a wav (bot.synthesize)."""
    global SCN
    t_start = time.time()
    root = root or (getattr(bot, "ROOT", None) if bot else None) or Path(__file__).resolve().parent
    key = key if key is not None else os.environ.get("GEMINI_API_KEY", "").strip()
    cfg = getattr(bot, "CONFIG", {}) if bot else {}
    budget = budget or cfg.get("art_budget_s", 1500)
    if bot is not None and any(not s.get("voice") for s in episode["scenes"]):
        bot.synthesize(episode["scenes"], lang, tmp)
    scenes = plan(episode, lang)
    total, nuni, lib = prepare(scenes, lang, root, seed, tmp, key, budget)
    title = re.sub(r"[^\w\s!?'.,&:-]", "", (episode.get("title") or "").split("|")[0], flags=re.UNICODE).strip()
    letters = Letters(lang)
    words = title.split()
    lines = [title] if len(title) <= 22 or len(words) < 3 else [" ".join(words[: len(words) // 2 + len(words) % 2]),
                                                              " ".join(words[len(words) // 2 + len(words) % 2:])]
    tl = []
    for j, ln in enumerate(lines):
        tl.append((letters.line(ln, 118, W * 0.9)[0], H * (0.13 + 0.15 * j)))
    teaser = [sticker(s0["stk_src"], H * 0.2, silhouette=True) for s0 in scenes if s0["kind"] == "hide" and s0.get("stk_src") is not None][:6]
    for s in scenes:
        s["title"] = tl
        s["teaser"] = teaser
    SCN = scenes
    ART.clear()
    ART.update({"nuni": nuni, "tmp": tmp, "prevlast": {}})
    log(f"Studio v3: {len(scenes)} scenes, {total:.0f}s, art {time.time() - t_start:.0f}s")
    audio = os.path.join(tmp, "mix.wav")
    build_audio(scenes, total, seed, audio)
    procs = max(1, min(os.cpu_count() or 1, 4))
    t1 = time.time()
    if procs > 1:
        with get_context("fork").Pool(procs) as pool:
            files = pool.map(render_scene, range(len(scenes)), chunksize=1)
    else:
        files = [render_scene(i) for i in range(len(scenes))]
    log(f"Frames encoded in {time.time() - t1:.0f}s ({procs} processes)")
    lst = os.path.join(tmp, "scenes.txt")
    Path(lst).write_text("".join(f"file '{f}'\n" for f in files))
    video = os.path.join(tmp, "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path], check=True)
    LAST.clear()
    LAST["chapters"] = chapters(scenes, lang)
    base = os.path.splitext(out_path)[0]
    try:
        hook = (episode.get("thumb_text") or "").strip() or (scenes[1]["caption"] if len(scenes) > 1 and scenes[1]["kind"] == "hide" else title)
        LAST["thumb"] = thumbnail(scenes, hook[:22], base + "_thumb.jpg", lang)
    except Exception as e:  # noqa: BLE001
        log("thumbnail failed:", e)
    try:
        LAST["short"] = make_short(out_path, scenes, base + "_short.mp4", lang, tmp)
    except Exception as e:  # noqa: BLE001
        log("short failed:", e)
    log(f"Rendered {out_path}: {len(scenes)} scenes, {total:.0f}s (studio v3, {time.time() - t_start:.0f}s total)")
    log("Extras:", json.dumps({k: (v if k != "chapters" else len(v)) for k, v in LAST.items()}))
    return total
