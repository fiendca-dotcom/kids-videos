"""Kids learning videos -> YouTube, fully automatic (single file).

Settings: edit CONFIG below. Texts/voices: edit LANGS below.
    python bot.py --dry-run   make he/ar/en test videos into out/ (no upload)
    python bot.py             make the next episode and upload it
"""
import asyncio
import datetime
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "languages": ["he", "ar", "en"],
    "privacy": "public",
    "category_id": "27",
    "width": 1920,
    "height": 1080,
    "fps": 24,
    "tts_rate": "-15%",
    "topics": [
        "counting",
        "colors",
        "shapes"
    ]
}

LANGS = {
    "he": {
        "voice": "he-IL-HilaNeural",
        "numbers": [
            "אחת",
            "שתיים",
            "שלוש",
            "ארבע",
            "חמש",
            "שש",
            "שבע",
            "שמונה",
            "תשע",
            "עשר"
        ],
        "colors": {
            "red": "אדום",
            "blue": "כחול",
            "yellow": "צהוב",
            "green": "ירוק",
            "orange": "כתום",
            "purple": "סגול",
            "pink": "ורוד",
            "brown": "חום"
        },
        "shapes": {
            "circle": "עיגול",
            "square": "ריבוע",
            "triangle": "משולש",
            "star": "כוכב",
            "heart": "לב",
            "diamond": "מעוין"
        },
        "counting": {
            "title": "בואו נספור עד {n}! | ספירה לילדים - פרק {ep}",
            "intro_screen": "בואו נספור!",
            "intro_say": "שלום ילדים! היום נספור ביחד עד {n}. מוכנים?",
            "end_say": "כל הכבוד! ספרנו עד {n}!"
        },
        "colors_topic": {
            "title": "לומדים צבעים! | צבעים לילדים - פרק {ep}",
            "intro_screen": "לומדים צבעים!",
            "intro_say": "שלום ילדים! בואו נלמד צבעים ביחד.",
            "item_say": "זה {name}. {name}!",
            "end_say": "כל הכבוד! למדנו הרבה צבעים!"
        },
        "shapes_topic": {
            "title": "לומדים צורות! | צורות לילדים - פרק {ep}",
            "intro_screen": "לומדים צורות!",
            "intro_say": "שלום ילדים! בואו נכיר צורות ביחד.",
            "item_say": "זה {name}. {name}!",
            "end_say": "כל הכבוד! הכרנו צורות חדשות!"
        },
        "outro_screen": "להתראות!",
        "outro_say": "להתראות ילדים! נתראה בפעם הבאה!",
        "description": "סרטון לימודי קצר לילדים קטנים: ספירה, צבעים וצורות עם קריינות ברורה ודמויות שמחות.\nסרטון חדש כמה פעמים בשבוע.",
        "tags": [
            "ילדים",
            "לימוד לילדים",
            "ספירה",
            "צבעים",
            "צורות",
            "גן ילדים",
            "פעוטות"
        ]
    },
    "ar": {
        "voice": "ar-SA-ZariyahNeural",
        "numbers": [
            "واحد",
            "اثنان",
            "ثلاثة",
            "أربعة",
            "خمسة",
            "ستة",
            "سبعة",
            "ثمانية",
            "تسعة",
            "عشرة"
        ],
        "colors": {
            "red": "أحمر",
            "blue": "أزرق",
            "yellow": "أصفر",
            "green": "أخضر",
            "orange": "برتقالي",
            "purple": "بنفسجي",
            "pink": "وردي",
            "brown": "بني"
        },
        "shapes": {
            "circle": "دائرة",
            "square": "مربع",
            "triangle": "مثلث",
            "star": "نجمة",
            "heart": "قلب",
            "diamond": "معين"
        },
        "counting": {
            "title": "هيا نعد حتى {n}! | تعلم الأرقام للأطفال - الحلقة {ep}",
            "intro_screen": "هيا نعد!",
            "intro_say": "مرحبا يا أطفال! اليوم سنعد معا حتى {n}. هل أنتم مستعدون؟",
            "end_say": "أحسنتم! عددنا حتى {n}!"
        },
        "colors_topic": {
            "title": "نتعلم الألوان! | الألوان للأطفال - الحلقة {ep}",
            "intro_screen": "نتعلم الألوان!",
            "intro_say": "مرحبا يا أطفال! هيا نتعلم الألوان معا.",
            "item_say": "هذا {name}. {name}!",
            "end_say": "أحسنتم! تعلمنا ألوانا كثيرة!"
        },
        "shapes_topic": {
            "title": "نتعلم الأشكال! | الأشكال للأطفال - الحلقة {ep}",
            "intro_screen": "نتعلم الأشكال!",
            "intro_say": "مرحبا يا أطفال! هيا نتعرف على الأشكال معا.",
            "item_say": "هذا {name}. {name}!",
            "end_say": "أحسنتم! تعرفنا على أشكال جديدة!"
        },
        "outro_screen": "مع السلامة!",
        "outro_say": "مع السلامة يا أطفال! نراكم في المرة القادمة!",
        "description": "فيديو تعليمي قصير للأطفال الصغار: الأرقام والألوان والأشكال بصوت واضح وشخصيات سعيدة.\nفيديو جديد عدة مرات في الأسبوع.",
        "tags": [
            "أطفال",
            "تعليم الأطفال",
            "أرقام",
            "ألوان",
            "أشكال",
            "روضة"
        ]
    },
    "en": {
        "voice": "en-US-AnaNeural",
        "numbers": [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten"
        ],
        "colors": {
            "red": "red",
            "blue": "blue",
            "yellow": "yellow",
            "green": "green",
            "orange": "orange",
            "purple": "purple",
            "pink": "pink",
            "brown": "brown"
        },
        "shapes": {
            "circle": "circle",
            "square": "square",
            "triangle": "triangle",
            "star": "star",
            "heart": "heart",
            "diamond": "diamond"
        },
        "counting": {
            "title": "Let's Count to {n}! | Counting for Kids - Episode {ep}",
            "intro_screen": "Let's count!",
            "intro_say": "Hello kids! Today we will count together to {n}. Are you ready?",
            "end_say": "Great job! We counted to {n}!"
        },
        "colors_topic": {
            "title": "Learn Colors! | Colors for Kids - Episode {ep}",
            "intro_screen": "Learn colors!",
            "intro_say": "Hello kids! Let's learn colors together.",
            "item_say": "This is {name}. {name}!",
            "end_say": "Great job! We learned lots of colors!"
        },
        "shapes_topic": {
            "title": "Learn Shapes! | Shapes for Kids - Episode {ep}",
            "intro_screen": "Learn shapes!",
            "intro_say": "Hello kids! Let's meet some shapes together.",
            "item_say": "This is a {name}. {name}!",
            "end_say": "Great job! We met new shapes!"
        },
        "outro_screen": "Bye bye!",
        "outro_say": "Bye bye kids! See you next time!",
        "description": "A short learning video for little kids: counting, colors and shapes with clear narration and happy characters.\nNew videos several times a week.",
        "tags": [
            "kids",
            "learning for kids",
            "counting",
            "colors",
            "shapes",
            "preschool",
            "toddlers"
        ]
    }
}



PALETTE = {
    "red": (235, 64, 52), "blue": (52, 120, 235), "yellow": (250, 205, 40),
    "green": (60, 185, 90), "orange": (250, 140, 30), "purple": (150, 80, 210),
    "pink": (245, 120, 180), "brown": (140, 90, 50),
}
BACKGROUNDS = [(255, 244, 214), (220, 240, 255), (232, 255, 228), (255, 228, 240), (240, 232, 255)]
SHAPES = ["circle", "square", "triangle", "star", "heart", "diamond"]

FONT_CANDIDATES = {
    "he": ["/usr/share/fonts/truetype/noto/NotoSansHebrew-Bold.ttf"],
    "ar": ["/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"],
    "any": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"],
}


def find_font(lang):
    for p in FONT_CANDIDATES.get(lang, []) + FONT_CANDIDATES["any"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No suitable font found (install fonts-dejavu-core / fonts-noto-core)")


def darker(c, f=0.6):
    return tuple(int(v * f) for v in c)


# ---------------------------------------------------------------- shapes
def shape_points(kind, cx, cy, s):
    if kind == "triangle":
        return [(cx, cy - s * 1.05), (cx + s * 1.15, cy + s * 0.85), (cx - s * 1.15, cy + s * 0.85)]
    if kind == "diamond":
        return [(cx, cy - s * 1.2), (cx + s * 0.9, cy), (cx, cy + s * 1.2), (cx - s * 0.9, cy)]
    if kind == "star":
        pts = []
        for i in range(10):
            r = s * 1.15 if i % 2 == 0 else s * 0.5
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts
    if kind == "heart":
        pts = []
        for i in range(60):
            t = 2 * math.pi * i / 60
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((cx + x * s / 15, cy - y * s / 15 - s * 0.1))
        return pts
    return None


def draw_shape(d, kind, cx, cy, s, color, face=True):
    if s < 2:
        return
    w = max(3, int(s * 0.07))
    out = darker(color)
    if kind == "circle":
        d.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color, outline=out, width=w)
    elif kind == "square":
        d.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=int(s * 0.2), fill=color, outline=out, width=w)
    else:
        d.polygon(shape_points(kind, cx, cy, s), fill=color, outline=out, width=w)
    if not face:
        return
    fy = {"triangle": cy + s * 0.25, "heart": cy - s * 0.05}.get(kind, cy)
    fs = s * (0.6 if kind in ("star", "triangle", "diamond") else 0.85)
    for dx in (-0.35, 0.35):
        ex, ey, er = cx + dx * fs, fy - 0.15 * fs, 0.17 * fs
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill="white")
        pr = er * 0.55
        d.ellipse([ex - pr, ey - pr + er * 0.15, ex + pr, ey + pr + er * 0.15], fill=(30, 30, 30))
    d.arc([cx - 0.32 * fs, fy - 0.05 * fs, cx + 0.32 * fs, fy + 0.4 * fs], 20, 160,
          fill=(30, 30, 30), width=max(2, int(fs * 0.07)))


def pop(t, dur=0.45):
    """0 -> overshoot -> 1 scale animation."""
    if t >= dur:
        return 1.0
    x = t / dur
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


# ---------------------------------------------------------------- scenes
@dataclass
class Scene:
    say: str
    painter: callable
    bg: tuple
    min_dur: float = 2.5
    dur: float = 0.0
    audio: str = ""


@dataclass
class Video:
    title: str
    scenes: list = field(default_factory=list)


class Painter:
    def __init__(self, W, H, font_path):
        self.W, self.H, self.font_path = W, H, font_path
        self._fonts = {}

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def text(self, d, txt, y, size, color=(255, 255, 255), stroke=(40, 40, 60), scale=1.0):
        size = max(8, int(size * scale))
        d.text((self.W / 2, y), txt, font=self.font(size), fill=color, anchor="mm",
               stroke_width=max(2, size // 14), stroke_fill=stroke)


def make_background(W, H, color, rng):
    img = Image.new("RGB", (W, H), color)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y, r = rng.randint(0, W), rng.randint(0, H), rng.randint(6, 22)
        c = rng.choice(list(PALETTE.values()))
        soft = tuple(int(a * 0.25 + b * 0.75) for a, b in zip(c, color))
        d.ellipse([x - r, y - r, x + r, y + r], fill=soft)
    return img


def build_counting(lang, ep, rng, P):
    n = rng.randint(5, 10)
    kind = rng.choice(SHAPES)
    cname = rng.choice(list(PALETTE))
    color = PALETTE[cname]
    L = lang["counting"]
    W, H = P.W, P.H
    rows = math.ceil(n / 5)
    s = 85 if rows == 2 else 105
    slots = []
    for i in range(n):
        r, c = divmod(i, 5)
        in_row = min(5, n - r * 5)
        x = W / 2 + (c - (in_row - 1) / 2) * 300
        y = (720 if rows == 1 else 560 + r * 290)
        slots.append((x, y))

    scenes = []

    rng_list = rng.sample(SHAPES, 3)
    intro_cols = rng.sample(list(PALETTE), 3)

    def intro(d, t):
        P.text(d, L["intro_screen"], 330, 170, scale=pop(t))
        for i, sh in enumerate(rng_list):
            bob = math.sin(t * 3 + i) * 20
            draw_shape(d, sh, W / 2 + (i - 1) * 380, 720 + bob, 110 * pop(max(0, t - 0.2 * i)), PALETTE[intro_cols[i]])

    scenes.append(Scene(L["intro_say"].format(n=n), intro, rng.choice(BACKGROUNDS)))

    bg = rng.choice(BACKGROUNDS)
    for k in range(1, n + 1):
        def paint(d, t, k=k):
            P.text(d, str(k), 210, 260, color=color, stroke=darker(color, 0.45), scale=pop(t))
            for i in range(k):
                x, y = slots[i]
                sc = pop(t) if i == k - 1 else 1.0
                bob = math.sin(t * 4 + i) * 8
                draw_shape(d, kind, x, y + bob, s * sc, color)
        scenes.append(Scene(lang["numbers"][k - 1], paint, bg, min_dur=1.8))

    def end(d, t):
        P.text(d, f"1 - {n}", 260, 200, color=color, stroke=darker(color, 0.45), scale=pop(t))
        for i in range(n):
            x, y = slots[i]
            jump = abs(math.sin(t * 5 + i * 0.6)) * 30
            draw_shape(d, kind, x, y - jump, s, color)
    scenes.append(Scene(L["end_say"].format(n=n), end, bg))
    return L["title"].format(n=n, ep=ep), scenes


def build_items(lang, ep, rng, P, topic):
    W, H = P.W, P.H
    L = lang["colors_topic" if topic == "colors" else "shapes_topic"]
    scenes = []
    intro_shapes = rng.sample(SHAPES, 3)
    intro_cols = rng.sample(list(PALETTE), 3)

    def intro(d, t):
        P.text(d, L["intro_screen"], 330, 170, scale=pop(t))
        for i, sh in enumerate(intro_shapes):
            bob = math.sin(t * 3 + i) * 20
            draw_shape(d, sh, W / 2 + (i - 1) * 380, 720 + bob, 110 * pop(max(0, t - 0.2 * i)), PALETTE[intro_cols[i]])
    scenes.append(Scene(L["intro_say"], intro, rng.choice(BACKGROUNDS)))

    if topic == "colors":
        items = rng.sample(list(PALETTE), 6)
    else:
        items = rng.sample(SHAPES, 5)

    for it in items:
        if topic == "colors":
            col, kind, name = PALETTE[it], rng.choice(SHAPES), lang["colors"][it]
            txt_col = col
        else:
            col, kind, name = PALETTE[rng.choice(list(PALETTE))], it, lang["shapes"][it]
            txt_col = (255, 255, 255)

        def paint(d, t, col=col, kind=kind, name=name, txt_col=txt_col):
            bob = math.sin(t * 3) * 15
            draw_shape(d, kind, W / 2, 440 + bob, 260 * pop(t), col)
            P.text(d, name, 900, 170, color=txt_col, stroke=darker(col, 0.45), scale=pop(max(0, t - 0.25)))
        scenes.append(Scene(L["item_say"].format(name=name), paint, rng.choice(BACKGROUNDS), min_dur=3.0))

    def end(d, t):
        for i, it in enumerate(items):
            col = PALETTE[it] if topic == "colors" else PALETTE[list(PALETTE)[i % 8]]
            kind = SHAPES[i % 6] if topic == "colors" else it
            x = W / 2 + (i - (len(items) - 1) / 2) * 290
            jump = abs(math.sin(t * 5 + i * 0.7)) * 40
            draw_shape(d, kind, x, 620 - jump, 100, col)
    scenes.append(Scene(L["end_say"], end, rng.choice(BACKGROUNDS)))
    return L["title"].format(ep=ep), scenes


def outro_scene(lang, rng, P):
    sh = rng.sample(SHAPES, 4)
    cols = rng.sample(list(PALETTE), 4)

    def paint(d, t):
        P.text(d, lang["outro_screen"], 330, 190, scale=pop(t))
        for i in range(4):
            wave = math.sin(t * 6 + i) * 25
            draw_shape(d, sh[i], P.W / 2 + (i - 1.5) * 330, 720 + wave, 100, PALETTE[cols[i]])
    return Scene(lang["outro_say"], paint, rng.choice(BACKGROUNDS))


# ---------------------------------------------------------------- audio
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())


async def _tts(text, voice, rate, path):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(path)


def narrate(scenes, voice, rate, tmp):
    try:
        import edge_tts  # noqa: F401
        have_tts = True
    except ImportError:
        have_tts = False
        print("WARNING: edge-tts not installed - video will be silent")
    for i, sc in enumerate(scenes):
        clip = os.path.join(tmp, f"say{i}.mp3")
        speech = 0.0
        if have_tts:
            for attempt in range(3):
                try:
                    asyncio.run(_tts(sc.say, voice, rate, clip))
                    speech = probe_duration(clip)
                    break
                except Exception as e:  # network hiccup -> retry
                    print(f"TTS retry {attempt + 1}: {e}")
            else:
                raise RuntimeError("TTS failed 3 times")
        else:
            speech = 0.45 * len(sc.say.split()) + 0.5
        sc.dur = max(sc.min_dur, speech + 1.0)
        wav = os.path.join(tmp, f"scene{i}.wav")
        if have_tts:
            run(["ffmpeg", "-y", "-i", clip, "-af", "adelay=350:all=1,apad", "-t", f"{sc.dur:.3f}",
                 "-ar", "44100", "-ac", "2", wav])
        else:
            run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-t", f"{sc.dur:.3f}", wav])
        sc.audio = wav
        sc.silent = not have_tts


def music_track(scenes, path, seed, sr=44100):
    """Fallback when there is no narration: a cheerful original melody + a 'pop' per scene."""
    import numpy as np
    import wave
    rng = random.Random(seed)
    total = sum(sc.dur for sc in scenes)
    n = int(total * sr)
    out = np.zeros(n)
    scale = [0, 2, 4, 7, 9, 12, 14, 16]  # major pentatonic
    base = 261.63 * 2 ** (rng.choice([0, 2, 5, 7]) / 12)
    beat = 60 / rng.choice([110, 120, 128])
    phrase = [rng.choice(scale) for _ in range(8)]
    chords = [0, 5, 7, 0]
    t = 0.0
    i = 0
    while t < total:
        note = phrase[i % 8] if (i // 8) % 4 != 3 else phrase[(7 - i) % 8]
        f = base * 2 ** (note / 12)
        ln = int(beat * 0.9 * sr)
        tt = np.arange(ln) / sr
        env = np.exp(-tt * 5)
        tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * env * 0.16
        if i % 2 == 0:
            bf = base / 2 * 2 ** (chords[(i // 8) % 4] / 12)
            tone += np.sin(2 * np.pi * bf * tt) * np.exp(-tt * 3) * 0.12
        a = int(t * sr)
        seg = tone[: max(0, min(ln, n - a))]
        out[a:a + len(seg)] += seg
        t += beat / 2 if i % 4 == 3 else beat
        i += 1
    start = 0.0
    for sc in scenes:
        a = int((start + 0.05) * sr)
        ln = int(0.18 * sr)
        tt = np.arange(ln) / sr
        f = 600 + 900 * tt / 0.18
        pop_s = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 25) * 0.35
        seg = pop_s[: max(0, min(ln, n - a))]
        out[a:a + len(seg)] += seg
        start += sc.dur
    fade = int(1.5 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)
    out = np.clip(out, -1, 1)
    pcm = (np.repeat(out[:, None], 2, axis=1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- render
def render(scenes, W, H, fps, out_path, tmp, seed):
    rng = random.Random(seed)
    bg_cache = {}
    silent = os.path.join(tmp, "video.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "22", "-pix_fmt", "yuv420p", silent],
                          stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for sc in scenes:
        if sc.bg not in bg_cache:
            bg_cache[sc.bg] = make_background(W, H, sc.bg, rng)
        base = bg_cache[sc.bg]
        for f in range(int(round(sc.dur * fps))):
            img = base.copy()
            sc.painter(ImageDraw.Draw(img), f / fps)
            ff.stdin.write(img.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg video encode failed")

    lst = os.path.join(tmp, "audio.txt")
    with open(lst, "w") as fh:
        for sc in scenes:
            fh.write(f"file '{sc.audio}'\n")
    audio = os.path.join(tmp, "audio.wav")
    if getattr(scenes[0], "silent", False):
        music_track(scenes, audio, seed)
    else:
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", audio])
    run(["ffmpeg", "-y", "-i", silent, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         "-shortest", "-movflags", "+faststart", out_path])


def make_video(topic, lang_code, episode, out_path, cfg):
    lang = LANGS[lang_code]
    W, H, fps = cfg["width"], cfg["height"], cfg["fps"]
    rng = random.Random(f"{topic}-{episode}")
    P = Painter(W, H, find_font(lang_code))
    if topic == "counting":
        title, scenes = build_counting(lang, episode, rng, P)
    else:
        title, scenes = build_items(lang, episode, rng, P, topic)
    scenes.append(outro_scene(lang, rng, P))

    tmp = tempfile.mkdtemp()
    try:
        narrate(scenes, lang["voice"], cfg.get("tts_rate", "-15%"), tmp)
        render(scenes, W, H, fps, out_path, tmp, episode)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = sum(s.dur for s in scenes)
    print(f"Rendered {out_path}: {len(scenes)} scenes, {total:.0f}s")
    return {"title": title[:100], "description": lang["description"], "tags": lang["tags"]}


# ---------------------------------------------------------------- upload
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _refresh_token():
    raw = os.environ["YT_REFRESH_TOKEN"].strip()
    if raw.startswith("{"):  # whole JSON response pasted - take refresh_token from it
        raw = json.loads(raw)["refresh_token"].strip()
    return raw


def youtube_client():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        token=None,
        refresh_token=_refresh_token(),
        client_id=os.environ["YT_CLIENT_ID"].strip(),
        client_secret=os.environ["YT_CLIENT_SECRET"].strip(),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
    )
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload(path, title, description, tags, language, privacy="public", category_id="27"):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    yt = youtube_client()
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": True,  # required for children's content (COPPA)
            "containsSyntheticMedia": True,   # AI voice narration disclosure
        },
    }
    media = MediaFileUpload(path, mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response, retries = None, 0
    while response is None:
        try:
            status, response = req.next_chunk()
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 5:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    vid = response["id"]
    print(f"Done: https://youtu.be/{vid} (privacy: {response['status']['privacyStatus']})")
    return vid


# ---------------------------------------------------------------- main
def main():
    ROOT.joinpath("out").mkdir(exist_ok=True)
    state_path = ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"episode": 0, "history": []}
    ep = state["episode"] + 1
    topic = CONFIG["topics"][(ep - 1) % len(CONFIG["topics"])]

    if "--dry-run" in sys.argv:
        for lang in ("he", "ar", "en"):
            meta = make_video(topic, lang, ep, str(ROOT / "out" / f"video_{lang}.mp4"), CONFIG)
            print(lang, "->", meta["title"])
        return

    from google.oauth2.credentials import Credentials  # noqa: F401
    langs = CONFIG["languages"]
    for lang in langs:
        out = ROOT / "out" / f"video_{lang}.mp4"
        print(f"Episode {ep}: {topic} ({lang})")
        meta = make_video(topic, lang, ep, str(out), CONFIG)
        vid = upload(str(out), meta["title"], meta["description"], meta["tags"], lang,
                     CONFIG.get("privacy", "public"), CONFIG.get("category_id", "27"))
        state["history"].append({"episode": ep, "lang": lang, "topic": topic, "video_id": vid,
                                 "title": meta["title"], "date": datetime.date.today().isoformat()})
    state["episode"] = ep
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
