"""Nuni the Owl - automatic kids learning videos for YouTube (v2, single file).

Each run makes one new episode in Hebrew, Arabic and English:
  * Gemini writes a fresh script (letters / animals / riddles / story)
  * Gemini TTS reads it in a warm, expressive voice (edge-tts fallback)
  * Frames: animated backgrounds, talking owl mascot, colour emoji art, captions
  * Original background music, 3-5 minutes per video

    python bot.py --dry-run   make the 3 videos into out/ (no upload)
    python bot.py             make + upload, then advance state.json
Env: GEMINI_API_KEY (optional - offline fallback without it),
     YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN (for upload)
"""
import asyncio
import base64
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
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "languages": ["en"],  # he/ar paused until their channels exist
    "formats": ["animals", "letters", "riddles", "story"],
    "privacy": "public",
    "voice": "edge",  # "edge" (Microsoft, free) or "gemini"
    "minimax_langs": ["en", "ar", "he"],  # used only when MINIMAX_API_KEY secret exists; else free voice
    "category_id": "27",
    "width": 1280,
    "height": 720,
    "fps": 24,
    "text_model": "gemini-2.5-flash",
    "tts_models": ["gemini-2.5-flash-preview-tts", "gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview"],
    "tts_gap_seconds": 6,  # pause between TTS calls (free-tier rate limit)
}

LANG = {
    "he": {
        "name": "Hebrew", "mascot": "נוני", "gemini_voice": "Leda", "edge_voice": "he-IL-HilaNeural",
        "alphabet": list("אבגדהוזחטיכלמנסעפצקרשת"),
        "tags": ["ילדים", "לימוד לילדים", "נוני הינשופה", "שירים לילדים", "גן ילדים", "פעוטות"],
        "desc": "נוני הינשופה מלמדת ילדים קטנים בדרך שמחה: אותיות, חיות, חידות וסיפורים.\nפרק חדש כמה פעמים בשבוע!",
    },
    "ar": {
        "name": "Arabic", "mascot": "نوني", "gemini_voice": "Leda", "edge_voice": "ar-SA-ZariyahNeural",
        "alphabet": list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي"),
        "tags": ["أطفال", "تعليم الأطفال", "نوني البومة", "حروف", "حيوانات", "روضة"],
        "desc": "نوني البومة تعلم الأطفال الصغار بطريقة ممتعة: الحروف والحيوانات والألغاز والقصص.\nحلقة جديدة عدة مرات في الأسبوع!",
    },
    "en": {
        "name": "English", "mascot": "Nuni", "gemini_voice": "Leda", "edge_voice": "en-US-AnaNeural",
        "alphabet": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "tags": ["kids", "learning for kids", "Nuni the owl", "ABC", "animals", "preschool"],
        "desc": "Nuni the owl helps little kids learn in a happy way: letters, animals, riddles and stories.\nNew episodes several times a week!",
    },
}

BGS = ["meadow", "forest", "sea", "night"]

# ----------------------------------------------------------------- fallback content
FALLBACK_ANIMALS = {
    "he": [("🐄", "פרה", "מוּ מוּ"), ("🐶", "כלב", "הַב הַב"), ("🐱", "חתול", "מיאו"), ("🦆", "ברווז", "גַע גַע"),
           ("🐑", "כבשה", "מֶה מֶה"), ("🐓", "תרנגול", "קוּקוּרִיקוּ"), ("🦁", "אריה", "רררר"), ("🐸", "צפרדע", "קְווה קְווה")],
    "ar": [("🐄", "بقرة", "موو"), ("🐶", "كلب", "هاو هاو"), ("🐱", "قطة", "مياو"), ("🦆", "بطة", "واك واك"),
           ("🐑", "خروف", "ماء ماء"), ("🐓", "ديك", "كوكوكوكو"), ("🦁", "أسد", "رووار"), ("🐸", "ضفدع", "نق نق")],
    "en": [("🐄", "Cow", "moo"), ("🐶", "Dog", "woof woof"), ("🐱", "Cat", "meow"), ("🦆", "Duck", "quack quack"),
           ("🐑", "Sheep", "baa"), ("🐓", "Rooster", "cock-a-doodle-doo"), ("🦁", "Lion", "roar"), ("🐸", "Frog", "ribbit")],
}
FALLBACK_TEXT = {
    "he": {"title": "מי עושה את הקול? | חיות וקולות לילדים עם נוני", "hi": "שלום חברים! אני נוני הינשופה! היום נשחק משחק קולות של חיות. מוכנים?",
           "q": "מי אומר {s}? מי זה?", "a": "נכון! זה {n}! ה{n} אומר {s}!", "bye": "כל הכבוד! הייתם מדהימים. להתראות, נתראה בפעם הבאה!"},
    "ar": {"title": "من يصدر هذا الصوت؟ | أصوات الحيوانات للأطفال مع نوني", "hi": "مرحبا يا أصدقاء! أنا نوني البومة! اليوم سنلعب لعبة أصوات الحيوانات. هل أنتم مستعدون؟",
           "q": "من يقول {s}؟ من هذا؟", "a": "صحيح! إنه {n}! {n} يقول {s}!", "bye": "أحسنتم! كنتم رائعين. مع السلامة، نراكم في المرة القادمة!"},
    "en": {"title": "Who Makes This Sound? | Animal Sounds for Kids with Nuni", "hi": "Hello friends! I'm Nuni the owl! Today we'll play an animal sounds game. Are you ready?",
           "q": "Who says {s}? Who is it?", "a": "Yes! It's the {n}! The {n} says {s}!", "bye": "Great job! You were amazing. Bye bye, see you next time!"},
}


def fallback_episode(lang):
    T = FALLBACK_TEXT[lang]
    sc = [{"say": T["hi"], "show": ["🦉"], "caption": "", "bg": "meadow", "hide": False, "intro": True}]
    bgs = ["meadow", "forest", "meadow", "sea", "meadow", "forest", "meadow", "sea"]
    for i, (e, n, s) in enumerate(FALLBACK_ANIMALS[lang]):
        sc.append({"say": T["q"].format(s=s), "show": [e], "caption": "?", "bg": bgs[i], "hide": True})
        sc.append({"say": T["a"].format(n=n, s=s), "show": [e], "caption": n, "bg": bgs[i], "hide": False, "reveal": True})
    sc.append({"say": T["bye"], "show": ["🦉", "💖"], "caption": "", "bg": "night", "hide": False, "outro": True})
    return {"title": T["title"], "scenes": sc}


# ----------------------------------------------------------------- Gemini
def gemini_call(model, body, key, timeout=180):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


FORMAT_RULES = {
    "animals": "Animal sounds game with 8 different animals (vary them from past episodes). For each animal make TWO scenes: "
               "(1) a question scene with hide=true where {m} imitates the animal sound and asks the kids who it is; "
               "(2) a reveal scene with reveal=true, the animal emoji, caption = animal name, and one tiny fun fact.",
    "letters": "Letters lesson for these letters, in order: {letters}. For each letter: one scene with big=<the letter>, "
               "caption=<the letter>, where {m} says the letter name and its sound; then TWO scenes each with a simple word that "
               "STARTS with that letter (show = one emoji that pictures the word, caption = the word, big = the letter). "
               "Finish with one quick review scene showing the letters.",
    "riddles": "'Who am I?' riddle game with 7 riddles about animals, fruits, vehicles or everyday things. For each riddle make TWO "
               "scenes: (1) question with hide=true and 2-3 very simple clues ending with 'Who am I?'; (2) reveal=true with the answer "
               "emoji and caption = the answer word.",
    "story": "A short, original, gentle story (14-18 scenes) starring {m} the owl and an animal friend, with a simple lesson "
             "(sharing, kindness, bravery, saying sorry, patience...). Each scene shows 1-3 emoji of the characters/objects "
             "in that moment, caption = 1-3 key words, bg chosen to fit the scene.",
}


_MODELS = None


def list_models(key):
    global _MODELS
    if _MODELS is None:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=200"
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read())
            _MODELS = [(m["name"].split("/")[-1], m.get("supportedGenerationMethods", [])) for m in data.get("models", [])]
            print("Gemini models:", ", ".join(n for n, _ in _MODELS if "flash" in n or "tts" in n), flush=True)
        except Exception as e:  # noqa: BLE001
            print("list models failed:", e, flush=True)
            _MODELS = []
    return _MODELS


def text_models(key):
    bad = ("tts", "image", "lite", "live", "audio", "embedding", "thinking", "omni")
    names = [n for n, m in list_models(key) if "generateContent" in m and "flash" in n and not any(x in n for x in bad)]
    pref = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-2.5-flash"]
    return [n for n in pref if n in names] + [n for n in names if n not in pref] or [CONFIG["text_model"]]


def pick_text_model(key):
    bad = ("tts", "image", "lite", "live", "audio", "embedding", "thinking")
    names = [n for n, m in list_models(key) if "generateContent" in m and "flash" in n and not any(x in n for x in bad)]
    for pref in ("gemini-flash-latest", "gemini-3", "gemini-2.5-flash"):
        for n in names:
            if n.startswith(pref):
                return n
    return names[0] if names else CONFIG["text_model"]


def pick_tts_models(key):
    names = [n for n, _ in list_models(key) if "tts" in n]
    ordered = [n for n in CONFIG["tts_models"] if n in names] + sorted([n for n in names if n not in CONFIG["tts_models"]], key=lambda n: "pro" in n)
    return ordered or CONFIG["tts_models"]


TREND_CHANNELS = [  # big kids channels - public RSS (no key needed). Add more channel IDs any time.
    "UCbCmjCuTUZos6Inko4u57UQ",  # CoComelon
    "UCLsooMJoIpl_7ux2jvdPB-Q",  # Super Simple Songs
    "UCcdwLMPsaU2ezNSJU1nFoBQ",  # Pinkfong
    "UCKAqou7V9FAWXpZd9xtOg3Q",  # Little Baby Bum
    "UCBnZ16ahKA2DZ_T5W0FPUXg",  # ChuChu TV
]
_YT_TRENDS = None


def youtube_trends(max_items=15, days=60):
    """Most-viewed recent uploads of top kids channels (titles + views) - inspiration only."""
    global _YT_TRENDS
    if _YT_TRENDS is not None:
        return _YT_TRENDS
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom", "m": "http://search.yahoo.com/mrss/"}
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    items = []
    for cid in TREND_CHANNELS:
        try:
            req = urllib.request.Request(f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}",
                                         headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(urllib.request.urlopen(req, timeout=30).read())
            for e in root.findall("a:entry", ns):
                title = e.findtext("a:title", "", ns)
                pub = datetime.datetime.fromisoformat(e.findtext("a:published", "", ns).replace("Z", "+00:00"))
                st = e.find("m:group/m:community/m:statistics", ns)
                views = int(st.get("views", 0)) if st is not None else 0
                if pub >= cutoff:
                    items.append((views, title))
        except Exception as ex:  # noqa: BLE001
            print("RSS failed for", cid, ex, flush=True)
    items.sort(reverse=True)
    lines = [f"- {t} ({v / 1e6:.1f}M views)" for v, t in items[:max_items]]
    _YT_TRENDS = "Most-viewed recent kids videos:\n" + "\n".join(lines) if lines else ""
    print(_YT_TRENDS, flush=True)
    return _YT_TRENDS


def gemini_trends(lang, key):
    """Ask Gemini (with Google Search) what kids' videos are going viral right now - patterns only."""
    L = LANG[lang]
    prompt = (f"Use Google Search to find which YouTube videos for children aged 2-6 are getting the most views in the "
              f"last 30 days, especially in {L['name']} and worldwide (look at top channels, viral titles and topics). "
              "Summarise in at most 8 short bullet points the PATTERNS that make them popular: topics, formats "
              "(songs, daily routines, stories, games, animals, colors...), hooks in the first seconds, title styles. "
              "Do not list copyrighted lyrics. Answer in English, max 1200 characters.")
    body = {"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}]}
    for model in text_models(key)[:3]:
        try:
            r = gemini_call(model, body, key, timeout=240)
            txt = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])[:1500]
            if txt.strip():
                print(f"Trends ({lang}, {model}):\n{txt}", flush=True)
                return txt
        except Exception as e:  # noqa: BLE001
            print("Trend research failed:", model, e, flush=True)
    return ""


def gemini_script(lang, fmt, ep, past_titles, letters, key, trends=""):
    L = LANG[lang]
    rules = FORMAT_RULES[fmt].format(m=L["mascot"], letters=" ".join(letters))
    prompt = f"""You write scripts for a YouTube channel for children aged 2-6. Host: {L['mascot']}, a friendly, curious owl.
Write episode {ep} in {L['name']} only (all text fields in {L['name']}, natural for native speakers, very simple words).
Format: {rules}
What is going viral with young kids right now (inspiration only): {trends or "n/a"}
Use those patterns (topics, hooks, energy, title style) so the episode feels current and catchy, and you may adapt the format toward them - but everything must be 100% ORIGINAL: never copy or mention existing characters, channels, brands, song lyrics or video titles.
Rules:
- First scene: intro (intro=true): {L['mascot']} greets the kids warmly and says what we'll do today.
- Last scene: outro (outro=true): praise the kids and say goodbye until next time. Do NOT ask to like or subscribe.
- 'say' = what {L['mascot']} says aloud: max 30 words, short sentences, cheerful, can include sounds like "Hooray!".
- 'show' = 1-3 standard single emoji characters that picture the scene (no text, no flags).
- 'caption' = max 4 words shown on screen. 'big' = optional single letter or number to show huge, else "".
- 'bg' = one of: meadow, forest, sea, night.
- Total 18-30 scenes. Safe, kind, no scary or violent content, no brands or known characters.
- Make it different from these past episodes: {past_titles[-12:]}
- 'title' = catchy YouTube title in {L['name']} (max 70 chars) that mentions {L['mascot']}.
Return JSON: {{"title": str, "scenes": [{{"say": str, "show": [str], "caption": str, "big": str, "bg": str,
"hide": bool, "reveal": bool, "intro": bool, "outro": bool}}]}}"""
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}}
    for attempt, model in enumerate(text_models(key)[:5]):
        try:
            print("Script model:", model, flush=True)
            r = gemini_call(model, body, key, timeout=300)
            data = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
            scenes = [s for s in data["scenes"] if s.get("say")]
            if len(scenes) >= 8:
                data["scenes"] = scenes
                return data
        except Exception as e:  # noqa: BLE001
            print(f"Gemini script attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)
    return None


# ----------------------------------------------------------------- speech
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())


def gemini_tts(text, lang, key, out_wav, raw_prompt=False):
    style = "" if raw_prompt else ("Read this aloud as a warm, playful, very cheerful host of a show for small children. "
             "Speak clearly, a little slowly, with lots of expression and smiles in your voice:\n")
    body = {"contents": [{"parts": [{"text": style + text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": LANG[lang]["gemini_voice"]}}}}}
    last = None
    for model in pick_tts_models(key):
        print("TTS model:", model, flush=True)
        for attempt in range(3):
            try:
                r = gemini_call(model, body, key, timeout=600)
                part = r["candidates"][0]["content"]["parts"][0]["inlineData"]
                pcm = base64.b64decode(part["data"])
                raw = out_wav + ".pcm"
                Path(raw).write_bytes(pcm)
                run(["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", raw, out_wav])
                return True
            except urllib.error.HTTPError as e:
                last = f"{model} HTTP {e.code}"
                body_txt = e.read().decode(errors="ignore")[:300]
                print(f"TTS {model} HTTP {e.code}: {body_txt}", flush=True)
                if e.code in (429, 500, 503) and "PerDay" not in body_txt:
                    time.sleep(30 * (attempt + 1))
                    continue
                break  # 404 etc. -> try next model
            except Exception as e:  # noqa: BLE001
                last = f"{model} {e}"
                time.sleep(5)
    print("Gemini TTS failed:", last)
    return False


async def _edge(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate="-10%", pitch="+8Hz").save(path)


def edge_tts_say(text, lang, out_wav):
    mp3 = out_wav + ".mp3"
    for attempt in range(3):
        try:
            asyncio.run(_edge(text, LANG[lang]["edge_voice"], mp3))
            run(["ffmpeg", "-y", "-i", mp3, out_wav])
            return True
        except Exception as e:  # noqa: BLE001
            print("edge-tts retry", attempt + 1, e)
            time.sleep(3)
    return False


MINIMAX_BOOST = {"en": "English", "ar": "Arabic", "he": "Hebrew"}
MINIMAX_VOICES = ["English_radiant_girl", "Lively_Girl", "English_expressive_narrator"]


def minimax_say(text, lang, out_wav, key, state):
    """MiniMax T2A v2. Returns True on success; sets state['dead'] on auth/quota errors."""
    import urllib.request
    import urllib.error
    models = ["speech-2.8-hd", "speech-2.6-hd"]
    voices = ([state["voice"]] if state.get("voice") else MINIMAX_VOICES)
    for model in ([state["model"]] if state.get("model") else models):
        for voice in voices:
            body = {"model": model, "text": text, "stream": False, "output_format": "hex",
                    "language_boost": MINIMAX_BOOST.get(lang, "auto"),
                    "voice_setting": {"voice_id": voice, "speed": 0.92, "vol": 1, "pitch": 1, "emotion": "happy"},
                    "audio_setting": {"sample_rate": 44100, "bitrate": 128000, "format": "mp3", "channel": 1}}
            req = urllib.request.Request("https://api.minimax.io/v1/t2a_v2", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            try:
                res = json.load(urllib.request.urlopen(req, timeout=120))
            except Exception as e:  # noqa: BLE001
                print("MiniMax error:", model, voice, str(e)[:120], flush=True)
                continue
            code = (res.get("base_resp") or {}).get("status_code")
            if code == 0 and (res.get("data") or {}).get("audio"):
                mp3 = out_wav + ".mp3"
                with open(mp3, "wb") as f:
                    f.write(bytes.fromhex(res["data"]["audio"]))
                run(["ffmpeg", "-y", "-i", mp3, out_wav])
                state["model"], state["voice"] = model, voice
                return True
            msg = (res.get("base_resp") or {}).get("status_msg", "")
            print("MiniMax failed:", model, voice, code, msg[:120], flush=True)
            if code in (1004, 1008, 2049):  # auth failure / insufficient balance / invalid key
                state["dead"] = True
                return False
    return False


def gemini_tts_script(lines, lang, key, out_wav):
    """ONE request for the whole episode (free-tier friendly)."""
    style = ("You are the warm, playful, very cheerful host of a show for small children. Read the lines below aloud, "
             "clearly and a little slowly, full of expression and smiles. After EACH line stay completely silent "
             "for two full seconds before the next line. Do not read anything else.\n\n")
    return gemini_tts(style + "\n\n".join(lines), lang, key, out_wav, raw_prompt=True)


def split_by_silence(a, sr, weights):
    """Cut one long narration into len(weights) parts at the best silent gaps."""
    n = len(weights)
    hop = int(0.02 * sr)
    rms = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) for i in range(0, len(a), hop)])
    thr = max(0.008, np.percentile(rms, 30) * 1.5)
    quiet = rms < thr
    gaps, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * 0.02 >= 0.3 and i > 0 and j < len(quiet):
                gaps.append(((i + j) / 2 * hop, (j - i)))
            i = j
        else:
            i += 1
    total = sum(weights)
    cum, targets = 0, []
    for w in weights[:-1]:
        cum += w
        targets.append(cum / total * len(a))
    if len(gaps) < n - 1:
        # not enough clear pauses: cut at the quietest moment near each expected boundary
        cuts, prev = [], 0
        win = int(len(rms) * 0.05)
        for tg in targets:
            c = int(tg / hop)
            lo, hi = max(prev + 5, c - win), min(len(rms) - 5, c + win)
            k = lo + int(np.argmin(rms[lo:hi])) if hi > lo else c
            cuts.append(k * hop)
            prev = k
        edges = [0] + cuts + [len(a)]
        return [a[x:y] for x, y in zip(edges[:-1], edges[1:])]
    # dynamic programming: pick n-1 gaps in order, minimise distance to targets, prefer long gaps
    G = len(gaps)
    INF = float("inf")
    cost = [[INF] * G for _ in range(n - 1)]
    back = [[-1] * G for _ in range(n - 1)]
    for g in range(G):
        cost[0][g] = abs(gaps[g][0] - targets[0]) / len(a) - gaps[g][1] * 0.002
    for k in range(1, n - 1):
        best, bi = INF, -1
        for g in range(G):
            if g - 1 >= 0 and cost[k - 1][g - 1] < best:
                best, bi = cost[k - 1][g - 1], g - 1
            if bi >= 0:
                cost[k][g] = best + abs(gaps[g][0] - targets[k]) / len(a) - gaps[g][1] * 0.002
                back[k][g] = bi
    g = min(range(G), key=lambda x: cost[n - 2][x])
    if cost[n - 2][g] == INF:
        return None
    cuts = []
    for k in range(n - 2, -1, -1):
        cuts.append(int(gaps[g][0]))
        g = back[k][g]
    cuts = sorted(cuts)
    edges = [0] + cuts + [len(a)]
    parts = []
    for x, y in zip(edges[:-1], edges[1:]):
        seg = a[x:y]
        nz = np.where(np.abs(seg) > thr)[0]
        parts.append(seg[max(0, nz[0] - hop):nz[-1] + hop] if len(nz) else seg[:hop])
    return parts


def write_wav(path, a, sr=44100):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())


def synthesize(scenes, lang, tmp):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    try:
        import edge_tts  # noqa: F401
        have_edge = True
    except ImportError:
        have_edge = False
    if key and CONFIG.get("voice") == "gemini":
        whole = os.path.join(tmp, "whole.wav")
        if gemini_tts_script([s["say"] for s in scenes], lang, key, whole):
            a = load_mono(whole)
            parts = split_by_silence(a, 44100, [max(8, len(s["say"])) for s in scenes])
            if parts:
                for i, (sc, p) in enumerate(zip(scenes, parts)):
                    sc["voice"] = os.path.join(tmp, f"v{i}.wav")
                    write_wav(sc["voice"], p)
                print("Voice engine: gemini (single request)", flush=True)
                return "gemini"
            print("Could not split Gemini narration cleanly - using fallback voice", flush=True)
    mm_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if mm_key and lang in CONFIG.get("minimax_langs", []):
        state = {}
        paths = []
        for i, sc in enumerate(scenes):
            path = os.path.join(tmp, f"v{i}.wav")
            if not minimax_say(sc["say"], lang, path, mm_key, state):
                break
            paths.append(path)
        if len(paths) == len(scenes):
            for sc, p in zip(scenes, paths):
                sc["voice"] = p
            print("Voice engine: minimax", state.get("model"), state.get("voice"), flush=True)
            return "minimax"
        print("MiniMax unavailable - falling back to free voice", flush=True)
    for engine in (["edge"] if have_edge else []) + ["silent"]:
        ok = True
        for i, sc in enumerate(scenes):
            path = os.path.join(tmp, f"v{i}.wav")
            if engine == "edge":
                ok = edge_tts_say(sc["say"], lang, path)
            else:
                dur = 0.42 * len(sc["say"].split()) + 0.6
                run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{dur:.2f}", path])
            if not ok:
                break
            sc["voice"] = path
        if ok:
            print("Voice engine:", engine, flush=True)
            return engine
    raise RuntimeError("no speech engine worked")


def load_mono(path, sr=44100):
    tmpw = path + ".44k.wav"
    run(["ffmpeg", "-y", "-i", path, "-ar", str(sr), "-ac", "1", tmpw])
    with wave.open(tmpw) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return a


# ----------------------------------------------------------------- music
def music_track(total, seed, sr=44100):
    rng = random.Random(seed)
    n = int(total * sr)
    out = np.zeros(n, dtype=np.float32)
    scale = [0, 2, 4, 7, 9, 12, 14, 16]
    base = 261.63 * 2 ** (rng.choice([0, 2, 5, 7]) / 12)
    beat = 60 / rng.choice([100, 108, 116])
    phrases = [[rng.choice(scale) for _ in range(8)] for _ in range(2)]
    prog = rng.choice([[0, 5, 7, 0], [0, 9, 5, 7], [0, 7, 9, 5]])
    t, i = 0.0, 0
    while t < total:
        bar = (i // 8) % 4
        note = phrases[(i // 32) % 2][i % 8]
        f = base * 2 ** (note / 12)
        ln = int(beat * 0.9 * sr)
        tt = np.arange(ln) / sr
        tone = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 4) * 0.10
        if i % 2 == 0:
            bf = base / 2 * 2 ** (prog[bar] / 12)
            tone += np.sin(2 * np.pi * bf * tt) * np.exp(-tt * 2.5) * 0.09
        if i % 4 == 2:  # soft shaker
            tone[: int(0.05 * sr)] += np.random.default_rng(i).normal(0, 0.02, int(0.05 * sr)) * np.exp(-tt[: int(0.05 * sr)] * 60)
        a = int(t * sr)
        seg = tone[: max(0, min(ln, n - a))]
        out[a:a + len(seg)] += seg
        t += beat / 2
        i += 1
    fade = int(2 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def chime(sr=44100):
    tt = np.arange(int(0.5 * sr)) / sr
    s = sum(np.sin(2 * np.pi * f * tt) for f in (1046.5, 1318.5, 1568)) * np.exp(-tt * 6) * 0.12
    return s.astype(np.float32)


def pop_sfx(sr=44100):
    tt = np.arange(int(0.15 * sr)) / sr
    f = 500 + 900 * tt / 0.15
    return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 25) * 0.25).astype(np.float32)


# ----------------------------------------------------------------- drawing helpers
FONT_FILES = {
    "he": ["/usr/share/fonts/truetype/noto/NotoSansHebrew-Bold.ttf"],
    "ar": ["/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"],
    "en": ["/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"],
    "any": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


def find_font(lang):
    for p in FONT_FILES.get(lang, []) + FONT_FILES["any"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("font")


class Art:
    def __init__(self, W, H, lang):
        self.W, self.H = W, H
        self.font_path = find_font(lang)
        self.fonts = {}
        self.efont = ImageFont.truetype(EMOJI_FONT, 109)
        self.ecache = {}

    def font(self, size):
        size = max(8, int(size))
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size)
        return self.fonts[size]

    def emoji(self, ch, size):
        size = max(4, int(size))
        k = (ch, size)
        if k not in self.ecache:
            base = self.ecache.get((ch, 0))
            if base is None:
                im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((10, 10), ch, font=self.efont, embedded_color=True)
                bb = im.getbbox()
                if not bb or (bb[2] - bb[0]) < 40:
                    im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
                    ImageDraw.Draw(im).text((10, 10), "⭐", font=self.efont, embedded_color=True)
                    bb = im.getbbox()
                base = im.crop(bb)
                self.ecache[(ch, 0)] = base
            r = size / max(base.size)
            self.ecache[k] = base.resize((max(1, int(base.width * r)), max(1, int(base.height * r))), Image.LANCZOS)
        return self.ecache[k]

    def text(self, img, txt, cx, cy, size, fill=(255, 255, 255), stroke=(50, 40, 90)):
        if not txt:
            return
        d = ImageDraw.Draw(img)
        f = self.font(size)
        while d.textlength(txt, font=f) > self.W * 0.86 and size > 20:
            size *= 0.9
            f = self.font(size)
        d.text((cx, cy), txt, font=f, fill=fill, anchor="mm", stroke_width=max(3, int(size / 11)), stroke_fill=stroke)


def ease_back(x):
    x = min(max(x, 0), 1)
    c = 1.7
    return max(0.0, 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient(W, H, top, bottom):
    g = np.linspace(0, 1, H)[:, None]
    arr = (np.array(top)[None, None, :] * (1 - g[..., None]) + np.array(bottom)[None, None, :] * g[..., None])
    arr = np.repeat(arr, W, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


class Backgrounds:
    def __init__(self, W, H, seed):
        self.W, self.H = W, H
        self.rng = random.Random(seed)
        self.static = {}
        self.clouds = [(self.rng.uniform(0, W), self.rng.uniform(40, 200), self.rng.uniform(0.6, 1.2), self.rng.uniform(8, 20))
                       for _ in range(5)]
        self.stars = [(self.rng.uniform(0, W), self.rng.uniform(0, H * 0.6), self.rng.uniform(1.5, 3.5), self.rng.uniform(0, 6))
                      for _ in range(90)]

    def base(self, kind):
        if kind in self.static:
            return self.static[kind]
        W, H = self.W, self.H
        if kind == "night":
            img = gradient(W, H, (20, 24, 70), (60, 50, 120))
            d = ImageDraw.Draw(img)
            d.ellipse([-200, H - 160, W * 0.7, H + 250], fill=(40, 70, 70))
            d.ellipse([W * 0.4, H - 130, W + 250, H + 250], fill=(30, 60, 60))
        elif kind == "sea":
            img = gradient(W, H, (120, 200, 255), (210, 240, 255))
            d = ImageDraw.Draw(img)
            d.rectangle([0, H * 0.58, W, H], fill=(40, 150, 220))
            d.ellipse([-100, H * 0.83, W * 0.6, H * 1.25], fill=(250, 220, 150))
        else:
            img = gradient(W, H, (110, 195, 255), (215, 240, 255))
            d = ImageDraw.Draw(img)
            d.ellipse([-250, H * 0.62, W * 0.65, H * 1.5], fill=(120, 205, 90))
            d.ellipse([W * 0.35, H * 0.66, W + 300, H * 1.5], fill=(95, 185, 75))
            r = random.Random(kind)
            for _ in range(40):
                x, y = r.uniform(0, W), r.uniform(H * 0.78, H * 0.98)
                c = r.choice([(255, 120, 160), (255, 220, 80), (255, 255, 255), (190, 140, 255)])
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=c)
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 200, 40))
            if kind == "forest":
                for x in [70, 230, W - 230, W - 80]:
                    d.rectangle([x - 18, H * 0.45, x + 18, H * 0.8], fill=(130, 85, 45))
                    for dx, dy, rr in [(0, -40, 95), (-55, 10, 75), (55, 10, 75)]:
                        d.ellipse([x + dx - rr, H * 0.42 + dy - rr, x + dx + rr, H * 0.42 + dy + rr], fill=(55, 150, 70))
        self.static[kind] = img
        return img

    def draw(self, kind, t):
        img = self.base(kind).copy()
        d = ImageDraw.Draw(img, "RGBA")
        W, H = self.W, self.H
        if kind == "night":
            d.ellipse([W - 250, 50, W - 150, 150], fill=(255, 250, 210))
            d.ellipse([W - 225, 40, W - 135, 130], fill=(30, 34, 85))
            for x, y, s, ph in self.stars:
                a = int(150 + 105 * math.sin(t * 2 + ph))
                d.ellipse([x - s, y - s, x + s, y + s], fill=(255, 255, 220, a))
            return img
        sx, sy = W - 160, 120
        ang = t * 0.4
        for k in range(12):
            a = ang + k * math.pi / 6
            d.polygon([(sx + 70 * math.cos(a - 0.08), sy + 70 * math.sin(a - 0.08)),
                       (sx + 115 * math.cos(a), sy + 115 * math.sin(a)),
                       (sx + 70 * math.cos(a + 0.08), sy + 70 * math.sin(a + 0.08))], fill=(255, 210, 60, 200))
        d.ellipse([sx - 60, sy - 60, sx + 60, sy + 60], fill=(255, 220, 70))
        for x0, y, sc, sp in self.clouds:
            x = (x0 + t * sp) % (W + 300) - 150
            for dx, dy, r in [(0, 0, 38), (40, -15, 45), (85, 0, 36), (40, 12, 40)]:
                rr = r * sc
                d.ellipse([x + dx * sc - rr, y + dy * sc - rr, x + dx * sc + rr, y + dy * sc + rr], fill=(255, 255, 255, 235))
        if kind == "sea":
            for row in range(4):
                y0 = H * 0.6 + row * 28
                pts = [(x, y0 + 6 * math.sin(x / 40 + t * 2 + row)) for x in range(0, W + 20, 20)]
                d.line(pts, fill=(255, 255, 255, 120), width=4)
        return img


class Owl:
    """Original mascot: Nuni the owl (code-drawn)."""

    def draw(self, img, cx, cy, s, t, mouth, wave=0.0):
        d = ImageDraw.Draw(img, "RGBA")
        bob = math.sin(t * 3) * 4 * s
        cy += bob
        body, belly, dark = (155, 105, 200), (235, 215, 250), (95, 60, 140)
        flap = math.sin(t * 12) * 10 * s * min(1, mouth * 3 + wave)
        d.ellipse([cx - 95 * s, cy - 15 * s - flap, cx - 45 * s, cy + 75 * s - flap], fill=dark)
        d.ellipse([cx + 45 * s, cy - 15 * s + (flap if wave == 0 else -flap * 2), cx + 95 * s, cy + 75 * s], fill=dark)
        d.polygon([(cx - 60 * s, cy - 95 * s), (cx - 75 * s, cy - 140 * s), (cx - 25 * s, cy - 105 * s)], fill=dark)
        d.polygon([(cx + 60 * s, cy - 95 * s), (cx + 75 * s, cy - 140 * s), (cx + 25 * s, cy - 105 * s)], fill=dark)
        d.ellipse([cx - 80 * s, cy - 120 * s, cx + 80 * s, cy + 100 * s], fill=body)
        d.ellipse([cx - 55 * s, cy - 20 * s, cx + 55 * s, cy + 92 * s], fill=belly)
        blink = (t % 3.7) < 0.12
        for ex in (-32, 32):
            x, y = cx + ex * s, cy - 55 * s
            d.ellipse([x - 30 * s, y - 30 * s, x + 30 * s, y + 30 * s], fill=(255, 255, 255), outline=dark, width=max(2, int(4 * s)))
            if blink:
                d.line([x - 22 * s, y, x + 22 * s, y], fill=(40, 30, 60), width=max(2, int(5 * s)))
            else:
                d.ellipse([x - 14 * s, y - 12 * s, x + 14 * s, y + 16 * s], fill=(40, 30, 60))
                d.ellipse([x - 7 * s, y - 8 * s, x, y - 1 * s], fill=(255, 255, 255))
        d.ellipse([cx - 78 * s, cy - 32 * s, cx - 50 * s, cy - 14 * s], fill=(255, 150, 190, 150))
        d.ellipse([cx + 50 * s, cy - 32 * s, cx + 78 * s, cy - 14 * s], fill=(255, 150, 190, 150))
        op = min(1.0, mouth * 4) * 16 * s
        d.polygon([(cx - 14 * s, cy - 26 * s), (cx + 14 * s, cy - 26 * s), (cx, cy - 10 * s)], fill=(255, 170, 40))
        d.polygon([(cx - 11 * s, cy - 18 * s), (cx + 11 * s, cy - 18 * s), (cx, cy - 4 * s + op)], fill=(240, 140, 30))
        for fx in (-25, 25):
            d.ellipse([cx + fx * s - 16 * s, cy + 90 * s, cx + fx * s + 16 * s, cy + 104 * s], fill=(255, 170, 40))


# ----------------------------------------------------------------- render
@dataclass
class Scene:
    data: dict
    dur: float = 0.0
    speech: float = 0.0
    env: np.ndarray = field(default_factory=lambda: np.zeros(1))


def render_video(episode, lang, out_path, seed, tmp):
    W, H, fps = CONFIG["width"], CONFIG["height"], CONFIG["fps"]
    raw = episode["scenes"]
    synthesize(raw, lang, tmp)
    scenes = []
    sr = 44100
    for sd in raw:
        a = load_mono(sd["voice"], sr)
        sp = len(a) / sr
        pause = 3.6 if sd.get("hide") else (1.4 if sd.get("reveal") else 0.8)
        dur = max(2.5, 0.35 + sp + pause)
        hop = sr // fps
        env = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) if i < len(a) else 0 for i in range(0, int(dur * sr), hop)])
        if env.max() > 0:
            env = env / max(env.max(), 1e-3)
        scenes.append(Scene(sd, dur, sp, env))
        sd["_audio"] = a

    total = sum(s.dur for s in scenes)
    n = int(total * sr) + sr
    voice = np.zeros(n, dtype=np.float32)
    sfx = np.zeros(n, dtype=np.float32)
    t0 = 0.0
    for s in scenes:
        a = s.data["_audio"]
        st = int((t0 + 0.35) * sr)
        voice[st:st + len(a)] += a[: max(0, n - st)]
        fx = chime(sr) if s.data.get("reveal") else pop_sfx(sr)
        st2 = int((t0 + (0.35 + s.speech + 0.2 if s.data.get("reveal") else 0.05)) * sr)
        if s.data.get("reveal"):
            st2 = int((t0 + 0.05) * sr)
        sfx[st2:st2 + len(fx)] += fx[: max(0, n - st2)]
        t0 += s.dur
    mus = music_track(n / sr, seed, sr)[:n]
    talk = np.convolve((np.abs(voice) > 0.02).astype(np.float32), np.ones(int(0.3 * sr)) / int(0.3 * sr), mode="same")
    mix = voice * 1.0 + sfx + mus * (1.0 - 0.65 * np.clip(talk * 3, 0, 1))
    mix = np.clip(mix / max(1.0, np.abs(mix).max() / 0.95), -1, 1)
    audio_path = os.path.join(tmp, "mix.wav")
    with wave.open(audio_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((mix * 32767).astype("<i2").tobytes())

    art = Art(W, H, lang)
    bgs = Backgrounds(W, H, seed)
    owl = Owl()
    silent = os.path.join(tmp, "video.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", silent],
                          stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gt = 0.0
    for idx, s in enumerate(scenes):
        sd = s.data
        bg = sd.get("bg") if sd.get("bg") in BGS else "meadow"
        show = [e for e in (sd.get("show") or []) if isinstance(e, str) and e.strip()][:3] or ["⭐"]
        big = (sd.get("big") or "").strip()[:2]
        cap = (sd.get("caption") or "").strip()
        is_intro, is_outro = sd.get("intro"), sd.get("outro")
        nf = int(round(s.dur * fps))
        for f in range(nf):
            t = f / fps
            img = bgs.draw(bg, gt + t)
            mouth = float(s.env[min(f, len(s.env) - 1)]) if t > 0.35 else 0.0
            if is_intro or is_outro:
                k = ease_back(t / 0.6)
                owl.draw(img, W * 0.5, H * 0.58, 1.6 * k + 0.01, gt + t, mouth, wave=1.0)
                if is_intro and episode.get("title"):
                    art.text(img, episode["title"].split("|")[0].strip(), W / 2, 90, 54)
                for i, e in enumerate(show[1:] if len(show) > 1 else []):
                    em = art.emoji(e, 120 * ease_back((t - 0.3 - i * 0.2) / 0.5))
                    img.paste(em, (int(W * 0.78 - em.width / 2), int(H * 0.45 - em.height / 2 + math.sin(t * 3 + i) * 10)), em)
            else:
                owl.draw(img, 150, H - 150, 0.85, gt + t, mouth)
                k = ease_back((t - 0.1) / 0.5)
                if sd.get("hide"):
                    pulse = 1 + 0.06 * math.sin(t * 6)
                    d = ImageDraw.Draw(img, "RGBA")
                    r = max(2, 150 * k * pulse)
                    d.ellipse([W / 2 - r, H * 0.5 - r, W / 2 + r, H * 0.5 + r], fill=(255, 255, 255, 220), outline=(255, 170, 40), width=10)
                    art.text(img, "?", W / 2, H * 0.5, 230 * k * pulse, fill=(255, 140, 40), stroke=(120, 60, 20))
                    cd_start = 0.35 + s.speech + 0.3
                    if t > cd_start:
                        num = 3 - int((t - cd_start) / 1.05)
                        if num >= 1:
                            frac = ((t - cd_start) % 1.05) / 1.05
                            art.text(img, str(num), W - 170, H - 150, 140 * (1.2 - 0.4 * frac), fill=(255, 230, 60), stroke=(150, 80, 0))
                else:
                    n_e = len(show)
                    size = {1: 300, 2: 240, 3: 200}[n_e] * (0.85 if big else 1.0)
                    xs = [W / 2] if n_e == 1 else [W / 2 + (i - (n_e - 1) / 2) * size * 1.15 for i in range(n_e)]
                    if big:
                        xs = [x + 150 for x in xs]
                        art.text(img, big, W / 2 - 230 if n_e == 1 else W * 0.3, H * 0.5, 300 * k, fill=(255, 90, 120), stroke=(255, 255, 255))
                    for i, (x, e) in enumerate(zip(xs, show)):
                        kk = ease_back((t - 0.1 - i * 0.15) / 0.5)
                        em = art.emoji(e, size * kk + 1)
                        y = H * 0.5 + math.sin((gt + t) * 2.5 + i) * 12
                        img.paste(em, (int(x - em.width / 2), int(y - em.height / 2)), em)
                    if sd.get("reveal") and t < 1.2:
                        d = ImageDraw.Draw(img, "RGBA")
                        for j in range(12):
                            a = j * math.pi / 6
                            rr = 120 + t * 260
                            x, y = W / 2 + rr * math.cos(a), H * 0.5 + rr * math.sin(a)
                            sz = 14 * (1.2 - t)
                            d.polygon([(x, y - sz), (x + sz * 0.3, y - sz * 0.3), (x + sz, y), (x + sz * 0.3, y + sz * 0.3),
                                       (x, y + sz), (x - sz * 0.3, y + sz * 0.3), (x - sz, y), (x - sz * 0.3, y - sz * 0.3)],
                                      fill=(255, 230, 80, 230))
                if cap and cap != "?":
                    art.text(img, cap, W / 2, 85, 70 * ease_back((t - 0.25) / 0.5))
            if t < 0.2 and idx > 0:
                img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), 0.6 * (1 - t / 0.2))
            ff.stdin.write(img.tobytes())
        gt += s.dur
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")
    run(["ffmpeg", "-y", "-i", silent, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         "-shortest", "-movflags", "+faststart", out_path])
    print(f"Rendered {out_path}: {len(scenes)} scenes, {total:.0f}s")
    return total


# ----------------------------------------------------------------- episode planning
def plan_episode(lang, fmt, ep, state):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    past = [h["title"] for h in state.get("history", []) if h.get("lang") == lang]
    letters = []
    if fmt == "letters":
        alpha = LANG[lang]["alphabet"]
        pos = state.setdefault("letter_pos", {}).get(lang, 0) % len(alpha)
        letters = alpha[pos:pos + 4] or alpha[:4]
    trends = youtube_trends()
    data = gemini_script(lang, fmt, ep, past, letters, key, trends) if key else None
    if not data:
        print(f"Using offline fallback script for {lang}")
        data = fallback_episode(lang)
        fmt = "animals"
    elif fmt == "letters":
        state["letter_pos"][lang] = (state["letter_pos"].get(lang, 0) + len(letters)) % len(LANG[lang]["alphabet"])
    return data, fmt


def make_video(lang, fmt, ep, state, out_path):
    data, fmt = plan_episode(lang, fmt, ep, state)
    tmp = tempfile.mkdtemp()
    try:
        render_video(data, lang, out_path, seed=ep * 7 + len(lang), tmp=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    title = data.get("title") or FALLBACK_TEXT[lang]["title"]
    return {"title": title[:100], "description": LANG[lang]["desc"], "tags": LANG[lang]["tags"], "format": fmt}


# ----------------------------------------------------------------- upload
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _refresh_token(lang=None):
    raw = (os.environ.get(f"YT_REFRESH_TOKEN_{(lang or '').upper()}") or os.environ["YT_REFRESH_TOKEN"]).strip()
    if raw.startswith("{"):
        raw = json.loads(raw)["refresh_token"].strip()
    return raw


def upload(path, title, description, tags, language, privacy="public", category_id="27"):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    creds = Credentials(token=None, refresh_token=_refresh_token(language), client_id=os.environ["YT_CLIENT_ID"].strip(),
                        client_secret=os.environ["YT_CLIENT_SECRET"].strip(),
                        token_uri="https://oauth2.googleapis.com/token", scopes=SCOPES)
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {"snippet": {"title": title, "description": description, "tags": tags, "categoryId": category_id,
                        "defaultLanguage": language, "defaultAudioLanguage": language},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": True, "containsSyntheticMedia": True}}
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
    print(f"Done: https://youtu.be/{response['id']} (privacy: {response['status']['privacyStatus']})")
    return response["id"]


# ----------------------------------------------------------------- main
def main():
    (ROOT / "out").mkdir(exist_ok=True)
    state_path = ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"episode": 0, "history": []}
    ep = state["episode"] + 1
    fmt = CONFIG["formats"][(ep - 1) % len(CONFIG["formats"])]
    dry = "--dry-run" in sys.argv
    langs = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--lang=")] or CONFIG["languages"]
    for lang in langs:
        out = ROOT / "out" / f"video_{lang}.mp4"
        print(f"=== Episode {ep}: {fmt} ({lang})")
        meta = make_video(lang, fmt, ep, state, str(out))
        print("Title:", meta["title"])
        if dry:
            continue
        vid = upload(str(out), meta["title"], meta["description"], meta["tags"], lang,
                     CONFIG["privacy"], CONFIG["category_id"])
        state["history"].append({"episode": ep, "lang": lang, "format": meta["format"], "video_id": vid,
                                 "title": meta["title"], "date": datetime.date.today().isoformat()})
    if not dry:
        state["episode"] = ep
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
