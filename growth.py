"""Growth brain for the Nuni channel.

Before every episode:
  1. trend research - what kids' videos are getting the most views per day right now (public RSS of big
     kids channels, no key needed) + a Google-Search-grounded Gemini summary of current patterns (cached 3 days)
  2. our own results - views per day of our uploaded videos (public RSS of our channel), averaged per format
  3. strategy - Gemini picks the format, a fresh specific theme (seasonal events count), the title angle,
     keywords and a first-5-seconds hook; the script writer then builds the episode around that brief
  4. SEO - description with chapters (timestamps) and hashtags
Everything fails soft: if anything is missing, the bot simply continues with its normal rotation.
"""
import datetime
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

KIDS_CHANNELS = [
    "UCbCmjCuTUZos6Inko4u57UQ",  # CoComelon
    "UCLsooMJoIpl_7ux2jvdPB-Q",  # Super Simple Songs
    "UCcdwLMPsaU2ezNSJU1nFoBQ",  # Pinkfong
    "UCKAqou7V9FAWXpZd9xtOg3Q",  # Little Baby Bum
    "UCBnZ16ahKA2DZ_T5W0FPUXg",  # ChuChu TV
    "UC5PYHgAzJ1wLEidB58SK6Xw",  # Blippi
    "UCpYye8D5fFMUPf9nSfgd4bA",  # BabyBus
    "UCG2CL6EUjG8TVT1Tpl9nJdg",  # Ms Rachel
    "UC4NALVCmcmL5ntpV0thoH6w",  # Dave and Ava
    "UCeTCWDGkW0rZ4bDQ6LJmNYQ",  # Bebefinn
]
FORMAT_INFO = {
    "animals": "animal sounds guessing game (silhouette + sound, then reveal)",
    "letters": "ABC / alphabet lesson with words and pictures",
    "riddles": "'Who am I?' riddle game with picture clues and silhouettes",
    "story": "short gentle story with a lesson",
}
NS = {"a": "http://www.w3.org/2005/Atom", "m": "http://search.yahoo.com/mrss/", "yt": "http://www.youtube.com/xml/schemas/2015"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def log(*a):
    print(*a, flush=True)


def _get(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def feed(channel_id):
    """[(title, views, age_days, video_id)] from a channel's public RSS."""
    out = []
    root = ET.fromstring(_get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"))
    now = datetime.datetime.now(datetime.timezone.utc)
    for e in root.findall("a:entry", NS):
        title = e.findtext("a:title", "", NS)
        vid = e.findtext("yt:videoId", "", NS)
        pub = datetime.datetime.fromisoformat(e.findtext("a:published", "", NS).replace("Z", "+00:00"))
        st = e.find("m:group/m:community/m:statistics", NS)
        views = int(st.get("views", 0)) if st is not None else 0
        out.append((title, views, max(0.25, (now - pub).total_seconds() / 86400), vid))
    return out


def trending(max_age=30, top=15):
    rows = []
    for cid in KIDS_CHANNELS:
        try:
            rows += [r for r in feed(cid) if r[2] <= max_age]
        except Exception as ex:  # noqa: BLE001
            log("  trend feed failed:", cid, str(ex)[:80])
    rows.sort(key=lambda r: r[1] / r[2], reverse=True)
    return [f"- {t} ({v / d / 1000:.0f}K views/day, {d:.0f} days old)" for t, v, d, _ in rows[:top]]


def channel_id(state):
    if state.get("channel_id"):
        return state["channel_id"]
    for h in reversed(state.get("history", [])):
        vid = h.get("video_id")
        if not vid:
            continue
        try:
            html = _get(f"https://www.youtube.com/watch?v={vid}").decode("utf-8", "ignore")
            m = re.search(r'"channelId":"(UC[\w-]{22})"', html) or re.search(r'"externalId":"(UC[\w-]{22})"', html)
            if m:
                state["channel_id"] = m.group(1)
                log("Our channel:", m.group(1))
                return m.group(1)
        except Exception as ex:  # noqa: BLE001
            log("  channel lookup failed:", str(ex)[:80])
        break
    return None


def own_results(state):
    cid = channel_id(state)
    if not cid:
        return [], {}
    try:
        rows = feed(cid)
    except Exception as ex:  # noqa: BLE001
        log("  own feed failed:", str(ex)[:80])
        return [], {}
    fmt_of = {h.get("video_id"): h.get("format") for h in state.get("history", [])}
    lines, per = [], {}
    for t, v, d, vid in rows[:15]:
        f = fmt_of.get(vid, "?")
        lines.append(f"- {t} [{f}] {v} views in {d:.0f} days ({v / d:.0f}/day)")
        if f in FORMAT_INFO:
            per.setdefault(f, []).append(v / d)
    avg = {f: sum(x) / len(x) for f, x in per.items()}
    state["format_scores"] = {f: round(x, 1) for f, x in avg.items()}
    return lines, avg


def web_trends(lang, key, bot, state):
    cache = state.get("trend_cache") or {}
    today = datetime.date.today()
    if cache.get("lang") == lang and cache.get("date") and (today - datetime.date.fromisoformat(cache["date"])).days < 3:
        return cache.get("text", "")
    txt = ""
    try:
        txt = bot.gemini_trends(lang, key) or ""
    except Exception as ex:  # noqa: BLE001
        log("  web trends failed:", str(ex)[:80])
    if txt:
        state["trend_cache"] = {"lang": lang, "date": today.isoformat(), "text": txt[:1500]}
    return txt


def plan(lang, fmt, ep, state, key, bot):
    """-> {"format", "theme", "title", "keywords", "hook", "brief"} or {}"""
    t0 = time.time()
    trend = trending()
    own, scores = own_results(state)
    web = web_trends(lang, key, bot, state)
    past = [h for h in state.get("history", []) if h.get("lang") == lang]
    last_fmt = past[-1].get("format") if past else None
    formats = bot.CONFIG["formats"]
    today = datetime.date.today()
    prompt = f"""You are the content strategist of a YouTube channel for children aged 2-6, hosted by Nuni, a cute baby owl.
Language of the episode: {bot.LANG[lang]['name']}. Today is {today.strftime('%A %d %B %Y')}.
Goal: the most views and watch time. Pick what to make for episode {ep}.

Formats we can produce: {json.dumps({f: FORMAT_INFO.get(f, f) for f in formats})}
Default next format (rotation): {fmt}. Last episode's format: {last_fmt}.
Our own results so far (views/day per format: {json.dumps(scores)}):
{chr(10).join(own) or "- no data yet"}
Fastest-growing videos on the biggest kids channels right now (views per day):
{chr(10).join(trend) or "- n/a"}
Web research on current viral patterns:
{web or "n/a"}
Our past titles (do not repeat their themes): {json.dumps([h.get('title') for h in past[-12:]], ensure_ascii=False)}

Rules: choose ONE format from the list (not the same as the last episode's format unless the data clearly says it wins),
and ONE fresh, concrete, kid-friendly theme that fits what is trending or a seasonal moment in the next 3 weeks
(e.g. holidays, winter, snow, a popular topic like dinosaurs, vehicles, colours, bedtime...). 100% original - never use
existing characters, brands, channels or song lyrics. The title must be catchy, searchable, max 70 characters, include "Nuni".
Return JSON: {{"format": str, "theme": str, "title": str, "keywords": [str], "hook": str, "thumb_text": str, "why": str}}
"hook" = Nuni's first sentence (max 15 words) that makes kids stay. "thumb_text" = max 3 big words for the thumbnail."""
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}}
    for model in bot.text_models(key)[:3]:
        try:
            r = bot.gemini_call(model, body, key, timeout=180)
            data = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
            if isinstance(data, list):
                data = data[0]
            if data.get("format") not in formats:
                data["format"] = fmt
            kw = [k for k in (data.get("keywords") or []) if isinstance(k, str)][:12]
            data["keywords"] = kw
            data["brief"] = (f"THIS EPISODE'S THEME (build the whole episode around it): {data.get('theme', '')}\n"
                             f"Title idea: {data.get('title', '')}\nOpening hook for the first scene: {data.get('hook', '')}\n"
                             f"Keywords to use naturally: {', '.join(kw)}\n")
            log(f"Strategy ({time.time() - t0:.0f}s): {data.get('format')} - {data.get('theme')} | {data.get('title')}")
            log("  why:", str(data.get("why", ""))[:300])
            return data
        except Exception as ex:  # noqa: BLE001
            log("  strategy failed:", model, str(ex)[:120])
    return {}


def description(data, lang, bot, chapters):
    L = bot.LANG[lang]
    desc = (data.get("description") or "").strip() or L["desc"]
    parts = [desc]
    if chapters:
        lines = []
        for t, lab in chapters:
            m, s = divmod(int(t), 60)
            lines.append(f"{m}:{s:02d} {lab}")
        parts.append("\n".join(lines))
    tags = [t for t in (data.get("tags") or []) if isinstance(t, str)]
    tags += [t for t in (data.get("keywords") or []) if isinstance(t, str)]
    tags += L["tags"]
    seen, clean = set(), []
    for t in tags:
        t = t.strip().lstrip("#")
        if t and t.lower() not in seen and len(t) < 40:
            seen.add(t.lower())
            clean.append(t)
    hashtags = " ".join("#" + re.sub(r"\W+", "", t) for t in clean[:4] if re.sub(r"\W+", "", t))
    if hashtags:
        parts.append(hashtags)
    text = "\n\n".join(parts)
    total, out_tags = 0, []
    for t in clean:
        if total + len(t) + 2 > 450:
            break
        out_tags.append(t)
        total += len(t) + 2
    return text[:4900], out_tags
