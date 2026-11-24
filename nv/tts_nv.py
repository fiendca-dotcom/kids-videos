"""Narrate nv/script.json with MiniMax (documentary narrator)."""
import json, os, time, urllib.request

KEY = os.environ["MINIMAX_API_KEY"].strip()
scenes = json.load(open("nv/script.json"))["scenes"]
os.makedirs("nv_audio", exist_ok=True)
VOICES = ["English_expressive_narrator", "Deep_Voice_Man", "English_radiant_girl"]
voice = None
for i, sc in enumerate(scenes):
    for v in ([voice] if voice else VOICES):
        body = {"model": "speech-2.8-hd", "text": sc["say"], "stream": False, "output_format": "hex",
                "language_boost": "English",
                "voice_setting": {"voice_id": v, "speed": 0.97, "vol": 1, "pitch": 0},
                "audio_setting": {"sample_rate": 44100, "bitrate": 128000, "format": "mp3", "channel": 1}}
        req = urllib.request.Request("https://api.minimax.io/v1/t2a_v2", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        res = json.load(urllib.request.urlopen(req, timeout=120))
        if (res.get("base_resp") or {}).get("status_code") == 0:
            open(f"nv_audio/s{i:02d}.mp3", "wb").write(bytes.fromhex(res["data"]["audio"]))
            voice = v
            print(i, "ok", v, flush=True)
            break
        print(i, v, res.get("base_resp"), flush=True)
        time.sleep(2)
    else:
        raise SystemExit(f"scene {i} failed")
