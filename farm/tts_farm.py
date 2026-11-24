"""Narrate farm/script.json with MiniMax (one mp3 per scene)."""
import json, os, time, urllib.request

KEY = os.environ["MINIMAX_API_KEY"].strip()
scenes = json.load(open("farm/script.json"))["scenes"]
os.makedirs("farm_audio", exist_ok=True)
for i, sc in enumerate(scenes):
    body = {"model": "speech-2.8-hd", "text": sc["say"], "stream": False, "output_format": "hex",
            "language_boost": "English",
            "voice_setting": {"voice_id": "English_radiant_girl", "speed": 0.92, "vol": 1, "pitch": 1, "emotion": "happy"},
            "audio_setting": {"sample_rate": 44100, "bitrate": 128000, "format": "mp3", "channel": 1}}
    for attempt in range(4):
        req = urllib.request.Request("https://api.minimax.io/v1/t2a_v2", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        res = json.load(urllib.request.urlopen(req, timeout=120))
        code = (res.get("base_resp") or {}).get("status_code")
        if code == 0:
            open(f"farm_audio/s{i:02d}.mp3", "wb").write(bytes.fromhex(res["data"]["audio"]))
            print(i, "ok", (res.get("extra_info") or {}).get("audio_length"), flush=True)
            break
        print(i, "retry", code, res.get("base_resp"), flush=True)
        time.sleep(5)
    else:
        raise SystemExit(f"scene {i} failed")
