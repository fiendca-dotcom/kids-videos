"""Generate candidate designs of Nuni (original owl mascot) with MiniMax image-01."""
import base64, json, os, urllib.request

KEY = os.environ["MINIMAX_API_KEY"].strip()
BASE = ("3D animated feature film style, high quality CGI render, soft studio lighting, subsurface scattering, "
        "adorable original baby owl character, chubby round fluffy body, very big round sparkling eyes, tiny orange beak, "
        "small soft wings, friendly happy smile, full body standing, front view, simple soft pastel background, "
        "kid friendly, cute, no text, no logo")
VARIANTS = {
    "A_teal": "soft teal and cream feathers, little tuft on head",
    "B_purple": "lavender purple and white feathers, round glasses-shaped markings around eyes",
    "C_orange": "warm orange and cream feathers, tiny red scarf",
    "D_blue": "sky blue and white feathers, star-shaped feather on chest",
}
os.makedirs("design", exist_ok=True)
for name, extra in VARIANTS.items():
    body = {"model": "image-01", "prompt": BASE + ", " + extra, "aspect_ratio": "1:1",
            "response_format": "base64", "n": 2, "prompt_optimizer": False}
    req = urllib.request.Request("https://api.minimax.io/v1/image_generation", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    res = json.load(urllib.request.urlopen(req, timeout=180))
    print(name, res.get("base_resp"), flush=True)
    for i, b64 in enumerate((res.get("data") or {}).get("image_base64") or []):
        with open(f"design/nuni_{name}_{i + 1}.jpg", "wb") as f:
            f.write(base64.b64decode(b64))
print(sorted(os.listdir("design")))
