"""Nuni local picture generator - runs inside the GitHub runner, no API keys and no paid services.

Model: Stable Diffusion 1.5 fine-tune "DreamShaper 8" (CreativeML OpenRAIL-M) + LCM-LoRA, so a picture needs only
~6 steps on a CPU. The models are downloaded once from Hugging Face and kept in the workflow cache.
Prompts are tuned to the look of the Nuni books: cute 3D picture-book style, warm golden light, soft pastel colours.

    python artgen.py --demo     make sample pictures into out/art_demo/ (used by the art-test workflow)
"""
import os
import sys
import time

MODEL = os.environ.get("ART_MODEL", "Lykon/dreamshaper-8")
LORA = os.environ.get("ART_LORA", "latent-consistency/lcm-lora-sdv1-5")
STYLE = "cute 3d cartoon, pixar style, picture book art, adorable, soft warm light, pastel colours, fluffy, detailed"
NEG = ("text, letters, words, watermark, signature, logo, frame, border, scary, horror, creepy, angry, ugly, deformed, "
       "disfigured, mutated, extra limbs, extra heads, bad anatomy, blurry, lowres, jpeg artifacts, nsfw, violence, blood, "
       "weapon, photo, photorealistic, pig, piglet")
_P = None


def log(*a):
    print(*a, flush=True)


def available():
    if os.environ.get("NO_LOCAL_ART"):
        return False
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def pipe():
    global _P
    if _P is None:
        import torch
        from diffusers import LCMScheduler, StableDiffusionPipeline
        torch.set_num_threads(os.cpu_count() or 2)
        t0 = time.time()
        kw = dict(torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
        try:
            p = StableDiffusionPipeline.from_pretrained(MODEL, variant="fp16", **kw)
        except Exception:  # noqa: BLE001
            p = StableDiffusionPipeline.from_pretrained(MODEL, **kw)
        p.scheduler = LCMScheduler.from_config(p.scheduler.config)
        p.load_lora_weights(LORA)
        p.fuse_lora()
        p.set_progress_bar_config(disable=True)
        _P = p
        log(f"Local art model ready in {time.time() - t0:.0f}s ({MODEL} + LCM, {os.cpu_count()} CPUs)")
    return _P


def generate(prompt, w, h, seed, steps=4, guidance=1.3):
    import torch
    p = pipe()
    g = torch.Generator("cpu").manual_seed(int(seed) % (2 ** 31))
    with torch.inference_mode():
        return p(prompt=prompt, negative_prompt=NEG, width=w, height=h, num_inference_steps=steps,
                 guidance_scale=guidance, generator=g).images[0]


def item_prompt(name):
    return f"one cute {name}, {STYLE}, full body, centered, isolated on plain white background"


def bg_prompt(place):
    return f"{place}, wide landscape, {STYLE}, empty scenery, no animals, no people, storybook background"


def release():
    global _P
    _P = None
    import gc
    gc.collect()


def demo(out="out/art_demo"):
    from PIL import Image, ImageDraw
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import studio
    os.makedirs(out, exist_ok=True)
    items = ["cow", "duckling", "snowman", "banana", "red toy train"]
    places = ["sunny farm meadow with a red barn", "snowy winter forest"]
    tiles = []
    for i, nm in enumerate(items):
        t0 = time.time()
        im = generate(item_prompt(nm), 512, 512, 1000 + i)
        im.save(f"{out}/item_{i}.png")
        cut = studio.cutout(im)
        log(f"item '{nm}': {time.time() - t0:.0f}s, cutout {'ok' if cut is not None else 'FAILED'}")
        st = studio.sticker(cut, 420) if cut is not None else im.convert("RGBA").resize((420, 420))
        bgc = Image.new("RGB", (520, 520), (90, 170, 90))
        st.thumbnail((500, 500))
        bgc.paste(st, (10, 10), st)
        tiles.append(bgc)
    for i, pl in enumerate(places):
        t0 = time.time()
        im = generate(bg_prompt(pl), 912, 512, 2000 + i)
        im.save(f"{out}/bg_{i}.png")
        log(f"background '{pl}': {time.time() - t0:.0f}s")
        tiles.append(im.resize((520, 292)).crop((0, 0, 520, 292)))
    sheet = Image.new("RGB", (4 * 520, 4 * 520), (30, 30, 30))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 4) * 520, (k // 4) * 520))
    sheet.save(f"{out}/sheet.jpg", quality=88)
    log("demo sheet:", f"{out}/sheet.jpg")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
