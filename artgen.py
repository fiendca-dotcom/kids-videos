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
XL_MODEL = os.environ.get("ART_XL_MODEL", "Lykon/dreamshaper-xl-lightning")


def big_machine():
    """SDXL needs ~11 GB RAM; public-repo GitHub runners have 16 GB and 4 CPUs."""
    if os.environ.get("ART_XL") != "1":  # SDXL is ~10 min/image on CPU: nightly library only
        return False
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        mem = 0
    return mem >= 14 and (os.cpu_count() or 1) >= 4
STYLE = "cute 3d cartoon, pixar style, picture book art, adorable, soft warm light, pastel colours, fluffy, detailed"
NEG = ("text, letters, words, watermark, signature, logo, frame, border, scary, horror, creepy, angry, ugly, deformed, "
       "disfigured, mutated, extra limbs, extra heads, bad anatomy, blurry, lowres, jpeg artifacts, nsfw, violence, blood, "
       "weapon, photo, photorealistic, pig, piglet, shadow, floor, ground, reflection, glow")
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


XL = False


def pipe():
    global _P, XL
    if _P is None and big_machine():
        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
            torch.set_num_threads(os.cpu_count() or 4)
            t0 = time.time()
            kw = dict(torch_dtype=torch.float32)
            try:
                p = StableDiffusionXLPipeline.from_pretrained(XL_MODEL, variant="fp16", **kw)
            except Exception:  # noqa: BLE001
                p = StableDiffusionXLPipeline.from_pretrained(XL_MODEL, **kw)
            p.scheduler = DPMSolverMultistepScheduler.from_config(
                p.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
            p.set_progress_bar_config(disable=True)
            _P, XL = p, True
            log(f"Local art model ready in {time.time() - t0:.0f}s ({XL_MODEL}, SDXL, {os.cpu_count()} CPUs)")
        except Exception as e:  # noqa: BLE001
            log("SDXL not available, using SD 1.5:", str(e)[:200])
            _P = None
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
        if XL:  # SDXL works natively around 1 megapixel
            k = (1024 * 1024 / (w * h)) ** 0.5
            W, H = int(w * k) // 64 * 64, int(h * k) // 64 * 64
            return p(prompt=prompt, negative_prompt=NEG, width=W, height=H, num_inference_steps=5,
                     guidance_scale=2.0, generator=g).images[0]
        return p(prompt=prompt, negative_prompt=NEG, width=w, height=h, num_inference_steps=steps,
                 guidance_scale=guidance, generator=g).images[0]


ANIMALS = set("""cow calf sheep lamb horse donkey goat duck duckling chick hen rooster dog puppy cat kitten rabbit frog
turtle snail butterfly bee ladybug caterpillar squirrel hedgehog mouse fox deer bear raccoon beaver elephant giraffe lion
tiger zebra monkey hippo rhino crocodile panda koala kangaroo camel parrot flamingo peacock swan toucan dinosaur fish whale
dolphin octopus crab starfish seal jellyfish seahorse shark penguin reindeer owl""".split())
HINTS = {
    "cow": "black and white spotted dairy cow with pink nose", "calf": "baby black and white spotted calf",
    "sheep": "fluffy white sheep with woolly coat", "horse": "brown horse with long mane", "duck": "brown mallard duck",
    "duckling": "fluffy yellow duckling", "chick": "fluffy yellow baby chick", "hen": "brown hen chicken",
    "rooster": "colourful rooster with red comb", "dog": "golden puppy dog", "cat": "orange tabby cat",
    "polar bear": "white polar bear", "arctic fox": "white arctic fox", "reindeer": "brown reindeer with antlers",
    "banana": "yellow banana fruit", "apple": "shiny red apple fruit with a leaf", "orange": "orange citrus fruit",
    "grapes": "bunch of purple grapes", "pear": "green pear fruit", "cherry": "two red cherries", "lemon": "yellow lemon fruit",
    "strawberry": "red strawberry", "watermelon": "slice of watermelon", "pineapple": "pineapple fruit",
    "carrot": "orange carrot with green leaves", "corn": "corn cob", "peas": "green pea pod", "tomato": "red tomato",
    "mushroom": "red mushroom with white spots", "snowman": "snowman with carrot nose, scarf and hat",
    "mitten": "pair of knitted red mittens", "winter hat": "knitted winter hat with pom pom", "scarf": "striped knitted scarf",
    "sled": "wooden sled", "gift box": "gift box with red ribbon bow", "hot cocoa": "mug of hot cocoa with marshmallows",
    "bell": "golden bell", "star": "golden star", "moon": "crescent moon", "sun": "smiling sun", "cloud": "fluffy white cloud",
    "snowflake": "big white and pale blue six-pointed snowflake crystal, flat ornament",
    "sparkling blue snowflake": "big white and pale blue six-pointed snowflake crystal, flat ornament",
    "ice skate": "white ice skate boot with silver blade", "penguin": "black and white baby penguin with orange feet",
    "fox": "orange fox with white chest and fluffy tail", "deer": "brown baby deer fawn with white spots",
    "bunny": "fluffy white bunny rabbit", "owl": "brown baby owl",
    "ball": "colourful beach ball", "key": "golden key", "book": "children's book", "dinosaur": "friendly green dinosaur",
}


def item_prompt(name):
    desc = HINTS.get(name)
    if desc is None:  # "cute penguin standing on ice" -> hint of its main word
        key = next((w for w in reversed(name.split()) if w in HINTS), None)
        desc = name if key is None else f"{name}, {HINTS[key]}"
    if name in ANIMALS or any(w in ANIMALS for w in name.split()):
        return f"one cute baby {desc}, {STYLE}, full body, centered, isolated on plain white background"
    return f"one {desc}, simple cute 3d cartoon object, {STYLE}, centered, isolated on plain white background"


def bg_prompt(place):
    return (f"{place}, children's storybook illustration, soft digital painting, painterly, warm golden light, "
            "pastel colours, cute cartoon environment, wide background, no animals, no people")


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
