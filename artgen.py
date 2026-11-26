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
    if os.environ.get("ART_XL") == "0":
        return False
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        mem = 0
    return mem >= 14 and (os.cpu_count() or 1) >= 4


STYLE = "pixar style 3d render, disney pixar animation, adorable, big expressive eyes, soft warm light, vibrant, highly detailed"
NEG = ("text, letters, words, watermark, signature, logo, frame, border, scary, horror, creepy, angry, ugly, deformed, "
       "disfigured, mutated, extra limbs, extra heads, bad anatomy, blurry, lowres, jpeg artifacts, nsfw, violence, blood, "
       "weapon, photo, photorealistic, pig, piglet, shadow, floor, ground, reflection, glow")
STYLE_OBJ = "pixar style 3d render, smooth clean toy-like shapes, soft warm light, vibrant colours, highly detailed"
OBJ_MARK = "simple cute 3d cartoon object"
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
            from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
            torch.set_num_threads(os.cpu_count() or 4)
            t0 = time.time()
            kw = dict(torch_dtype=torch.float32)
            try:
                p = StableDiffusionXLPipeline.from_pretrained(XL_MODEL, variant="fp16", **kw)
            except Exception:  # noqa: BLE001
                p = StableDiffusionXLPipeline.from_pretrained(XL_MODEL, **kw)
            # Lightning models are distilled for Euler with "trailing" timesteps and low guidance;
            # other samplers / higher guidance give burnt colours and halos
            p.scheduler = EulerDiscreteScheduler.from_config(p.scheduler.config, timestep_spacing="trailing")
            try:  # tiny VAE: decoding a 1 MP picture on CPU drops from ~60 s to ~2 s
                from diffusers import AutoencoderTiny
                p.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float32)
            except Exception as e:  # noqa: BLE001
                log("tiny VAE not available:", str(e)[:120])
            p.unet.to(memory_format=torch.channels_last)
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
        if XL:  # SDXL works natively around 1 megapixel; objects are cut out and shown <= 600 px
            k = ((832 * 832 if w == h else 1024 * 1024) / (w * h)) ** 0.5
            W, H = int(w * k) // 64 * 64, int(h * k) // 64 * 64
            neg = NEG + (", face, eyes, mouth, cartoon face" if OBJ_MARK in prompt else "")
            return p(prompt=prompt, negative_prompt=neg, width=W, height=H, num_inference_steps=5,
                     guidance_scale=1.4, generator=g).images[0]
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


# white subjects vanish on a white backdrop - paint those on soft blue so the cutter sees their edge
WHITE = set("""snowman snowflake polar sheep lamb seal swan cloud bunny rabbit arctic milk ghost egg goat igloo
marshmallow dove unicorn ice skate skates iceberg tooth toothbrush""".split())


def backdrop(name):
    return ("plain flat soft sky-blue background" if any(w in WHITE for w in name.lower().split())
            else "plain flat pure white background")


def item_prompt(name):
    desc = HINTS.get(name)
    if desc is None:  # "cute penguin standing on ice" -> hint of its main word
        key = next((w for w in reversed(name.split()) if w in HINTS), None)
        desc = name if key is None else f"{name}, {HINTS[key]}"
    if name in ANIMALS or any(w in ANIMALS for w in name.split()):
        return f"one cute baby {desc}, full body, whole body visible, centered, isolated on {backdrop(name)}, no shadow, {STYLE}"
    if "snowman" in name:
        return f"one {desc}, full body from head to base, centered, isolated on {backdrop(name)}, no shadow, {STYLE}"
    return f"one {desc}, whole object visible, centered, isolated on {backdrop(name)}, no shadow, {OBJ_MARK}, {STYLE_OBJ}"


def bg_prompt(place):
    return (f"{place}, pixar style 3d animated movie background, disney pixar environment art, warm golden light, "
            "vibrant colours, highly detailed, wide shot, empty scene, no animals, no people")


_CLIP = None
DISTRACT = ("cow dog cat duck bird fish horse sheep rabbit bear penguin fox lion elephant frog turtle owl apple banana "
            "potato carrot ball car truck train house tree flower snowman hat sock cup star").split()


def matches(im, name):
    """Quality gate: does the picture really show `name`? CLIP must rank it above ~30 other subjects
    and above 'deformed / blurry / pig'. Unknown problems (no CLIP) never block a picture."""
    global _CLIP
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if _CLIP is None:
            mid = "openai/clip-vit-base-patch32"
            _CLIP = (CLIPModel.from_pretrained(mid).eval(), CLIPProcessor.from_pretrained(mid))
        model, proc = _CLIP
        words = set(name.lower().split())
        labels = [name] + [d for d in DISTRACT if d not in words and not any(d in w or w in d for w in words)]
        labels += ["a blurry deformed shape", "a pig"]
        texts = [f"a cartoon picture of {'an' if lb[0] in 'aeiou' else 'a'} {lb}" if not lb.startswith("a ") else lb
                 for lb in labels]
        with torch.inference_mode():
            x = proc(text=texts, images=im.convert("RGB"), return_tensors="pt", padding=True)
            pr = model(**x).logits_per_image.softmax(-1)[0]
        ok = int(pr.argmax()) == 0
        if not ok:
            log(f"  quality gate: '{name}' looks like '{labels[int(pr.argmax())]}' ({float(pr.max()):.2f})")
        return ok
    except Exception as e:  # noqa: BLE001
        log("  quality gate skipped:", str(e)[:120])
        return True


def release():
    global _P, _CLIP
    _P = None
    _CLIP = None
    import gc
    gc.collect()


DEMO = ["polar bear", "snowman", "penguin", "sled", "hot cocoa", "christmas tree", "frog", "donkey", "snowflake",
        "reindeer"]


def demo(out="out/art_demo", names=None):
    """The real episode path (generate -> AI cut-out -> quality gates -> sticker / silhouette) for 10 subjects
    that came out broken before. Writes each picture and one contact sheet."""
    import tempfile
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import studio
    names = names or DEMO
    os.makedirs(out, exist_ok=True)
    lib = studio.ArtLib(tempfile.mkdtemp(), "", 10 ** 6)
    lib.models, lib.poll_on = [], False
    bg = Image.open(sorted(lib_bgs())[0]).convert("RGB") if lib_bgs() else None
    T = 520
    sheet = Image.new("RGB", (5 * T, 2 * (T + 220)), (245, 240, 230))
    for k, nm in enumerate(names):
        t0 = time.time()
        cut = lib.item(nm, exact=True)
        log(f"item '{nm}': {'ok' if cut is not None else 'FAILED'} ({time.time() - t0:.0f}s)")
        tile = (bg.resize((T * 16 // 9, T)).crop((T * 7 // 18, 0, T * 7 // 18 + T, T)) if bg
                else Image.new("RGB", (T, T), (120, 170, 210)))
        low = Image.new("RGB", (T, 220), (245, 240, 230))
        if cut is not None:
            cut.save(f"{out}/{k + 1:02d}_{nm.replace(' ', '_')}.png")
            st = studio.sticker(cut, int(T * 0.78))
            st.thumbnail((T - 10, T - 10))
            tile.paste(st, ((T - st.width) // 2, (T - st.height) // 2), st)
            sil = studio.sticker(cut, 190, silhouette=True)
            sil.thumbnail((210, 210))
            low.paste(sil, ((T - sil.width) // 2, (220 - sil.height) // 2), sil)
        x, y = (k % 5) * T, (k // 5) * (T + 220)
        sheet.paste(tile, (x, y))
        sheet.paste(low, (x, y + T))
    sheet.save(f"{out}/sheet.jpg", quality=90)
    log("demo sheet:", f"{out}/sheet.jpg")


def lib_bgs():
    import glob
    return glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/lib/bg/*.jpg"))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
