"""Coloring-book page generator (local SDXL model from artgen.py, no API keys).

    python coloring.py --shard=0/8      make every 8th page starting at 0 into out/coloring/
    python coloring.py --cover          make colourful cover art into out/coloring/

Pages are original drawings: princesses, unicorns and mermaids, as clean black line art on white.
"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artgen  # noqa: E402

OUT = "out/coloring"
LINE = ("coloring book page for kids, black and white line art, thick clean bold outlines, no shading, "
        "no gray, white background, cute cartoon, simple shapes, big areas to color")
NEG = ("color, colored, shading, grayscale, gradient, shadow, hatching, texture, dark fill, solid black areas, "
       "text, letters, words, watermark, signature, frame, border, photo, realistic, blurry, scary, ugly, "
       "deformed, extra limbs, bad anatomy, nsfw, nude, nudity, topless, nipples, bare chest, pig, two horns, multiple horns, 3d render, colorful")

PAGES = [
    # princesses
    "a smiling little princess with a crown and a big ball gown, standing in front of a castle",
    "a little princess having a tea party with teddy bears at a small table",
    "a fairytale castle with tall towers, flags and a bridge over a moat",
    "a little princess riding a pony through a flower meadow",
    "a little princess reading a book under a big tree with birds",
    "a little princess dancing at a ball with sparkles and stars",
    "a royal carriage shaped like a pumpkin pulled by two horses",
    "a little princess watering roses in a palace garden with butterflies",
    "a little princess holding a magic wand with stars around her",
    "a little princess with a kitten in a cozy bedroom with a canopy bed",
    "a little princess baking cupcakes in a castle kitchen",
    "a little princess and a friendly baby dragon having a picnic",
    "a jeweled crown and a tiara on a cushion with gems and hearts",
    "a little princess on a swing hanging from a tree with flowers",
    "a little princess and a frog wearing a tiny crown by a pond",
    "a little princess waving from a castle window at night with a big round moon and a few stars",
    # unicorns
    "a cute baby unicorn with a flowing mane standing on a fluffy cloud with a rainbow",
    "a cute baby unicorn with one horn running on a grassy hill with a few big flowers",
    "a cute unicorn sleeping on a cloud under a crescent moon and stars",
    "a cute unicorn eating an ice cream cone",
    "a cute unicorn with butterfly wings flying over hills",
    "a mother unicorn and a baby unicorn cuddling with hearts",
    "a cute baby unicorn with one horn wearing a flower crown next to big cute mushrooms",
    "a cute unicorn blowing out candles on a birthday cake",
    "a cute baby unicorn with one horn standing on a fluffy cloud under a rainbow",
    "a cute unicorn reading a book with glasses in a cozy library",
    "a cute baby unicorn with one horn and a little bunny holding balloons together",
    "a cute unicorn splashing in a pond with lily pads and a frog",
    "a cute unicorn in the snow building a snowman",
    "a cute unicorn with a big bow sitting among cupcakes and candy",
    "a cute unicorn with one horn blowing dandelion seeds in a sunny meadow",
    "a cute baby unicorn with one horn sleeping on a crescent moon with stars",
    # mermaids
    "a cute little mermaid wearing a seashell top sitting on a rock in the sea with waves and seagulls",
    "a cute little mermaid wearing a seashell top swimming with a happy dolphin",
    "a cute little mermaid wearing a seashell top and a sea turtle exploring a coral reef",
    "an underwater castle made of shells and coral with little fish",
    "a cute little mermaid wearing a seashell top combing her long hair with a seashell comb",
    "a cute little mermaid wearing a seashell top blowing bubbles with a smiling starfish and a little fish",
    "a cute little mermaid wearing a seashell top playing with a friendly octopus",
    "a cute little mermaid wearing a seashell top riding a seahorse",
    "a cute chibi mermaid girl with a long fish tail wearing a seashell top swimming next to a smiling crab and big seashells",
    "a cute little mermaid wearing a seashell top reading a book in an underwater garden",
    "a cute little mermaid wearing a seashell top waving from inside a giant seashell",
    "a cute little mermaid wearing a seashell top and a baby whale blowing bubbles",
    "a cute little mermaid wearing a seashell top finding a treasure chest with jewels",
    "a cute little mermaid wearing a seashell top with a starfish crown and jellyfish friends",
    "a cute chibi mermaid girl with a long fish tail wearing a seashell top swimming under a big round moon with bubbles and stars",
    "a cute little mermaid wearing a seashell top and a puffer fish making music with shells",
]


def clean(im):
    """Render -> pure black lines on white; returns (image, black ratio).

    Thin dark strokes are kept (black-hat filter); big dark fills are turned into outlines,
    so a page that came out shaded still becomes a usable colouring page."""
    import cv2
    g = np.array(im.convert("L").resize((im.width * 2, im.height * 2), Image.LANCZOS))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    simple = g < 165
    if 0.03 <= simple.mean() <= 0.20:  # already clean line art: keep it as drawn
        ink = cv2.medianBlur(simple.astype(np.uint8) * 255, 5) > 0
        return Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)), float(ink.mean())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    lines = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k) > 50
    dark = (g < 100).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    rim = cv2.morphologyEx(dark, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)) > 0
    ink = (lines | rim).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 60
    ink = keep[lab]
    out = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
    return out, float(ink.mean())


def page(i, attempt):
    prompt = f"{PAGES[i]}, {LINE}"
    neg = NEG + (", legs, feet, knees, toes, human legs" if "mermaid" in PAGES[i] else "")
    old = artgen.NEG
    artgen.NEG = NEG
    try:
        if not artgen.XL:
            artgen.pipe()
        if artgen.XL:  # 0.8 MP is plenty for line art and keeps memory well inside 16 GB
            import torch
            g = torch.Generator("cpu").manual_seed(5000 + i * 37 + attempt * 991)
            with torch.inference_mode():
                return artgen.pipe()(prompt=prompt, negative_prompt=neg, width=768, height=1024,
                                     num_inference_steps=5, guidance_scale=1.4, generator=g).images[0]
        return artgen.generate(prompt, 768, 1024, 5000 + i * 37 + attempt * 991)
    finally:
        artgen.NEG = old


def main():
    os.environ.setdefault("ART_XL", "1")
    os.makedirs(OUT, exist_ok=True)
    if "--cover" in sys.argv:
        artgen.NEG = artgen.NEG + ", legs, feet, nude, topless"
        for k, (name, p) in enumerate([
            ("cover_unicorn", "a cute baby unicorn with rainbow mane, isolated on plain white background"),
            ("cover_mermaid", "a cute little mermaid wearing a seashell top girl with short curly blonde hair, a lavender purple tail and a "
             "starfish hairclip, full body, waving, isolated on plain white background"),
            ("cover_princess", "a cute little girl princess with black curly hair in a mint green ball gown with a flower crown, full body "
             "from head to shoes, standing and smiling, isolated on plain white background"),
            ("cover_castle", "a small whimsical pastel cottage castle with round towers and heart shaped windows on a "
             "green hill, wide scene, sunny sky"),
        ]):
            w, h = (1024, 768) if name == "cover_castle" else (832, 832)
            artgen.generate(f"{p}, {artgen.STYLE}", w, h, 900 + k).save(f"{OUT}/{name}.png")
            print("made", name, flush=True)
        return
    if "--one=" in " ".join(sys.argv):
        one(int(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--one="))))
        return
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, n = map(int, shard.split("/"))
    todo = list(range(k, len(PAGES), n))
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pages=")), "")
    if not only and os.path.exists("coloring_todo.txt"):  # re-run only the pages listed there
        only = open("coloring_todo.txt").read().strip()
    if only:
        todo = [int(x) for x in only.split(",") if x.strip()][k::n]
    import subprocess
    env = dict(os.environ, MALLOC_ARENA_MAX="2")
    for i in todo:  # one process per page: the model is re-loaded, but memory never piles up
        subprocess.run([sys.executable, os.path.abspath(__file__), f"--one={i}"], env=env, check=False)


def colorful(im):
    """True when the model drew a coloured picture instead of line art."""
    hsv = np.array(im.convert("HSV").resize((192, 256)))
    return float((hsv[..., 1] > 60).mean()) > 0.15


def one(i):
    best = None
    for attempt in range(6):
        t0 = time.time()
        raw = page(i, attempt)
        bw, ratio = clean(raw)
        col = colorful(raw)
        print(f"page {i:02d} try {attempt}: black {ratio:.2f} colour {col} ({time.time() - t0:.0f}s)", flush=True)
        score = abs(ratio - 0.09) + (1.0 if col else 0.0)
        if best is None or score < best[3]:
            best = (bw, ratio, raw, score)
        if not col and 0.03 <= ratio <= 0.20:
            break
    best[0].save(f"{OUT}/page_{i:02d}.png")
    best[2].save(f"{OUT}/raw_{i:02d}.jpg", quality=85)


if __name__ == "__main__":
    main()
