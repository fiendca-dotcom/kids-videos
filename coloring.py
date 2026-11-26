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
       "deformed, extra limbs, bad anatomy, nsfw, pig")

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
    "a little princess looking at the moon and stars from a castle balcony",
    # unicorns
    "a cute baby unicorn with a flowing mane standing on a fluffy cloud with a rainbow",
    "a cute unicorn galloping through a field of flowers",
    "a cute unicorn sleeping on a cloud under a crescent moon and stars",
    "a cute unicorn eating an ice cream cone",
    "a cute unicorn with butterfly wings flying over hills",
    "a mother unicorn and a baby unicorn cuddling with hearts",
    "a cute unicorn wearing a flower crown in a magical forest with mushrooms",
    "a cute unicorn blowing out candles on a birthday cake",
    "a cute unicorn jumping over a rainbow with clouds",
    "a cute unicorn reading a book with glasses in a cozy library",
    "a cute unicorn and a bunny playing with balloons",
    "a cute unicorn splashing in a pond with lily pads and a frog",
    "a cute unicorn in the snow building a snowman",
    "a cute unicorn with a big bow sitting among cupcakes and candy",
    "a cute unicorn looking into a magic mirror with sparkles",
    "a cute unicorn sitting on a crescent moon with stars",
    # mermaids
    "a cute little mermaid sitting on a rock in the sea with waves and seagulls",
    "a cute little mermaid swimming with a happy dolphin",
    "a cute little mermaid and a sea turtle exploring a coral reef",
    "an underwater castle made of shells and coral with little fish",
    "a cute little mermaid combing her long hair with a seashell comb",
    "a cute little mermaid holding a big pearl in an open clam shell",
    "a cute little mermaid playing with a friendly octopus",
    "a cute little mermaid riding a seahorse",
    "a cute little mermaid and a crab having a picnic on the beach",
    "a cute little mermaid reading a book in an underwater garden",
    "a cute little mermaid waving from inside a giant seashell",
    "a cute little mermaid and a baby whale blowing bubbles",
    "a cute little mermaid finding a treasure chest with jewels",
    "a cute little mermaid with a starfish crown and jellyfish friends",
    "a cute little mermaid looking at the moon from the sea at night",
    "a cute little mermaid and a puffer fish making music with shells",
]


def clean(im):
    """Grayscale render -> pure black lines on white; returns (image, black ratio)."""
    g = im.convert("L")
    g = g.resize((g.width * 2, g.height * 2), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2))
    a = np.array(g)
    bw = (a > 165).astype(np.uint8) * 255
    out = Image.fromarray(bw).filter(ImageFilter.MedianFilter(5))
    ratio = 1 - np.array(out).mean() / 255
    return out, float(ratio)


def page(i, attempt):
    prompt = f"{PAGES[i]}, {LINE}"
    old = artgen.NEG
    artgen.NEG = NEG
    try:
        return artgen.generate(prompt, 768, 1024, 5000 + i * 37 + attempt * 991)
    finally:
        artgen.NEG = old


def main():
    os.environ.setdefault("ART_XL", "1")
    os.makedirs(OUT, exist_ok=True)
    if "--cover" in sys.argv:
        for k, (name, p) in enumerate([
            ("cover_unicorn", "a cute baby unicorn with rainbow mane, isolated on plain white background"),
            ("cover_mermaid", "a cute little mermaid with red hair and teal tail waving, isolated on plain white background"),
            ("cover_princess", "a cute little princess in a pink gown with a crown, isolated on plain white background"),
            ("cover_castle", "a magical pastel fairytale castle on clouds with a rainbow, wide scene"),
        ]):
            w, h = (1024, 768) if name == "cover_castle" else (832, 832)
            artgen.generate(f"{p}, {artgen.STYLE}", w, h, 900 + k).save(f"{OUT}/{name}.png")
            print("made", name, flush=True)
        return
    shard = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shard=")), "0/1")
    k, n = map(int, shard.split("/"))
    for i in range(k, len(PAGES), n):
        best = None
        for attempt in range(3):
            t0 = time.time()
            raw = page(i, attempt)
            bw, ratio = clean(raw)
            print(f"page {i:02d} try {attempt}: black {ratio:.2f} ({time.time() - t0:.0f}s)", flush=True)
            if best is None or abs(ratio - 0.12) < abs(best[1] - 0.12):
                best = (bw, ratio, raw)
            if 0.04 <= ratio <= 0.22:
                break
        best[0].save(f"{OUT}/page_{i:02d}.png")
        best[2].save(f"{OUT}/raw_{i:02d}.jpg", quality=85)


if __name__ == "__main__":
    main()
