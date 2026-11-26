"""Builds the Nuni picture library with the local model (artgen.py), a little every night, until every subject exists.

Episodes then use these pictures instantly, and the script writer is told which subjects are already illustrated.
    python artlib.py --budget=3000     work for at most 3000 seconds, then stop (the next night continues)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SUBJECTS = [
    # winter & holidays first (season), then farm, then everything else
    "snowman", "snowflake", "penguin", "polar bear", "reindeer", "arctic fox", "mitten", "winter hat", "scarf",
    "sled", "christmas tree", "gift box", "hot cocoa", "gingerbread cookie", "candle", "bell", "ice skates",
    "cow", "calf", "sheep", "lamb", "horse", "donkey", "goat", "duck", "duckling", "chick", "hen", "rooster",
    "dog", "puppy", "cat", "kitten", "rabbit", "frog", "turtle", "snail", "butterfly", "bee", "ladybug",
    "caterpillar", "squirrel", "hedgehog", "mouse", "fox", "deer", "bear", "raccoon", "beaver",
    "elephant", "giraffe", "lion", "tiger", "zebra", "monkey", "hippo", "rhino", "crocodile", "panda", "koala",
    "kangaroo", "camel", "parrot", "flamingo", "peacock", "swan", "toucan", "dinosaur",
    "fish", "whale", "dolphin", "octopus", "crab", "starfish", "seal", "jellyfish", "seahorse", "shark",
    "apple", "banana", "strawberry", "orange", "grapes", "watermelon", "pear", "cherry", "pineapple", "lemon",
    "carrot", "pumpkin", "tomato", "corn", "broccoli", "peas", "mushroom",
    "wooden toy train", "yellow school bus", "red fire truck", "police car", "airplane", "rocket", "sailboat",
    "tractor", "bicycle", "hot air balloon", "helicopter", "submarine", "race car", "dump truck",
    "teddy bear", "ball", "kite", "umbrella", "balloon", "star", "moon", "sun", "cloud", "rainbow",
    "cupcake", "birthday cake", "ice cream cone", "honey pot", "cookie", "pizza", "milk bottle",
    "toothbrush", "soap", "rubber duck", "bed", "book", "crayons", "drum", "guitar", "piano", "trumpet",
    "alarm clock", "key", "shoe", "rain boots", "glasses", "sunflower", "tulip", "tree", "leaf", "acorn",
    "pinecone", "sandcastle", "seashell", "bucket and spade", "flower pot", "watering can", "wheelbarrow",
]
PLACES = [
    "snowy winter forest", "snowy village with little houses", "sunny farm meadow with a red barn",
    "farm pond with water lilies", "vegetable garden", "flower garden", "green forest clearing",
    "sandy beach by the sea", "underwater coral reef", "starry night sky over hills", "jungle with big leaves",
    "autumn forest with orange leaves", "playground in the park", "cozy kitchen", "cozy bedroom",
    "rainbow over green hills", "mountain lake", "desert with cactus", "inside a wooden barn with hay",
    "little town street", "outer space with planets", "savanna with acacia trees",
]


def main():
    budget = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--budget=")), 3000))
    import artgen
    import studio
    if not artgen.available():
        print("Local picture generator not installed - nothing to do", flush=True)
        return
    lib = studio.ArtLib(ROOT, "", budget)
    lib.models, lib.poll_on = [], False  # local model only
    t0 = time.time()
    todo_i = [s for s in SUBJECTS if not (lib.dir / "items" / f"{studio.slug(s)}.webp").exists()]
    todo_b = [p for p in PLACES if not (lib.dir / "bg" / f"{studio.slug(p)}.jpg").exists()]
    print(f"Library: {len(SUBJECTS) - len(todo_i)}/{len(SUBJECTS)} subjects, "
          f"{len(PLACES) - len(todo_b)}/{len(PLACES)} places", flush=True)
    queue = []
    while todo_i or todo_b:
        for _ in range(4):
            if todo_i:
                queue.append(("item", todo_i.pop(0)))
        if todo_b:
            queue.append(("bg", todo_b.pop(0)))
    made = 0
    for kind, name in queue:
        if time.time() - t0 > budget - 300:
            break
        t1 = time.time()
        r = lib.item(name, exact=True) if kind == "item" else lib.background(name, exact=True)
        made += r is not None
        print(f"  {kind} '{name}': {'ok' if r is not None else 'failed'} ({time.time() - t1:.0f}s)", flush=True)
    print(f"Made {made} pictures in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
