"""Pixar-style colouring pages: render a cute 3D scene with SDXL, then trace it into clean line art.

The tracer is the small "informative drawings" line-art network (weights from lllyasviel/Annotators,
the same model ControlNet's line-art preprocessor uses), so the characters keep the soft, rounded
Pixar look while the page itself is pure black outlines on white.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

STYLE = ("pixar style 3d animation render, adorable, big expressive eyes, round soft shapes, "
         "soft even studio light, clean simple background, highly detailed, masterpiece")
NEG = ("text, letters, words, watermark, signature, logo, frame, border, scary, creepy, ugly, deformed, "
       "extra limbs, extra fingers, bad anatomy, two horns, multiple horns, blurry, low quality, nsfw, nude, "
       "topless, nipples, pig, cluttered background, busy background, dark, night photo")
MERMAID_NEG = ", legs, feet, knees, toes, human legs, red hair"

_N = nn.InstanceNorm2d


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), _N(c), nn.ReLU(inplace=True),
                                        nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), _N(c))

    def forward(self, x):
        return x + self.conv_block(x)


class _Gen(nn.Module):
    def __init__(self, n_res=3):
        super().__init__()
        self.model0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), _N(64), nn.ReLU(inplace=True))
        m1, c = [], 64
        for _ in range(2):
            m1 += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1), _N(c * 2), nn.ReLU(inplace=True)]
            c *= 2
        self.model1 = nn.Sequential(*m1)
        self.model2 = nn.Sequential(*[_Res(c) for _ in range(n_res)])
        m3 = []
        for _ in range(2):
            m3 += [nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1, output_padding=1), _N(c // 2),
                   nn.ReLU(inplace=True)]
            c //= 2
        self.model3 = nn.Sequential(*m3)
        self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Sigmoid())

    def forward(self, x):
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


_LINE = {}


def _liner(coarse):
    if coarse not in _LINE:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("lllyasviel/Annotators", "sk_model2.pth" if coarse else "sk_model.pth")
        g = _Gen()
        g.load_state_dict(torch.load(path, map_location="cpu"))
        _LINE[coarse] = g.eval()
    return _LINE[coarse]


def trace(img, coarse=True, size=1024):
    """RGB render -> grayscale sketch (dark lines on white), at size px on the long side."""
    im = img.convert("RGB")
    k = size / max(im.size)
    im = im.resize((int(im.width * k) // 8 * 8, int(im.height * k) // 8 * 8), Image.LANCZOS)
    x = torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None] / 255.0
    with torch.inference_mode():
        y = _liner(coarse)(x)[0, 0].numpy()
    y = (y * 255).clip(0, 255).astype(np.uint8)
    if y.mean() < 128:  # some checkpoints answer white-on-black
        y = 255 - y
    return Image.fromarray(y)


def to_page(sketch, scale=2):
    """Sketch -> bold, smooth, pure black-on-white colouring page; returns (page, ink ratio, noise)."""
    import cv2
    g = np.array(sketch.resize((sketch.width * scale, sketch.height * scale), Image.BICUBIC))
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    ink = (g < 150).astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # bolder outlines for crayons
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    small = st[1:, cv2.CC_STAT_AREA] < 90
    keep = np.r_[False, ~small]
    ink = keep[lab]
    sm = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.0) > 0.45  # smooth jaggies
    out = Image.fromarray(np.where(sm, 0, 255).astype(np.uint8))
    return out, float(sm.mean()), int(small.sum())


def render(prompt, seed, neg):
    import artgen
    p = artgen.pipe()
    g = torch.Generator("cpu").manual_seed(seed)
    with torch.inference_mode():
        return p(prompt=f"{prompt}, {STYLE}", negative_prompt=neg, width=832, height=1088,
                 num_inference_steps=6, guidance_scale=1.5, generator=g).images[0]


def colourful(im):
    hsv = np.array(im.convert("HSV").resize((192, 256)))
    return float((hsv[..., 1] > 50).mean())


def one(i, subject, out):
    neg = NEG + (MERMAID_NEG if "mermaid" in subject else "")
    best = None
    for attempt in range(4):
        t0 = time.time()
        raw = render(subject, 7000 + i * 53 + attempt * 1013, neg)
        page, ink, noise = to_page(trace(raw, coarse=True))
        # a good page: enough outline to colour, not a scribble; few stray specks
        score = abs(ink - 0.075) * 10 + noise / 400.0
        print(f"page {i:02d} try {attempt}: ink {ink:.3f} noise {noise} score {score:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or score < best[0]:
            best = (score, page, raw)
        if attempt >= 1 and best[0] < 0.35:
            break
    best[1].save(f"{out}/page_{i:02d}.png")
    best[2].save(f"{out}/raw_{i:02d}.jpg", quality=88)
    trace(best[2], coarse=False).save(f"{out}/fine_{i:02d}.png")  # alternative tracing, for review
