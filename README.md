# Nuni & Friends - kids video bot

Automatic YouTube channel for little kids starring **Nuni**, a tiny teal-blue baby owl.
GitHub Actions write an episode with Gemini, illustrate it, voice it, render the video and upload it (plus a Short).

## How it works
| File | What it does |
|---|---|
| `bot.py` | Main pipeline: plan → script (Gemini) → voice → render → upload → `state.json` |
| `studio.py` | Picture-book renderer (scenes, captions, thumbnail, Short) |
| `growth.py` | Episode strategy and SEO description/tags |
| `artgen.py`, `artlib.py` | Local Stable Diffusion art and the cached picture library in `assets/lib` |
| `coloring.py`, `pixarline.py` | Coloring-book page generation |
| `state.json` | Episode counter and upload history (committed by the bot) |

## Workflows
- **publish** - Sun/Tue/Thu 14:00 UTC, or manual (`test: true` = dry run, no upload).
- **art-library** - nightly, grows the picture library.
- **check-auth** - checks that the YouTube tokens still work.
- **coloring-book** - manual, builds coloring pages in 9 shards.

## Secrets
`GEMINI_API_KEY`, `MINIMAX_API_KEY` (optional voice), `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` (+ `_EN` / `_HE` / `_AR` per channel).

## Safety rules built in
- Every upload is marked *Made for Kids* and *contains synthetic media*.
- If Gemini fails, the offline fallback episode is **not** published (`CONFIG["upload_fallback"]`).
- `state.json` is saved after every upload and committed even when a later step fails, so an episode is never uploaded twice.

## Run locally
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
GEMINI_API_KEY=... python bot.py --dry-run --lang=en   # builds out/video_en.mp4 without uploading
```
