"""
download_models.py
------------------
Run this script once to download all required models before running the analysis.

Usage:
    python3 download_models.py
"""

import os
import requests

MODELS_DIR = os.path.expanduser("~/models")

MODELS = [
    {
        "name": "Discogs-Effnet embeddings",
        "url": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
        "filename": "discogs-effnet-bs64-1.pb",
    },
    {
        "name": "Genre Discogs400 model",
        "url": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
        "filename": "genre_discogs400-discogs-effnet-1.pb",
    },
    {
        "name": "Genre Discogs400 labels",
        "url": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
        "filename": "genre_discogs400-discogs-effnet-1.json",
    },
    {
        "name": "Voice/Instrumental model",
        "url": "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
        "filename": "voice_instrumental-discogs-effnet-1.pb",
    },
    {
        "name": "Voice/Instrumental labels",
        "url": "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json",
        "filename": "voice_instrumental-discogs-effnet-1.json",
    },
    {
        "name": "Danceability model",
        "url": "https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb",
        "filename": "danceability-discogs-effnet-1.pb",
    },
    {
        "name": "Danceability labels",
        "url": "https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.json",
        "filename": "danceability-discogs-effnet-1.json",
    },
    {
        "name": "CLAP audio-text embeddings (~1.8GB, may take a while)",
        "url": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt",
        "filename": "music_speech_epoch_15_esc_89.25.pt",
    },
]


def download_file(url, dest_path, name):
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping: {name}")
        return

    print(f"  Downloading: {name}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.1f}% ({downloaded/1e6:.1f} / {total/1e6:.1f} MB)", end="", flush=True)
        print(f"\r    Done: {name}                          ")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading models to: {MODELS_DIR}\n")

    for model in MODELS:
        dest = os.path.join(MODELS_DIR, model["filename"])
        download_file(model["url"], dest, model["name"])

    print("\nAll models downloaded successfully!")
    print(f"Models are saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()