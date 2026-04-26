"""
analyze_collection.py
---------------------
Analyzes a music collection and extracts descriptors for each track.
Supports GPU acceleration and multiprocessing with proper model sharing.

Usage:
    python analyze_collection.py --audio_dir /path/to/audio --models_dir /path/to/models --output results.json --workers 4 --gpu
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

SAVE_EVERY = 25
ALGOS = None  # global, initialized per worker process

# LOAD MODELS (called once per worker process)
def load_models(models_dir, use_gpu=False):
    # Must be set before TF initializes, set here inside the worker
    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import essentia
    import essentia.standard as es
    import laion_clap

    essentia.log.infoActive = False
    essentia.log.warningActive = False

    algos = {}

    algos["rhythm"]   = es.RhythmExtractor2013(method="multifeature")
    algos["key_temp"] = es.KeyExtractor(profileType="temperley")
    algos["key_krum"] = es.KeyExtractor(profileType="krumhansl")
    algos["key_edma"] = es.KeyExtractor(profileType="edma")
    algos["loudness"] = es.LoudnessEBUR128()

    algos["effnet"] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=os.path.join(models_dir, "discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:1"
    )

    with open(os.path.join(models_dir, "genre_discogs400-discogs-effnet-1.json")) as f:
        algos["genre_labels"] = json.load(f)["classes"]
    algos["genre"] = es.TensorflowPredict2D(
        graphFilename=os.path.join(models_dir, "genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )

    with open(os.path.join(models_dir, "voice_instrumental-discogs-effnet-1.json")) as f:
        algos["voice_labels"] = json.load(f)["classes"]
    algos["voice"] = es.TensorflowPredict2D(
        graphFilename=os.path.join(models_dir, "voice_instrumental-discogs-effnet-1.pb"),
        input="model/Placeholder",
        output="model/Softmax"
    )

    with open(os.path.join(models_dir, "danceability-discogs-effnet-1.json")) as f:
        algos["dance_labels"] = json.load(f)["classes"]
    algos["dance"] = es.TensorflowPredict2D(
        graphFilename=os.path.join(models_dir, "danceability-discogs-effnet-1.pb"),
        input="model/Placeholder",
        output="model/Softmax"
    )

    import torch
    if use_gpu and not torch.cuda.is_available():
        print("Warning: --gpu requested but PyTorch CUDA is not available. CLAP will run on CPU.")
        use_gpu = False

    device = "cuda" if use_gpu else "cpu"
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    clap.load_ckpt(ckpt=os.path.join(models_dir, "music_speech_epoch_15_esc_89.25.pt"))
    algos["clap"] = clap

    return algos


# ANALYZE A SINGLE TRACK (uses the passed-in algos dict)
def analyze_track(filepath, algos):
    import essentia.standard as es

    result = {}

    audio_stereo, sr, nch, _, _, _ = es.AudioLoader(filename=filepath)()

    if nch == 2:
        audio_mono = es.MonoMixer()(audio_stereo, nch)
    else:
        audio_mono = audio_stereo[:, 0] if audio_stereo.ndim == 2 else audio_stereo

    audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_mono)
    audio_48k = es.Resample(inputSampleRate=44100, outputSampleRate=48000)(audio_mono)

    # Tempo
    bpm, _, _, _, _ = algos["rhythm"](audio_mono)
    result["bpm"] = float(bpm)

    # Key / Scale (3 profiles)
    for profile, algo_key in [("temperley", "key_temp"),
                               ("krumhansl", "key_krum"),
                               ("edma",      "key_edma")]:
        key, scale, strength = algos[algo_key](audio_mono)
        result[f"key_{profile}"]          = key
        result[f"scale_{profile}"]        = scale
        result[f"key_strength_{profile}"] = float(strength)

    # Loudness
    _, _, integrated_loudness, _ = algos["loudness"](audio_stereo)
    result["loudness_lufs"] = float(integrated_loudness)

    # Discogs-Effnet embeddings
    effnet_frames = algos["effnet"](audio_16k)
    effnet_mean   = np.mean(effnet_frames, axis=0)

    # Genre
    genre_mean    = np.mean(algos["genre"](effnet_frames), axis=0)
    top_idx       = int(np.argmax(genre_mean))
    result["genre_top"]         = algos["genre_labels"][top_idx]
    result["genre_activations"] = genre_mean.tolist()

    # Voice / Instrumental
    voice_mean = np.mean(algos["voice"](effnet_frames), axis=0)
    result["voice_instrumental_prob"] = float(voice_mean[0])
    result["voice_prob"]              = float(voice_mean[1])
    result["voice_label"]             = "voice" if voice_mean[1] > voice_mean[0] else "instrumental"

    # Danceability
    dance_mean = np.mean(algos["dance"](effnet_frames), axis=0)
    result["danceability_prob"] = float(dance_mean[1])

    # Effnet mean embedding
    result["effnet_embedding"] = effnet_mean.tolist()

    # CLAP embedding
    clap_emb = algos["clap"].get_audio_embedding_from_data(
        x=[audio_48k.astype(np.float32)], use_tensor=False
    )
    result["clap_embedding"] = clap_emb[0].tolist()

    return result


# MULTIPROCESSING SUPPORT
def init_worker(models_dir, use_gpu):
    """Called once per worker process to load models into the global."""
    global ALGOS
    ALGOS = load_models(models_dir, use_gpu)


def worker(args):
    """Runs in a worker process; uses the process-global ALGOS."""
    global ALGOS
    filepath, audio_dir = args
    try:
        res = analyze_track(filepath, ALGOS)   # ← fixed: pass ALGOS, not models_dir
        return os.path.relpath(filepath, audio_dir), res, None
    except Exception as e:
        return os.path.relpath(filepath, audio_dir), None, str(e)


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Analyze a music collection.")
    parser.add_argument("--audio_dir",  required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--output",     default="results.json")
    parser.add_argument("--workers",    type=int, default=1)
    parser.add_argument("--gpu",        action="store_true")
    args = parser.parse_args()

    # Collect MP3 files
    files = []
    for root, _, fs in os.walk(args.audio_dir):
        for f in fs:
            if f.lower().endswith(".mp3"):
                files.append(os.path.join(root, f))
    files.sort()
    print(f"Found {len(files)} MP3 files.\n")

    # Resume support
    results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                results = json.load(f)
            print(f"Resuming: {len(results)} tracks already done.\n")
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: output file exists but is empty or corrupt — starting fresh.\n")

    tasks = [
        (fp, args.audio_dir)
        for fp in files
        if os.path.relpath(fp, args.audio_dir) not in results
    ]
    print(f"Processing {len(tasks)} remaining tracks with {args.workers} worker(s), GPU={args.gpu}\n")

    errors = []

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.models_dir, args.gpu),
    ) as pool:
        for i, (track_id, res, err) in enumerate(
            tqdm(pool.imap_unordered(worker, tasks), total=len(tasks))
        ):
            if res is not None:
                results[track_id] = res
            else:
                errors.append((track_id, err))
                tqdm.write(f"  Error on {track_id}: {err}")

            # Periodic save so progress isn't lost on crash
            if i % SAVE_EVERY == 0:
                with open(args.output, "w") as f:
                    json.dump(results, f)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f)

    print(f"\nDone! {len(results)} tracks analyzed, {len(errors)} errors.")
    if errors:
        print("Tracks with errors:")
        for tid, e in errors:
            print(f"  - {tid}: {e}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()