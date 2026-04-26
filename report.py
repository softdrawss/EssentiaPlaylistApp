"""
report.py
---------
Generates a statistical overview report from the music collection analysis.

Usage:
    python3 report.py --input ~/collection_analysis.json --output_dir ./report_output
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted")

DEFAULT_INPUT  = os.path.expanduser("~/collection_analysis.json")
DEFAULT_OUTPUT = "./report_output"


# Helpers
def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} tracks.")
    return raw


def save_fig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# 1. Genre / Music Style
def plot_genres(data, output_dir):
    print("\n[1] Genre distribution...")

    # Use the pre-computed genre_top field (e.g. "Pop---Ballad")
    top_genres = [t["genre_top"] for t in data.values() if "genre_top" in t]

    if not top_genres:
        print("  No genre_top field found, skipping.")
        return

    print(f"  Found genre_top for {len(top_genres)} tracks.")

    # Extract broad parent genre (format: "genre---style")
    broad_genres = [g.split("---")[0] if "---" in g else g for g in top_genres]

    # Broad genre distribution
    broad_counts = Counter(broad_genres)
    broad_df = pd.DataFrame(broad_counts.most_common(), columns=["Genre", "Count"])

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=broad_df, x="Genre", y="Count", ax=ax, color=COLORS[0])
    ax.set_title("Broad Genre Distribution (top predicted style per track)")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Tracks")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, output_dir, "01_broad_genre_distribution.png")

    # Full style TSV
    style_counts = Counter(top_genres)
    style_df = pd.DataFrame(style_counts.most_common(), columns=["Style", "Count"])
    tsv_path = os.path.join(output_dir, "genre_styles_full.tsv")
    style_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Saved full style TSV: {tsv_path}")

    # Top 20 styles
    top20 = style_df.head(20)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=top20, x="Style", y="Count", ax=ax, color=COLORS[1])
    ax.set_title("Top 20 Music Styles (top predicted style per track)")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Tracks")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, output_dir, "01b_top20_styles.png")

    # Print top 10 for terminal summary
    print("  Top 10 broad genres:")
    for genre, count in broad_counts.most_common(10):
        print(f"    {genre}: {count} ({100*count/len(broad_genres):.1f}%)")

    print("  Top 10 full styles:")
    for style, count in style_counts.most_common(10):
        print(f"    {style}: {count} ({100*count/len(top_genres):.1f}%)")


# 2. Tempo Distribution
def plot_tempo(data, output_dir):
    print("\n[2] Tempo distribution...")
    bpms = [t["bpm"] for t in data.values() if "bpm" in t]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(bpms, bins=60, kde=True, ax=ax, color=COLORS[2])
    ax.set_title("Tempo Distribution")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Number of Tracks")
    ax.axvline(np.median(bpms), color="red", linestyle="--",
               label=f"Median: {np.median(bpms):.1f} BPM")
    ax.legend()
    save_fig(fig, output_dir, "02_tempo_distribution.png")

    print(f"  BPM — mean: {np.mean(bpms):.1f}, median: {np.median(bpms):.1f}, "
          f"min: {np.min(bpms):.1f}, max: {np.max(bpms):.1f}")


# 3. Danceability Distribution
def plot_danceability(data, output_dir):
    print("\n[3] Danceability distribution...")
    dance = [t["danceability_prob"] for t in data.values() if "danceability_prob" in t]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(dance, bins=40, kde=True, ax=ax, color=COLORS[3])
    ax.set_title("Danceability Distribution (probability of being danceable)")
    ax.set_xlabel("Danceability Probability")
    ax.set_ylabel("Number of Tracks")
    ax.axvline(0.5, color="red", linestyle="--", label="Threshold (0.5)")
    ax.legend()
    save_fig(fig, output_dir, "03_danceability_distribution.png")

    danceable = sum(1 for d in dance if d > 0.5)
    print(f"  Danceable (>0.5): {danceable}/{len(dance)} ({100*danceable/len(dance):.1f}%)")


# 4. Key / Scale Distribution
def plot_keys(data, output_dir):
    print("\n[4] Key/scale distribution...")

    profiles  = ["temperley", "krumhansl", "edma"]
    key_order = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Key Distribution by Profile", fontsize=14)

    for ax, profile in zip(axes, profiles):
        keys_major, keys_minor = [], []
        for t in data.values():
            k = t.get(f"key_{profile}")
            s = t.get(f"scale_{profile}")
            if k and s:
                if s == "major":
                    keys_major.append(k)
                else:
                    keys_minor.append(k)

        major_counts = Counter(keys_major)
        minor_counts = Counter(keys_minor)

        major_vals = [major_counts.get(k, 0) for k in key_order]
        minor_vals = [minor_counts.get(k, 0) for k in key_order]

        x = np.arange(len(key_order))
        width = 0.4
        ax.bar(x - width/2, major_vals, width, label="Major", color=COLORS[0])
        ax.bar(x + width/2, minor_vals, width, label="Minor", color=COLORS[4])
        ax.set_title(f"Profile: {profile}")
        ax.set_xticks(x)
        ax.set_xticklabels(key_order)
        ax.set_xlabel("Key")
        ax.set_ylabel("Number of Tracks")
        ax.legend()

    plt.tight_layout()
    save_fig(fig, output_dir, "04_key_scale_distribution.png")

    # Agreement between all three profiles
    agreements = 0
    total = 0
    for t in data.values():
        keys   = [t.get(f"key_{p}")   for p in profiles]
        scales = [t.get(f"scale_{p}") for p in profiles]
        if all(k is not None for k in keys) and all(s is not None for s in scales):
            total += 1
            if len(set(keys)) == 1 and len(set(scales)) == 1:
                agreements += 1

    if total > 0:
        print(f"  All 3 profiles agree on key+scale: {agreements}/{total} "
              f"({100*agreements/total:.1f}%)")

    # Key strength comparison across profiles
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, profile in enumerate(profiles):
        strengths = [t[f"key_strength_{profile}"]
                     for t in data.values()
                     if f"key_strength_{profile}" in t]
        sns.kdeplot(strengths, ax=ax, label=f"{profile} (mean={np.mean(strengths):.3f})",
                    color=COLORS[i])
    ax.set_title("Key Estimation Confidence (strength) by Profile")
    ax.set_xlabel("Key Strength")
    ax.set_ylabel("Density")
    ax.legend()
    save_fig(fig, output_dir, "04c_key_strength_comparison.png")

    # Major vs minor pie charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Major vs Minor Scale Distribution by Profile", fontsize=14)
    for ax, profile in zip(axes, profiles):
        scales = [t.get(f"scale_{profile}")
                  for t in data.values()
                  if t.get(f"scale_{profile}")]
        scale_counts = Counter(scales)
        ax.pie(
            scale_counts.values(),
            labels=scale_counts.keys(),
            autopct="%1.1f%%",
            colors=[COLORS[0], COLORS[4]],
            startangle=90
        )
        ax.set_title(f"Profile: {profile}")
    save_fig(fig, output_dir, "04b_major_minor_pie.png")


# 5. Loudness Distribution
def plot_loudness(data, output_dir):
    print("\n[5] Loudness distribution...")
    loudness = [t["loudness_lufs"] for t in data.values()
                if "loudness_lufs" in t and t["loudness_lufs"] > -70]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(loudness, bins=50, kde=True, ax=ax, color=COLORS[5])
    ax.set_title("Integrated Loudness Distribution (LUFS)")
    ax.set_xlabel("Loudness (LUFS)")
    ax.set_ylabel("Number of Tracks")

    for lufs, label, color in [
        (-14, "Streaming target (-14 LUFS)", "green"),
        (-9,  "Loud master (-9 LUFS)",        "orange"),
        (-23, "Broadcast standard (-23 LUFS)", "red"),
    ]:
        ax.axvline(lufs, color=color, linestyle="--", alpha=0.7, label=label)
    ax.legend(fontsize=9)
    save_fig(fig, output_dir, "05_loudness_distribution.png")

    print(f"  LUFS — mean: {np.mean(loudness):.1f}, median: {np.median(loudness):.1f}, "
          f"min: {np.min(loudness):.1f}, max: {np.max(loudness):.1f}")


# 6. Voice vs Instrumental
def plot_voice(data, output_dir):
    print("\n[6] Voice vs Instrumental...")
    labels = [t["voice_label"] for t in data.values() if "voice_label" in t]
    counts = Counter(labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    axes[0].pie(
        counts.values(),
        labels=[l.capitalize() for l in counts.keys()],
        autopct="%1.1f%%",
        colors=[COLORS[0], COLORS[1]],
        startangle=90,
        explode=[0.05] * len(counts),
    )
    axes[0].set_title("Voice vs Instrumental Distribution")

    # Histogram of raw voice probability
    voice_probs = [t["voice_prob"] for t in data.values() if "voice_prob" in t]
    sns.histplot(voice_probs, bins=40, kde=True, ax=axes[1], color=COLORS[2])
    axes[1].axvline(0.5, color="red", linestyle="--", label="Decision boundary (0.5)")
    axes[1].set_title("Voice Probability Distribution")
    axes[1].set_xlabel("Voice Probability")
    axes[1].set_ylabel("Number of Tracks")
    axes[1].legend()

    plt.tight_layout()
    save_fig(fig, output_dir, "06_voice_instrumental.png")

    for label, count in counts.items():
        print(f"  {label.capitalize()}: {count} ({100*count/len(labels):.1f}%)")


# 7. Summary Stats Table
def print_summary(data):
    print("\n" + "=" * 50)
    print("COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Total tracks analyzed: {len(data)}")

    top_genres = [t["genre_top"] for t in data.values() if "genre_top" in t]
    if top_genres:
        broad = [g.split("---")[0] if "---" in g else g for g in top_genres]
        top_broad, top_broad_n = Counter(broad).most_common(1)[0]
        top_style, top_style_n = Counter(top_genres).most_common(1)[0]
        print(f"Top broad genre: {top_broad} ({top_broad_n} tracks, "
              f"{100*top_broad_n/len(top_genres):.1f}%)")
        print(f"Top style:       {top_style} ({top_style_n} tracks, "
              f"{100*top_style_n/len(top_genres):.1f}%)")
        print(f"Unique styles:   {len(set(top_genres))}")

    bpms = [t["bpm"] for t in data.values() if "bpm" in t]
    if bpms:
        print(f"Tempo: {np.mean(bpms):.1f} BPM avg "
              f"(range: {np.min(bpms):.0f}–{np.max(bpms):.0f})")

    dance = [t["danceability_prob"] for t in data.values() if "danceability_prob" in t]
    if dance:
        print(f"Danceability: {100*sum(d>0.5 for d in dance)/len(dance):.1f}% danceable")

    voice = [t["voice_label"] for t in data.values() if "voice_label" in t]
    if voice:
        vocal_pct = 100 * sum(v == "voice" for v in voice) / len(voice)
        print(f"Vocal content: {vocal_pct:.1f}% with vocals")

    loudness = [t["loudness_lufs"] for t in data.values()
                if "loudness_lufs" in t and t["loudness_lufs"] > -70]
    if loudness:
        print(f"Loudness: {np.mean(loudness):.1f} LUFS avg")

    print("=" * 50)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.input)

    plot_genres(data, args.output_dir)
    plot_tempo(data, args.output_dir)
    plot_danceability(data, args.output_dir)
    plot_keys(data, args.output_dir)
    plot_loudness(data, args.output_dir)
    plot_voice(data, args.output_dir)
    print_summary(data)

    print(f"\nReport saved to: {args.output_dir}")


if __name__ == "__main__":
    main()