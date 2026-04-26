"""
app1_descriptors.py
-------------------
Streamlit app for generating playlists based on music descriptors.

Usage:
    streamlit run app1_descriptors.py
"""

import os
import json
import numpy as np
import streamlit as st
import pandas as pd

# Config
ANALYSIS_FILE = os.path.expanduser("~/collection_analysis.json")
M3U_OUTPUT    = "playlist.m3u8"
TOP_N         = 10  # number of tracks to show in player

st.set_page_config(page_title="🎵 Playlist by Descriptors", layout="wide")


# Load data (cached)
@st.cache_data
def load_data():
    with open(ANALYSIS_FILE) as f:
        raw = json.load(f)

    records = []
    for track_id, info in raw.items():
        record = {"track_id": track_id, "filepath": info.get("filepath", track_id)}

        # Tempo
        record["bpm"] = info.get("bpm", 0.0)

        # Key/scale — use temperley as default
        record["key"]   = info.get("key_temperley", "?")
        record["scale"] = info.get("scale_temperley", "?")

        # Loudness
        record["loudness_lufs"] = info.get("loudness_lufs", -99.0)

        # Voice/instrumental
        record["voice_label"]    = info.get("voice_label", "unknown")
        record["voice_prob"]     = info.get("voice_prob", 0.5)

        # Danceability
        record["danceability_prob"] = info.get("danceability_prob", 0.5)

        # Genre (top genre and all activations)
        record["genre_top"] = info.get("genre_top", "unknown")
        record["genre_activations"] = info.get("genre_activations", [])
        record["genre_labels"]      = info.get("genre_labels", [])

        records.append(record)

    df = pd.DataFrame(records)
    return df


@st.cache_data
def get_genre_labels(df):
    for _, row in df.iterrows():
        if row["genre_labels"]:
            return row["genre_labels"]
    return []


# Sidebar filters 
st.title("🎵 Playlist Generator: Descriptor Search")
st.markdown("Filter your music collection by tempo, key, danceability, and more.")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Analysis file not found at `{ANALYSIS_FILE}`. Please run `analyze_collection.py` first.")
    st.stop()

st.sidebar.header("🎛️ Filters")

# Genre filter
st.sidebar.subheader("🎼 Music Style")
genre_labels = get_genre_labels(df)

# Extract broad genres
broad_genres = sorted(set(
    g.split("---")[0] for g in genre_labels if "---" in g
))
selected_broad = st.sidebar.multiselect(
    "Filter by broad genre (leave empty = all)",
    options=broad_genres,
    default=[]
)

genre_threshold = st.sidebar.slider(
    "Activation threshold for style matching",
    min_value=0.0, max_value=1.0, value=0.1, step=0.01
)

# Tempo filter
st.sidebar.subheader("🥁 Tempo")
bpm_min = float(df["bpm"].min()) if len(df) > 0 else 0.0
bpm_max = float(df["bpm"].max()) if len(df) > 0 else 300.0
bpm_range = st.sidebar.slider(
    "BPM range",
    min_value=0.0,
    max_value=300.0,
    value=(bpm_min, bpm_max),
    step=1.0
)

# Voice/Instrumental filter
st.sidebar.subheader("🎤 Voice / Instrumental")
voice_option = st.sidebar.radio(
    "Show tracks:",
    options=["All", "Vocal only", "Instrumental only"],
    index=0
)

# Danceability filter
st.sidebar.subheader("💃 Danceability")
dance_range = st.sidebar.slider(
    "Danceability probability range",
    min_value=0.0, max_value=1.0,
    value=(0.0, 1.0), step=0.01
)

# Key / Scale filter
st.sidebar.subheader("🎹 Key & Scale")
keys = ["Any", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
selected_key   = st.sidebar.selectbox("Key", keys)
selected_scale = st.sidebar.radio("Scale", ["Any", "major", "minor"])

# Number of results
st.sidebar.subheader("📋 Results")
n_results = st.sidebar.slider("Max results", 10, 200, 50, step=10)
sort_by = st.sidebar.selectbox(
    "Sort by",
    ["danceability_prob", "bpm", "loudness_lufs", "voice_prob"]
)
sort_desc = st.sidebar.checkbox("Sort descending", value=True)


# Apply filters
filtered = df.copy()

# Tempo
filtered = filtered[
    (filtered["bpm"] >= bpm_range[0]) &
    (filtered["bpm"] <= bpm_range[1])
]

# Voice
if voice_option == "Vocal only":
    filtered = filtered[filtered["voice_label"] == "voice"]
elif voice_option == "Instrumental only":
    filtered = filtered[filtered["voice_label"] == "instrumental"]

# Danceability
filtered = filtered[
    (filtered["danceability_prob"] >= dance_range[0]) &
    (filtered["danceability_prob"] <= dance_range[1])
]

# Key
if selected_key != "Any":
    filtered = filtered[filtered["key"] == selected_key]

# Scale
if selected_scale != "Any":
    filtered = filtered[filtered["scale"] == selected_scale]

# Genre (filter by activation threshold for selected broad genres)
if selected_broad and genre_labels:
    def matches_genre(row):
        if not row["genre_activations"] or not row["genre_labels"]:
            return False
        for i, label in enumerate(row["genre_labels"]):
            broad = label.split("---")[0] if "---" in label else label
            if broad in selected_broad:
                if i < len(row["genre_activations"]) and row["genre_activations"][i] >= genre_threshold:
                    return True
        return False
    filtered = filtered[filtered.apply(matches_genre, axis=1)]

# Sort
filtered = filtered.sort_values(sort_by, ascending=not sort_desc).head(n_results)


# Results display
st.markdown(f"### 🎧 Found **{len(filtered)}** matching tracks")

if len(filtered) == 0:
    st.warning("No tracks match your filters. Try loosening the criteria.")
else:
    # Audio players for top N tracks
    st.markdown(f"#### Top {min(TOP_N, len(filtered))} tracks")
    cols = st.columns(2)
    for i, (_, row) in enumerate(filtered.head(TOP_N).iterrows()):
        col = cols[i % 2]
        with col:
            st.markdown(f"**{i+1}. {os.path.basename(row['filepath'])}**")
            st.markdown(
                f"🥁 {row['bpm']:.0f} BPM | 🎹 {row['key']} {row['scale']} | "
                f"💃 {row['danceability_prob']:.2f} | 🎤 {row['voice_label']} | "
                f"🔊 {row['loudness_lufs']:.1f} LUFS"
            )
            st.markdown(f"🎼 {row['genre_top']}")
            if os.path.exists(row["filepath"]):
                st.audio(row["filepath"])
            else:
                st.caption(f"_(audio file not found at {row['filepath']})_")
            st.divider()

    # Full results table
    with st.expander("📋 View full results table"):
        display_df = filtered[[
            "track_id", "bpm", "key", "scale",
            "danceability_prob", "voice_label", "loudness_lufs", "genre_top"
        ]].copy()
        display_df.columns = [
            "Track", "BPM", "Key", "Scale",
            "Danceability", "Voice", "LUFS", "Top Genre"
        ]
        display_df["BPM"] = display_df["BPM"].round(1)
        display_df["Danceability"] = display_df["Danceability"].round(3)
        display_df["LUFS"] = display_df["LUFS"].round(1)
        st.dataframe(display_df, use_container_width=True)

    # Save M3U8 playlist
    if st.button("💾 Save playlist as M3U8"):
        with open(M3U_OUTPUT, "w") as f:
            f.write("#EXTM3U\n")
            for _, row in filtered.iterrows():
                f.write(f"#EXTINF:-1,{os.path.basename(row['filepath'])}\n")
                f.write(f"{row['filepath']}\n")
        st.success(f"Playlist saved to `{M3U_OUTPUT}`. Open with VLC!")