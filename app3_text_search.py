"""
app3_text_search.py
-------------------
Streamlit app for generating playlists based on freeform text queries.
Uses LAION-CLAP text-audio embeddings for matching.

Usage:
    streamlit run app3_text_search.py
"""

import os
import json
import numpy as np
import streamlit as st

# Config
ANALYSIS_FILE = os.path.expanduser("~/collection_analysis.json")
CLAP_CKPT     = os.path.expanduser("~/models/music_speech_epoch_15_esc_89.25.pt")
TOP_N         = 10

st.set_page_config(page_title="🔍 Playlist by Text Search", layout="wide")


# Load audio data (cached)
@st.cache_data
def load_audio_data():
    with open(ANALYSIS_FILE) as f:
        raw = json.load(f)

    track_ids  = []
    filepaths  = []
    clap_embs  = []
    metadata   = []

    for track_id, info in raw.items():
        if "clap_embedding" not in info:
            continue
        track_ids.append(track_id)
        filepaths.append(info.get("filepath", track_id))
        clap_embs.append(info["clap_embedding"])
        metadata.append({
            "bpm":               info.get("bpm", 0),
            "key":               info.get("key_temperley", "?"),
            "scale":             info.get("scale_temperley", "?"),
            "genre_top":         info.get("genre_top", "unknown"),
            "voice_label":       info.get("voice_label", "unknown"),
            "danceability_prob": info.get("danceability_prob", 0),
        })

    clap_matrix = np.array(clap_embs, dtype=np.float32)
    clap_matrix /= (np.linalg.norm(clap_matrix, axis=1, keepdims=True) + 1e-9)

    return track_ids, filepaths, clap_matrix, metadata


# Load CLAP model (cached)
@st.cache_resource
def load_clap_model():
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device="cpu")
    model.load_ckpt(ckpt=CLAP_CKPT)
    return model


# UI
st.title("🔍 Playlist Generator — Text Search")
st.markdown("Describe the music you want to hear and find matching tracks using CLAP text-audio embeddings.")

# Load audio data
try:
    track_ids, filepaths, clap_matrix, metadata = load_audio_data()
except FileNotFoundError:
    st.error(f"Analysis file not found at `{ANALYSIS_FILE}`. Please run `analyze_collection.py` first.")
    st.stop()

st.success(f"Loaded {len(track_ids)} tracks.")

# Load CLAP model
with st.spinner("Loading CLAP model (first time may take ~30 seconds)..."):
    try:
        clap_model = load_clap_model()
    except Exception as e:
        st.error(f"Failed to load CLAP model: {e}")
        st.stop()

st.success("CLAP model ready!")

# Text query input
st.subheader("💬 Enter your text query")

# Example queries for inspiration
examples = [
    "upbeat electronic dance music",
    "calm acoustic guitar",
    "heavy metal with distorted guitar",
    "jazz with saxophone and piano",
    "sad melancholic piano ballad",
    "fast tempo drum and bass",
    "relaxing ambient music for studying",
    "energetic hip hop with heavy bass",
]

st.markdown("**Example queries:**")
example_cols = st.columns(4)
for i, example in enumerate(examples):
    if example_cols[i % 4].button(example, key=f"ex_{i}"):
        st.session_state["text_query"] = example

text_query = st.text_input(
    "Your query:",
    value=st.session_state.get("text_query", ""),
    placeholder="e.g. 'upbeat dance music with synths'",
    key="text_query"
)

n_results = st.slider("Number of results", 5, 50, TOP_N, step=5)

# Search
if text_query.strip():
    with st.spinner(f"Searching for: *{text_query}*..."):
        # Get text embedding from CLAP
        text_emb = clap_model.get_text_embedding([text_query], use_tensor=False)
        text_vec = np.array(text_emb[0], dtype=np.float32)
        text_vec /= (np.linalg.norm(text_vec) + 1e-9)

        # Cosine similarity with all audio embeddings
        scores = clap_matrix @ text_vec
        top_indices = np.argsort(scores)[::-1][:n_results]

    st.markdown(f"### 🎧 Top {n_results} results for: *\"{text_query}\"*")

    for rank, idx in enumerate(top_indices):
        name  = os.path.basename(filepaths[idx])
        score = scores[idx]
        meta  = metadata[idx]

        with st.container():
            col_info, col_audio = st.columns([2, 3])
            with col_info:
                st.markdown(f"**{rank+1}. {name}**")
                st.markdown(f"Similarity score: `{score:.4f}`")
                st.markdown(
                    f"🥁 {meta['bpm']:.0f} BPM | 🎹 {meta['key']} {meta['scale']}  \n"
                    f"💃 {meta['danceability_prob']:.2f} | 🎤 {meta['voice_label']}  \n"
                    f"🎼 {meta['genre_top']}"
                )
            with col_audio:
                if os.path.exists(filepaths[idx]):
                    st.audio(filepaths[idx])
                else:
                    st.caption(f"_(audio not found at {filepaths[idx]})_")
            st.divider()

    # Save M3U8
    if st.button("💾 Save playlist as M3U8"):
        fname = f"playlist_text_{text_query[:30].replace(' ', '_')}.m3u8"
        with open(fname, "w") as f:
            f.write("#EXTM3U\n")
            for idx in top_indices:
                f.write(f"#EXTINF:-1,{os.path.basename(filepaths[idx])}\n")
                f.write(f"{filepaths[idx]}\n")
        st.success(f"Saved `{fname}` — open with VLC!")

else:
    st.info("👆 Enter a text query above or click one of the example buttons to search.")

    # Show some stats while waiting
    st.markdown("---")
    st.markdown("### 📊 Collection Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tracks", len(track_ids))
    col2.metric("With CLAP Embeddings", len(clap_matrix))
    col3.metric("Embedding Dimensions", clap_matrix.shape[1] if len(clap_matrix) > 0 else 0)