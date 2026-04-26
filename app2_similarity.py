"""
app2_similarity.py
------------------
Streamlit app for generating playlists based on track similarity.
Uses Discogs-Effnet and CLAP embeddings with cosine similarity.

Usage:
    streamlit run app2_similarity.py
"""

import os
import json
import numpy as np
import streamlit as st

# Config
ANALYSIS_FILE = os.path.expanduser("~/collection_analysis.json")
TOP_N = 10

st.set_page_config(page_title="🎵 Playlist by Similarity", layout="wide")


# Load data (cached)
@st.cache_data
def load_data():
    with open(ANALYSIS_FILE) as f:
        raw = json.load(f)

    track_ids   = []
    filepaths   = []
    effnet_embs = []
    clap_embs   = []
    metadata    = []

    for track_id, info in raw.items():
        # Only include tracks that have both embeddings
        if "effnet_embedding" not in info or "clap_embedding" not in info:
            continue

        track_ids.append(track_id)
        filepaths.append(info.get("filepath", track_id))
        effnet_embs.append(info["effnet_embedding"])
        clap_embs.append(info["clap_embedding"])
        metadata.append({
            "bpm":              info.get("bpm", 0),
            "key":              info.get("key_temperley", "?"),
            "scale":            info.get("scale_temperley", "?"),
            "genre_top":        info.get("genre_top", "unknown"),
            "voice_label":      info.get("voice_label", "unknown"),
            "danceability_prob": info.get("danceability_prob", 0),
        })

    # Convert to numpy arrays for fast similarity computation
    effnet_matrix = np.array(effnet_embs, dtype=np.float32)
    clap_matrix   = np.array(clap_embs,   dtype=np.float32)

    # L2 normalize for cosine similarity via dot product
    effnet_matrix /= (np.linalg.norm(effnet_matrix, axis=1, keepdims=True) + 1e-9)
    clap_matrix   /= (np.linalg.norm(clap_matrix,   axis=1, keepdims=True) + 1e-9)

    return track_ids, filepaths, effnet_matrix, clap_matrix, metadata


def cosine_similarity(query_vec, matrix):
    """Compute cosine similarity between query vector and all rows in matrix."""
    return matrix @ query_vec


# UI
st.title("🎵 Playlist Generator: Track Similarity")
st.markdown("Select a query track and find the most similar tracks using audio embeddings.")

try:
    track_ids, filepaths, effnet_matrix, clap_matrix, metadata = load_data()
except FileNotFoundError:
    st.error(f"Analysis file not found at `{ANALYSIS_FILE}`. Please run `analyze_collection.py` first.")
    st.stop()

st.success(f"Loaded {len(track_ids)} tracks with embeddings.")

# Query track selector
st.subheader("🔍 Select Query Track")

# Allow search by track ID
search_query = st.text_input("Search track by filename (or leave empty to browse)", "")

if search_query:
    matching = [i for i, tid in enumerate(track_ids)
                if search_query.lower() in os.path.basename(filepaths[i]).lower()]
    if not matching:
        st.warning("No tracks found matching your search.")
        display_ids = list(range(min(100, len(track_ids))))
    else:
        display_ids = matching
else:
    display_ids = list(range(min(200, len(track_ids))))

display_options = {
    os.path.basename(filepaths[i]): i for i in display_ids
}

selected_name = st.selectbox("Choose a query track:", list(display_options.keys()))
query_idx = display_options[selected_name]

# Show query track info
st.markdown("#### 🎧 Query Track")
q_meta = metadata[query_idx]
st.markdown(
    f"**{selected_name}**  \n"
    f"🥁 {q_meta['bpm']:.0f} BPM | 🎹 {q_meta['key']} {q_meta['scale']} | "
    f"💃 {q_meta['danceability_prob']:.2f} | 🎤 {q_meta['voice_label']}  \n"
    f"🎼 {q_meta['genre_top']}"
)
if os.path.exists(filepaths[query_idx]):
    st.audio(filepaths[query_idx])

st.divider()

# Compute similarity
effnet_query = effnet_matrix[query_idx]
clap_query   = clap_matrix[query_idx]

effnet_scores = cosine_similarity(effnet_query, effnet_matrix)
clap_scores   = cosine_similarity(clap_query,   clap_matrix)

# Exclude the query track itself
effnet_scores[query_idx] = -1
clap_scores[query_idx]   = -1

effnet_top = np.argsort(effnet_scores)[::-1][:TOP_N]
clap_top   = np.argsort(clap_scores)[::-1][:TOP_N]

# Show results side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔵 Discogs-Effnet Similarity")
    st.caption("Based on music style embeddings trained on Discogs metadata (400 genres)")
    for rank, idx in enumerate(effnet_top):
        name = os.path.basename(filepaths[idx])
        score = effnet_scores[idx]
        meta = metadata[idx]
        st.markdown(
            f"**{rank+1}. {name}** (similarity: {score:.3f})  \n"
            f"🥁 {meta['bpm']:.0f} BPM | 🎹 {meta['key']} {meta['scale']} | "
            f"🎼 {meta['genre_top']}"
        )
        if os.path.exists(filepaths[idx]):
            st.audio(filepaths[idx])
        else:
            st.caption(f"_(audio not found)_")
        st.divider()

with col2:
    st.markdown("### 🟢 CLAP Similarity")
    st.caption("Based on LAION-CLAP audio embeddings (music + speech model)")
    for rank, idx in enumerate(clap_top):
        name = os.path.basename(filepaths[idx])
        score = clap_scores[idx]
        meta = metadata[idx]
        st.markdown(
            f"**{rank+1}. {name}** (similarity: {score:.3f})  \n"
            f"🥁 {meta['bpm']:.0f} BPM | 🎹 {meta['key']} {meta['scale']} | "
            f"🎼 {meta['genre_top']}"
        )
        if os.path.exists(filepaths[idx]):
            st.audio(filepaths[idx])
        else:
            st.caption(f"_(audio not found)_")
        st.divider()

# Save playlists
st.divider()
col_save1, col_save2 = st.columns(2)

with col_save1:
    if st.button("💾 Save Effnet playlist as M3U8"):
        with open("playlist_effnet.m3u8", "w") as f:
            f.write("#EXTM3U\n")
            for idx in effnet_top:
                f.write(f"#EXTINF:-1,{os.path.basename(filepaths[idx])}\n")
                f.write(f"{filepaths[idx]}\n")
        st.success("Saved `playlist_effnet.m3u8`!")

with col_save2:
    if st.button("💾 Save CLAP playlist as M3U8"):
        with open("playlist_clap.m3u8", "w") as f:
            f.write("#EXTM3U\n")
            for idx in clap_top:
                f.write(f"#EXTINF:-1,{os.path.basename(filepaths[idx])}\n")
                f.write(f"{filepaths[idx]}\n")
        st.success("Saved `playlist_clap.m3u8`!")