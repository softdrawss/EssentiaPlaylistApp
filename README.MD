# ACTSM 2026 - Essentia playlists assignment
This is the repository with all the code for my Advanced Computing Techniques for Sound and Music assignment to create apps to generate playlists.

## How to recreate all the steps for the project:
### 1. Install all the dependencies needed to execute the code
In my case, I did the project on a Windows and executed the code with WSL, but I tried to annotate the indications in order to execute the code in mac or Linux:
 
- Python **3.9–3.11** (Essentia with TensorFlow does not support Python 3.12+ yet)
- `ffmpeg` installed on your system:
  - **Ubuntu/Debian:** `sudo apt install ffmpeg`
  - **macOS:** `brew install ffmpeg`
  - **Windows (WSL):** `sudo apt install ffmpeg`

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS / WSL
# venv\Scripts\activate         # Windows CMD (not recommended, use WSL)
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for GPU users:** The `torch` line in `requirements.txt` installs the CPU version by default. For CUDA support, replace it with the appropriate wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.
 
### 4. Download pre-trained models
You can execute the following script to download all models automatically:
 
```bash
python download_models.py
```

This fetches all Essentia classification models and the CLAP checkpoint (aprox 1.8 GB total) into `~/models/`. Already-downloaded files are skipped, so it is safe to re-run.

If you prefer to download the models manually, you can reate a `models/` folder at the root of this repo and download the following files into it:
| File | Source |
|------|--------|
| `discogs-effnet-bs64-1.pb` | [Essentia Models Discogs-Effnet](https://essentia.upf.edu/models.html#discogs-effnet) |
| `genre_discogs400-discogs-effnet-1.pb` | [Essentia Models Genre Discogs400](https://essentia.upf.edu/models.html#genre-discogs400) |
| `genre_discogs400-discogs-effnet-1.json` | Same page as above |
| `voice_instrumental-discogs-effnet-1.pb` | [Essentia Models Voice/Instrumental](https://essentia.upf.edu/models.html#voice-instrumental-discogs-effnet) |
| `voice_instrumental-discogs-effnet-1.json` | Same page as above |
| `danceability-discogs-effnet-1.pb` | [Essentia Models Danceability](https://essentia.upf.edu/models.html#danceability-discogs-effnet) |
| `danceability-discogs-effnet-1.json` | Same page as above |
| `music_speech_epoch_15_esc_89.25.pt` | [LAION-CLAP releases](https://github.com/LAION-AI/CLAP/releases) |

I alraedy added `models/` in the `.gitignore` so it should not be an issue in case someone wants to fork this project.

### 5. Run the script `analyze_collection.py` to extract the features
Scans `--audio_dir` recursively for MP3 files, runs all models, and writes results to a JSON file.
 
```bash
python analyze_collection.py \
  --audio_dir  /path/to/your/music \
  --models_dir ~/models \
  --output     ~/collection_analysis.json \
  --workers    4
```
 
Add `--gpu` to run CLAP inference on a CUDA GPU:
 
```bash
python analyze_collection.py \
  --audio_dir  /path/to/your/music \
  --models_dir ~/models \
  --output     ~/collection_analysis.json \
  --workers    1 \
  --gpu
```
| Argument | Default | Description |
|----------|---------|-------------|
| `--audio_dir` | REQUIRED | Root folder to scan for MP3 files |
| `--models_dir` | REQUIRED | Folder containing the downloaded model files |
| `--output` | `results.json` | Output JSON file path |
| `--workers` | `1` | Number of parallel worker processes |
| `--gpu` | off | Use CUDA GPU for CLAP inference |


> IMPORTANT! If the script is interrupted, re-running the same command skips already-analyzed tracks and picks up where it left off. 

If you are running the script in local it takes quite some time to execute the code, so be aware of this. 
 
For the project, I used [MusAV](https://repositori.upf.edu/items/ea4c4a4c-958f-4004-bdc2-e1f6ad7e6829#page=1). It is already downloaded in the project, so you can use it to test everything.

If you want to see of use the results I already extracted, you can find them [HERE](./collection_analysis.json).

### 6. Run the script `analyze_collection.py` to generate the plots
The script reads the JSON generated in the step before and saves plots to `--output_dir`.
 
```bash
python report.py \
  --input      ~/collection_analysis.json \
  --output_dir ./report_output
```
 
Output files:
 
| File | Description |
|------|-------------|
| `01_broad_genre_distribution.png` | Bar chart of parent genre counts |
| `01b_top20_styles.png` | Top 20 fine-grained music styles |
| `genre_styles_full.tsv` | Full style frequency table |
| `02_tempo_distribution.png` | BPM histogram with median line |
| `03_danceability_distribution.png` | Danceability probability histogram |
| `04_key_scale_distribution.png` | Key distribution by all three profiles |
| `04b_major_minor_pie.png` | Major vs minor split per profile |
| `04c_key_strength_comparison.png` | Key estimation confidence per profile |
| `05_loudness_distribution.png` | LUFS histogram with mastering reference lines |
| `06_voice_instrumental.png` | Voice vs instrumental pie + probability histogram |

### 7. Execute the apps
To execute the apps:
```bash
# App 1: descriptor filters
streamlit run app1_descriptors.py

# App 2: track similarity
streamlit run app2_similarity.py

# App 3: text search
streamlit run app3_text_search.py
```

The console will show you the link where the app has been hosted in local. Usually, the links are similar to this one:

`http://localhost:8501`

#### Listening to playlists
If you want to export your results, they will be exported as an M3U8 file. 

You can open the file in the File Explorer and open it with VCL, but in case it gives you any issue, you can run it with:
 
- **[VLC](https://www.videolan.org/):** `Media -> Open File` -> select the `.m3u8`

Probably there are other apps that you can use, but this is the one I'm used to and works :)
 
## 🗒️ Notes
Probably some of these are self-explanatory, but just in case there is any doubt that you may encounter while executing the code:
- Only **MP3** files are processed; subdirectories are scanned recursively.
- The `venv/` and `~/models/` folders are excluded from the repository, run the setup steps above to recreate them.
- Analysis speed: roughly 3 seconds per track for Essentia models on a mid-range CPU. CLAP adds extra time, especially without a GPU.
- If you hit multiprocessing errors with CLAP and GPU, set `--workers 1`.
