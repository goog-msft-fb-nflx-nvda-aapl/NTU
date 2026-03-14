# Homework 2: Controllable Text-to-Music Generation

---

## 1. Background & Methods Overview
*The following section summarizes the key concepts and models discussed in the course lectures relevant to this homework.*

### 1.1 Diffusion & Control Models
- **Diffusion Models:** Foundation for text-to-music generation.
- **ControlNet:** Adds conditional control to diffusion models.
  - **Architecture:** Each block contains 4 ResNet layers and 2 Vision Transformers (ViTs).
  - **Branches:** 
    - *Original Branch:* UNet (25 blocks), processes image latent & text.
    - *ControlNet Branch:* UNet Encoder (12 blocks), zero convolution layers, processes condition latents & text.
  - **Reference:** Zhang et al., "Adding conditional control to text-to-image diffusion models," ICCV 2023.

### 1.2 Common Time-Varying Conditions in Music
- **Melody:** 
  - STFT-based Chromagram (MusicGen, Music ControlNet)
  - F0 (Jasco)
  - CQT top-4 (Stable-Audio ControlNet, MuseControlLite)
- **Rhythm:** 
  - Beat time, Downbeat time (Music ControlNet, MuseControlLite)
  - Separated drum track (Coco-Mulla)
- **Others:** Chords, Dynamics.

### 1.3 Suggested Models
| Model | Control Capabilities | Type | Open Source | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **MusicGen-Melody** | Melody (Prefix conditioning) | Transformer | Yes | Copet et al., Neurips 2023 |
| **Coco-Mulla** | Separated drum track, Full-mix | LLM | Yes | Lin et al., arXiv 2024 |
| **Music ControlNet** | Attribute Control (Pixel domain) | Diffusion | **No** | Wu et al., TASLP 2024 |
| **MuseControlLite** | Attribute & Audio Control (Latent diffusion) | Diffusion | **Yes** | Tsai et al., ICML 2025 |
| **Jasco** | Text, Melody | - | - | - |
| **MusicGen-Style** | Text, Style | - | - | - |

**Resources:**
- [Music ControlNet Web](https://musiccontrolnet.github.io/web/)
- [MuseControlLite Web](https://musecontrollite.github.io/web/)
- [MuseControlLite Colab](https://colab.research.google.com/drive/1rR-Ncng_gSeb6hX0LY20SA4O9RCF-ZF3)

---

## 2. Problem Formulation
**Goal:** Generate the desired music!

The target music is provided in `target_music_list_60s`. The list contains:
- 3 Piano solos
- 3 Drum solos
- 1 Chinese flute solo
- 2 Chinese flute and piano duets

---

## 3. Methods
You are required to implement the following two methods:

### 3.1 Retrieval
Retrieve the most similar songs from the provided `reference_music_list` for each music in `target_music_list`.

**Suggestions for Audio Encoders:**
Options include but are not limited to:
1. Stable-Audio-Open VAE encode
2. Music2latent
3. CLAP
4. MuQ

### 3.2 Controllable Text-to-Music Generation
**Methodology:**
1. **Audio Captioning:** Obtain the text description of music in `target_music_list` using Audio Language Models (ALMs).
2. **Text-to-Music:**
   - **Simple:** Text condition only.
   - **Medium:** Any condition extracted from music (e.g., Melody, Rhythm).
   - **Strong:** Adjust the classifier-free guidance.

**Suggestions for ALMs:**
- Audio Flamingo 3
- Qwen-audio
- LP-MusicCaps

**Suggestions for Controllable Text-to-Music Models:**
- **MuseControlLite:** Text, rhythm, dynamics, melody.
- **MusicGen:** Text, melody.
- **MusicGen-Style:** Text, style.
- **Jasco:** Text, chords, melody, separated drum tracks, full-mix audio.
- **Coco-Mulla:** Pitch, chords, drum track.

> **Note:** The music in `target_music_list_60s` are 60 seconds long. You can generate music up to the model's limit (e.g., MuseControlLite generates 47-second music).

---

## 4. Evaluation Metrics

### 4.1 Retrieval Metrics
For each song in the `target_music_list`, report:
1. **CLAP:** Calculate cosine similarity between:
   - Generated music and Target music.
2. **Meta Audiobox Aesthetics:**
   - **CE:** Content Enjoyment
   - **CU:** Content Usefulness
   - **PC:** Production Complexity
   - **PQ:** Production Quality
3. **Melody Similarity (Accuracy)**

### 4.2 Generation Metrics
For each song in the `target_music_list`, report:
1. **CLAP:** Calculate cosine similarity between:
   - Target music and Input text
   - Input text and Generated music
   - Generated music and Target music
2. **Meta Audiobox Aesthetics:**
   - **CE:** Content Enjoyment
   - **CU:** Content Usefulness
   - **PC:** Production Complexity
   - **PQ:** Production Quality
3. **Melody Similarity (Accuracy)**

> **Important:** You should slice the target music to the length same as the music your model generates. (i.e., If you use MuseControlLite, trim the target music to the first 47 seconds).

---

## 5. Rules
1. **Captioning:** Do the music captioning with ALMs.
2. **Conditions:** You **cannot** directly use the music from `target_music_list` as a condition to generate music. Only conditions extracted using MIR (Music Information Retrieval) tools are allowed.
   - You cannot use an auto-encoder to encode and decode the target audio and use it as submission.
   - You cannot use the "audio condition" in MuseControlLite.
3. **Flexibility:** You can use different methods for each music in `target_music_list`.

---

## 6. Scoring
- **HW2 accounts for 20% of the total grade.**
- **Report:** 100% of the homework grade.

---

## 7. Submission

### 7.1 Report
Upload `studentID_report.md` (e.g., `r13921031_report.md`) to **NTU Cool**.
For each music in `target_music_list`, please report:
- **Implementation Details:** (e.g., Model used, time-varying conditions, text input).
- **Generated Music:** Ensure that you set it to public (provide links).
- **Evaluation Results:** 
  - CLAP
  - Meta Audiobox Aesthetics
  - Melody Accuracy
- **Source Code URL:** Link to your code.
- **Reflection:** Any thoughts that you would like to share.

> **Requirement:** Please create a report that is clear and can be understood without the need for oral presentations.

### 7.2 Source Code
- Upload all your source code and models to a cloud drive.
- Open access permissions.
- Put the link in the report.
- You should provide a detailed `README` file so that the results can be reproduced.

---

### file structure ###
```
$ tree
.
├── =3.6.1
├── home
│   └── fundwotsai
│       └── Deep_MIR_hw2
│           ├── Melody_acc.py
│           ├── outputs
│           │   ├── captions
│           │   ├── eval
│           │   ├── generated
│           │   └── retrieved
│           ├── referecne_music_list_60s
│           │   ├── 04 聽泉.mp3
...
│           │   └── 黃土情.mp3
│           ├── retrieval_results
│           │   └── retrieval_results.json
│           └── target_music_list_60s
│               ├── 10_country_114_beat_4-4.wav
...
│               └── 竹笛｜这世界那么多人_cover 莫文蔚_60s.mp3
└── step1_retrieval.py
```
---

### Hardware ###
```
$ nvidia-smi
Wed Mar 11 23:46:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200                    On  |   00000000:03:00.0 Off |                    0 |
| N/A   39C    P0            122W /  700W |    1977MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H200                    On  |   00000000:29:00.0 Off |                    0 |
| N/A   30C    P0             76W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H200                    On  |   00000000:59:00.0 Off |                    0 |
| N/A   31C    P0             75W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H200                    On  |   00000000:63:00.0 Off |                    0 |
| N/A   37C    P0             78W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H200                    On  |   00000000:7B:00.0 Off |                    0 |
| N/A   36C    P0             76W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H200                    On  |   00000000:A3:00.0 Off |                    0 |
| N/A   31C    P0             76W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H200                    On  |   00000000:D3:00.0 Off |                    0 |
| N/A   35C    P0             77W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H200                    On  |   00000000:E5:00.0 Off |                    0 |
| N/A   30C    P0             75W /  700W |       4MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A   3815742      C   python3                                       656MiB |
|    0   N/A  N/A   3817470      C   python3                                       656MiB |
|    0   N/A  N/A   3817873      C   python3                                       644MiB |
+-----------------------------------------------------------------------------------------+
```

````markdown
# Deep MIR HW2 Environment Setup

## Hardware
```
| Component | Specification |
|---|---|
| GPU | NVIDIA H200 |
| GPU Memory | 143 GB |
| Driver | 550.163.01 |
| CUDA Runtime | 12.4 |
| GPUs Available | 8 |
```
Command used to verify:

```bash
nvidia-smi
````

---

# Conda Environment

Environment name:

```
deep_mir_hw2
```

Python version:

```
Python 3.10
```

Creation:

```bash
conda create -n deep_mir_hw2 python=3.10
conda activate deep_mir_hw2
```

---

# Core Deep Learning Stack
```
| Package     | Version      | Notes                              |
| ----------- | ------------ | ---------------------------------- |
| torch       | 2.1.0+cu121  | Required by AudioCraft             |
| torchaudio  | 2.1.0+cu121  | Audio I/O and processing           |
| torchvision | 0.16.0+cu121 | Required dependency                |
| numpy       | 1.26.4       | Required for PyTorch compatibility |
| triton      | 2.1.0        | PyTorch backend                    |
```
Verification:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# CUDA Libraries

Installed via pip:

| Package            | Version  |
| ------------------ | -------- |
| nvidia-cudnn-cu12  | 8.9.2.26 |
| nvidia-cublas-cu12 | 12.1.3.1 |

Verified:

```bash
python -c "import torch; print(torch.backends.cudnn.version())"
```

Result:

```
8902
```

---

# Audio and MIR Libraries
```
| Package   | Version |
| --------- | ------- |
| librosa   | latest  |
| soundfile | latest  |
| scipy     | latest  |
| audioread | latest  |
| av        | 13.1.0  |
```
These provide audio decoding and MIR utilities.

---

# Retrieval Model
```
| Package    | Version |
| ---------- | ------- |
| laion-clap | 1.1.7   |
```
Used for:

* audio embedding
* similarity retrieval
* CLAP evaluation metric

Verification:

```bash
python -c "import laion_clap"
```

---

# Music Generation

Installed:

| Package      | Version |
| ------------ | ------- |
| audiocraft   | 1.3.0   |
| transformers | 4.35.2  |
| hydra-core   | 1.3.2   |
| omegaconf    | 2.3.0   |
| encodec      | 0.1.1   |
| demucs       | 4.0.1   |

Model used:

```
facebook/musicgen-melody
```

Test:

```bash
python -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-melody', device='cuda')"
```

---

# Final Verified State

Working components:

* GPU acceleration
* CLAP embedding
* MusicGen melody model
* Audio decoding
* PyTorch CUDA backend

Test command:

```bash
python -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-melody', device='cuda')"
```

Output:

```
loaded
```

Environment is now ready for:

1. audio retrieval
2. captioning
3. controllable music generation
4. evaluation metrics

```
```