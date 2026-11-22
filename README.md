<h1 align="center">TSRN-RTVD: Real-Time Video Deblurring System</h1>

<p align="center">
  <a href="https://orcid.org/0009-0007-6825-9720">Nikita Alutis</a> ·
  <a href="https://orcid.org/0009-0000-9714-2256">Danila Evsyukov</a> ·
  <a href="https://orcid.org/0009-0004-6156-195X">Egor Chistov</a> ·
  <a href="https://orcid.org/0009-0008-0647-3649">Mikhail Voronin</a> ·
  <a href="https://orcid.org/0000-0002-6173-3561">Evgeney Bogatyrev</a> ·
  <a href="https://orcid.org/0000-0002-8893-9340">Dmitriy Vatolin</a>
</p>
<h3 align="center">ACM MM 2026 Demo</h3>
<p align="center">
  <a href="https://doi.org/10.1145/3767308.3839003">📄 Paper</a> ·
  <a href="https://youtu.be/3alMwVrVALU">🎬 Demo</a> ·
  <a href="https://huggingface.co/egorchistov/deblurring-tsrn-rtvd">📦 Models</a>
</p>

## 🏅 Overview

**TSRN-RTVD** is a real-time video deblurring system for streaming handheld footage. It restores video with one-frame look-ahead using a compact trajectory-guided recurrent model.

## 📥 Installation

```shell
pip install git+https://github.com/msu-video-group/tsrn-rtvd
```

## 🎥 Live Demo

A webcam is required to run the live demo.

```shell
tsrn-rtvd-demo
```

Press `q` or `Esc` to exit. Press `r` to reset the streaming state from the current frame.

## 🚀 Usage

```python
import numpy as np
from tsrn_rtvd import TSRNRTVD

model = TSRNRTVD.from_pretrained("egorchistov/deblurring-tsrn-rtvd", device="cuda")

first = np.zeros((720, 1280, 3), dtype=np.uint8)
next_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

model.reset(first)
result = model.process_next(next_frame)
deblurred_bgr = result.output_bgr
```

## 🛠️ Dev Installation

For training, evaluation, or trajectory estimation, install from a clone and run the commands below from the repository root.

```shell
git clone https://github.com/msu-video-group/tsrn-rtvd.git
cd tsrn-rtvd
pip install -e '.[dev]'
pre-commit install
```

## 🗂️ Datasets

Download the [paired GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) and extract it to `data/GOPRO_Large/`, with frames under `{train,test}/<scene>/{blur,sharp}/*.png`. Precomputed trajectories for all 33 scenes are included in [data/GOPRO_trajectories](data/GOPRO_trajectories).

## 🧪 Training and Evaluation

```shell
# Evaluate the released model on GoPro (30.08 dB PSNR)
python tools/evaluate_gopro.py

# Train the restoration model
tsrn-rtvd-train

# Evaluate a trained checkpoint with precomputed trajectories
tsrn-rtvd-eval --checkpoint path/to/best.pth --oracle-z-ref 10
```

### 🧭 Estimate GoPro trajectories (optional)

To recompute trajectories, download the original 240 fps frames (**GOPRO_Large_all**) from the [GoPro dataset page](https://seungjunnah.github.io/Datasets/gopro.html) and run the estimator on a CUDA GPU.

```shell
python tools/estimate_gopro_trajectories.py \
  --raw-root data/GOPRO_Large_all \
  --images-root data/GOPRO_Large \
  --output-root outputs/GOPRO_estimated \
  --split train --scene GOPR0372_07_01
```

Omit `--scene` to process the whole split.

## 📖 Citation

```bibtex
@inproceedings{Alutis_2026,
  author = {Alutis, Nikita and Evsyukov, Danila and Chistov, Egor and Voronin, Mikhail and Bogatyrev, Evgeney and Vatolin, Dmitriy},
  title = {{TSRN-RTVD}: Real-Time Video Deblurring System},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  publisher = {ACM},
  year = {2026},
  pages = {17172--17174},
  doi = {10.1145/3767308.3839003}
}
```
