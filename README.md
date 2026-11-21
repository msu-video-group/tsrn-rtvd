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

```shell
git clone https://github.com/msu-video-group/tsrn-rtvd.git
cd tsrn-rtvd
pip install -e '.[dev]'
pre-commit install
```

## 🗂️ Datasets

Download [GOPRO_Large](https://seungjunnah.github.io/Datasets/gopro.html) and extract it with the original `train` and `test` directories. Set the paths in [tsrnn_bt1.yaml](src/tsrn_rtvd/configs/tsrnn_bt1.yaml) before training or evaluation.

| Config key | Example root | Expected files relative to that root |
| --- | --- | --- |
| `data.root` | `/data/GOPRO_Large` | `{train,test}/<scene>/blur/*.png` and `{train,test}/<scene>/sharp/*.png` |
| `data.traj_root` | `/data/GOPRO_trajectories` | `{train,test}/<scene>/trajectories.npz` |

For trajectory-based training or oracle evaluation, provide the precomputed trajectories separately, using the same split and scene names as the images.

## 🧪 Training and Evaluation

Train TSRN-RTVD:

```shell
tsrn-rtvd-train
```

Evaluate a trained checkpoint on the GoPro test split:

```shell
tsrn-rtvd-eval --checkpoint path/to/best.pth
```

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
