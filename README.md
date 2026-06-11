# TSRN-RTVD: Real-Time Video Deblurring System

## Overview

**TSRN-RTVD** is a real-time video deblurring system for streaming handheld footage. It restores video with one-frame look-ahead using a compact trajectory-guided recurrent model.

## Installation

```shell
pip install git+https://github.com/msu-video-group/tsrn-rtvd
```

## Live Demo

A webcam is required to run the live demo.

```shell
tsrn-rtvd-demo
```

Press `q` or `Esc` to exit. Press `r` to reset the streaming state from the current frame.

## Usage

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

## Dev Installation

```shell
git clone https://github.com/msu-video-group/tsrn-rtvd.git
cd tsrn-rtvd
pip install -e .[dev]
pre-commit install
```

## Training And Evaluation

Download the GoPro deblurring dataset and set `data.root` in `src/tsrn_rtvd/configs/tsrnn_bt1.yaml` to the dataset root. For trajectory-based training or oracle evaluation, also set `data.traj_root` to the precomputed GoPro trajectory directory.

Train TSRN-RTVD:

```shell
tsrn-rtvd-train
```

Evaluate TSRN-RTVD:

```shell
tsrn-rtvd-eval --checkpoint path/to/best.pth
```
