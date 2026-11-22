"""Evaluate the released streaming model on the complete GoPro test split."""

import argparse
from pathlib import Path

import cv2

from tsrn_rtvd import TSRNRTVD
from tsrn_rtvd.demo.frame_io import bgr_to_tensor
from tsrn_rtvd.utils.metrics import compute_psnr

PAPER_REVISION = "2c9a43a8111ea398f87dd295bc6693f5189d7568"


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot read {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/GOPRO_Large"))
    args = parser.parse_args()

    split = args.root / "test"
    scenes = sorted(p for p in split.iterdir() if p.is_dir())
    inputs = [sorted((s / "blur").glob("*.png")) for s in scenes]
    if len(scenes) != 11 or sum(map(len, inputs)) != 1111 or not all(inputs):
        parser.error(
            f"Expected 11 scenes and 1111 blurred frames under {split.resolve()}"
        )
    model = TSRNRTVD.from_pretrained(device="cuda", revision=PAPER_REVISION)
    total = 0.0
    count = 0
    for scene, frames in zip(scenes, inputs):
        model.reset(read_image(frames[0]))
        score = 0.0
        for current, following in zip(frames, frames[1:] + frames[-1:]):
            result = model.process_next(read_image(following), return_tensor=True)
            target = bgr_to_tensor(
                read_image(scene / "sharp" / current.name), model.device, False
            )
            score += compute_psnr(result.output_rgb_tensor.float().clamp(0, 1), target)
        print(
            f"{scene.name}: {len(frames)} frames, {score / len(frames):.4f} dB",
            flush=True,
        )
        total += score
        count += len(frames)
    print(f"GoPro test: {count} frames, PSNR {total / count:.4f} dB")


if __name__ == "__main__":
    main()
