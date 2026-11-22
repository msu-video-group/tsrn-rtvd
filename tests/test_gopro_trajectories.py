import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from tsrn_rtvd.data.dataset import SceneTraj
from tsrn_rtvd.data.gopro_sequence_dataset import GoProSequenceDataset
from tsrn_rtvd.tiny_delta3d.dataset import GoproTrajectoryDataset


def test_required_trajectory_files(tmp_path):
    images = tmp_path / "images" / "train" / "scene"
    blur = images / "blur"
    blur.mkdir(parents=True)
    (images / "sharp").mkdir()
    (blur / "000001.png").touch()
    labels = tmp_path / "labels" / "train"
    labels.mkdir(parents=True)
    expected = labels / "scene" / "trajectories.npz"
    with pytest.raises(FileNotFoundError) as train:
        GoProSequenceDataset(images.parent, labels, seq_len=1)
    with pytest.raises(FileNotFoundError) as oracle:
        SceneTraj(expected)
    with pytest.raises(FileNotFoundError) as predictor:
        GoproTrajectoryDataset(images.parent.parent, labels.parent, "train")
    assert str(expected) in str(train.value) and str(expected) in str(oracle.value)
    assert str(expected.with_suffix(".txt")) in str(predictor.value)


def test_invalid_trajectory_file(tmp_path):
    path = tmp_path / "trajectories.npz"
    np.savez(path, poses=np.ones((1, 7)))
    with pytest.raises(ValueError) as error:
        SceneTraj(path)
    assert str(path) in str(error.value)


def test_generator_resumes_without_loading_weights(tmp_path, monkeypatch, capsys):
    output = tmp_path / "out" / "train" / "scene"
    output.mkdir(parents=True)
    script = (
        Path(__file__).resolve().parents[1] / "tools" / "estimate_gopro_trajectories.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--raw-root",
            str(tmp_path / "raw"),
            "--images-root",
            str(tmp_path / "images"),
            "--output-root",
            str(output.parents[1]),
            "--split",
            "train",
            "--scene",
            "scene",
        ],
    )
    # A partial result must survive a retry.
    (output / "trajectories.txt").write_text("labels")
    with pytest.raises(FileExistsError, match="Incomplete scene"):
        runpy.run_path(str(script), run_name="__main__")
    np.savez(output / "trajectories.npz", poses=np.ones((1, 7)))
    runpy.run_path(str(script), run_name="__main__")
    assert "skipping" in capsys.readouterr().out
    assert (output / "trajectories.txt").read_text() == "labels"
