from __future__ import annotations

import argparse

import torch

from .config import load_config
from .losses import DeltaStandardizer, tinydelta_loss
from .model import build_model_from_config, count_parameters, make_deploy_copy
from .preprocess import FixedDerivativePreprocess
from .utils import configure_torch_for_speed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic tensor sanity check for TinyDelta-3D."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    configure_torch_for_speed()
    cfg = load_config(args.config)
    model = build_model_from_config(cfg).eval()
    preprocess = FixedDerivativePreprocess().eval()
    x_rgb = torch.rand(args.batch, cfg.data.K, 3, cfg.data.height, cfg.data.width)
    x = preprocess(x_rgb)
    pred, log_var = model(x)
    y = torch.randn(args.batch, cfg.data.K - 1, 3)
    loss = tinydelta_loss(pred, log_var, y, DeltaStandardizer())
    deploy = make_deploy_copy(model)
    pred_deploy, _ = deploy(x)
    print(f"config={cfg.name}")
    print(f"x_rgb={tuple(x_rgb.shape)} x_pre={tuple(x.shape)}")
    print(f"neighbor_indices={model.neighbor_indices}")
    print(f"pred={tuple(pred.shape)} log_var={tuple(log_var.shape)}")
    print(f"params_train_graph={count_parameters(model):,}")
    print(f"params_deploy_graph={count_parameters(deploy):,}")
    print(f"loss={loss.total.item():.6f}")
    print(f"max_abs_diff_deploy={float((pred - pred_deploy).abs().max().detach()):.8f}")


if __name__ == "__main__":
    main()
