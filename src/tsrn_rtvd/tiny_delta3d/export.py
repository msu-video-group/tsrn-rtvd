from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .model import build_model_from_config, count_parameters, switch_to_deploy
from .utils import configure_torch_for_speed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fused TinyDelta-3D checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Also save a traced TorchScript module next to --out.",
    )
    args = parser.parse_args()

    configure_torch_for_speed()
    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model_from_config(cfg).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    switch_to_deploy(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stats": ckpt.get("stats"),
            "config_name": cfg.name,
            "deploy": True,
        },
        out,
    )
    print(f"saved fused checkpoint: {out}")
    print(f"deploy_params={count_parameters(model):,}")
    if args.torchscript:
        x = torch.randn(1, cfg.data.K, cfg.model.in_ch, cfg.data.height, cfg.data.width)
        traced = torch.jit.trace(model, x)
        ts_path = out.with_suffix(".torchscript.pt")
        traced.save(str(ts_path))
        print(f"saved torchscript: {ts_path}")


if __name__ == "__main__":
    main()
