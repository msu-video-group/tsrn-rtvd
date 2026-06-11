from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        padding: int,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class RepDWConv3x3(nn.Module):
    """MobileOne-style reparameterizable depthwise 3x3 block.

    Training graph:
      BN(DW 3x3) + padded BN(DW 1x1) + BN(identity) if stride==1, then ReLU.

    Deploy graph:
      one fused depthwise Conv2d(3x3) with bias, then ReLU.
    """

    def __init__(self, channels: int, stride: int = 1, deploy: bool = False) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("RepDWConv3x3 currently supports stride 1 or 2.")
        self.channels = channels
        self.stride = stride
        self.deploy = deploy
        if deploy:
            self.fused = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels,
                bias=True,
            )
        else:
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    stride=stride,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    1,
                    stride=stride,
                    padding=0,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.branch_id = nn.BatchNorm2d(channels) if stride == 1 else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.fused(x))
        y = self.branch_3x3(x)
        y = y + self.branch_1x1(x)
        if self.branch_id is not None:
            y = y + self.branch_id(x)
        return self.act(y)

    @staticmethod
    def _fuse_conv_bn(
        conv: nn.Conv2d, bn: nn.BatchNorm2d
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        if conv.bias is None:
            bias = torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
        else:
            bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused_w = w * scale
        fused_b = beta + (bias - running_mean) * gamma / std
        return fused_w, fused_b

    def _fuse_identity_bn(
        self, bn: nn.BatchNorm2d
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = torch.zeros(
            self.channels,
            1,
            3,
            3,
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        kernel[:, 0, 1, 1] = 1.0
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        std = torch.sqrt(running_var + bn.eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused_w = kernel * scale
        fused_b = beta - running_mean * gamma / std
        return fused_w, fused_b

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.fused.weight, self.fused.bias
        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        k1 = F.pad(k1, [1, 1, 1, 1])
        kernel = k3 + k1
        bias = b3 + b1
        if self.branch_id is not None:
            kid, bid = self._fuse_identity_bn(self.branch_id)
            kernel = kernel + kid
            bias = bias + bid
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        fused = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        fused.weight.data.copy_(kernel)
        fused.bias.data.copy_(bias)
        self.fused = fused
        del self.branch_3x3
        del self.branch_1x1
        if hasattr(self, "branch_id"):
            del self.branch_id
        self.deploy = True


class DSRepBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int, deploy: bool = False
    ) -> None:
        super().__init__()
        self.dw = RepDWConv3x3(in_ch, stride=stride, deploy=deploy)
        self.pw = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        D: int = 128,
        stem_channels: int = 16,
        stage_channels: Sequence[int] = (24, 48, 96),
        stage_depths: Sequence[int] = (2, 3, 4),
        pooling: str = "gap",
        grid_pool: tuple[int, int] = (2, 3),
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(stage_depths):
            raise ValueError("stage_channels and stage_depths must match length.")
        if pooling not in {"gap", "attn_grid"}:
            raise ValueError("pooling must be 'gap' or 'attn_grid'.")
        self.pooling = pooling
        self.grid_pool = grid_pool
        self.stem = ConvBNAct(in_ch, stem_channels, kernel=3, stride=2, padding=1)
        blocks: list[nn.Module] = []
        in_c = stem_channels
        for out_c, depth in zip(stage_channels, stage_depths):
            if depth < 1:
                raise ValueError("Each stage depth must be >=1.")
            blocks.append(DSRepBlock(in_c, out_c, stride=2, deploy=deploy))
            for _ in range(depth - 1):
                blocks.append(DSRepBlock(out_c, out_c, stride=1, deploy=deploy))
            in_c = out_c
        self.stages = nn.Sequential(*blocks)
        self.proj = ConvBNAct(in_c, D, kernel=1, stride=1, padding=0)
        if pooling == "attn_grid":
            self.grid = nn.AdaptiveAvgPool2d(grid_pool)
            self.cell_score = nn.Linear(D, 1)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.proj(x)
        if self.pooling == "gap":
            return self.gap(x).flatten(1)
        x = self.grid(x)  # [N, D, gh, gw]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [N, cells, D]
        weights = torch.softmax(self.cell_score(x), dim=1)
        return (weights * x).sum(dim=1)


class TemporalDSBlock(nn.Module):
    def __init__(self, D: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Temporal kernel must be odd.")
        self.dw = nn.Conv1d(
            D,
            D,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=D,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(D)
        self.act = nn.ReLU(inplace=True)
        self.pw = nn.Conv1d(D, D, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw(y)
        y = self.bn2(y)
        return x + y


class TemporalMixer(nn.Module):
    def __init__(self, D: int = 128, num_blocks: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[TemporalDSBlock(D, kernel_size) for _ in range(num_blocks)]
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # [B, K, D] -> [B, D, K] -> [B, K, D]
        u = t.transpose(1, 2).contiguous()
        u = self.blocks(u)
        return u.transpose(1, 2).contiguous()


class TinyDelta3D(nn.Module):
    def __init__(
        self,
        K: int = 3,
        in_ch: int = 3,
        D: int = 128,
        stem_channels: int = 16,
        stage_channels: Sequence[int] = (24, 48, 96),
        stage_depths: Sequence[int] = (2, 3, 4),
        pooling: str = "gap",
        grid_pool: tuple[int, int] = (2, 3),
        temporal_blocks: int = 2,
        temporal_kernel: int = 3,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if K % 2 != 1:
            raise ValueError("K must be odd.")
        self.K = K
        self.center = K // 2
        self.neighbor_indices = [i for i in range(K) if i != self.center]
        self.D = D

        self.encoder = FrameEncoder(
            in_ch=in_ch,
            D=D,
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            stage_depths=stage_depths,
            pooling=pooling,
            grid_pool=grid_pool,
            deploy=deploy,
        )
        self.temporal_in = nn.Sequential(
            nn.Linear(2 * D + 1, D),
            nn.LayerNorm(D),
            nn.ReLU(inplace=True),
        )
        self.temporal = TemporalMixer(
            D=D, num_blocks=temporal_blocks, kernel_size=temporal_kernel
        )
        head_hidden = 128 if D >= 128 else D
        head_mid = 64 if D >= 128 else max(32, D // 2)
        self.head = nn.Sequential(
            nn.Linear(3 * D + 1, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, head_mid),
            nn.ReLU(inplace=True),
            nn.Linear(head_mid, 4),
        )
        r = torch.linspace(-1.0, 1.0, steps=K)
        self.register_buffer("time_offsets", r.view(1, K, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,K,C,H,W], got {tuple(x.shape)}")
        B, K, Cin, H, W = x.shape
        if K != self.K:
            raise ValueError(f"Expected K={self.K}, got K={K}")
        x = x.reshape(B * K, Cin, H, W)
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        f = self.encoder(x)  # [B*K, D]
        f = f.reshape(B, K, -1)

        c = f[:, self.center : self.center + 1, :]
        fdiff = f - c
        r = self.time_offsets.to(dtype=f.dtype).expand(B, -1, -1)
        t = torch.cat([f, fdiff, r], dim=-1)
        t = self.temporal_in(t)
        t = self.temporal(t)

        center_tok = t[:, self.center : self.center + 1, :]
        neigh_tok = t[:, self.neighbor_indices, :]
        center_rep = center_tok.expand(-1, K - 1, -1)
        r_neigh = r[:, self.neighbor_indices, :]
        pair = torch.cat(
            [center_rep, neigh_tok, neigh_tok - center_rep, r_neigh], dim=-1
        )
        out = self.head(pair)
        return out[..., :3], out[..., 3:4]


def build_model_from_config(cfg) -> TinyDelta3D:
    return TinyDelta3D(
        K=cfg.data.K,
        in_ch=cfg.model.in_ch,
        D=cfg.model.D,
        stem_channels=cfg.model.stem_channels,
        stage_channels=cfg.model.stage_channels,
        stage_depths=cfg.model.stage_depths,
        pooling=cfg.model.pooling,
        grid_pool=tuple(cfg.model.grid_pool),
        temporal_blocks=cfg.model.temporal_blocks,
        temporal_kernel=cfg.model.temporal_kernel,
        deploy=cfg.model.deploy,
    )


def switch_to_deploy(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, RepDWConv3x3):
            module.switch_to_deploy()
    return model


def make_deploy_copy(model: nn.Module) -> nn.Module:
    deploy_model = deepcopy(model).eval()
    switch_to_deploy(deploy_model)
    return deploy_model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
