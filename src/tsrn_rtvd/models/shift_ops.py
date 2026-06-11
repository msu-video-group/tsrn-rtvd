"""
Fast GPU-side feature alignment for TS-RNN models.

The original shift operators in ``models.new_blocks`` and ``models.tsrnn_blocks``
convert ``dp`` to NumPy and loop over the batch. On CUDA this forces a
host/device synchronization for every alignment call. The replacements here keep
all displacement math on-device and use batched ``grid_sample``.

Modes
-----
roll_match
    Checkpoint-safe mode. Uses round(dp / scale), matching the old integer-roll
    operating point, but without ``dp.detach().cpu().numpy()`` and the Python
    batch loop.
exact
    Finetune mode. Builds one full-displacement ``grid_sample``. For modules
    with a fractional MLP, the full displacement starts from ``dp / scale`` plus
    a fixed compatibility residual ``round(dp / scale) - dp / scale`` so the
    first iteration remains close to the old checkpoint; the MLP can learn to
    cancel that residual during a short recover finetune.
"""

from __future__ import annotations

import types
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn.functional as F

_DEF_MAX = 32
_GRID_CACHE_MAX = 32
_GRID_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _grid_key(B: int, H: int, W: int, feat: torch.Tensor) -> tuple:
    device = feat.device
    return (device.type, device.index, str(feat.dtype), B, H, W)


def _base_grid(B: int, H: int, W: int, feat: torch.Tensor) -> torch.Tensor:
    """Return a cached pixel-center grid in NDC coordinates."""
    key = _grid_key(B, H, W, feat)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        _GRID_CACHE.move_to_end(key)
        return cached

    ly = torch.linspace(-1.0, 1.0, H, device=feat.device, dtype=feat.dtype)
    lx = torch.linspace(-1.0, 1.0, W, device=feat.device, dtype=feat.dtype)
    gy, gx = torch.meshgrid(ly, lx, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    _GRID_CACHE[key] = grid
    if len(_GRID_CACHE) > _GRID_CACHE_MAX:
        _GRID_CACHE.popitem(last=False)
    return grid


def translate(feat: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
    """
    Batched zero-padded translation.

    ``feat`` is ``(B, C, H, W)``. ``shift_px`` is ``(B, 2)`` in feature pixels:
    positive dx moves content right, positive dy moves content down.

    For ``align_corners=True`` a one-pixel translation corresponds to
    ``2 / (W - 1)`` or ``2 / (H - 1)`` in normalized grid coordinates. With an
    integer ``shift_px`` this matches ``torch.roll`` + zero-filled borders up to
    floating-point roundoff.
    """
    B, _, H, W = feat.shape
    grid0 = _base_grid(B, H, W, feat)
    sx_n = (shift_px[:, 0].to(dtype=feat.dtype) * 2.0 / max(W - 1, 1)).view(B, 1, 1)
    sy_n = (shift_px[:, 1].to(dtype=feat.dtype) * 2.0 / max(H - 1, 1)).view(B, 1, 1)
    grid = torch.stack([grid0[..., 0] - sx_n, grid0[..., 1] - sy_n], dim=-1)
    return F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _scaled_shift(
    dp: torch.Tensor, scale: float, max_shift: int, mode: str
) -> torch.Tensor:
    """Full-res dp -> feature-space shift in pixels."""
    s = (dp[:, :2] / scale).clamp(-max_shift, max_shift)
    if mode == "roll_match":
        s = torch.round(s)
    elif mode != "exact":
        raise ValueError(f"Unsupported fast-shift mode: {mode!r}")
    return s


def _legacy_round_residual(s: torch.Tensor) -> torch.Tensor:
    """Fixed init residual: continuous base + residual == rounded base."""
    return torch.round(s).detach() - s.detach()


def _make_roll_shift_zero_gpu(
    mode: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Factory for the direct TSRNNLite roll_shift_zero replacement."""

    def roll_shift_zero_gpu(h: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        s = _scaled_shift(dp, scale=4.0, max_shift=12, mode=mode)
        return translate(h, s)

    return roll_shift_zero_gpu


def roll_shift_zero_gpu(h: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    """Backward-compatible checkpoint-safe replacement for direct imports."""
    return _make_roll_shift_zero_gpu("roll_match")(h, dp)


def _shiftwarp_fast_forward(
    self, feat: torch.Tensor, dp: torch.Tensor, mode: str
) -> torch.Tensor:
    """
    Fast ``new_blocks.ShiftWarp``.

    Old path:
        integer ``roll_and_zero(feat, round(dp / scale))``
        followed by ``_bilinear_translate(shifted, subpix(dp))``.

    Fast path:
        one ``grid_sample`` on ``dp / scale + subpix(dp)``. In ``roll_match``
        mode the base displacement is rounded; in ``exact`` mode it is the full
        continuous camera displacement and is intended for short recover-finetune.
    """
    s = _scaled_shift(dp, scale=float(self.scale), max_shift=_DEF_MAX, mode=mode)
    if mode == "exact":
        # Start near old checkpoint behavior while still expressing the warp as
        # a full-displacement grid_sample. During recover finetune, subpix(dp)
        # can learn to cancel this fixed residual and exploit the continuous dp.
        s = s + _legacy_round_residual(s)
    delta = 0.5 * torch.tanh(self.subpix(dp))
    return translate(feat, s + delta)


def _hybridshift_fast_forward(
    self, h: torch.Tensor, dp: torch.Tensor, mode: str
) -> torch.Tensor:
    """
    Fast ``tsrnn_blocks.HybridShift``.

    ``roll_match`` preserves the old operating point by first applying the
    rounded displacement and then using the historical fractional-grid sign.

    ``exact`` is the finetune path: it predicts the fractional correction from
    the unshifted hidden descriptor and applies ``dp / 4 + delta`` with a single
    zero-padded ``grid_sample``. This intentionally removes the integer roll and
    the second bilinear pass; use it with ``--init-checkpoint`` and 3-5 recover
    epochs.
    """
    s = _scaled_shift(dp, scale=4.0, max_shift=12, mode=mode)

    if mode == "exact":
        # Same compatibility trick as ShiftWarp: initial full warp is close to
        # old rounded alignment, then the MLP can recover/subpixel-adapt.
        s = s + _legacy_round_residual(s)
        h_pool = h.mean(dim=[2, 3])
        delta = torch.tanh(self.mlp(torch.cat([h_pool, dp[:, :2]], dim=1))) * 0.5
        return translate(h, s + delta)

    # Checkpoint-safe compatibility path.
    h_roll = translate(h, s)
    B, _, H, W = h_roll.shape
    h_pool = h_roll.mean(dim=[2, 3])
    delta = torch.tanh(self.mlp(torch.cat([h_pool, dp[:, :2]], dim=1))) * 0.5

    # Keep the original HybridShift fractional convention: grid + offset with
    # 2/W, 2/H normalization. This path is for old checkpoints, not for the new
    # finetuned exact alignment.
    base = _base_grid(B, H, W, h_roll)
    dx_n = (delta[:, 0].to(dtype=h_roll.dtype) * 2.0 / W).view(B, 1, 1, 1)
    dy_n = (delta[:, 1].to(dtype=h_roll.dtype) * 2.0 / H).view(B, 1, 1, 1)
    offset = torch.cat(
        [dx_n.expand(B, H, W, 1), dy_n.expand(B, H, W, 1)],
        dim=-1,
    )
    return F.grid_sample(
        h_roll,
        base + offset,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def apply_speed_patches(
    model: torch.nn.Module, mode: str = "roll_match"
) -> torch.nn.Module:
    """
    Patch all supported shift modules in-place and return ``model``.

    ``roll_match`` is checkpoint-safe. ``exact`` is a finetune/recover mode.
    """
    if mode not in {"roll_match", "exact"}:
        raise ValueError(f"Unsupported fast-shift mode: {mode!r}")

    n_shiftwarp = 0
    n_hybrid = 0
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "ShiftWarp":
            module.forward = types.MethodType(
                lambda self, feat, dp, _mode=mode: _shiftwarp_fast_forward(
                    self, feat, dp, _mode
                ),
                module,
            )
            n_shiftwarp += 1
        elif cls_name == "HybridShift":
            module.forward = types.MethodType(
                lambda self, h, dp, _mode=mode: _hybridshift_fast_forward(
                    self, h, dp, _mode
                ),
                module,
            )
            n_hybrid += 1

    # Patch direct references used by tsrnn_net.TSRNNLiteV1/TSRNNv2.
    try:
        from . import tsrnn_blocks, tsrnn_net

        roll_impl = _make_roll_shift_zero_gpu(mode)
        tsrnn_blocks.roll_shift_zero = roll_impl
        tsrnn_net.roll_shift_zero = roll_impl
    except Exception:
        pass

    model._shift_fast_mode = mode
    print(
        f"[shift_ops] fast alignment enabled: mode={mode}, "
        f"ShiftWarp={n_shiftwarp}, HybridShift={n_hybrid}, roll_shift_zero=GPU"
    )
    return model


def to_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    """Enable channels-last inference layout for models that allocate state."""
    model = model.to(memory_format=torch.channels_last)
    model._channels_last_inference = True
    return model


def maybe_compile_step(
    model: torch.nn.Module, mode: str = "reduce-overhead"
) -> torch.nn.Module:
    """Compile ``model.step`` when the current PyTorch build supports it."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch version")
    model.step = torch.compile(model.step, mode=mode, fullgraph=False)  # type: ignore[method-assign]
    print(f"[shift_ops] torch.compile enabled for model.step, mode={mode}")
    return model
