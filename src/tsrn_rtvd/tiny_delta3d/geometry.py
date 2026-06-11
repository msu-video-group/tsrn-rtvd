from __future__ import annotations

import torch


def quat_xyzw_to_rotmat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert quaternion(s) in qx,qy,qz,qw order to rotation matrices.

    Args:
        q: Tensor with shape [..., 4] in x, y, z, w order.
    Returns:
        Tensor with shape [..., 3, 3].
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    x, y, z, w = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = torch.stack(
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1
    )
    row1 = torch.stack(
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1
    )
    row2 = torch.stack(
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def center_relative_translations(
    positions_world: torch.Tensor,
    quats_xyzw: torch.Tensor,
    center: int,
    neighbor_indices: list[int],
    target_frame: str = "center_camera",
) -> torch.Tensor:
    """Build [K-1, 3] center-to-neighbor translation targets.

    positions_world: [K, 3]
    quats_xyzw: [K, 4]
    """
    p_center = positions_world[center]
    deltas = positions_world[neighbor_indices] - p_center.unsqueeze(0)
    if target_frame == "world":
        return deltas
    if target_frame != "center_camera":
        raise ValueError(f"Unknown target_frame={target_frame!r}")
    r_center = quat_xyzw_to_rotmat(quats_xyzw[center])
    return torch.einsum("ij,nj->ni", r_center.transpose(-1, -2), deltas)
