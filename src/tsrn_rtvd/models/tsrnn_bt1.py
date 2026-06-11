"""
TSRNN_BT1 – Delay-1 tri-frame (t-1, t, t+1) + second-order causal state.

Each step consumes: B_prev, B_cur, B_next (1-frame lookahead).
Streaming: cache encoded features of (prev, cur) and encode only the
new next frame per output step.

State:
  h4/h4_prev/h8/h8_prev/dp_prev — identical to E30
  feat_prev: (X2, F4, F8) of B_{t-1}   — for streaming
  feat_cur:  (X2, F4, F8) of B_t       — for streaming
  B_cur_raw: (B, 3, H, W) of B_t       — needed for residual output

Training forward(): encodes all L frames in one batched call, then
unrolls sequentially. At each step t neighbour features come from
the pre-encoded tensors (no redundant encoding).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .new_blocks import (
    RB,
    PoseEmbed,
    SharedEncoder,
    ShiftWarp,
    UpBlock,
    lrelu,
    make_rb_stack,
    pad_dp,
    weighted_sum,
)

P = 4


class TSRNN_BT1(nn.Module):
    def __init__(self, update8_blocks: int = 10, update4_blocks: int = 10):
        super().__init__()
        self.encoder = SharedEncoder()

        self.sw4 = ShiftWarp(scale=4, P=P)
        self.sw8 = ShiftWarp(scale=8, P=P)

        # Separate pose embeds for past/future at each scale
        self.pe8_neg = PoseEmbed(P, 96)
        self.pe8_pos = PoseEmbed(P, 96)
        self.pe4_neg = PoseEmbed(P, 64)
        self.pe4_pos = PoseEmbed(P, 64)

        # ── 1/8 stage ──────────────────────────────────────────
        # Compress neighbor evidence:  cat(N8_neg, N8_pos): 192 → 96
        self.cn8 = nn.Sequential(nn.Conv2d(192, 96, 1, bias=True), lrelu())
        # Compress past-state:         cat(A8_1, A8_2): 192 → 96
        self.ch8 = nn.Sequential(nn.Conv2d(192, 96, 1, bias=True), lrelu())
        # Compress pose:               cat(E8_neg, E8_pos): 192 → 96
        self.cp8 = nn.Sequential(nn.Conv2d(192, 96, 1, bias=True), lrelu())
        # Z8 = cat(F8, CN8, CH8, CP8) → 384
        self.update8_in = nn.Sequential(nn.Conv2d(384, 96, 3, 1, 1, bias=True), lrelu())
        self.update8_rbs = make_rb_stack(96, update8_blocks)

        # ── 1/4 stage ──────────────────────────────────────────
        self.lift8to4 = nn.Sequential(nn.Conv2d(96, 64, 3, 1, 1, bias=True), lrelu())
        self.cn4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=True), lrelu())
        self.ch4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=True), lrelu())
        self.cp4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=True), lrelu())
        # Z4 = cat(F4, U8to4, CN4, CH4, CP4) → 320
        self.update4_in = nn.Sequential(nn.Conv2d(320, 64, 3, 1, 1, bias=True), lrelu())
        self.update4_rbs = make_rb_stack(64, update4_blocks)

        # ── Decoder ────────────────────────────────────────────
        self.up_d2 = UpBlock(64, 32, 48, nRB=4)
        self.up_d1 = nn.Sequential(nn.Conv2d(48, 24, 3, 1, 1, bias=True), lrelu())
        self.dec_d1 = make_rb_stack(24, 3)
        self.head = nn.Conv2d(24, 3, 3, 1, 1, bias=True)

    # ── Internal fuse helpers ──────────────────────────────────

    def _fuse8(self, F8, N8_neg, N8_pos, A8_1, A8_2, dp4_neg, dp4_pos):
        E8_neg = self.pe8_neg.as_map(dp4_neg, F8)
        E8_pos = self.pe8_pos.as_map(dp4_pos, F8)
        CN8 = self.cn8(torch.cat([N8_neg, N8_pos], dim=1))
        CH8 = self.ch8(torch.cat([A8_1, A8_2], dim=1))
        CP8 = self.cp8(torch.cat([E8_neg, E8_pos], dim=1))
        Z8 = torch.cat([F8, CN8, CH8, CP8], dim=1)  # (B, 384, ·)
        return self.update8_rbs(self.update8_in(Z8))

    def _fuse4(self, F4, U8to4, N4_neg, N4_pos, A4_1, A4_2, dp4_neg, dp4_pos):
        E4_neg = self.pe4_neg.as_map(dp4_neg, F4)
        E4_pos = self.pe4_pos.as_map(dp4_pos, F4)
        CN4 = self.cn4(torch.cat([N4_neg, N4_pos], dim=1))
        CH4 = self.ch4(torch.cat([A4_1, A4_2], dim=1))
        CP4 = self.cp4(torch.cat([E4_neg, E4_pos], dim=1))
        Z4 = torch.cat([F4, U8to4, CN4, CH4, CP4], dim=1)  # (B, 320, ·)
        return self.update4_rbs(self.update4_in(Z4))

    def _decode(self, H4, X2, B_t):
        D2 = self.up_d2(H4, X2)
        D2u = F.interpolate(D2, scale_factor=2, mode="bilinear", align_corners=False)
        D1 = self.dec_d1(self.up_d1(D2u))
        return B_t + self.head(D1)

    # ── Single step using pre-encoded features ─────────────────

    def _step_from_feats(
        self,
        feat_prev,
        feat_cur,
        feat_next,
        B_cur,
        state,
        dp4_neg,  # dp_{t-1→t}
        dp4_pos,  # dp_{t+1→t}
        dp4_cum,  # dp_{t-2→t}
    ) -> tuple[torch.Tensor, dict]:

        X2_c, F4_c, F8_c = feat_cur
        _, F4_n, F8_n = feat_next
        _, F4_p, F8_p = feat_prev

        H4_1, H4_2 = state["h4"], state["h4_prev"]
        H8_1, H8_2 = state["h8"], state["h8_prev"]

        # Align neighbor frames to centre viewpoint
        N8_neg = self.sw8(F8_p, dp4_neg)
        N8_pos = self.sw8(F8_n, dp4_pos)
        N4_neg = self.sw4(F4_p, dp4_neg)
        N4_pos = self.sw4(F4_n, dp4_pos)

        # Align past states
        A8_1 = self.sw8(H8_1, dp4_neg)
        A8_2 = self.sw8(H8_2, dp4_cum)
        A4_1 = self.sw4(H4_1, dp4_neg)
        A4_2 = self.sw4(H4_2, dp4_cum)

        H8_new = self._fuse8(F8_c, N8_neg, N8_pos, A8_1, A8_2, dp4_neg, dp4_pos)
        U8to4 = self.lift8to4(
            F.interpolate(H8_new, scale_factor=2, mode="bilinear", align_corners=False)
        )
        H4_new = self._fuse4(F4_c, U8to4, N4_neg, N4_pos, A4_1, A4_2, dp4_neg, dp4_pos)
        pred = self._decode(H4_new, X2_c, B_cur)

        new_state = {
            "h4": H4_new,
            "h4_prev": H4_1,
            "h8": H8_new,
            "h8_prev": H8_1,
            "dp_prev": dp4_neg.detach(),
        }
        return pred, new_state

    # ── Streaming step API ─────────────────────────────────────
    # step() encodes only B_next (the newly-arrived frame).
    # feat_prev and feat_cur come from state.

    def step(
        self,
        B_next: torch.Tensor,  # (B, 3, H, W) newly arrived frame
        state: dict,
        dp: torch.Tensor,  # (B, 2) dp_{t-1→t}
        dp_next: torch.Tensor,  # (B, 2) dp_{t+1→t}  (= -dp_seq[t+1])
        **_,
    ) -> tuple[torch.Tensor, dict]:

        dp4_neg = pad_dp(dp, P)  # dp_{t-1→t}
        dp4_pos = pad_dp(dp_next, P)  # dp_{t+1→t}
        dp4_cum = state["dp_prev"] + dp4_neg

        feat_next = tuple(f.detach() for f in self.encoder(B_next))
        feat_prev = state["feat_prev"]
        feat_cur = state["feat_cur"]
        B_cur = state["B_cur"]

        pred, new_state = self._step_from_feats(
            feat_prev,
            feat_cur,
            feat_next,
            B_cur,
            state,
            dp4_neg,
            dp4_pos,
            dp4_cum,
        )
        new_state["feat_prev"] = feat_cur
        new_state["feat_cur"] = feat_next
        new_state["B_cur"] = B_next
        return pred, new_state

    # ── Training forward ───────────────────────────────────────

    def forward(
        self,
        blur_seq: torch.Tensor,  # (B, L, 3, H, W)
        dp_seq: torch.Tensor,  # (B, L, 2) dp_{t-1→t}
    ) -> torch.Tensor:
        B, L, _, H, W = blur_seq.shape

        # Encode all frames in one batched call
        flat = blur_seq.view(B * L, 3, H, W)
        X2_all, F4_all, F8_all = self.encoder(flat)

        def split(f, C, h, w):
            return list(f.view(B, L, C, h, w).unbind(dim=1))

        X2s = split(X2_all, 32, H // 2, W // 2)
        F4s = split(F4_all, 64, H // 4, W // 4)
        F8s = split(F8_all, 96, H // 8, W // 8)

        state = self.init_state(B, H, W, blur_seq.device)
        preds = []
        for t in range(L):
            tp = max(0, t - 1)
            tn = min(L - 1, t + 1)
            dp4_neg = pad_dp(dp_seq[:, t], P)
            # dp_{t+1→t} = -dp_{t→t+1} = -dp_seq[:, t+1]
            dp4_pos = pad_dp(
                -dp_seq[:, tn] if tn != t else torch.zeros_like(dp_seq[:, t]), P
            )
            dp4_cum = state["dp_prev"] + dp4_neg

            feat_prev = (X2s[tp], F4s[tp], F8s[tp])
            feat_cur = (X2s[t], F4s[t], F8s[t])
            feat_next = (X2s[tn], F4s[tn], F8s[tn])

            pred, state = self._step_from_feats(
                feat_prev,
                feat_cur,
                feat_next,
                blur_seq[:, t],
                state,
                dp4_neg,
                dp4_pos,
                dp4_cum,
            )
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def init_state(self, B, H, W, device) -> dict:
        memory_format = (
            torch.channels_last
            if getattr(self, "_channels_last_inference", False)
            else torch.contiguous_format
        )
        z4 = torch.zeros(B, 64, H // 4, W // 4, device=device).contiguous(
            memory_format=memory_format
        )
        z8 = torch.zeros(B, 96, H // 8, W // 8, device=device).contiguous(
            memory_format=memory_format
        )
        dummy_frame = torch.zeros(B, 3, H, W, device=device).contiguous(
            memory_format=memory_format
        )
        with torch.no_grad():
            feat0 = tuple(f.detach() for f in self.encoder(dummy_frame))
        return {
            "h4": z4,
            "h4_prev": z4.clone(),
            "h8": z8,
            "h8_prev": z8.clone(),
            "dp_prev": torch.zeros(B, P, device=device),
            "feat_prev": feat0,
            "feat_cur": feat0,
            "B_cur": dummy_frame,
        }
