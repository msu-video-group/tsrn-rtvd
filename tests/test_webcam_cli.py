from __future__ import annotations

from tsrn_rtvd.demo.cli import build_parser


def test_demo_parser_defaults_are_publishable() -> None:
    args = build_parser().parse_args([])

    assert args.config is None
    assert args.checkpoint is None
    assert args.traj_config is None
    assert args.traj_checkpoint is None
    assert args.traj_stats is None
    assert args.fast_shift == "exact"
    assert args.hf_repo_id == "egorchistov/deblurring-tsrn-rtvd"
