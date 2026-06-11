"""BT1-only compatibility shims.

Some older scripts imported pad_dp from this compatibility module; in the optimized BT1
model the implementation lives in models.new_blocks.
"""

from .new_blocks import pad_dp

__all__ = ["pad_dp"]
