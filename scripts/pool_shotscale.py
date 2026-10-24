"""Pool the shot-scaling jobs (Addendum B2) into one .npz per grid value.

Usage: python scripts/pool_shotscale.py results/hardware/shotscale_<time>
Writes <run dir>/pooled_phoenix52_square_G<grid>.npz (array, num_bits), concatenating the
shots of every part in job order.
"""

import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
for grid in ("1.0", "2.2"):
    parts = sorted(run.glob(f"shotscale_phoenix52_part*__phoenix52_square_G{grid}.npz"))
    arrays = [np.load(p) for p in parts]
    num_bits = {int(z["num_bits"]) for z in arrays}
    assert len(num_bits) == 1, num_bits
    pooled = np.concatenate([z["array"] for z in arrays])
    np.savez_compressed(run / f"pooled_phoenix52_square_G{grid}.npz", array=pooled, num_bits=num_bits.pop())
    print(f"grid {grid}: {len(parts)} parts, {len(pooled):,} shots")
