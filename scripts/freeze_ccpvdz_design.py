"""Freeze the 52-qubit N2/cc-pVDZ pair lists (square on phoenix, heavy-hex on kingston) once.

Usage: python scripts/freeze_ccpvdz_design.py
Writes results/hardware/design_ccpvdz_frozen.json from the saved device snapshots.
"""

import json

from qchem_sqd.backends import offline_backend
from qchem_sqd.ccpvdz_arms import DESIGN_FILE, freeze_design

if __name__ == "__main__":
    if DESIGN_FILE.exists():
        raise SystemExit(f"{DESIGN_FILE} already exists; it is frozen. Delete it deliberately to refreeze.")
    design = freeze_design(offline_backend("ibm_phoenix"), offline_backend("ibm_kingston"))
    for g, entry in design["grid"].items():
        print(json.dumps({"grid": g, "square_pairs": len(entry["pairs_square"]),
                          "heavy_hex_pairs": [p[0] for p in entry["pairs_heavy_hex"]]}))
    print(f"wrote {DESIGN_FILE}")
