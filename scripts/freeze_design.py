"""Freeze the logical hardware design (pairs per pattern, optimized parameters) once, on 1 thread.

Usage: python scripts/freeze_design.py
Writes results/hardware/design_frozen.json from the saved device snapshots. Every later build
(pilot, all repeats, both days) reuses this file instead of re-optimizing.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--maxiter", type=int, default=200)
parser.add_argument("--opt-bond-lengths", type=float, nargs="+", help="option B: optimized arm only at these")
args = parser.parse_args()

from qchem_sqd.repro import pin_threads  # noqa: E402

pin_threads(1)  # before numpy/pyscf/ffsim load: the optimized angles depend on the thread count

import json  # noqa: E402

from qchem_sqd.backends import offline_backend  # noqa: E402
from qchem_sqd.hardware import DESIGN_FILE, freeze_design  # noqa: E402

if __name__ == "__main__":
    if DESIGN_FILE.exists():
        raise SystemExit(f"{DESIGN_FILE} already exists; it is frozen. Delete it deliberately to refreeze.")
    design = freeze_design(offline_backend("ibm_phoenix"), offline_backend("ibm_kingston"), maxiter=args.maxiter,
                           opt_bond_lengths=args.opt_bond_lengths)
    for r, entry in design["bond_lengths"].items():
        print(json.dumps({"R": r, "ccsd_route": entry["ccsd_route"], "pairs_square": len(entry["pairs_square"]),
                          "pairs_heavy_hex": entry["pairs_heavy_hex"], "e_opt_minus_rhf_mha":
                          1000 * (entry["e_opt_ansatz"] - entry["e_rhf"])}))
    print(f"wrote {DESIGN_FILE}")
