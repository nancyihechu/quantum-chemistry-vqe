"""Compute RHF / CCSD / active-space FCI reference energies with symmetry-adapted orbitals.

Usage: python scripts/run_references.py n2_631g_16o [--bond-lengths 1.0 1.2 1.8 2.4]
Writes results/references/<system>_sym.json (the older <system>.json files used
unsymmetrized orbitals, which break symmetry for N2 from 1.6 A).
"""

import argparse
import json
from pathlib import Path

from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "references"
CHECK_AND_HARDWARE_GEOMETRIES = (1.0, 1.2, 1.6, 1.8, 2.0, 2.4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="+")
    parser.add_argument("--bond-lengths", type=float, nargs="+", default=list(CHECK_AND_HARDWARE_GEOMETRIES))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.systems:
        records = []
        for r in args.bond_lengths:
            record, _ = reference_energies(SYSTEMS[name], r)
            records.append(record)
            print(json.dumps(record), flush=True)
        (OUT_DIR / f"{name}_sym.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
