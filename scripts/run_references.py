"""Compute RHF / CCSD / active-space FCI reference energies over each bond-length scan.

Usage: python scripts/run_references.py n2_631g_8o [n2_631g_16o ...]
Writes results/references/<system>.json
"""

import json
import sys
from pathlib import Path

from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "references"


def main(names: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        system = SYSTEMS[name]
        records = []
        for r in system.bond_lengths:
            record, _ = reference_energies(system, r)
            records.append(record)
            print(json.dumps(record), flush=True)
        (OUT_DIR / f"{name}.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or list(SYSTEMS))
