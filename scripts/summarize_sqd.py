"""Print SQD errors vs FCI (mean +/- std over seeds, in mHa) for a run_sqd_sim.py output file.

Usage: python scripts/summarize_sqd.py results/sqd_sim/n2_631g_8o_noisy.json
"""

import json
import sys

import numpy as np


def error_stats(runs: list[dict], e_fci: float) -> str:
    errors = 1000 * (np.array([run["e_sqd"] for run in runs]) - e_fci)
    dims = np.mean([run["subspace_dim"] for run in runs])
    return f"{errors.mean():+8.3f} ± {errors.std(ddof=1) if len(errors) > 1 else 0:6.3f} (dim {dims:8.0f})"


def main(path: str) -> None:
    records = json.loads(open(path).read())
    for rec in records:
        fci = rec["e_fci"]
        ccsd = f"{1000 * (rec['e_ccsd'] - fci):+8.2f}" if rec.get("e_ccsd") is not None else "    n/a"
        print(f"R={rec['bond_length']:.1f} Å   CCSD {ccsd}   random {error_stats(rec['random'], fci)}")
        for name, layout in rec["layouts"].items():
            for kind in ("ideal", "noisy"):
                if kind in layout:
                    print(f"    {name:11s} {kind:5s} ab={layout['n_pairs_ab']:2d} 2q={layout['isa_2q_count']:4d}   "
                          f"{error_stats(layout[kind], fci)}")


if __name__ == "__main__":
    main(sys.argv[1])
