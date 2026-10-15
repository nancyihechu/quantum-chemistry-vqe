"""Tabulate SQD error vs FCI (mHa, mean over seeds) per max_dim for a run_sqd_dimsweep.py output.

Usage: python scripts/summarize_dimsweep.py results/sqd_dimsweep/n2_631g_16o.json
"""

import json
import sys
from collections import defaultdict

import numpy as np


def main(path: str) -> None:
    for rec in json.loads(open(path).read()):
        fci = rec["e_fci"]
        print(f"\nR={rec['bond_length']:.1f} Å   (RHF {1000 * (rec['e_rhf'] - fci):+.1f}, "
              f"CCSD {1000 * (rec['e_ccsd'] - fci):+.1f} mHa vs FCI)")
        sources = rec["sources"]
        max_dims = sorted({run["max_dim"] for run in sources["random"]["runs"]})
        print(f"  {'source':21s} {'E_ansatz':>9s} {'2q':>5s} " + " ".join(f"{f'd={d}':>15s}" for d in max_dims))
        for name, src in sources.items():
            by_dim = defaultdict(list)
            for run in src["runs"]:
                by_dim[run["max_dim"]].append(1000 * (run["e_sqd"] - fci))
            e_ansatz = f"{1000 * (src['e_ansatz'] - fci):+9.1f}" if "e_ansatz" in src else f"{'-':>9s}"
            two_q = f"{src['isa_2q_count']:5d}" if "isa_2q_count" in src else f"{'-':>5s}"
            cells = " ".join(f"{np.mean(by_dim[d]):+7.2f}±{np.std(by_dim[d]):5.2f}  " for d in max_dims)
            print(f"  {name:21s} {e_ansatz} {two_q} {cells}")


if __name__ == "__main__":
    main(sys.argv[1])
