"""Pipeline check: our SQD on the published ibm_torino N2/cc-pVDZ shots (arXiv:2405.05068).

Uses the repository's FCIDUMP and raw shots, our qiskit-addon-sqd version (they used 0.1.0),
and compares with their published SQD energy at the same subspace size d.

Usage: python scripts/reproduce_torino.py --grid 1.0 --max-dim 1200 --samples-per-batch 1500
       (grid in units of 1.0975135 A; max_dim 1200 -> d = 1.44M)
Writes results/ccpvdz/reproduce_R<grid>_d<max_dim>_spb<samples>.json
"""

import argparse
import json
import time
from pathlib import Path

from qchem_sqd import fcidump_system as fs
from qchem_sqd.repro import SEEDS
from qchem_sqd.sqd import random_valid_fraction, run_sqd, valid_fraction

OUT = Path(__file__).resolve().parents[1] / "results" / "ccpvdz"
PUBLISHED_D = {1200: "1.44M", 2000: "4M", 3000: "9M", 4000: "16M"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=float, required=True)
    parser.add_argument("--max-dim", type=int, default=1200)
    parser.add_argument("--samples-per-batch", type=int, default=1500)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()

    data = fs.load(args.grid)
    bits = fs.torino_bit_array(args.grid)
    hci = fs.reference_energies()[args.grid]["hci"]
    published = fs.published_sqd()[args.grid]
    t0 = time.perf_counter()
    run = run_sqd(data, bits, samples_per_batch=args.samples_per_batch, num_batches=args.num_batches,
                  max_iterations=args.max_iterations, max_dim=args.max_dim, seed=SEEDS["sqd"])
    label = PUBLISHED_D.get(args.max_dim)
    record = {
        "grid": args.grid, "angstrom": fs.angstrom(args.grid), "max_dim": args.max_dim,
        "settings": {k: getattr(args, k) for k in ("samples_per_batch", "num_batches", "max_iterations")},
        "shots": bits.num_shots, "valid_fraction": valid_fraction(bits, data.norb, data.nelec),
        "random_valid_fraction": random_valid_fraction(data.norb, data.nelec),
        "e_hci": hci, "e_sqd": run["e_sqd"], "error_vs_hci_mha": 1000 * (run["e_sqd"] - hci),
        "error_vs_hci_by_round_mha": [1000 * (e - hci) for e in run["energy_history"]],
        "dim_by_round": run["dim_history"],
        "published_sqd": published.get(label) if label else None,
        "published_error_vs_hci_mha": 1000 * (published[label] - hci) if label in published else None,
        "ours_minus_published_mha": 1000 * (run["e_sqd"] - published[label]) if label in published else None,
        "seconds": time.perf_counter() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"reproduce_R{args.grid:.1f}_d{args.max_dim}_spb{args.samples_per_batch}.json"
    path.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
