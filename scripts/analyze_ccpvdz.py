"""SQD on N2/cc-pVDZ (10e, 26o) samples at fixed subspace size, against the published HCI energy.

Sources:
  torino     the published ibm_torino shots, processed with exactly our pipeline and settings
             (the only comparison point for hardware; published SQD energies are never used)
  random_hfinit    uniform random 52-bit strings with the same shot count; at 52 qubits almost none
                   are valid (expected fraction 9.6e-7), so SQD starts from RHF orbital occupations
  random_particle  uniform random strings with the correct alpha and beta electron counts
                   (every shot valid; tests sample diversity without device information)
  <file.npz> our hardware shots (as saved by run_hardware.py fetch)

Reported per run: error vs the published HCI energy after iteration 1 (postselection only)
and after each recovery round, subspace dimension per round, and valid fraction.

Usage: python scripts/analyze_ccpvdz.py --grid 1.0 --source torino --max-dim 2000
Writes results/ccpvdz/sqd_<source>_R<grid>_d<max_dim>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform, generate_counts_bipartite_hamming

from qchem_sqd import fcidump_system as fs
from qchem_sqd.repro import SEEDS
from qchem_sqd.sqd import random_valid_fraction, run_sqd, valid_fraction

OUT = Path(__file__).resolve().parents[1] / "results" / "ccpvdz"
SQD_SETTINGS = {"samples_per_batch": 1500, "num_batches": 3, "max_iterations": 5}


def load_source(source: str, grid: float, shots: int | None, seed: int) -> tuple[BitArray, str]:
    if source == "torino":
        return fs.torino_bit_array(grid), "torino"
    n = shots or fs.torino_bit_array(grid).num_shots
    if source == "random_hfinit":
        return generate_bit_array_uniform(n, 52, rand_seed=seed), f"random_hfinit_seed{seed}"
    if source == "random_particle":
        counts = generate_counts_bipartite_hamming(n, 52, hamming_right=5, hamming_left=5, rand_seed=seed)
        return BitArray.from_counts(counts), f"random_particle_seed{seed}"
    z = np.load(source)
    return BitArray(z["array"], int(z["num_bits"])), Path(source).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=float, required=True)
    parser.add_argument("--source", required=True, help="torino, random_hfinit, random_particle, or a .npz of hardware shots")
    parser.add_argument("--max-dim", type=int, required=True, help="strings per spin; 1200 -> 1.44M, 2000 -> 4M")
    parser.add_argument("--shots", type=int, help="random only; default = torino shot count")
    parser.add_argument("--seed", type=int, default=SEEDS["random_bitstrings"])
    parser.add_argument("--subsample", type=int, help="keep a fixed random subset of this many shots (matched-shot comparison)")
    args = parser.parse_args()

    data = fs.load(args.grid)
    bits, label = load_source(args.source, args.grid, args.shots, args.seed)
    if args.subsample and args.subsample < bits.num_shots:
        keep = np.sort(np.random.default_rng(SEEDS["sampler"]).choice(bits.num_shots, args.subsample, replace=False))
        bits, label = BitArray(bits.array[keep], bits.num_bits), f"{label}_sub{args.subsample}"
    hci = fs.reference_energies()[args.grid]["hci"]
    t0 = time.perf_counter()
    rhf = np.array([1.0] * data.nelec[0] + [0.0] * (data.norb - data.nelec[0]))
    initial = (rhf, rhf) if args.source == "random_hfinit" else None
    run = run_sqd(data, bits, max_dim=args.max_dim, seed=SEEDS["sqd"], initial_occupancies=initial, **SQD_SETTINGS)
    record = {
        "grid": args.grid, "angstrom": fs.angstrom(args.grid), "source": label, "max_dim": args.max_dim,
        "d": args.max_dim ** 2, "settings": SQD_SETTINGS, "shots": bits.num_shots,
        "initial_occupancies": "RHF" if initial is not None else None,
        "valid_fraction": valid_fraction(bits, data.norb, data.nelec),
        "random_valid_fraction": random_valid_fraction(data.norb, data.nelec),
        "e_hci": hci, "error_vs_hci_mha": 1000 * (run["e_sqd"] - hci),
        "error_vs_hci_by_round_mha": [1000 * (e - hci) for e in run["energy_history"]],
        "dim_by_round": run["dim_history"],
        "seconds": time.perf_counter() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sqd_{label}_R{args.grid:.1f}_d{args.max_dim}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "settings"}, indent=1))


if __name__ == "__main__":
    main()
