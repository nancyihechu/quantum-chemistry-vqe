"""Version check: the original-generation SQD loop (qiskit-addon-sqd 0.3.0) on the ibm_torino shots.

qiskit-addon-sqd 0.1.0, used in arXiv:2405.05068, is not publicly available (PyPI and the
official repository start at 0.3.0, 2024-09-06), so 0.3.0 is the closest available proxy.
This follows the 0.3.0 tutorial loop: postselect and subsample, solve each batch in the
product space of its strings (no cap, no carry-over between iterations), recover
configurations from the batch-averaged occupancies, repeat. With closed-shell symmetrized
strings, samples_per_batch = 600 gives at most 1200 strings, i.e. d <= 1.44M.

Run in the separate Python 3.11 environment /cluster/pixstor/dkhf3-lab/users/nancy/envs/sqd-v030:
  python -I scripts/reproduce_torino_v030.py --grid 1.0 --samples-per-batch 600 --num-batches 10
Writes results/ccpvdz/reproduce_v030_R<grid>_spb<spb>_nb<num_batches>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import bitstring_matrix_to_sorted_addresses, flip_orbital_occupancies, solve_fermion
from qiskit_addon_sqd.subsampling import postselect_and_subsample

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/cluster/pixstor/dkhf3-lab/users/nancy/GITHUB/external")
FCIDUMP = EXTERNAL / "sqd_data_repository" / "integrals" / "N2_cc-pVDZ"
COUNTS = EXTERNAL / "torino_counts_npz"
REFERENCES = EXTERNAL / "sqd_data_repository" / "classical_reference_energies" / "N2_cc-pVDZ" / "energies_n2_cc-pvdz.txt"
PUBLISHED = EXTERNAL / "sqd_data_repository" / "experiments" / "N2_cc-pVDZ" / "sqd_hardware_energetics" / "sqd_energies_vs_bond_length.txt"
UNIT = 1.0975135


def reference(grid: float, path: Path, column: int, scale: float = 1.0) -> float:
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if parts and abs(float(parts[0]) / scale - grid) < 1e-6:
            return float(parts[column])
    raise KeyError(grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=float, required=True)
    parser.add_argument("--samples-per-batch", type=int, default=600)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()

    ints = fcidump.read(str(FCIDUMP / f"R_26_{args.grid:.2f}_fcidump.txt"))
    norb, nelec = ints["NORB"], ints["NELEC"]
    n_alpha = n_beta = nelec // 2
    hcore, eri, ecore = ints["H1"], ao2mo.restore(1, ints["H2"], norb), ints["ECORE"]

    z = np.load(COUNTS / f"torino_R{args.grid:.1f}.npz")
    keys = ["".join(map(str, row)) for row in z["bitstrings"]]
    bitstring_matrix_full, probs_full = counts_to_arrays(dict(zip(keys, z["weights"])))

    e_hci = reference(args.grid, REFERENCES, 5)
    e_published = reference(args.grid, PUBLISHED, 1, scale=UNIT)  # d = 1.44M column
    rng = np.random.default_rng(args.seed)
    occupancies, history, dims = None, [], []
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        if occupancies is None:
            bs_mat, probs = bitstring_matrix_full, probs_full
        else:
            bs_mat, probs = recover_configurations(bitstring_matrix_full, probs_full, occupancies, n_alpha, n_beta,
                                                   rand_seed=int(rng.integers(2**31)))
        batches = postselect_and_subsample(bs_mat, probs, n_alpha, n_beta, args.samples_per_batch, args.num_batches,
                                           rand_seed=int(rng.integers(2**31)))
        energies, occs, batch_dims = [], [], []
        for batch in batches:
            addresses = bitstring_matrix_to_sorted_addresses(batch, open_shell=False)
            energy, _, avg_occs, _ = solve_fermion(addresses, hcore, eri, spin_sq=0, max_davidson=200)
            energies.append(energy + ecore)
            occs.append(np.concatenate(avg_occs))
            batch_dims.append(len(addresses[0]) * len(addresses[1]))
        best = int(np.argmin(energies))
        history.append(energies[best])
        dims.append(batch_dims[best])
        occupancies = flip_orbital_occupancies(np.mean(occs, axis=0))
        print(json.dumps({"iteration": len(history), "error_vs_hci_mha": 1000 * (energies[best] - e_hci),
                          "dim": batch_dims[best]}), flush=True)
    record = {
        "qiskit_addon_sqd": "0.3.0", "grid": args.grid, "settings": vars(args), "e_hci": e_hci,
        "error_vs_hci_mha": 1000 * (min(history) - e_hci),
        "error_vs_hci_by_iteration_mha": [1000 * (e - e_hci) for e in history], "dim_by_iteration": dims,
        "published_error_vs_hci_mha": 1000 * (e_published - e_hci), "seconds": time.perf_counter() - t0,
    }
    out = ROOT / "results" / "ccpvdz" / f"reproduce_v030_R{args.grid:.1f}_spb{args.samples_per_batch}_nb{args.num_batches}.json"
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
