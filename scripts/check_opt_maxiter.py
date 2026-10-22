"""Option C for the optimized 32-qubit arm: is the regularized optimize=True state below RHF
at every study bond length when L-BFGS-B gets more iterations?

Same settings as the frozen design (regularization 1e-2, phoenix square pairs from
results/hardware/design_frozen.json, 1 thread), with a larger maxiter.

Usage: python scripts/check_opt_maxiter.py --maxiter 1000
Writes results/checks/opt_maxiter<maxiter>.json
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--maxiter", type=int, required=True)
args = parser.parse_args()

from qchem_sqd.repro import pin_threads  # noqa: E402

pin_threads(1)

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import ffsim  # noqa: E402

from qchem_sqd.hardware import BOND_LENGTHS, DESIGN_FILE, OPT_REGULARIZATION, SQD_SYSTEM  # noqa: E402
from qchem_sqd.lucj import ansatz_energy  # noqa: E402
from qchem_sqd.molecules import SYSTEMS  # noqa: E402
from qchem_sqd.reference import reference_energies  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    design = json.loads(DESIGN_FILE.read_text())
    fci = {r["bond_length"]: r["e_fci"] for r in json.loads((ROOT / "results/references/n2_631g_16o_sym.json").read_text())}
    rows = []
    for r in BOND_LENGTHS:
        record, data = reference_energies(SYSTEMS[SQD_SYSTEM], r, run_fci=False)
        pairs = [tuple(p) for p in design["bond_lengths"][str(r)]["pairs_square"]]
        aa = [(p, p + 1) for p in range(data.norb - 1)]
        t0 = time.perf_counter()
        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            data.ccsd_t2, t1=data.ccsd_t1, n_reps=1, interaction_pairs=(aa, pairs), optimize=True,
            method="L-BFGS-B", regularization=OPT_REGULARIZATION, options={"maxiter": args.maxiter})
        e = ansatz_energy(data, op)
        row = {"bond_length": r, "maxiter": args.maxiter, "e_opt": e, "minus_rhf_mha": 1000 * (e - record["e_rhf"]),
               "minus_fci_mha": 1000 * (e - fci[r]), "below_rhf": e < record["e_rhf"],
               "params": op.to_parameters(interaction_pairs=(aa, pairs)).tolist(),
               "with_final_orbital_rotation": op.final_orbital_rotation is not None,
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "params"}), flush=True)
    (ROOT / "results" / "checks" / f"opt_maxiter{args.maxiter}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
