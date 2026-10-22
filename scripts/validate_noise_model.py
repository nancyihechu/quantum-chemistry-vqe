"""Check the first-order valid-fraction model against exact Aer noisy sampling at 16 qubits.

Same snapshot noise model on both sides (Aer's from_backend has no idle noise, so the
model is run with aer_equivalent=True for the comparison; the idle-T1 version is printed too).

Usage: python scripts/validate_noise_model.py --shots 5000
Writes results/checks/noise_model_validation.json
"""

import argparse
import json
from pathlib import Path

from qiskit.primitives import BitArray
from qiskit_aer import AerSimulator

from qchem_sqd.backends import offline_backend
from qchem_sqd.lucj import build_lucj
from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.noise_estimates import estimate_valid_fraction
from qchem_sqd.reference import reference_energies
from qchem_sqd.repro import SEEDS
from qchem_sqd.sqd import valid_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=5000)
    args = parser.parse_args()
    rows = []
    for name, connectivity in (("ibm_phoenix", "square"), ("ibm_kingston", "heavy-hex"), ("ibm_fez", "heavy-hex")):
        backend = offline_backend(name)
        simulator = AerSimulator.from_backend(backend, seed_simulator=SEEDS["aer"])
        for r in (1.0, 2.4):
            _, data = reference_energies(SYSTEMS["n2_631g_8o"], r)
            lucj = build_lucj(data, backend, connectivity)
            counts = simulator.run(lucj.isa, shots=args.shots).result().get_counts()
            aer = valid_fraction(BitArray.from_counts(counts), data.norb, data.nelec)
            model = estimate_valid_fraction(lucj.isa, backend, aer_equivalent=True)
            with_idle = estimate_valid_fraction(lucj.isa, backend)
            row = {"backend": name, "bond_length": r, "aer_valid": aer, "aer_std": (aer * (1 - aer) / args.shots) ** 0.5,
                   "model_valid": model["valid_fraction"], "ratio": model["valid_fraction"] / aer,
                   "model_valid_with_idle": with_idle["valid_fraction"], **{k: model[k] for k in ("2q", "1q", "measure")}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / "checks" / "noise_model_validation.json"
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
