"""SQD error vs subspace dimension: circuit samples against random bitstrings.

Uncapped SQD rewards sample *diversity*, which uniform random bitstrings
maximise, so it cannot tell whether circuit samples carry information. Capping
each spin sector at the same max_dim for every sample source asks the fair
question: per configuration kept, which source finds the better subspace?

Sample sources (all noiseless, same shot count):
  random               uniform bitstrings (control)
  square_r1            plain LUCJ, ibm_phoenix layout, 1 layer
  heavyhex_r1          plain LUCJ, ibm_kingston layout, 1 layer
  square_r2            plain LUCJ, ibm_phoenix layout, 2 layers
  square_r1_optimized  ffsim optimize=True LUCJ (unphysical, labelled control)

Usage: python scripts/run_sqd_dimsweep.py n2_631g_16o --shots 100000 --seeds 3
Needs results/references/<system>_sym.json (FCI). Writes results/sqd_dimsweep/<system>.json
"""

import argparse
import json
from pathlib import Path

import ffsim
from qiskit_addon_sqd.counts import generate_bit_array_uniform

from qchem_sqd.backends import offline_backend
from qchem_sqd.lucj import ansatz_energy, build_lucj
from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies
from qchem_sqd.sqd import run_sqd

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAX_DIMS = (30, 60, 100, 150, 200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-dims", type=int, nargs="+", default=list(MAX_DIMS))
    args = parser.parse_args()

    system = SYSTEMS[args.system]
    references = json.loads((RESULTS / "references" / f"{args.system}_sym.json").read_text())
    fci = {rec["bond_length"]: rec["e_fci"] for rec in references}
    square, heavy_hex = offline_backend("ibm_phoenix"), offline_backend("ibm_kingston")
    circuit_sources = {
        "square_r1": dict(backend=square, connectivity="square", n_reps=1),
        "heavyhex_r1": dict(backend=heavy_hex, connectivity="heavy-hex", n_reps=1),
        "square_r2": dict(backend=square, connectivity="square", n_reps=2),
        "square_r1_optimized": dict(backend=square, connectivity="square", n_reps=1, optimize=True),
    }

    records = []
    for r in system.bond_lengths:
        record, data = reference_energies(system, r, run_fci=False)
        record["e_fci"] = fci[r]
        record["sources"] = {}

        samplers = {"random": lambda seed: generate_bit_array_uniform(args.shots, 2 * data.norb, rand_seed=seed)}
        for name, kwargs in circuit_sources.items():
            lucj = build_lucj(data, **kwargs)
            record["sources"][name] = {**lucj.metrics(), "e_ansatz": ansatz_energy(data, lucj.ucj_op)}
            samplers[name] = lambda seed, circuit=lucj.logical: (
                ffsim.qiskit.FfsimSampler(default_shots=args.shots, seed=seed).run([circuit]).result()[0].data.meas
            )
        record["sources"]["random"] = {}

        for name, sample in samplers.items():
            runs = record["sources"][name]["runs"] = []
            for seed in range(args.seeds):
                bits = sample(seed)
                for max_dim in args.max_dims:
                    out = run_sqd(data, bits, seed=seed, max_dim=max_dim)
                    out["seed"] = seed
                    runs.append(out)

        records.append(record)
        print(json.dumps(record), flush=True)

    out_dir = RESULTS / "sqd_dimsweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.system}.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
