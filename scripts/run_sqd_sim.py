"""SQD bond-length scan on simulators, to validate the pipeline before using QPU time.

For every geometry: build the LUCJ circuit for ibm_phoenix (square) and ibm_kingston
(heavy-hex) from saved calibration snapshots, record circuit metrics, then for each seed run SQD on
  - ideal samples of the LUCJ circuit (ffsim),
  - noisy samples from the ibm_phoenix and ibm_kingston noise models (Aer, --noisy),
  - uniformly random bitstrings with the same shot count (control).
If the random control matches the circuit samples, the active space is too
small for SQD to say anything about sample quality.

Usage: python scripts/run_sqd_sim.py n2_631g_8o --shots 20000 --seeds 5 [--noisy]
Reuses FCI energies from results/references/<system>_sym.json when present.
Writes results/sqd_sim/<system>[_noisy].json
"""

import argparse
import json
import time
from pathlib import Path

import ffsim
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from qchem_sqd.backends import offline_backend
from qchem_sqd.lucj import build_lucj
from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies
from qchem_sqd.sqd import run_sqd, valid_fraction

RESULTS = Path(__file__).resolve().parents[1] / "results"


def sqd_record(data, bits, seed: int) -> dict:
    out = run_sqd(data, bits, seed=seed)
    out["valid_fraction"] = valid_fraction(bits, data.norb, data.nelec)
    out["seed"] = seed
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noisy", action="store_true", help="also sample with the device noise models")
    args = parser.parse_args()

    system = SYSTEMS[args.system]
    reference_file = RESULTS / "references" / f"{args.system}_sym.json"
    saved_fci = {}
    if reference_file.exists():
        saved_fci = {rec["bond_length"]: rec["e_fci"] for rec in json.loads(reference_file.read_text())}

    devices = ((offline_backend("ibm_phoenix"), "square"), (offline_backend("ibm_kingston"), "heavy-hex"))

    records = []
    for r in system.bond_lengths:
        record, data = reference_energies(system, r, run_fci=r not in saved_fci)
        if r in saved_fci:
            record["e_fci"] = saved_fci[r]

        record["random"] = [
            sqd_record(data, generate_bit_array_uniform(args.shots, 2 * data.norb, rand_seed=seed), seed)
            for seed in range(args.seeds)
        ]

        record["layouts"] = {}
        for backend, connectivity in devices:
            t0 = time.perf_counter()
            lucj = build_lucj(data, backend, connectivity)
            layout = record["layouts"][backend.name] = lucj.metrics()
            layout["build_seconds"] = time.perf_counter() - t0

            layout["ideal"] = []
            for seed in range(args.seeds):
                sampler = ffsim.qiskit.FfsimSampler(default_shots=args.shots, seed=seed)
                bits = sampler.run([lucj.logical]).result()[0].data.meas
                layout["ideal"].append(sqd_record(data, bits, seed))

            if args.noisy:
                noisy_sim = AerSimulator.from_backend(backend)
                layout["noisy"] = []
                for seed in range(args.seeds):
                    t0 = time.perf_counter()
                    sampler = AerSampler.from_backend(noisy_sim, seed=seed)
                    bits = sampler.run([lucj.isa], shots=args.shots).result()[0].data.meas
                    layout["noisy"].append(sqd_record(data, bits, seed))
                    layout["noisy"][-1]["sim_seconds"] = time.perf_counter() - t0

        records.append(record)
        print(json.dumps(record), flush=True)

    out_dir = RESULTS / "sqd_sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_noisy" if args.noisy else ""
    (out_dir / f"{args.system}{suffix}.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
