"""SQD bond-length scan on simulators, to validate the pipeline before using QPU time.

For every geometry: build the LUCJ circuit for FakeMiami (square) and FakeFez
(heavy-hex), record circuit metrics, sample it ideally (ffsim) and, optionally,
with the FakeMiami noise model (Aer), then run SQD on each sample set.

Usage: python scripts/run_sqd_sim.py n2_631g_8o --shots 20000 [--noisy]
Writes results/sqd_sim/<system>[_noisy].json
"""

import argparse
import json
import time
from pathlib import Path

import ffsim
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMiami

from qchem_sqd.lucj import build_lucj
from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies
from qchem_sqd.sqd import run_sqd, valid_fraction

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "sqd_sim"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--noisy", action="store_true", help="also sample with the FakeMiami noise model")
    parser.add_argument("--no-fci", action="store_true", help="skip FCI (for active spaces too large to solve exactly)")
    args = parser.parse_args()

    system = SYSTEMS[args.system]
    miami, fez = FakeMiami(), FakeFez()
    noisy_sampler = AerSampler.from_backend(AerSimulator.from_backend(miami)) if args.noisy else None

    records = []
    for r in system.bond_lengths:
        record, data = reference_energies(system, r, run_fci=not args.no_fci)
        record["layouts"] = {}
        for backend, connectivity in ((miami, "square"), (fez, "heavy-hex")):
            lucj = build_lucj(data, backend, connectivity)
            layout = record["layouts"][backend.name] = lucj.metrics()

            bits = ffsim.qiskit.FfsimSampler(default_shots=args.shots, seed=7).run([lucj.logical]).result()[0].data.meas
            layout["ideal"] = run_sqd(data, bits)
            layout["ideal"]["valid_fraction"] = valid_fraction(bits, data.norb, data.nelec)

            if noisy_sampler is not None and backend is miami:
                t0 = time.perf_counter()
                bits = noisy_sampler.run([lucj.isa], shots=args.shots).result()[0].data.meas
                layout["noisy"] = run_sqd(data, bits)
                layout["noisy"]["valid_fraction"] = valid_fraction(bits, data.norb, data.nelec)
                layout["noisy"]["sim_seconds"] = time.perf_counter() - t0

        records.append(record)
        print(json.dumps(record), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_noisy" if args.noisy else ""
    (OUT_DIR / f"{args.system}{suffix}.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
