"""Does a second LUCJ layer (n_reps=2) use more alpha-beta pairs, and does it stay physical?

For each plain (CCSD-truncated) operator with the frozen square and heavy-hex pair lists:
effective pairs per layer and overall (|J_ab[p,p]| > 1e-8), two-qubit gate count on the device
layout, and, for the 32-qubit system, the exact ansatz energy against RHF and FCI. The 52-qubit
state (C(26,5)^2 = 4.3e9 amplitudes) is too large to evaluate exactly, so only pairs and gate
counts are reported there.

Usage: python scripts/check_nreps2.py
Writes results/checks/nreps2.json
"""

import json
from pathlib import Path

import ffsim
import numpy as np

from qchem_sqd import fcidump_system as fs
from qchem_sqd.backends import offline_backend
from qchem_sqd.ccpvdz_arms import DESIGN_FILE as DESIGN52
from qchem_sqd.hardware import BOND_LENGTHS, DESIGN_FILE, SQD_SYSTEM, transpile_lucj_layout
from qchem_sqd.lucj import ansatz_energy
from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies

ROOT = Path(__file__).resolve().parents[1]


def per_layer_pairs(op, tol=1e-8) -> list[list[int]]:
    mats = np.abs(np.asarray(op.diag_coulomb_mats)[:, 1])
    return [[int(p) for p in np.flatnonzero(np.diag(m) > tol)] for m in mats]


def analyse(data, pairs_by_pattern, backends, e_rhf=None, e_fci=None, energies=True) -> dict:
    aa = [(p, p + 1) for p in range(data.norb - 1)]
    out = {}
    for pattern, (pairs, backend, connectivity) in pairs_by_pattern.items():
        for n_reps in (1, 2):
            op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(data.ccsd_t2, t1=data.ccsd_t1, n_reps=n_reps,
                                                          interaction_pairs=(aa, pairs))
            layers = per_layer_pairs(op)
            lucj = transpile_lucj_layout(data, op, backend, connectivity, pairs)
            row = {"pairs_allowed": len(pairs), "effective_pairs_by_layer": layers,
                   "effective_pairs_union": sorted(set().union(*map(set, layers))),
                   "isa_2q_count": lucj.metrics()["isa_2q_count"], "isa_2q_depth": lucj.metrics()["isa_2q_depth"]}
            if energies:
                e = ansatz_energy(data, op)
                row.update({"minus_rhf_mha": 1000 * (e - e_rhf), "minus_fci_mha": 1000 * (e - e_fci), "below_rhf": e < e_rhf})
            out[f"{pattern}_nreps{n_reps}"] = row
            print(json.dumps({"pattern": pattern, "n_reps": n_reps, **{k: v for k, v in row.items()
                                                                        if k != "effective_pairs_by_layer"}}), flush=True)
    return out


def main() -> None:
    phoenix, kingston = offline_backend("ibm_phoenix"), offline_backend("ibm_kingston")
    design, design52 = json.loads(DESIGN_FILE.read_text()), json.loads(DESIGN52.read_text())
    fci = {r["bond_length"]: r["e_fci"] for r in json.loads((ROOT / "results/references/n2_631g_16o_sym.json").read_text())}
    results = {"n2_631g_16o": {}, "n2_ccpvdz_26o": {}}
    for r in BOND_LENGTHS:
        record, data = reference_energies(SYSTEMS[SQD_SYSTEM], r, run_fci=False)
        entry = design["bond_lengths"][str(r)]
        print(f"# 32 qubits, R = {r} A", flush=True)
        results["n2_631g_16o"][str(r)] = analyse(
            data, {"square": ([tuple(p) for p in entry["pairs_square"]], phoenix, "square"),
                   "heavy_hex": ([tuple(p) for p in entry["pairs_heavy_hex"]], kingston, "heavy-hex")},
            None, record["e_rhf"], fci[r])
    for g in fs.STUDY_GRID:
        data, _, _ = fs.continued_ccsd(g)
        entry = design52["grid"][str(g)]
        print(f"# 52 qubits, grid {g}", flush=True)
        results["n2_ccpvdz_26o"][str(g)] = analyse(
            data, {"square": ([tuple(p) for p in entry["pairs_square"]], phoenix, "square"),
                   "heavy_hex": ([tuple(p) for p in entry["pairs_heavy_hex"]], kingston, "heavy-hex")},
            None, energies=False)
    (ROOT / "results" / "checks" / "nreps2.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
