"""52-qubit N2/cc-pVDZ (10e, 26o) arms on the published FCIDUMPs, built on device snapshots.

Arms at grid 1.0, 1.5, 2.2 (1.098, 1.646, 2.415 A), 1-layer LUCJ from CCSD:
  phoenix_square      every alpha-beta pair ffsim's square layout accepts on ibm_phoenix
  phoenix_hhpattern   heavy-hex pattern (every 4th pair) on exactly the same phoenix qubits
  kingston_heavyhex   heavy-hex pattern on ibm_kingston, with auxiliary qubits as ffsim places them

The shared phoenix qubits come from a layout pass on a throwaway random operator that is
non-zero on every accepted pair, so the noise-aware VF2PostLayout cannot drop a pair one
arm needs; both phoenix arms are then transpiled onto those qubits with routing disabled.

Usage: python scripts/build_ccpvdz_arms.py
Writes results/ccpvdz/arms.json and results/ccpvdz/circuits_<arm>_R<grid>.qpy
"""

import json
from pathlib import Path

import ffsim
from qiskit import qpy

from qchem_sqd import fcidump_system as fs
from qchem_sqd.backends import offline_backend
from qchem_sqd.hardware import effective_pairs, logical_circuit, physical_qubits, transpile_on_qubits
from qchem_sqd.lucj import LUCJCircuit, ansatz_energy
from qchem_sqd.noise_estimates import estimate_valid_fraction
from qchem_sqd.repro import SEEDS

OUT = Path(__file__).resolve().parents[1] / "results" / "ccpvdz"


def ffsim_layout(data, ucj_op, backend, connectivity, pairs_ab):
    pm, accepted = ffsim.qiskit.generate_lucj_pass_manager(
        backend, data.norb, connectivity, interaction_pairs=([(p, p + 1) for p in range(data.norb - 1)], pairs_ab),
        optimization_level=3, seed_transpiler=SEEDS["transpiler"])
    circuit = logical_circuit(data, ucj_op)
    isa = pm.run(circuit)
    return LUCJCircuit(logical=circuit, isa=isa, ucj_op=ucj_op, pairs_ab=[tuple(p) for p in accepted],
                       duration_s=float(isa.estimate_duration(backend.target, unit="s"))), accepted


def describe(lucj: LUCJCircuit, backend, n_data_qubits: int) -> dict:
    m = lucj.metrics()
    est = estimate_valid_fraction(lucj.isa, backend)
    est_aer = estimate_valid_fraction(lucj.isa, backend, aer_equivalent=True)
    return {**m, "auxiliary_qubits": m["isa_qubits_used"] - n_data_qubits, "isa_depth": lucj.isa.depth(),
            "pairs_ab": [p[0] for p in lucj.pairs_ab], "effective_pairs_ab": effective_pairs(lucj.ucj_op),
            "valid_fraction_model": est["valid_fraction"], "valid_fraction_model_no_idle": est_aer["valid_fraction"]}


def main() -> None:
    phoenix, kingston = offline_backend("ibm_phoenix"), offline_backend("ibm_kingston")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"snapshots": {b.name: str(b.snapshot_folder) for b in (phoenix, kingston)}, "grid": {}}
    for g in fs.STUDY_GRID:
        data, mf, record = fs.continued_ccsd(g)
        norb, aa = data.norb, [(p, p + 1) for p in range(data.norb - 1)]
        # 1. which square pairs phoenix can host
        anchor_probe = ffsim.random.random_ucj_op_spin_balanced(norb, interaction_pairs=(aa, None), seed=1)
        _, square_pairs = ffsim_layout(data, anchor_probe, phoenix, "square", None)
        square_pairs = [tuple(p) for p in square_pairs]
        candidates = [p for p in square_pairs if p[0] % 4 == 0]
        # 2. heavy-hex pattern = every 4th of those that kingston also hosts (with auxiliary qubits)
        hh_probe = ffsim.random.random_ucj_op_spin_balanced(norb, interaction_pairs=(aa, candidates), seed=2)
        _, hh_pairs = ffsim_layout(data, hh_probe, kingston, "heavy-hex", candidates)
        hh_pairs = [tuple(p) for p in hh_pairs]
        # 3. shared phoenix qubits from a random operator non-zero on every square pair
        anchor = ffsim.random.random_ucj_op_spin_balanced(norb, interaction_pairs=(aa, square_pairs), seed=3,
                                                          with_final_orbital_rotation=True)
        anchor_lucj, _ = ffsim_layout(data, anchor, phoenix, "square", square_pairs)
        qubits = physical_qubits(anchor_lucj.isa)
        plain = lambda pairs: ffsim.UCJOpSpinBalanced.from_t_amplitudes(data.ccsd_t2, t1=data.ccsd_t1, n_reps=1,
                                                                        interaction_pairs=(aa, pairs))
        arms = {
            "phoenix_square": transpile_on_qubits(data, plain(square_pairs), phoenix, square_pairs, qubits),
            "phoenix_hhpattern": transpile_on_qubits(data, plain(hh_pairs), phoenix, hh_pairs, qubits),
            "kingston_heavyhex": ffsim_layout(data, plain(hh_pairs), kingston, "heavy-hex", hh_pairs)[0],
        }
        entry = {"angstrom": fs.angstrom(g), **record, "e_hci": fs.reference_energies()[g]["hci"],
                 "phoenix_square_pairs_fit": len(square_pairs), "arms": {}}
        for arm, lucj in arms.items():
            backend = kingston if arm.startswith("kingston") else phoenix
            entry["arms"][arm] = describe(lucj, backend, 2 * norb)
            entry["arms"][arm]["e_ansatz"] = ansatz_energy(data, lucj.ucj_op) if norb <= 16 else None
            with open(OUT / f"circuits_{arm}_R{g:.1f}.qpy", "wb") as fh:
                qpy.dump([lucj.isa, lucj.logical], fh)
        results["grid"][str(g)] = entry
        print(json.dumps({"grid": g, "pairs_fit": len(square_pairs), "hh_pairs": [p[0] for p in hh_pairs],
                          **{a: {k: v for k, v in d.items() if k in ("isa_2q_count", "isa_2q_depth", "isa_depth",
                                 "isa_duration_us", "auxiliary_qubits", "effective_pairs_ab", "valid_fraction_model")}
                             for a, d in entry["arms"].items()}}), flush=True)
    (OUT / "arms.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
