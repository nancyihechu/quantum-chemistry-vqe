"""LUCJ ansatz circuits seeded from CCSD amplitudes, laid out for square or heavy-hex devices."""

from dataclasses import dataclass
from typing import Literal

import ffsim
import numpy as np
from pyscf import ao2mo
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers import BackendV2

Connectivity = Literal["square", "heavy-hex"]


@dataclass
class LUCJCircuit:
    logical: QuantumCircuit  # before transpilation, usable by FfsimSampler
    isa: QuantumCircuit  # transpiled for the backend
    ucj_op: ffsim.UCJOpSpinBalanced
    pairs_ab: list[tuple[int, int]]
    duration_s: float  # scheduled length of one shot on the target device

    def metrics(self) -> dict:
        ops = self.isa.count_ops()
        two_q = sum(n for name, n in ops.items() if name in ("cz", "ecr", "cx"))
        return {
            "n_pairs_ab": len(self.pairs_ab),
            "isa_depth": self.isa.depth(),
            "isa_2q_depth": self.isa.depth(lambda inst: inst.operation.num_qubits == 2),
            "isa_2q_count": two_q,
            "isa_duration_us": 1e6 * self.duration_s,
            "isa_qubits_used": len({self.isa.find_bit(q).index for inst in self.isa.data for q in inst.qubits}),
        }


def build_lucj(
    data: ffsim.MolecularData,
    backend: BackendV2,
    connectivity: Connectivity,
    n_reps: int = 1,
    optimize: bool = False,
    seed: int = 1234,
    pairs_ab: list[tuple[int, int]] | None = None,
    measure: bool = True,
) -> LUCJCircuit:
    """Build the LUCJ circuit with the alpha-beta pairs the device layout can host.

    Same-spin pairs are nearest neighbours along each spin chain. Alpha-beta
    pairs default to every orbital on a square lattice and every fourth orbital
    on heavy-hex; ffsim drops any the layout cannot accommodate. Passing
    ``pairs_ab`` explicitly (e.g. the heavy-hex pattern on a square device)
    isolates the effect of the interaction pattern from the device itself.
    ``measure=False`` leaves the circuit unmeasured, for the Estimator.
    """
    if data.ccsd_t2 is None:
        raise ValueError("MolecularData needs CCSD amplitudes (run_ccsd with store_t2=True)")
    norb, nelec = data.norb, data.nelec
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]

    pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
        backend,
        norb,
        connectivity,
        interaction_pairs=(pairs_aa, pairs_ab),
        optimization_level=3,
        seed_transpiler=seed,
    )
    # Plain truncation of the CCSD-derived UCJ keeps the state physical (energy
    # between FCI and RHF). optimize=True fits the operator, not the energy: for
    # N2 it gives states far above RHF whose energies also vary between machines,
    # so it is kept only as a labelled control.
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        data.ccsd_t2, t1=data.ccsd_t1, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab), optimize=optimize
    )

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    if measure:
        circuit.measure_all()

    isa = pass_manager.run(circuit)
    return LUCJCircuit(
        logical=circuit,
        isa=isa,
        ucj_op=ucj_op,
        pairs_ab=pairs_ab,
        duration_s=float(isa.estimate_duration(backend.target, unit="s")),
    )


def ansatz_energy(data: ffsim.MolecularData, ucj_op: ffsim.UCJOpSpinBalanced) -> float:
    """Exact energy of the noiseless LUCJ state; a physical ansatz lies between FCI and RHF."""
    norb, nelec = data.norb, data.nelec
    hamiltonian = ffsim.linear_operator(
        ffsim.MolecularHamiltonian(
            data.one_body_integrals, ao2mo.restore(1, data.two_body_integrals, norb), data.core_energy
        ),
        norb=norb,
        nelec=nelec,
    )
    state = ffsim.apply_unitary(ffsim.hartree_fock_state(norb, nelec), ucj_op, norb=norb, nelec=nelec)
    return float(np.vdot(state, hamiltonian @ state).real)
