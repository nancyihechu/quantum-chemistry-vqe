"""Hardware design, frozen circuit parameters, QPU estimates and job specifications.

Arms (32-qubit N2/6-31G, 1-layer LUCJ, symmetry-adapted orbitals, CCSD-seeded):
  phoenix_square      ibm_phoenix, square pattern (alpha-beta pairs phoenix hosts), plain truncation
  phoenix_hhpattern   ibm_phoenix, heavy-hex pattern, on exactly the qubits of phoenix_square
  phoenix_square_opt  ibm_phoenix, square pattern, ffsim optimize=True (regularization 1e-2,
                      200 L-BFGS-B iterations, 1 thread), on exactly the qubits of phoenix_square
  kingston_heavyhex   ibm_kingston, heavy-hex pattern (same logical circuit as phoenix_hhpattern)

The logical design (pairs per pattern and the optimized parameters) is computed once by
``freeze_design`` with threads pinned to 1 and stored in DESIGN_FILE; every later build
(pilot, every repeat, every day) rebuilds circuits from that file. Only the physical
layout is chosen at build time from the day's calibration, and recorded.

Bond lengths 1.0, 1.2, 1.8, 2.4 A; 5 repeats over at least 2 days; 1e5 shots per circuit.
Each phoenix job interleaves its three arms PUB by PUB (by bond length) so they share a
calibration window; one phoenix job and one kingston job per repeat.

QPU time estimate per job: executions from the executor Sampler's own client-side
``prepare`` (twirling randomizations x shots), each costing its circuit duration plus
the device's default repetition delay, plus JOB_OVERHEAD_S per job (an assumption).
"""

import json
from importlib.metadata import version
from dataclasses import dataclass, field
from pathlib import Path

import ffsim
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.executor_sampler import Sampler
from qiskit_ibm_runtime.executor_sampler.prepare import prepare as prepare_sampler

from .backends import check_allowed, connectivity_of
from .lucj import LUCJCircuit, ansatz_energy
from .molecules import SYSTEMS
from .reference import reference_energies
from .repro import SEEDS

JOB_OVERHEAD_S = 5.0
SQD_SYSTEM = "n2_631g_16o"
BOND_LENGTHS = (1.0, 1.2, 1.8, 2.4)
PILOT_BOND_LENGTHS = (1.0, 2.4)
REPEATS = 5
SHOTS = 100_000
OPT_REGULARIZATION = 1e-2
OPT_MAXITER = 200
DESIGN_FILE = Path(__file__).resolve().parents[2] / "results" / "hardware" / "design_frozen.json"
PHOENIX_ARMS = ("phoenix_square", "phoenix_hhpattern", "phoenix_square_opt")
SAMPLER_OPTIONS = {
    "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
    "twirling": {"enable_gates": True, "enable_measure": True},
}


@dataclass
class Pub:
    label: str
    circuit: QuantumCircuit  # ISA circuit for the job's backend
    meta: dict = field(default_factory=dict)
    logical: QuantumCircuit | None = None  # pre-transpilation circuit, for noiseless checks
    data: ffsim.MolecularData | None = None  # Hamiltonian, for SQD on returned samples


@dataclass
class JobSpec:
    name: str
    instance: str
    backend_name: str
    options: dict
    pubs: list[Pub]

    def sampler(self, mode) -> Sampler:
        return Sampler(mode=mode, options=self.options)


def _pairs_aa(norb: int) -> list[tuple[int, int]]:
    return [(p, p + 1) for p in range(norb - 1)]


def _as_pairs(pairs) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in pairs]


def logical_circuit(data, ucj_op) -> QuantumCircuit:
    register = QuantumRegister(2 * data.norb, name="q")
    circuit = QuantumCircuit(register)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(data.norb, data.nelec), register)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), register)
    circuit.measure_all()
    return circuit


def physical_qubits(isa: QuantumCircuit) -> list[int]:
    """Physical qubit of each logical qubit, in logical order."""
    return list(isa.layout.initial_index_layout(filter_ancillas=True))


def _lucj(data, ucj_op, pairs_ab, isa, backend) -> LUCJCircuit:
    return LUCJCircuit(logical=None, isa=isa, ucj_op=ucj_op, pairs_ab=pairs_ab,
                       duration_s=float(isa.estimate_duration(backend.target, unit="s")))


def transpile_lucj_layout(data, ucj_op, backend, connectivity: str, pairs_ab) -> LUCJCircuit:
    """ffsim's LUCJ layout for these pairs; fails rather than dropping any frozen pair."""
    pass_manager, accepted = ffsim.qiskit.generate_lucj_pass_manager(
        backend, data.norb, connectivity, interaction_pairs=(_pairs_aa(data.norb), list(pairs_ab)),
        optimization_level=3, seed_transpiler=SEEDS["transpiler"],
    )
    if _as_pairs(accepted) != _as_pairs(pairs_ab):
        raise RuntimeError(f"{backend.name} cannot host the frozen pairs {pairs_ab}; it accepted {accepted}")
    circuit = logical_circuit(data, ucj_op)
    lucj = _lucj(data, ucj_op, pairs_ab, pass_manager.run(circuit), backend)
    lucj.logical = circuit
    return lucj


def transpile_on_qubits(data, ucj_op, backend, pairs_ab, qubits: list[int]) -> LUCJCircuit:
    """Transpile onto fixed physical qubits (logical qubit i -> qubits[i]) with routing disabled:
    the transpiler raises if any SWAP would be needed, so no arm can hide extra routing."""
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=qubits,
                                                routing_method="none", seed_transpiler=SEEDS["transpiler"])
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    circuit = logical_circuit(data, ucj_op)
    lucj = _lucj(data, ucj_op, pairs_ab, pass_manager.run(circuit), backend)
    lucj.logical = circuit
    return lucj


# ---------- frozen logical design ----------

def freeze_design(phoenix, kingston, bond_lengths=BOND_LENGTHS, path: Path = DESIGN_FILE,
                  maxiter: int = OPT_MAXITER, opt_bond_lengths=None) -> dict:
    """Pairs per pattern and optimized parameters, computed once. Run with threads pinned to 1.

    ``opt_bond_lengths`` limits the optimized arm to those geometries (pre-registered option B);
    by default it runs at every bond length.
    """
    from .repro import thread_info

    opt_bond_lengths = list(bond_lengths if opt_bond_lengths is None else opt_bond_lengths)
    design = {"system": SQD_SYSTEM, "optimize": {"regularization": OPT_REGULARIZATION, "maxiter": maxiter,
                                                 "method": "L-BFGS-B"}, "opt_bond_lengths": opt_bond_lengths,
              "thread_info": thread_info(), "snapshots": {b.name: str(getattr(b, "snapshot_folder", "live"))
                                                         for b in (phoenix, kingston)},
              "versions": {pkg: version(pkg) for pkg in ("ffsim", "pyscf", "qiskit", "qiskit-addon-sqd")},
              "bond_lengths": {}}
    for r in bond_lengths:
        record, data = reference_energies(SYSTEMS[SQD_SYSTEM], r, run_fci=False)
        norb, aa = data.norb, _pairs_aa(data.norb)
        _, square_pairs = ffsim.qiskit.generate_lucj_pass_manager(phoenix, norb, "square", interaction_pairs=(aa, None),
                                                                  optimization_level=3, seed_transpiler=SEEDS["transpiler"])
        candidates = [p for p in _as_pairs(square_pairs) if p[0] % 4 == 0]
        _, hh_pairs = ffsim.qiskit.generate_lucj_pass_manager(kingston, norb, "heavy-hex", interaction_pairs=(aa, candidates),
                                                              optimization_level=3, seed_transpiler=SEEDS["transpiler"])
        opt_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            data.ccsd_t2, t1=data.ccsd_t1, n_reps=1, interaction_pairs=(aa, _as_pairs(square_pairs)), optimize=True,
            method="L-BFGS-B", regularization=OPT_REGULARIZATION, options={"maxiter": maxiter},
        )
        design["bond_lengths"][str(r)] = {
            "ccsd_route": record.get("ccsd_route"), "e_ccsd": record.get("e_ccsd"), "e_rhf": record["e_rhf"],
            "pairs_square": _as_pairs(square_pairs), "pairs_heavy_hex": _as_pairs(hh_pairs),
            "opt_params": opt_op.to_parameters(interaction_pairs=(aa, _as_pairs(square_pairs))).tolist(),
            "opt_with_final_orbital_rotation": opt_op.final_orbital_rotation is not None,
            "e_opt_ansatz": ansatz_energy(data, opt_op),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design, indent=1))
    return design


def frozen_ops(design: dict, r: float, data) -> dict:
    """The three logical operators for one bond length, rebuilt from the frozen design."""
    entry, aa = design["bond_lengths"][str(r)], _pairs_aa(data.norb)
    square, hh = _as_pairs(entry["pairs_square"]), _as_pairs(entry["pairs_heavy_hex"])
    plain = lambda pairs: ffsim.UCJOpSpinBalanced.from_t_amplitudes(data.ccsd_t2, t1=data.ccsd_t1, n_reps=1,
                                                                    interaction_pairs=(aa, pairs))
    opt = ffsim.UCJOpSpinBalanced.from_parameters(np.array(entry["opt_params"]), norb=data.norb, n_reps=1,
                                                  interaction_pairs=(aa, square),
                                                  with_final_orbital_rotation=entry["opt_with_final_orbital_rotation"])
    return {"square": (plain(square), square), "heavy_hex": (plain(hh), hh), "square_opt": (opt, square)}


def effective_pairs(ucj_op, tol: float = 1e-8) -> list[int]:
    """Orbitals p whose alpha-beta coupling J_ab[p, p] is non-zero in any layer.

    A pattern can allow a pair that the operator never uses: the 1-layer truncation of the
    CCSD-derived UCJ in symmetry-adapted orbitals is exactly zero on several pairs.
    """
    ab = np.abs(np.asarray(ucj_op.diag_coulomb_mats)[:, 1]).max(axis=0)
    return [int(p) for p in np.flatnonzero(np.diag(ab) > tol)]


def _meta(arm: str, r: float, record: dict, data, lucj: LUCJCircuit) -> dict:
    return {
        "arm": arm, "system": SQD_SYSTEM, "bond_length": r, "norb": data.norb, "nelec": list(data.nelec),
        "e_fci": record.get("e_fci"), "e_rhf": record["e_rhf"], "e_ccsd": record.get("e_ccsd"),
        "ccsd_route": record.get("ccsd_route"), "e_ansatz": ansatz_energy(data, lucj.ucj_op),
        "physical_qubits": physical_qubits(lucj.isa), "pairs_ab": lucj.pairs_ab,
        "effective_pairs_ab": effective_pairs(lucj.ucj_op), **lucj.metrics(),
    }


def build_arms(phoenix, kingston, design: dict, bond_lengths=BOND_LENGTHS, fci: dict | None = None) -> dict:
    """{arm: [Pub per bond length]} for the four arms, from the frozen design."""
    for backend in (phoenix, kingston):
        check_allowed(backend)
    assert connectivity_of(phoenix) == "square" and connectivity_of(kingston) == "heavy-hex"
    arms = {arm: [] for arm in (*PHOENIX_ARMS, "kingston_heavyhex")}
    for r in bond_lengths:
        record, data = reference_energies(SYSTEMS[SQD_SYSTEM], r, run_fci=False)
        record["e_fci"] = (fci or {}).get(r)
        if record.get("ccsd_route") != design["bond_lengths"][str(r)]["ccsd_route"]:
            raise RuntimeError(f"R={r}: CCSD route changed since the design was frozen")
        ops = frozen_ops(design, r, data)
        def on_layout(key, backend, connectivity):
            op, pairs = ops[key]
            return transpile_lucj_layout(data, op, backend, connectivity, pairs)

        def on_qubits(key, qubits):
            op, pairs = ops[key]
            return transpile_on_qubits(data, op, phoenix, pairs, qubits)

        # ffsim's pass manager ends with a noise-aware VF2PostLayout that only keeps the pairs a
        # circuit actually uses, so the shared phoenix qubits are chosen from the arm with the most
        # non-zero alpha-beta couplings; the other phoenix arms must then fit with routing disabled.
        active = {key: effective_pairs(ops[key][0]) for key in ("square", "square_opt", "heavy_hex")}
        candidates = ("square_opt", "square") if r in design.get("opt_bond_lengths", BOND_LENGTHS) else ("square",)
        anchor_key = max(candidates, key=lambda k: len(active[k]))
        anchor = on_layout(anchor_key, phoenix, "square")
        qubits = physical_qubits(anchor.isa)
        built = {
            "phoenix_square": on_qubits("square", qubits),
            "phoenix_hhpattern": on_qubits("heavy_hex", qubits),
            "kingston_heavyhex": on_layout("heavy_hex", kingston, "heavy-hex"),
        }
        if r in design.get("opt_bond_lengths", BOND_LENGTHS):
            built["phoenix_square_opt"] = on_qubits("square_opt", qubits)
        for arm in (a for a in PHOENIX_ARMS if a in built):
            if physical_qubits(built[arm].isa) != qubits:
                raise RuntimeError(f"R={r}: {arm} did not land on the shared phoenix qubits")
        for arm, lucj in built.items():
            arms[arm].append(Pub(f"{arm}_R{r}", lucj.isa, _meta(arm, r, record, data, lucj), lucj.logical, data))
    return arms


def study_jobs(arms: dict, instance: str, repeats, shots: int = SHOTS, prefix: str = "") -> list[JobSpec]:
    """Per repeat: one phoenix job (three arms interleaved by bond length) and one kingston job."""
    options = {**SAMPLER_OPTIONS, "default_shots": shots}
    by_r = {}
    for arm in PHOENIX_ARMS:
        for pub in arms[arm]:
            by_r.setdefault(pub.meta["bond_length"], []).append(pub)
    interleaved = [pub for r in sorted(by_r) for pub in by_r[r]]
    jobs = []
    for k in repeats:
        jobs.append(JobSpec(f"{prefix}phoenix_rep{k}", instance, "ibm_phoenix", options, interleaved))
        jobs.append(JobSpec(f"{prefix}kingston_rep{k}", instance, "ibm_kingston", options, arms["kingston_heavyhex"]))
    return jobs


def _rep_delay_s(backend) -> float:
    value = getattr(backend.configuration(), "default_rep_delay", None) or 250e-6
    return value * 1e-6 if value > 1e-2 else value  # snapshots store microseconds, live configs seconds


def estimate_qpu_seconds(job: JobSpec, backend) -> dict:
    program, _ = prepare_sampler([p.circuit for p in job.pubs], job.sampler(backend).options, None, backend=backend)
    rep_delay = _rep_delay_s(backend)
    durations = [p.meta["isa_duration_us"] * 1e-6 for p in job.pubs]
    per_item_shots = [int(np.prod(getattr(item, "shape", (1,)) or (1,))) * program.shots for item in program.items]
    if len(per_item_shots) != len(durations):  # the sampler program holds one item per PUB, in PUB order
        raise RuntimeError(f"{job.name}: {len(per_item_shots)} program items for {len(durations)} PUBs")
    seconds = sum(n * (d + rep_delay) for n, d in zip(per_item_shots, durations))
    return {"job": job.name, "backend": job.backend_name, "pubs": len(job.pubs), "executions": sum(per_item_shots),
            "rep_delay_us": 1e6 * rep_delay, "longest_circuit_us": 1e6 * max(durations),
            "qpu_seconds": seconds + JOB_OVERHEAD_S}
