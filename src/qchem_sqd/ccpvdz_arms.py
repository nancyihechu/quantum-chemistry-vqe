"""52-qubit N2/cc-pVDZ (10e, 26o) hardware arms on the published FCIDUMPs (arXiv:2405.05068).

Arms at grid 1.0, 1.5, 2.2 (units of 1.0975135 A), 1-layer LUCJ from CCSD, plain truncation:
  phoenix52_square      alpha-beta pairs ffsim's square layout accepts on ibm_phoenix
  phoenix52_hhpattern   every 4th of those that ibm_kingston also hosts, on the same phoenix qubits
  kingston52_heavyhex   that heavy-hex pattern on ibm_kingston, auxiliary qubits as ffsim places them

The pair lists are frozen once (DESIGN_FILE). The shared phoenix qubits come from a layout pass
on a throwaway random operator that is non-zero on every square pair (the noise-aware
VF2PostLayout keeps only the pairs a circuit uses), and both phoenix arms are transpiled onto
them with routing disabled. Amplitude-guided pair selection is deliberately not used.
"""

import json
from pathlib import Path

import ffsim
import numpy as np

from . import fcidump_system as fs
from .backends import check_allowed
from .hardware import Pub, effective_pairs, logical_circuit, physical_qubits, transpile_on_qubits
from .lucj import LUCJCircuit
from .noise_estimates import estimate_valid_fraction
from .repro import SEEDS

DESIGN_FILE = Path(__file__).resolve().parents[2] / "results" / "hardware" / "design_ccpvdz_frozen.json"
CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "results" / "hardware" / "phoenix52_shots_calibrated.json"
KINGSTON_SHOTS = 100_000  # phoenix shots are derived to match kingston's QPU time per circuit
PHOENIX_ARMS = ("phoenix52_square", "phoenix52_hhpattern")


def _aa(norb: int) -> list[tuple[int, int]]:
    return [(p, p + 1) for p in range(norb - 1)]


def _pairs(pairs) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in pairs]


def ffsim_layout(data, ucj_op, backend, connectivity: str, pairs_ab):
    pm, accepted = ffsim.qiskit.generate_lucj_pass_manager(
        backend, data.norb, connectivity, interaction_pairs=(_aa(data.norb), pairs_ab),
        optimization_level=3, seed_transpiler=SEEDS["transpiler"])
    circuit = logical_circuit(data, ucj_op)
    isa = pm.run(circuit)
    lucj = LUCJCircuit(logical=circuit, isa=isa, ucj_op=ucj_op, pairs_ab=_pairs(accepted),
                       duration_s=float(isa.estimate_duration(backend.target, unit="s")))
    return lucj, _pairs(accepted)


def freeze_design(phoenix, kingston, grids=fs.STUDY_GRID, path: Path = DESIGN_FILE) -> dict:
    design = {"system": "N2/cc-pVDZ (10e,26o)", "source": str(fs.FCIDUMP_DIR), "grid_unit_angstrom": fs.UNIT_ANGSTROM,
              "snapshots": {b.name: str(getattr(b, "snapshot_folder", "live")) for b in (phoenix, kingston)}, "grid": {}}
    for g in grids:
        data = fs.load(g)
        aa = _aa(data.norb)
        probe = ffsim.random.random_ucj_op_spin_balanced(data.norb, interaction_pairs=(aa, None), seed=1)
        _, square = ffsim_layout(data, probe, phoenix, "square", None)
        candidates = [p for p in square if p[0] % 4 == 0]
        probe_hh = ffsim.random.random_ucj_op_spin_balanced(data.norb, interaction_pairs=(aa, candidates), seed=2)
        _, heavy_hex = ffsim_layout(data, probe_hh, kingston, "heavy-hex", candidates)
        design["grid"][str(g)] = {"pairs_square": square, "pairs_heavy_hex": heavy_hex}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design, indent=1))
    return design


def build_arms(phoenix, kingston, design: dict, grids=fs.STUDY_GRID) -> dict:
    """{arm: [Pub per grid value]} from the frozen pair lists, on the given backends."""
    for backend in (phoenix, kingston):
        check_allowed(backend)
    arms = {arm: [] for arm in (*PHOENIX_ARMS, "kingston52_heavyhex")}
    hci = fs.reference_energies()
    for g in grids:
        data, mf, record = fs.continued_ccsd(g)
        entry, aa = design["grid"][str(g)], _aa(data.norb)
        square, heavy_hex = _pairs(entry["pairs_square"]), _pairs(entry["pairs_heavy_hex"])
        plain = lambda pairs: ffsim.UCJOpSpinBalanced.from_t_amplitudes(data.ccsd_t2, t1=data.ccsd_t1, n_reps=1,
                                                                        interaction_pairs=(aa, pairs))
        anchor = ffsim.random.random_ucj_op_spin_balanced(data.norb, interaction_pairs=(aa, square), seed=3,
                                                          with_final_orbital_rotation=True)
        anchor_lucj, accepted = ffsim_layout(data, anchor, phoenix, "square", square)
        if accepted != square:
            raise RuntimeError(f"grid {g}: phoenix no longer hosts the frozen square pairs")
        qubits = physical_qubits(anchor_lucj.isa)
        kingston_lucj, accepted_hh = ffsim_layout(data, plain(heavy_hex), kingston, "heavy-hex", heavy_hex)
        if accepted_hh != heavy_hex:
            raise RuntimeError(f"grid {g}: kingston no longer hosts the frozen heavy-hex pairs")
        built = {
            "phoenix52_square": (transpile_on_qubits(data, plain(square), phoenix, square, qubits), phoenix),
            "phoenix52_hhpattern": (transpile_on_qubits(data, plain(heavy_hex), phoenix, heavy_hex, qubits), phoenix),
            "kingston52_heavyhex": (kingston_lucj, kingston),
        }
        for arm, (lucj, backend) in built.items():
            m = lucj.metrics()
            meta = {"arm": arm, "system": "n2_ccpvdz_26o", "grid": g, "bond_length": fs.angstrom(g),
                    "norb": data.norb, "nelec": list(data.nelec), "e_hci": hci[g]["hci"], "e_rhf": record["e_rhf"],
                    "e_ccsd": record["e_ccsd"], "ccsd_route": record["ccsd_route"], **m,
                    "isa_depth": lucj.isa.depth(), "auxiliary_qubits": m["isa_qubits_used"] - 2 * data.norb,
                    "physical_qubits": physical_qubits(lucj.isa), "pairs_ab": lucj.pairs_ab,
                    "effective_pairs_ab": effective_pairs(lucj.ucj_op),
                    "valid_fraction_model": estimate_valid_fraction(lucj.isa, backend)["valid_fraction"]}
            arms[arm].append(Pub(f"{arm}_G{g}", lucj.isa, meta, lucj.logical, data))
    return arms


def matched_phoenix_shots(arms: dict, phoenix, kingston, kingston_shots: int = KINGSTON_SHOTS) -> int:
    """Phoenix shots per circuit that cost the same QPU time as ``kingston_shots`` on kingston.

    Cost per shot = scheduled circuit duration + the device's default repetition delay,
    averaged over the circuits of each device; rounded to the nearest 1e4 shots.
    """
    from .hardware import _rep_delay_s

    def mean_cost(pubs, backend):
        return float(np.mean([p.meta["isa_duration_us"] * 1e-6 + _rep_delay_s(backend) for p in pubs]))

    kingston_cost = mean_cost(arms["kingston52_heavyhex"], kingston)
    phoenix_cost = mean_cost([p for a in PHOENIX_ARMS for p in arms[a]], phoenix)
    return int(round(kingston_shots * kingston_cost / phoenix_cost, -4))


def calibrated_phoenix_shots(pilot_usage: dict, kingston_shots: int = KINGSTON_SHOTS) -> dict:
    """Pre-registered matched-QPU-time rule, from the pilot's IBM-reported usage.

    ``pilot_usage`` = {"ibm_phoenix": (usage_seconds, shots), "ibm_kingston": (usage_seconds, shots)} for the
    two one-circuit pilot jobs. Phoenix study shots = kingston shots x (kingston usage per shot) /
    (phoenix usage per shot), rounded to the nearest 1e4.
    """
    per_shot = {name: usage / shots for name, (usage, shots) in pilot_usage.items()}
    shots = int(round(kingston_shots * per_shot["ibm_kingston"] / per_shot["ibm_phoenix"], -4))
    return {"phoenix_shots": shots, "kingston_shots": kingston_shots, "usage_per_shot_s": per_shot,
            "pilot_usage": {k: {"usage_s": u, "shots": n} for k, (u, n) in pilot_usage.items()}}


def ccpvdz_jobs(arms: dict, instance: str, repeats, phoenix, kingston, prefix: str = "",
                phoenix_shots: int | None = None) -> list:
    """Per repeat: one phoenix job (square and heavy-hex pattern interleaved by grid) and one
    kingston job, with shot counts that give each circuit the same estimated QPU time. Kept
    separate from the 32-qubit jobs because one executor program carries a single shot count."""
    from .hardware import SAMPLER_OPTIONS, JobSpec

    phoenix_shots = phoenix_shots or matched_phoenix_shots(arms, phoenix, kingston)
    interleaved = [pub for pair in zip(*(arms[a] for a in PHOENIX_ARMS)) for pub in pair]
    jobs = []
    for k in repeats:
        jobs.append(JobSpec(f"{prefix}phoenix52_rep{k}", instance, "ibm_phoenix",
                            {**SAMPLER_OPTIONS, "default_shots": phoenix_shots}, interleaved))
        jobs.append(JobSpec(f"{prefix}kingston52_rep{k}", instance, "ibm_kingston",
                            {**SAMPLER_OPTIONS, "default_shots": KINGSTON_SHOTS}, arms["kingston52_heavyhex"]))
    return jobs


def pilot_jobs(arms: dict, instance: str, phoenix, kingston, grid: float = 1.0) -> list:
    """One 52-qubit circuit per device for the pilot (equilibrium grid point), at matched QPU time."""
    from .hardware import SAMPLER_OPTIONS, JobSpec

    phoenix_shots = matched_phoenix_shots(arms, phoenix, kingston)
    pick = lambda arm: [next(p for p in arms[arm] if p.meta["grid"] == grid)]
    return [
        JobSpec("pilot_phoenix52", instance, "ibm_phoenix", {**SAMPLER_OPTIONS, "default_shots": phoenix_shots},
                pick("phoenix52_square")),
        JobSpec("pilot_kingston52", instance, "ibm_kingston", {**SAMPLER_OPTIONS, "default_shots": KINGSTON_SHOTS},
                pick("kingston52_heavyhex")),
    ]


SHOTSCALE_GRIDS = (1.0, 2.2)
SHOTSCALE_JOBS = 5
SHOTSCALE_SHOTS_PER_JOB = 2_000_000


def shotscale_jobs(arms: dict, instance: str, n_jobs: int = SHOTSCALE_JOBS,
                   shots_per_job: int = SHOTSCALE_SHOTS_PER_JOB) -> list:
    """Pre-registered (Addendum B2) phoenix shot-scaling run: arm E at grid 1.0 and 2.2.

    n_jobs jobs, each holding both grid values interleaved at shots_per_job shots per circuit,
    so the pooled data have n_jobs x shots_per_job shots per grid value (10M by default).
    """
    from .hardware import SAMPLER_OPTIONS, JobSpec

    pubs = [p for p in arms["phoenix52_square"] if p.meta["grid"] in SHOTSCALE_GRIDS]
    if sorted(p.meta["grid"] for p in pubs) != sorted(SHOTSCALE_GRIDS):
        raise RuntimeError("shot-scaling run needs arm E at grid 1.0 and 2.2")
    options = {**SAMPLER_OPTIONS, "default_shots": shots_per_job}
    return [JobSpec(f"shotscale_phoenix52_part{k}", instance, "ibm_phoenix", options, pubs) for k in range(1, n_jobs + 1)]
