"""Exploratory (not a pre-registered test): why is arm B (phoenix heavy-hex pattern) about 75 mHa
better than A (phoenix square) and D (kingston heavy-hex) at R = 2.4 A, d = 30, in all day-1 repeats?

For each arm and repeat, SQD is rerun with the frozen settings (samples_per_batch 300,
num_batches 3, max_iterations 5, symmetrize_spin, seed 24, max_dim 30) while recording the
strings kept by the cap at every iteration. Compared with the top-30 strings of the exact FCI
wavefunction, and probed by (i) removing each kept string from B's final subspace and
(ii) adding each B-only string to A's final subspace, re-diagonalizing each time.

Usage: python scripts/explore_b24_d30.py
Writes results/exploratory/b24_d30.json
"""

import json
from pathlib import Path

import numpy as np
from pyscf import ao2mo
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

from qchem_sqd.baselines import fci_vector, subspace_energy
from qchem_sqd.molecules import SYSTEMS
from qchem_sqd.reference import reference_energies
from qchem_sqd.repro import SEEDS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "hardware" / "study_20261022T2126Z"
ARMS = {"A": ("phoenix_rep{k}", "phoenix_square_R2.4"), "B": ("phoenix_rep{k}", "phoenix_hhpattern_R2.4"),
        "D": ("kingston_rep{k}", "kingston_heavyhex_R2.4")}
NORB, NOCC = 16, 5
HF = (1 << NOCC) - 1


def describe(s: int) -> dict:
    occ = [p for p in range(NORB) if s >> p & 1]
    return {"string": int(s), "occupied": occ, "excitation_from_hf": bin(s ^ HF).count("1") // 2}


def run_with_strings(data, bits, max_dim=30):
    history = []

    def callback(results):
        best = min(results, key=lambda r: r.energy)
        state = best.sci_state
        history.append({"energy": best.energy + data.core_energy, "strings": sorted(map(int, state.ci_strs_a)),
                        "dim": int(state.amplitudes.size)})

    result = diagonalize_fermionic_hamiltonian(
        data.one_body_integrals, ao2mo.restore(1, data.two_body_integrals, data.norb), bits,
        samples_per_batch=300, norb=data.norb, nelec=data.nelec, num_batches=3, max_iterations=5,
        symmetrize_spin=True, max_dim=max_dim, callback=callback, seed=SEEDS["sqd"])
    amps = result.sci_state.amplitudes
    a_strs, b_strs = result.sci_state.ci_strs_a, result.sci_state.ci_strs_b
    weight = {}
    for i, s in enumerate(a_strs):
        weight[int(s)] = weight.get(int(s), 0.0) + float(np.sum(np.abs(amps[i, :]) ** 2))
    for j, s in enumerate(b_strs):
        weight[int(s)] = weight.get(int(s), 0.0) + float(np.sum(np.abs(amps[:, j]) ** 2))
    return history, weight


def sample_frequencies(bits: BitArray) -> dict:
    """How often each alpha and beta string occurs among valid shots (both spins pooled)."""
    b = bits.to_bool_array(order="little")
    pow2 = 1 << np.arange(NORB)
    a = (b[:, :NORB] * pow2).sum(1)
    bb = (b[:, NORB:] * pow2).sum(1)
    valid = (b[:, :NORB].sum(1) == NOCC) & (b[:, NORB:].sum(1) == NOCC)
    vals, counts = np.unique(np.concatenate([a[valid], bb[valid]]), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def main() -> None:
    record, data = reference_energies(SYSTEMS["n2_631g_16o"], 2.4, run_fci=False)
    e_fci, civec = fci_vector(data)
    from pyscf.fci import cistring

    strs = cistring.make_strings(range(NORB), NOCC)
    marg = (np.abs(civec) ** 2).sum(1) + (np.abs(civec) ** 2).sum(0)
    fci_weight = {int(s): float(w) for s, w in zip(strs, marg)}
    best30 = [int(s) for s in strs[np.argsort(-marg)[:30]]]
    out = {"e_fci": e_fci, "e_rhf": record["e_rhf"], "best30": [describe(s) | {"fci_weight": fci_weight[s]} for s in best30],
           "best30_error_mha": 1000 * (subspace_energy(data, np.array(best30))[0] - e_fci), "runs": {}}
    finals = {}
    for arm, (job, label) in ARMS.items():
        for k in (1, 2, 3):
            z = np.load(RUN / f"{job.format(k=k)}__{label}.npz")
            bits = BitArray(z["array"], int(z["num_bits"]))
            history, sqd_weight = run_with_strings(data, bits)
            freq = sample_frequencies(bits)
            final = history[-1]["strings"]
            finals[(arm, k)] = final
            out["runs"][f"{arm}{k}"] = {
                "error_by_iteration_mha": [1000 * (h["energy"] - e_fci) for h in history],
                "dim_by_iteration": [h["dim"] for h in history],
                "kept_iter1": history[0]["strings"], "kept_final": final,
                "overlap_with_best30_iter1": len(set(history[0]["strings"]) & set(best30)),
                "overlap_with_best30_final": len(set(final) & set(best30)),
                "fci_weight_captured_final": sum(fci_weight.get(s, 0.0) for s in final),
                "kept_strings": [describe(s) | {"fci_weight": fci_weight.get(s, 0.0), "sample_count": freq.get(s, 0),
                                                "sqd_weight": sqd_weight.get(s, 0.0)} for s in final],
            }
            print(f"{arm}{k}: errors by iteration {[round(x, 1) for x in out['runs'][f'{arm}{k}']['error_by_iteration_mha']]} "
                  f"| best30 overlap iter1 {out['runs'][f'{arm}{k}']['overlap_with_best30_iter1']} final "
                  f"{out['runs'][f'{arm}{k}']['overlap_with_best30_final']} | FCI weight kept "
                  f"{out['runs'][f'{arm}{k}']['fci_weight_captured_final']:.3f}", flush=True)
    probes = {}
    for k in (1, 2, 3):
        b_final, a_final = finals[("B", k)], finals[("A", k)]
        e_b = subspace_energy(data, np.array(b_final))[0]
        e_a = subspace_energy(data, np.array(a_final))[0]
        leave_out = sorted(((1000 * (subspace_energy(data, np.array([s for s in b_final if s != x]))[0] - e_b), x)
                            for x in b_final), reverse=True)[:5]
        add_to_a = sorted(((1000 * (subspace_energy(data, np.array(a_final + [x]))[0] - e_a), x)
                           for x in set(b_final) - set(a_final)))[:5]
        probes[k] = {"b_only": sorted(set(b_final) - set(a_final)), "a_only": sorted(set(a_final) - set(b_final)),
                     "leave_one_out_from_B_top5": [{"delta_mha": d} | describe(x) for d, x in leave_out],
                     "add_one_to_A_top5": [{"delta_mha": d} | describe(x) for d, x in add_to_a]}
        print(f"repeat {k}: B-only strings {len(probes[k]['b_only'])}, A-only {len(probes[k]['a_only'])}", flush=True)
        for d, x in leave_out[:3]:
            print(f"   removing {describe(x)} from B raises E by {d:.1f} mHa (FCI weight {fci_weight.get(x, 0):.3f})")
        for d, x in add_to_a[:3]:
            print(f"   adding {describe(x)} to A changes E by {d:.1f} mHa (FCI weight {fci_weight.get(x, 0):.3f})")
    out["probes"] = probes
    path = ROOT / "results" / "exploratory"
    path.mkdir(parents=True, exist_ok=True)
    (path / "b24_d30.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
