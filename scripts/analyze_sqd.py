"""Analysis for the SQD hardware study (pre-registered in docs/preregistration.md).

For every arm, bond length and repeat, at each per-spin cap d (subspace <= d^2):
  - SQD energy error vs FCI after iteration 1 (postselection only, "before recovery")
    and after each configuration-recovery round
  - the valid electron-count fraction, against the uniform-random value (0.444% here)
Baselines at the same d, per bond length:
  - uniform random bitstrings with the same shot count, through the same SQD loop
  - HCI: top-d strings of a selected-CI wavefunction
  - best FCI subspace: top-d strings of the exact wavefunction

Usage: python scripts/analyze_sqd.py results/hardware/<run>/manifest.json [--full-baselines]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import ffsim
import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qchem_sqd.baselines import fci_vector, hci_wavefunction, subspace_comparison  # noqa: E402
from qchem_sqd.molecules import SYSTEMS  # noqa: E402
from qchem_sqd.reference import reference_energies  # noqa: E402
from qchem_sqd.repro import SEEDS  # noqa: E402
from qchem_sqd.sqd import random_valid_fraction, run_sqd, valid_fraction  # noqa: E402

MAX_DIMS = (30, 60, 100, 150, 200)
MAX_ITERATIONS = 5  # iteration 1 = before recovery, then 4 recovery rounds


def sqd_by_cap(data, bits: BitArray, e_fci: float, caps=MAX_DIMS, seed: int = SEEDS["sqd"]) -> dict:
    out = {"valid_fraction": valid_fraction(bits, data.norb, data.nelec), "caps": {}}
    for d in caps:
        run = run_sqd(data, bits, max_dim=d, max_iterations=MAX_ITERATIONS, seed=seed)
        out["caps"][d] = {
            "error_mha_by_round": [1000 * (e - e_fci) for e in run["energy_history"]],
            "dim_by_round": run["dim_history"],
            "error_mha_final": 1000 * (run["e_sqd"] - e_fci),
        }
    return out


def baselines(data, e_fci: float, shots: int, caps=MAX_DIMS, full: bool = True) -> dict:
    """Random-bitstring SQD (3 seeds) and, when full, HCI and best-FCI subspaces."""
    out = {"random": [], "random_valid_fraction_expected": random_valid_fraction(data.norb, data.nelec)}
    for k in range(3):
        bits = generate_bit_array_uniform(shots, 2 * data.norb, rand_seed=SEEDS["random_bitstrings"] + k)
        out["random"].append(sqd_by_cap(data, bits, e_fci, caps, seed=SEEDS["sqd"] + k))
    if full:
        e_hci, hvec = hci_wavefunction(data)  # from the integrals only; never sees the FCI vector
        e_exact, civec = fci_vector(data)
        out["fci_check_mha"] = 1000 * (e_exact - e_fci)
        out["hci_full_error_mha"] = 1000 * (e_hci - e_fci)
        out["hci_n_strings"] = int(len(set(map(int, hvec._strs[0])) | set(map(int, hvec._strs[1]))))
        out["best_fci"], out["hci"], out["hci_bestfci_overlap"] = {}, {}, {}
        for d in caps:
            cmp = subspace_comparison(data, hvec, civec, d)
            out["best_fci"][d] = 1000 * (cmp["e_best_fci"] - e_fci)
            out["hci"][d] = 1000 * (cmp["e_hci"] - e_fci)
            out["hci_bestfci_overlap"][d] = cmp["overlap"]
    return out


def print_table(analysis: dict, caps=MAX_DIMS) -> None:
    """Mean over repeats of the final error, and the before-recovery error, per arm and cap."""
    for r, block in sorted(analysis.items()):
        print(f"\nR = {r} A   (errors vs FCI in mHa; 'pre' = before recovery, 'fin' = after the last round)")
        header = "".join(f"{f'd={d}':>18s}" for d in caps)
        print(f"  {'source':20s} {'valid frac':>10s} {header}")
        rows = dict(block["arms"])
        rows["random"] = block["baselines"]["random"]
        for name, runs in rows.items():
            vf = np.mean([run["valid_fraction"] for run in runs])
            cells = ""
            for d in caps:
                pre = np.mean([run["caps"][d]["error_mha_by_round"][0] for run in runs])
                fin = np.mean([run["caps"][d]["error_mha_final"] for run in runs])
                cells += f"   pre{pre:6.1f} fin{fin:6.1f}"
            print(f"  {name:20s} {100 * vf:9.3f}% {cells}")
        b = block["baselines"]
        print(f"  {'random expected':20s} {100 * b['random_valid_fraction_expected']:9.3f}%")
        for name in ("hci", "best_fci"):
            if name in b:
                print(f"  {name:20s} {'':10s} " + "".join(f"{b[name][d]:18.1f}" for d in caps))
        if "hci_bestfci_overlap" in b:
            overlap = b["hci_bestfci_overlap"]
            cells = "".join(f"{str(overlap[d]) + '/' + str(d) + ' shared':>18s}" for d in caps)
            print(f"  {'HCI vs best-FCI':20s} {'':10s} {cells}")


def analyze_offline(jobs, shots: int = 100_000, caps=None, full: bool = False, all_bond_lengths: bool = False) -> dict:
    """Pipeline test without hardware: noiseless ffsim samples of each arm (first bond length, or all)."""
    caps = caps or (MAX_DIMS if all_bond_lengths else (30, 60))
    pubs = {}
    for job in jobs:
        for pub in job.pubs:
            pubs.setdefault((pub.meta["bond_length"], pub.meta["arm"]), pub)
    analysis = {}
    for (r, arm), pub in sorted(pubs.items()):
        if not all_bond_lengths and r != min(k[0] for k in pubs):
            continue
        e_fci = pub.meta["e_fci"]
        bits = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=SEEDS["sampler"]).run([pub.logical]).result()[0].data.meas
        block = analysis.setdefault(r, {"arms": {}, "baselines": None})
        block["arms"][arm] = [sqd_by_cap(pub.data, bits, e_fci, caps)]
        if block["baselines"] is None:
            block["baselines"] = baselines(pub.data, e_fci, shots, caps, full=full)
    print("\n== Offline analysis (noiseless samples, not hardware) ==")
    print_table(analysis, caps)
    return analysis


def analyze_manifest(manifest_path: Path, full: bool, out_name: str = "analysis.json") -> None:
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    system = SYSTEMS["n2_631g_16o"]
    grouped = defaultdict(lambda: defaultdict(list))  # r -> arm -> [(repeat, bits, meta)]
    for entry in manifest["jobs"]:
        for meta in entry["pubs"]:
            if meta.get("system") != "n2_631g_16o":  # 52-qubit circuits are analyzed by analyze_ccpvdz.py
                continue
            path = run_dir / f"{entry['name']}__{meta['label']}.npz"
            if path.exists():
                z = np.load(path)
                grouped[meta["bond_length"]][meta["arm"]].append((entry["name"], BitArray(z["array"], int(z["num_bits"])), meta))
    analysis = {}
    for r, arms in sorted(grouped.items()):
        _, data = reference_energies(system, r, run_fci=False)
        e_fci = next(iter(arms.values()))[0][2]["e_fci"]
        shots = next(iter(arms.values()))[0][1].num_shots
        analysis[r] = {"arms": {arm: [dict(sqd_by_cap(data, bits, e_fci), job=name) for name, bits, _ in runs]
                                for arm, runs in arms.items()},
                       "baselines": baselines(data, e_fci, shots, full=full)}
    (run_dir / out_name).write_text(json.dumps(analysis, indent=1, default=str))
    print_table(analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--full-baselines", action="store_true", help="also HCI and best-FCI (needs the FCI vector)")
    parser.add_argument("--out", default="analysis.json", help="output file name inside the run directory")
    args = parser.parse_args()
    analyze_manifest(Path(args.manifest), args.full_baselines, args.out)
