"""Correctness checks before hardware, with symmetry-adapted orbitals and pinned threads.

Per geometry:
  rhf        symmetric vs unsymmetrized RHF energy, and whether the latter breaks symmetry
  ccsd       convergence, and CCSD-FCI
  t_order    correlation energy rebuilt from (t1, t2) and the active-space integrals must
             equal pyscf's: confirms the amplitude orbital order matches the Hamiltonian
  sign       energy of exp(lam*(T - T^dag))|HF> for lam = +/-0.05 from the full UCJ: the
             correct sign lowers the energy, with slope 2*E_lin (first-order theory)
  lucj       plain LUCJ (square and heavy-hex pair patterns) energy vs RHF and FCI
  optimize   optimize=True with regularization=1e-2 and fixed maxiter: energy and parameters,
             to be compared across runs with different --threads

Usage: python scripts/software_checks.py n2_631g_8o --threads 1 --out results/checks/n2_8o_t1.json
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--bond-lengths", type=float, nargs="+", default=[1.0, 1.2, 1.6, 1.8, 2.0, 2.4])
parser.add_argument("--maxiter", type=int, default=200)
parser.add_argument("--regularization", type=float, default=1e-2)
parser.add_argument("--fci", action="store_true", help="compute FCI (else read results/references when present)")
parser.add_argument("--out", required=True)
args = parser.parse_args()

from qchem_sqd.repro import pin_threads, thread_info  # noqa: E402

pin_threads(args.threads)  # before numpy/pyscf/ffsim are imported

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import ffsim  # noqa: E402
import numpy as np  # noqa: E402
from pyscf import ao2mo  # noqa: E402

from qchem_sqd.lucj import ansatz_energy  # noqa: E402
from qchem_sqd.molecules import SYSTEMS, build_molecule, molecular_data, run_rhf  # noqa: E402
from qchem_sqd.reference import compute_fci, run_ccsd  # noqa: E402
from qchem_sqd.molecules import active_orbitals  # noqa: E402


def symmetry_breaking(mol_sym, dm) -> float:
    """Largest density element coupling different irreps (0 for a symmetric density)."""
    S, blocks = mol_sym.intor("int1e_ovlp"), mol_sym.symm_orb
    return float(max(np.abs(c1.T @ S @ dm @ S @ c2).max()
                     for i, c1 in enumerate(blocks) for j, c2 in enumerate(blocks) if i != j))


def correlation_from_amplitudes(data, t1, t2) -> float:
    """Closed-shell CCSD energy expression in the active-space MO basis (canonical HF: f_ia = 0)."""
    nocc = data.nelec[0]
    eri = ao2mo.restore(1, data.two_body_integrals, data.norb)
    ovov = eri[:nocc, nocc:, :nocc, nocc:]  # (ia|jb)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    return float(np.einsum("ijab,iajb->", tau, 2 * ovov) - np.einsum("ijab,ibja->", tau, ovov))


def linear_energy(data, t2) -> float:
    """<HF|H T2|HF> for closed shells, the first-order slope term."""
    nocc = data.nelec[0]
    eri = ao2mo.restore(1, data.two_body_integrals, data.norb)
    ovov = eri[:nocc, nocc:, :nocc, nocc:]
    return float(np.einsum("ijab,iajb->", t2, 2 * ovov) - np.einsum("ijab,ibja->", t2, ovov))


def main() -> None:
    system = SYSTEMS[args.system]
    ref_path = Path("results/references") / f"{args.system}_sym.json"
    saved_fci = {r["bond_length"]: r["e_fci"] for r in json.loads(ref_path.read_text())} if ref_path.exists() else {}
    records = []
    for r in args.bond_lengths:
        t_start = time.perf_counter()
        rec = {"system": args.system, "bond_length": r, "threads": args.threads, "thread_info": thread_info()}
        data, mf = molecular_data(system, r, symmetry=True)
        mf_nosym = run_rhf(build_molecule(system, r, symmetry=False))
        rec["rhf"] = {
            "e_sym": mf.e_tot,
            "e_nosym": mf_nosym.e_tot,
            "nosym_cross_irrep_density": symmetry_breaking(mf.mol, mf_nosym.make_rdm1()),
            "irrep_occupation": {k: int(v) for k, v in mf.get_irrep_nelec().items() if v},
        }

        rec["ccsd"] = run_ccsd(data, mf, active_orbitals(system, mf.mol), system, r)
        e_corr_pyscf = rec["ccsd"]["e_ccsd"] - mf.e_tot if rec["ccsd"]["e_ccsd"] is not None else None
        e_corr_rebuilt = correlation_from_amplitudes(data, data.ccsd_t1, data.ccsd_t2)
        rec["t_order"] = {"e_corr_pyscf": e_corr_pyscf, "e_corr_rebuilt": e_corr_rebuilt,
                          "abs_diff": None if e_corr_pyscf is None else abs(e_corr_pyscf - e_corr_rebuilt)}

        e_fci = saved_fci.get(r)
        if e_fci is None and args.fci:
            e_fci = compute_fci(data)
        rec["e_fci"] = e_fci

        lam, e_lin = 0.05, linear_energy(data, data.ccsd_t2)
        sign = {"lambda": lam, "e_hf": mf.e_tot, "predicted_slope_term": 2 * lam * e_lin}
        for label, s in (("plus", 1.0), ("minus", -1.0)):
            op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(s * lam * data.ccsd_t2)  # full UCJ, t2 only
            sign[f"e_{label}"] = ansatz_energy(data, op)
        sign["observed_slope_term"] = 0.5 * (sign["e_plus"] - sign["e_minus"])
        sign["correct_sign"] = sign["e_plus"] < sign["e_minus"]
        rec["sign"] = sign

        norb = data.norb
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        patterns = {"square": [(p, p) for p in range(norb)], "heavy_hex": [(p, p) for p in range(norb) if p % 4 == 0]}
        rec["lucj"] = {}
        for name, pairs_ab in patterns.items():
            op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(data.ccsd_t2, t1=data.ccsd_t1, n_reps=1,
                                                          interaction_pairs=(pairs_aa, pairs_ab))
            e = ansatz_energy(data, op)
            rec["lucj"][name] = {"e": e, "minus_rhf_mha": 1000 * (e - mf.e_tot),
                                 "minus_fci_mha": None if e_fci is None else 1000 * (e - e_fci)}

        t0 = time.perf_counter()
        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            data.ccsd_t2, t1=data.ccsd_t1, n_reps=1, interaction_pairs=(pairs_aa, patterns["square"]),
            optimize=True, method="L-BFGS-B", regularization=args.regularization, options={"maxiter": args.maxiter},
        )
        params = op.to_parameters()
        e = ansatz_energy(data, op)
        rec["optimize"] = {"regularization": args.regularization, "maxiter": args.maxiter, "e": e,
                           "minus_rhf_mha": 1000 * (e - mf.e_tot),
                           "minus_fci_mha": None if e_fci is None else 1000 * (e - e_fci),
                           "param_norm": float(np.linalg.norm(params)), "params": params.tolist(),
                           "seconds": time.perf_counter() - t0}
        rec["seconds"] = time.perf_counter() - t_start
        records.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "optimize"} | {"optimize_e": e}), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    sys.exit(main())
