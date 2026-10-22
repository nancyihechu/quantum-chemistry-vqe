"""Classical baselines at a fixed subspace size, solved exactly like SQD.

All subspaces are spin-symmetric products S x S of one list S of single-spin
strings (as SQD with symmetrize_spin=True), with |S| <= d, so the dimension
is at most d**2. Each is diagonalized with qiskit_addon_sqd's solve_sci, the
same solver SQD uses, so only the choice of strings differs:

  best_fci   the d strings with the largest marginal weight in the exact FCI
             wavefunction: a near-optimal subspace of that size (a reference
             point, not something a real calculation could know)
  hci        the d strings with the largest marginal weight in a converged
             heat-bath-style selected CI (pyscf selected_ci)
  random     handled in sqd.py via uniform random bitstrings + recovery
"""

import numpy as np
import pyscf.fci
from pyscf import ao2mo
from pyscf.fci import cistring, selected_ci
from qiskit_addon_sqd.fermion import solve_sci


def _full_eri(data) -> np.ndarray:
    return ao2mo.restore(1, data.two_body_integrals, data.norb)


def subspace_energy(data, strings: np.ndarray) -> tuple[float, int]:
    """Ground energy in the product space strings x strings (total energy, dimension)."""
    strings = np.sort(np.unique(np.asarray(strings, dtype=np.int64)))
    result = solve_sci((strings, strings), data.one_body_integrals, _full_eri(data), data.norb, data.nelec)
    return float(result.energy + data.core_energy), int(len(strings) ** 2)


def _top_strings(weights: np.ndarray, strings: np.ndarray, d: int) -> np.ndarray:
    order = np.argsort(-weights, kind="stable")
    return strings[order[:d]]


def fci_vector(data) -> tuple[float, np.ndarray]:
    """Exact energy and CI matrix C[alpha_string, beta_string] in pyscf string order."""
    energy, civec = pyscf.fci.direct_spin1.kernel(
        data.one_body_integrals, data.two_body_integrals, data.norb, data.nelec,
        ecore=data.core_energy, conv_tol=1e-10, max_cycle=200,
    )
    return float(energy), np.asarray(civec)


def best_fci_subspace(data, civec: np.ndarray, d: int) -> tuple[float, int]:
    """Top-d spin-symmetric marginal weights of the exact wavefunction."""
    return subspace_energy(data, best_fci_strings(data, civec, d))


def hci_wavefunction(data, select_cutoff: float = 1e-4):
    """Selected CI (heat-bath criterion on Hamiltonian matrix elements) in pyscf."""
    solver = selected_ci.SCI()
    solver.select_cutoff = select_cutoff
    solver.ci_coeff_cutoff = select_cutoff
    solver.conv_tol = 1e-8
    energy, civec = solver.kernel(data.one_body_integrals, data.two_body_integrals, data.norb, data.nelec,
                                  ecore=data.core_energy)
    return float(energy), civec


def best_fci_strings(data, civec: np.ndarray, d: int) -> np.ndarray:
    strs = cistring.make_strings(range(data.norb), data.nelec[0])
    weights = (np.abs(civec) ** 2).sum(axis=1) + (np.abs(civec) ** 2).sum(axis=0)
    return _top_strings(weights, strs, d)


def hci_strings(civec, d: int) -> np.ndarray:
    """Top-d spin-symmetric marginal weights of the selected-CI wavefunction.

    Uses only the selected-CI vector, which pyscf builds from the integrals; no FCI input.
    """
    strs_a, strs_b = civec._strs
    matrix = np.asarray(civec)
    weight_of = {}
    for s, w in zip(strs_a, (np.abs(matrix) ** 2).sum(axis=1)):
        weight_of[int(s)] = weight_of.get(int(s), 0.0) + w
    for s, w in zip(strs_b, (np.abs(matrix) ** 2).sum(axis=0)):
        weight_of[int(s)] = weight_of.get(int(s), 0.0) + w
    strings = np.array(list(weight_of), dtype=np.int64)
    weights = np.array([weight_of[int(s)] for s in strings])
    return _top_strings(weights, strings, d)


def hci_subspace(data, civec, d: int) -> tuple[float, int]:
    return subspace_energy(data, hci_strings(civec, d))


def subspace_comparison(data, hci_civec, fci_civec: np.ndarray, d: int) -> dict:
    """HCI and best-FCI subspaces at size d, and how many of their d strings coincide."""
    hci, best = hci_strings(hci_civec, d), best_fci_strings(data, fci_civec, d)
    return {"e_hci": subspace_energy(data, hci)[0], "e_best_fci": subspace_energy(data, best)[0],
            "overlap": int(len(set(map(int, hci)) & set(map(int, best))))}
