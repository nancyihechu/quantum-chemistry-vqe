"""Sample-based quantum diagonalization: recover energies from measured bitstrings."""

from math import comb

import ffsim
import numpy as np
from pyscf import ao2mo
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian


def run_sqd(
    data: ffsim.MolecularData,
    bit_array: BitArray,
    samples_per_batch: int = 300,
    num_batches: int = 3,
    max_iterations: int = 5,
    seed: int = 24,
    max_dim: int | None = None,
    initial_occupancies: tuple | None = None,
) -> dict:
    """Configuration recovery + subspace diagonalization; returns total energies.

    ``max_dim`` caps each spin sector (subspace <= max_dim**2), keeping carried-over
    strings first and then the most frequently sampled ones. Iteration 1 only
    postselects bitstrings with the right electron counts (before recovery);
    each later iteration is one configuration-recovery round. ``energy_history``
    and ``dim_history`` hold the best batch of every iteration.
    """
    history, dims = [], []

    def record_iteration(results):
        best = min(results, key=lambda r: r.energy)
        history.append(best.energy + data.core_energy)
        dims.append(int(best.sci_state.amplitudes.size))

    result = diagonalize_fermionic_hamiltonian(
        data.one_body_integrals,
        ao2mo.restore(1, data.two_body_integrals, data.norb),  # unpack to full (norb,)*4
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=data.norb,
        nelec=data.nelec,
        num_batches=num_batches,
        max_iterations=max_iterations,
        symmetrize_spin=True,
        max_dim=max_dim,
        initial_occupancies=initial_occupancies,
        callback=record_iteration,
        seed=seed,
    )
    return {
        "e_sqd": float(result.energy + data.core_energy),
        "subspace_dim": int(result.sci_state.amplitudes.size),
        "energy_history": history,
        "dim_history": dims,
        "samples_per_batch": samples_per_batch,
        "num_batches": num_batches,
        "max_dim": max_dim,
    }


def random_valid_fraction(norb: int, nelec: tuple[int, int]) -> float:
    """Valid fraction for uniform random bitstrings: C(norb, n_a) C(norb, n_b) / 2^(2 norb).

    0.444% for N2/6-31G with 16 orbitals and (5, 5) electrons.
    """
    return comb(norb, nelec[0]) * comb(norb, nelec[1]) / 2 ** (2 * norb)


def valid_fraction(bit_array: BitArray, norb: int, nelec: tuple[int, int]) -> float:
    """Fraction of shots with the correct alpha and beta electron counts (a noise diagnostic)."""
    return float(valid_mask(bit_array, norb, nelec).mean())


def valid_mask(bit_array: BitArray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Per-shot flag: correct alpha and beta electron counts."""
    bits = bit_array.to_bool_array(order="little")
    n_alpha = bits[:, :norb].sum(axis=1)
    n_beta = bits[:, norb : 2 * norb].sum(axis=1)
    return (n_alpha == nelec[0]) & (n_beta == nelec[1])
