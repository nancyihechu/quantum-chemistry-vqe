"""Sample-based quantum diagonalization: recover energies from measured bitstrings."""

import ffsim
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
) -> dict:
    """Configuration recovery + subspace diagonalization; returns total energies.

    ``max_dim`` caps each spin sector (subspace <= max_dim**2), keeping carried-over
    strings first and then the most frequently sampled ones.
    """
    history = []

    def record_iteration(results):
        history.append(min(r.energy for r in results) + data.core_energy)

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
        callback=record_iteration,
        seed=seed,
    )
    return {
        "e_sqd": float(result.energy + data.core_energy),
        "subspace_dim": int(result.sci_state.amplitudes.size),
        "energy_history": history,
        "samples_per_batch": samples_per_batch,
        "num_batches": num_batches,
        "max_dim": max_dim,
    }


def valid_fraction(bit_array: BitArray, norb: int, nelec: tuple[int, int]) -> float:
    """Fraction of shots with the correct alpha and beta electron counts (a noise diagnostic)."""
    bits = bit_array.to_bool_array(order="little")
    n_alpha = bits[:, :norb].sum(axis=1)
    n_beta = bits[:, norb : 2 * norb].sum(axis=1)
    return float(((n_alpha == nelec[0]) & (n_beta == nelec[1])).mean())
