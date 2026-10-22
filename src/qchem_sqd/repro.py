"""Reproducibility: pinned thread counts and fixed seeds.

Call ``pin_threads`` at the very top of a script, before numpy, scipy or pyscf
are imported, so that BLAS/OpenMP thread pools start with the pinned size.
L-BFGS-B (used by ffsim's optimize=True) is deterministic for a fixed thread
count but BLAS reductions can differ between thread counts.
"""

import os

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

SEEDS = {
    "transpiler": 1234,
    "sampler": 7,
    "sqd": 24,
    "random_bitstrings": 101,
    "aer": 11,
}


def pin_threads(n: int) -> None:
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)
    try:  # also covers pools already started by an earlier import
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except ImportError:
        pass
    try:
        from pyscf import lib

        lib.num_threads(n)
    except ImportError:
        pass


def thread_info() -> dict:
    info = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    try:
        from threadpoolctl import threadpool_info

        info["pools"] = [{"api": p["internal_api"], "threads": p["num_threads"]} for p in threadpool_info()]
    except ImportError:
        pass
    return info
