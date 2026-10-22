"""First-order estimate of the valid electron-count fraction under device noise.

The LUCJ circuit conserves alpha and beta particle numbers, so a single X- or
Y-type error (or a readout flip) on a data qubit changes a Hamming weight and
the shot fails postselection; Z-type errors do not. To first order

    valid ~= prod_i (1 - lambda_i)

over error events i, with lambda from the device calibration:
  2q gate, error per gate e:  depolarizing p = 4e/3, P(X/Y on either qubit) = 12/16 p = e
  1q gate (sx, x), error e:   depolarizing p = 2e,   P(X/Y) = 2/4 p = e
  measurement, error r:       lambda = r
  idle amplitude damping:     lambda = occupation * (1 - exp(-t_idle / T1))   (optional)

Two error compensating each other, and errors on auxiliary qubits that never
reach a data qubit, are ignored, so the estimate is slightly pessimistic for
heavy-hex circuits with auxiliary qubits. ``aer_equivalent=True`` drops the idle
term, matching Aer's from_backend noise model for validation.
"""

import numpy as np
from qiskit import QuantumCircuit


def _error(target, name: str, qargs: tuple) -> float:
    try:
        props = target[name][qargs]
    except KeyError:
        return 0.0
    return float(props.error) if props is not None and props.error is not None else 0.0


def estimate_valid_fraction(isa: QuantumCircuit, backend, occupation: float = 0.5,
                            aer_equivalent: bool = False) -> dict:
    target = backend.target
    log_valid, counts = 0.0, {"2q": 0, "1q": 0, "measure": 0}
    busy_until = {}
    for inst in isa.data:
        name = inst.operation.name
        qargs = tuple(isa.find_bit(q).index for q in inst.qubits)
        if name in ("barrier", "delay", "rz", "id"):
            continue
        e = _error(target, name, qargs)
        if name == "measure":
            counts["measure"] += 1
        elif len(qargs) == 2:
            counts["2q"] += 1
        else:
            counts["1q"] += 1
        log_valid += np.log1p(-min(e, 0.999))
        busy_until.update({q: None for q in qargs})
    if not aer_equivalent:
        duration = float(isa.estimate_duration(target, unit="s"))
        for q in busy_until:
            t1 = getattr(backend.qubit_properties(q), "t1", None)
            if t1:
                log_valid += np.log1p(-occupation * (1 - np.exp(-duration / t1)))
    return {"valid_fraction": float(np.exp(log_valid)), **counts}
