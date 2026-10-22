"""Convert the published ibm_torino N2/cc-pVDZ counts (arXiv:2405.05068 data repository) safely.

The repository ships counts as a pickled numpy object array. Instead of
np.load(allow_pickle=True), which would execute arbitrary code in the file, this
uses a restricted unpickler that only permits numpy's array reconstruction
helpers; anything else raises. Output: one .npz per bond length with plain
arrays (bitstrings as uint8 rows, counts), which never need pickle again.

Grouping follows the repository's load_data_dict.py: 24 bond lengths 0.7..3.0
(units of 1.0975135 A), consecutive len(data)//24 dictionaries per bond length.
Each dictionary holds 8192 distinct bitstrings with normalized weights of about
1/8192 each (one shot per bitstring), so weights are stored as floats and the
shot count is the number of bitstrings.

Usage: python -I scripts/convert_torino_counts.py <npy file> <output dir>
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module.replace("numpy.core", "numpy._core"), name)


def load_object_npy(path: Path):
    with open(path, "rb") as fh:
        version = npformat.read_magic(fh)
        read_header = npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0
        shape, fortran, dtype = read_header(fh)
        if dtype != np.dtype(object):
            raise ValueError(f"expected an object array, got {dtype}")
        return RestrictedUnpickler(fh).load(), shape


def main(src: str, out_dir: str) -> None:
    data, shape = load_object_npy(Path(src))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_bonds = 24
    per_bond = len(data) // n_bonds
    bond_lengths = np.round(np.arange(0.70, 3.01, 0.10), 1)
    for i, r in enumerate(bond_lengths):
        merged: dict[str, float] = {}
        for counts in data[i * per_bond:(i + 1) * per_bond]:
            if not isinstance(counts, dict):
                raise TypeError(f"element is {type(counts).__name__}, not dict")
            for key, value in counts.items():
                if not isinstance(key, str) or set(key) - {"0", "1"}:
                    raise ValueError(f"unexpected key {key!r}")
                merged[key] = merged.get(key, 0.0) + float(value)
        keys = list(merged)
        bits = np.frombuffer("".join(keys).encode(), dtype=np.uint8).reshape(len(keys), -1) - ord("0")
        weights = np.array([merged[k] for k in keys])
        spread = (weights.max() - weights.min()) / weights.mean()
        if spread > 0.1:  # a repeated bitstring would show up as about 100%
            raise ValueError(f"R={r:.1f}: weights are not near-uniform (relative spread {spread:.1e})")
        np.savez_compressed(out / f"torino_R{r:.1f}.npz", bitstrings=bits.astype(np.uint8), weights=weights)
        print(f"R={r:.1f}: {len(keys)} shots (distinct bitstrings), {bits.shape[1]} bits, "
              f"{per_bond} dictionaries, weight spread {spread:.1e}")
    print(f"array shape {shape}, {len(data)} dictionaries")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
