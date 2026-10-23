"""Mean and spread across study repeats for one or more arms, next to the pre-registered references.

Reads results only; no analysis setting is chosen here.
  32 qubits: <run dir>/<analysis file> from analyze_sqd.py (final error vs FCI, mHa, per cap d)
  52 qubits: results/ccpvdz/sqd_<job>__<arm>_G<g>_R<g>_d<max_dim>.json from analyze_ccpvdz.py
             (final error vs the repository HCI energy, mHa), with torino and both random
             baselines through the same pipeline.
Spread = sample standard deviation over repeats (n - 1); range = min to max.

Usage: python scripts/summarize_repeats.py results/hardware/study_20261022T2126Z \
           --analysis analysis_kingston_day1_interim.json --arms kingston_heavyhex kingston52_heavyhex
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPS_32 = ("30", "60", "100", "150", "200")


def stats(values) -> str:
    v = np.array(values, dtype=float)
    if len(v) == 0:
        return "n/a"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():7.1f} +/- {sd:5.1f} [{v.min():.1f}, {v.max():.1f}] n={len(v)}"


def summary_32(run_dir: Path, analysis_file: str, arms) -> None:
    analysis = json.loads((run_dir / analysis_file).read_text())
    print("== 32 qubits (N2/6-31G): final error vs FCI, mHa; mean +/- SD [min, max] over repeats ==")
    for r, block in sorted(analysis.items(), key=lambda kv: float(kv[0])):
        print(f"\nR = {r} A")
        sources = {a: runs for a, runs in block["arms"].items() if a in arms}
        sources["random (3 seeds)"] = block["baselines"]["random"]
        for name, runs in sources.items():
            vf = [100 * run["valid_fraction"] for run in runs]
            print(f"  {name:20s} valid {np.mean(vf):6.2f}%")
            for d in CAPS_32:
                pre = [run["caps"][d]["error_mha_by_round"][0] for run in runs]
                fin = [run["caps"][d]["error_mha_final"] for run in runs]
                print(f"    d={d:>3s}  before recovery {stats(pre)}   final {stats(fin)}")
        b = block["baselines"]
        if "hci" in b:
            print("  HCI / best FCI      " + "  ".join(f"d={d}: {b['hci'][d]:.1f} / {b['best_fci'][d]:.1f}" for d in CAPS_32))


def summary_52(arms) -> None:
    print("\n== 52 qubits (N2/cc-pVDZ): final error vs HCI, mHa; mean +/- SD [min, max] over repeats ==")
    out = ROOT / "results" / "ccpvdz"
    for g in ("1.0", "1.5", "2.2"):
        print(f"\ngrid {g}")
        for d in (1200, 2000):
            rows = {}
            for arm in arms:
                files = sorted(glob.glob(str(out / f"sqd_*rep*__{arm}_G{g}_R{g}_d{d}.json")))
                rows[arm] = [json.loads(Path(f).read_text()) for f in files]
            for ref in ("torino", "random_hfinit_seed101", "random_particle_seed101"):
                f = out / f"sqd_{ref}_R{g}_d{d}.json"
                rows[ref] = [json.loads(f.read_text())] if f.exists() else []
            print(f"  d = {d}^2")
            for name, runs in rows.items():
                if not runs:
                    continue
                pre = [r["error_vs_hci_by_round_mha"][0] for r in runs]
                fin = [r["error_vs_hci_mha"] for r in runs]
                vf = np.mean([100 * r["valid_fraction"] for r in runs])
                print(f"    {name:24s} valid {vf:7.4f}%  before {stats(pre)}   final {stats(fin)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--analysis", default="analysis.json")
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    summary_32(Path(args.run_dir), args.analysis, set(args.arms))
    summary_52([a for a in args.arms if "52" in a])


if __name__ == "__main__":
    main()
