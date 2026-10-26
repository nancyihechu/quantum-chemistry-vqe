"""Pre-registered 5-repeat analysis (docs/preregistration.md): H1 to H6, equivalence tests (Addendum B1),
day effects, and the shot-scaling curves (Addenda B2 and C).

Reads results only; every analysis setting was fixed by the pre-registration.
  32 qubits: <run dir>/analysis_day<k>.json from analyze_sqd.py, one per study day.
  52 qubits: results/ccpvdz/sqd_*.json from analyze_ccpvdz.py.

Rules (per comparison, per geometry, per cap; paired difference = first minus second arm, per repeat):
  superiority  mean has the predicted sign (negative) and |mean| > 2 SE
  equivalence  90% CI (mean +/- t(0.95, n-1) SE) inside (-delta, +delta); delta 10 mHa (32q), 5 mHa (52q)
  verdict      supported / equivalent / opposite at a majority of caps (32q: 3 of 5; 52q: both), else inconclusive
H6 compares arm repeats with a single baseline run: difference = arm minus baseline per repeat, same rules.

Usage: python scripts/analyze_preregistered.py [--out results/hardware/analysis_preregistered.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "results" / "ccpvdz"
DAYS = {1: ROOT / "results/hardware/study_20261022T2126Z", 2: ROOT / "results/hardware/study_20261024T1824Z"}
SHOTSCALE = ROOT / "results/hardware/shotscale_20261024T1909Z"
KINGSTONSCALE = ROOT / "results/hardware/kingstonscale_20261024T1910Z"
CAPS_32 = ("30", "60", "100", "150", "200")
CAPS_52 = (1200, 2000)
GRIDS = ("1.0", "1.5", "2.2")
GRID_52_SHOTS = {"E": 1_040_000, "F": 1_040_000, "G": 100_000}
ARMS_52 = {"E": "phoenix52_square", "F": "phoenix52_hhpattern", "G": "kingston52_heavyhex"}
JOB_52 = {"E": "phoenix52", "F": "phoenix52", "G": "kingston52"}
RANDOM_VALID_32 = 0.444  # percent, uniform random 32-qubit strings


def rep_of(job: str) -> int:
    return int(job.split("_rep")[1].split("_")[0])


def day_of(rep: int) -> int:
    return 1 if rep <= 3 else 2


# ---------- tests ----------

def paired(diffs: dict[int, float], delta: float) -> dict:
    """diffs: repeat -> difference (first minus second). Predicted sign: negative."""
    reps = sorted(diffs)
    v = np.array([diffs[r] for r in reps], dtype=float)
    n = len(v)
    if n < 2:
        return {"n": n, "mean": float(v.mean()) if n else None}
    mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.95, n - 1) * se)
    day = {k: [diffs[r] for r in reps if day_of(r) == k] for k in (1, 2)}
    return {
        "n": n, "reps": reps, "diffs": v.tolist(), "mean": mean, "se": se,
        "ci90": [mean - half, mean + half],
        "superior": mean < 0 and abs(mean) > 2 * se,
        "opposite": mean > 0 and abs(mean) > 2 * se,
        "equivalent": -delta < mean - half and mean + half < delta,
        "day1_minus_day2": float(np.mean(day[1]) - np.mean(day[2])) if day[1] and day[2] else None,
    }


def verdict(per_cap: list[dict], need: int) -> str:
    if any(c.get("n", 0) < 2 for c in per_cap):
        return "incomplete"
    for key, word in (("superior", "supported"), ("opposite", "opposite"), ("equivalent", "equivalent")):
        if sum(c[key] for c in per_cap) >= need:
            return word
    return "inconclusive"


def fmt_test(t: dict) -> str:
    if t.get("n", 0) < 2:
        return f"n={t.get('n', 0)} (incomplete)"
    flags = "S" if t["superior"] else ("O" if t["opposite"] else "-")
    flags += "E" if t["equivalent"] else "-"
    return f"{t['mean']:+7.1f} +/- {t['se']:4.1f}  CI90 [{t['ci90'][0]:+6.1f}, {t['ci90'][1]:+6.1f}]  {flags}  n={t['n']}"


def msd(v) -> str:
    v = np.array(v, dtype=float)
    if len(v) == 0:
        return "n/a"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():7.1f} +/- {sd:4.1f}"


# ---------- 32 qubits ----------

def load_32() -> dict:
    """R -> arm -> rep -> run; R -> 'random' -> list of runs (seeds)."""
    out: dict = {}
    for day, run_dir in DAYS.items():
        path = run_dir / f"analysis_day{day}.json"
        if not path.exists():
            print(f"(missing {path.relative_to(ROOT)})")
            continue
        for r, block in json.loads(path.read_text()).items():
            slot = out.setdefault(r, {"arms": {}, "random": {}, "best_fci": block["baselines"].get("best_fci")})
            for arm, runs in block["arms"].items():
                for run in runs:
                    slot["arms"].setdefault(arm, {})[rep_of(run["job"])] = run
            slot["random"][day] = block["baselines"]["random"]
    return out


def final_32(run, d):
    return run["caps"][d]["error_mha_final"]


def compare_32(data, r, a, b, delta=10.0):
    arms = data[r]["arms"]
    if a not in arms or b not in arms:
        return None
    common = sorted(set(arms[a]) & set(arms[b]))
    per_cap = [paired({k: final_32(arms[a][k], d) - final_32(arms[b][k], d) for k in common}, delta) for d in CAPS_32]
    return {"caps": dict(zip(CAPS_32, per_cap)), "verdict": verdict(per_cap, 3)}


def section_32(data, result) -> None:
    print("\n" + "=" * 100)
    print("32 QUBITS (N2/6-31G, 16o): final SQD error vs FCI, mHa")
    print("Paired difference = first arm minus second, per repeat. Flags: S superiority, O opposite sign > 2 SE, E TOST equivalence (delta 10 mHa).")
    print("Verdict needs 3 of 5 caps.")

    print("\nPer-arm final error, mean +/- SD over repeats")
    print(f"{'R (A)':6s} {'arm':22s} {'n':>2s} " + " ".join(f"{'d=' + d:>16s}" for d in CAPS_32))
    for r in sorted(data, key=float):
        for arm, reps in data[r]["arms"].items():
            row = " ".join(f"{msd([final_32(run, d) for run in reps.values()]):>16s}" for d in CAPS_32)
            print(f"{r:6s} {arm:22s} {len(reps):2d} {row}")
        rnd = data[r]["random"].get(1, [])
        print(f"{r:6s} {'random (3 seeds)':22s} {len(rnd):2d} " + " ".join(f"{msd([x['caps'][d]['error_mha_final'] for x in rnd]):>16s}" for d in CAPS_32))

    comparisons = [
        ("H1 (primary)", "phoenix_square", "phoenix_hhpattern", ("1.0", "1.2")),
        ("Exploratory A vs B", "phoenix_square", "phoenix_hhpattern", ("1.8", "2.4")),
        ("Exploratory B vs D (device)", "phoenix_hhpattern", "kingston_heavyhex", ("1.0", "1.2", "1.8", "2.4")),
        ("Exploratory C vs A (optimized)", "phoenix_square_opt", "phoenix_square", ("1.0", "1.2")),
    ]
    result["32q"] = {}
    for title, a, b, rs in comparisons:
        print(f"\n{title}: {a} minus {b}")
        print(f"{'R (A)':6s} {'cap':>5s}  {'mean +/- SE':>17s}  {'90% CI':>17s}  flags  day1-day2")
        for r in rs:
            c = compare_32(data, r, a, b)
            if c is None:
                continue
            result["32q"][f"{title} R={r}"] = c
            for d, t in c["caps"].items():
                dd = f"{t['day1_minus_day2']:+6.1f}" if t.get("day1_minus_day2") is not None else "  n/a"
                print(f"{r:6s} {d:>5s}  {fmt_test(t)}  {dd}")
            print(f"{r:6s} verdict: {c['verdict']}")

    print("\nH2: valid fraction (percent, threshold 10 x 0.444 = 4.44) and before-recovery error vs random at d = 30 and 60")
    print(f"{'R (A)':6s} {'arm':22s} {'valid % min..max':>17s} {'>=4.44 all':>10s}  {'before d=30 arm / random':>26s}  {'before d=60 arm / random':>26s}  pass")
    h2 = {}
    for r in sorted(data, key=float):
        rnd = data[r]["random"].get(1, [])
        rnd_pre = {d: np.mean([x["caps"][d]["error_mha_by_round"][0] for x in rnd]) for d in ("30", "60")}
        for arm, reps in data[r]["arms"].items():
            vf = [100 * run["valid_fraction"] for run in reps.values()]
            pre = {d: [run["caps"][d]["error_mha_by_round"][0] for run in reps.values()] for d in ("30", "60")}
            ok_vf = min(vf) >= 10 * RANDOM_VALID_32
            ok_pre = all(max(pre[d]) < rnd_pre[d] for d in ("30", "60"))
            h2[f"{r} {arm}"] = {"valid_min": min(vf), "valid_max": max(vf), "pre_max": {d: max(pre[d]) for d in pre},
                                "random_pre": rnd_pre, "pass": ok_vf and ok_pre}
            print(f"{r:6s} {arm:22s} {min(vf):7.2f}..{max(vf):6.2f}  {str(ok_vf):>10s}  "
                  + "  ".join(f"{np.mean(pre[d]):11.1f} / {rnd_pre[d]:11.1f}" for d in ("30", "60"))
                  + f"  {ok_pre and ok_vf}")
    result["32q"]["H2"] = h2
    print("(pass: every repeat's valid fraction >= 4.44% and every repeat's before-recovery error below the random mean at both caps)")


# ---------- 52 qubits ----------

def load_json(name):
    path = CC / name
    return json.loads(path.read_text()) if path.exists() else None


def arm_runs(key, g, d, suffix=""):
    """rep -> run for arm key E/F/G (suffix e.g. '_sub100000' or '_vsub185')."""
    out = {}
    for path in sorted(CC.glob(f"sqd_{JOB_52[key]}_rep*__{ARMS_52[key]}_G{g}{suffix}_R{g}_d{d}.json")):
        run = json.loads(path.read_text())
        out[rep_of(path.name.split("__")[0])] = run
    return out


def section_52(result) -> None:
    print("\n" + "=" * 100)
    print("52 QUBITS (N2/cc-pVDZ, 10e 26o): final SQD error vs repository HCI, mHa. d = 1.44M (1200^2) and 4M (2000^2)")
    print("E phoenix square, F phoenix heavy-hex pattern, G kingston heavy-hex, T torino (published shots, our pipeline).")
    print("Flags: S superiority, O opposite sign > 2 SE, E TOST equivalence (delta 5 mHa). Verdict needs both caps.")
    res = result.setdefault("52q", {})

    print("\nPer-arm results, mean +/- SD over repeats")
    print(f"{'grid':4s} {'d':>5s} {'arm':24s} {'n':>2s} {'shots':>9s} {'valid %':>8s} {'before':>16s} {'final':>16s}")
    for g in GRIDS:
        for d in CAPS_52:
            rows = [(f"{k} {ARMS_52[k]}", arm_runs(k, g, d)) for k in "EFG"]
            rows.append(("E sub 1e5", arm_runs("E", g, d, "_sub100000")))
            for label, runs in rows:
                v = list(runs.values())
                if not v:
                    continue
                print(f"{g:4s} {d:5d} {label:24s} {len(v):2d} {np.mean([x['shots'] for x in v]):9.0f} "
                      f"{100 * np.mean([x['valid_fraction'] for x in v]):8.3f} "
                      f"{msd([x['error_vs_hci_by_round_mha'][0] for x in v]):>16s} {msd([x['error_vs_hci_mha'] for x in v]):>16s}")
            for ref in ("torino", "random_hfinit_seed101", "random_particle_seed101"):
                x = load_json(f"sqd_{ref}_R{g}_d{d}.json")
                if x:
                    print(f"{g:4s} {d:5d} {ref:24s} {1:2d} {x['shots']:9d} {100 * x['valid_fraction']:8.3f} "
                          f"{x['error_vs_hci_by_round_mha'][0]:16.1f} {x['error_vs_hci_mha']:16.1f}")

    def run_pair(title, first, second, grids):
        print(f"\n{title}")
        print(f"{'grid':4s} {'d':>5s}  {'mean +/- SE':>17s}  {'90% CI':>17s}  flags  day1-day2")
        out = {}
        for g in grids:
            per_cap = []
            for d in CAPS_52:
                a, b = first(g, d), second(g, d)  # b: rep -> run (paired arms), one baseline run, or None
                if not b:
                    diffs = {}
                elif "error_vs_hci_mha" in b:
                    diffs = {k: run["error_vs_hci_mha"] - b["error_vs_hci_mha"] for k, run in a.items()}
                else:
                    diffs = {k: a[k]["error_vs_hci_mha"] - b[k]["error_vs_hci_mha"] for k in sorted(set(a) & set(b))}
                t = paired(diffs, 5.0)
                per_cap.append(t)
                dd = f"{t['day1_minus_day2']:+6.1f}" if t.get("day1_minus_day2") is not None else "  n/a"
                print(f"{g:4s} {d:5d}  {fmt_test(t)}  {dd}")
            v = verdict(per_cap, 2)
            out[g] = {"caps": dict(zip(map(str, CAPS_52), per_cap)), "verdict": v}
            print(f"{g:4s} verdict: {v}")
        res[title] = out

    run_pair("H3 (primary, matched QPU time): E minus G", lambda g, d: arm_runs("E", g, d), lambda g, d: arm_runs("G", g, d), GRIDS)
    run_pair("H4 (primary, matched shots): E subsampled to 1e5 minus G", lambda g, d: arm_runs("E", g, d, "_sub100000"),
             lambda g, d: arm_runs("G", g, d), GRIDS)
    run_pair("H5: E minus F (grid 1.0 tested; 1.5 and 2.2 exploratory)", lambda g, d: arm_runs("E", g, d), lambda g, d: arm_runs("F", g, d), GRIDS)

    print("\nH6: each arm minus a baseline run at the same d and shot count (random) or the same valid-shot count (T).")
    for k in "EFG":
        n = GRID_52_SHOTS[k]
        for rnd in ("random_hfinit", "random_particle"):
            run_pair(f"H6 {k} minus {rnd} ({n:,} shots)", lambda g, d, k=k: arm_runs(k, g, d),
                     lambda g, d, rnd=rnd, n=n: load_json(f"sqd_{rnd}_seed101_n{n}_R{g}_d{d}.json"), GRIDS)
        nvalid = {g: round(load_json(f"sqd_torino_R{g}_d1200.json")["shots"] * load_json(f"sqd_torino_R{g}_d1200.json")["valid_fraction"]) for g in GRIDS}
        run_pair(f"H6 {k} (subsampled to T's valid shots) minus T", lambda g, d, k=k: arm_runs(k, g, d, f"_vsub{nvalid[g]}"),
                 lambda g, d: load_json(f"sqd_torino_R{g}_d{d}.json"), GRIDS)

    print("\nValid shots per QPU second (IBM-reported usage per circuit = job usage / circuits in job)")
    print(f"{'grid':4s} {'arm':22s} {'valid shots':>12s} {'s per circuit':>14s} {'valid per QPU s':>16s}")
    usage = {}
    for run_dir in DAYS.values():
        for job in json.loads((run_dir / "manifest.json").read_text())["jobs"]:
            usage[job["name"]] = job["usage_seconds"] / len(job["pubs"])
    vps = {}
    for g in GRIDS:
        for k in "EFG":
            runs = arm_runs(k, g, 1200)
            vals = [(x["shots"] * x["valid_fraction"], usage[f"{JOB_52[k]}_rep{r}"]) for r, x in runs.items()]
            if vals:
                v, s = np.mean([a for a, _ in vals]), np.mean([b for _, b in vals])
                vps[f"{g} {k}"] = v / s
                print(f"{g:4s} {k + ' ' + ARMS_52[k]:22s} {v:12.0f} {s:14.1f} {v / s:16.1f}")
    res["valid_per_qpu_s"] = vps


# ---------- shot scaling ----------

def section_scaling(result) -> None:
    print("\n" + "=" * 100)
    print("SHOT SCALING (exploratory, Addenda B2 and C): error vs HCI, mHa; mean +/- SD over subsample seeds 7, 8, 9")
    curves = [("phoenix E (pooled, 10M)", f"pooled_phoenix52_square_G{{g}}", ("1.0", "2.2"), (100_000, 300_000, 1_000_000, 3_000_000), 10_000_000),
              ("kingston G (1M)", "kingstonscale_kingston52__kingston52_heavyhex_G{g}", ("2.2",), (100_000, 300_000), 1_000_000)]
    out = {}
    for title, stem, grids, sizes, full in curves:
        for g in grids:
            print(f"\n{title}, grid {g}")
            print(f"{'shots':>10s} {'d':>5s} {'n':>2s} {'valid shots':>12s} {'before':>16s} {'final':>16s}")
            for d in CAPS_52:
                for n in sizes + (full,):
                    names = ([f"sqd_{stem.format(g=g)}_sub{n}{'' if s == 7 else f'_s{s}'}_R{g}_d{d}.json" for s in (7, 8, 9)]
                             if n != full else [f"sqd_{stem.format(g=g)}_R{g}_d{d}.json"])
                    runs = [x for x in map(load_json, names) if x]
                    if not runs:
                        print(f"{n:10,d} {d:5d}  0 (pending)")
                        continue
                    pre = [x["error_vs_hci_by_round_mha"][0] for x in runs]
                    fin = [x["error_vs_hci_mha"] for x in runs]
                    valid = np.mean([x["shots"] * x["valid_fraction"] for x in runs])
                    out[f"{title} G{g} n{n} d{d}"] = {"n_runs": len(runs), "before": pre, "final": fin, "valid_shots": valid}
                    print(f"{n:10,d} {d:5d} {len(runs):2d} {valid:12.0f} {msd(pre):>16s} {msd(fin):>16s}")
    result["shot_scaling"] = out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results/hardware/analysis_preregistered.json"))
    args = parser.parse_args()
    result: dict = {}
    section_32(load_32(), result)
    section_52(result)
    section_scaling(result)
    Path(args.out).write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
