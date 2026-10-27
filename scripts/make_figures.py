"""Paper figures from the pre-registered analysis (results/hardware/analysis_preregistered.json) and result files.

Fig 1  52 qubits: valid fraction and before-recovery error for T, G, E, F by geometry, uniform random marked.
Fig 2  paired differences A minus B (32 qubits) and E minus F (52 qubits) against d, 90% CI, equivalence bands.
Fig 3  52 qubits: final and before-recovery error against shot count, phoenix E and kingston G, random_hfinit lines.
Fig 4  32 qubits: final error against d for A, B, D and baselines at 1.0 and 2.4 A;
       52 qubits: arm minus random_hfinit and arm minus T (matched valid shots) by geometry.

Colors follow the hardware arm in every figure (validated colorblind-safe as a set, all pairs):
square pattern on phoenix (A, E) blue, heavy-hex pattern on phoenix (B, F) orange, kingston heavy-hex (D, G) aqua,
torino T dark ink. Marker shape repeats the arm identity; baselines are neutral gray or black.

Usage: python scripts/make_figures.py [--out results/figures]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_preregistered import CAPS_32, CAPS_52, GRIDS, arm_runs, load_32, load_json  # noqa: E402

COLUMN_IN = 3.5  # IEEE one-column width
ANGSTROM_52 = {"1.0": 1.098, "1.5": 1.646, "2.2": 2.415}
INK, MUTED, GRIDLINE = "#0b0b0b", "#898781", "#e1e0d9"
STYLE = {  # arm -> (label, color, marker)
    "A": ("A phoenix, square", "#2a78d6", "s"),
    "B": ("B phoenix, heavy-hex pattern", "#eb6834", "h"),
    "D": ("D kingston, heavy-hex", "#1baf7a", "^"),
    "E": ("E phoenix, square", "#2a78d6", "s"),
    "F": ("F phoenix, heavy-hex pattern", "#eb6834", "h"),
    "G": ("G kingston, heavy-hex", "#1baf7a", "^"),
    "T": ("T torino (published shots)", "#3a3a38", "D"),
}
ARM_32 = {"A": "phoenix_square", "B": "phoenix_hhpattern", "D": "kingston_heavyhex"}
GEOM_INK = {"1.0": "#0b0b0b", "1.2": "#52514e", "1.8": "#898781", "2.4": "#b5b3ab"}  # one neutral ramp, dark to light
GEOM_MARK = {"1.0": "o", "1.2": "s", "1.8": "^", "2.4": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e", "axes.labelcolor": INK,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 1.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"written {out / name}.pdf/.png")


def panel_label(ax, text: str) -> None:
    ax.text(0.01, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontweight="bold")


def geom_axis(ax) -> None:
    ax.set_xticks(range(len(GRIDS)), [f"{ANGSTROM_52[g]:.3f}" for g in GRIDS])
    ax.set_xlim(-0.5, len(GRIDS) - 0.5)
    ax.grid(False, axis="x")


def ci(t: dict):
    return np.array([[t["mean"] - t["ci90"][0]], [t["ci90"][1] - t["mean"]]])


# ---------- Fig 1 ----------

def fig1(out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_IN, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    offsets = dict(zip("TGEF", (-0.27, -0.09, 0.09, 0.27)))
    rand = {g: {d: load_json(f"sqd_random_hfinit_seed101_R{g}_d{d}.json") for d in CAPS_52} for g in GRIDS}
    for i, g in enumerate(GRIDS):
        for k in "TGEF":
            label, color, marker = STYLE[k]
            for row, d in ((0, 1200), (1, 1200), (2, 2000)):
                runs = [load_json(f"sqd_torino_R{g}_d{d}.json")] if k == "T" else list(arm_runs(k, g, d).values())
                y = [100 * r["valid_fraction"] for r in runs] if row == 0 else [r["error_vs_hci_by_round_mha"][0] for r in runs]
                x = i + offsets[k]
                ax = axes[row]
                if len(y) > 1:
                    ax.scatter(np.full(len(y), x), y, s=6, color=color, alpha=0.45, linewidths=0, zorder=2)
                ax.plot([x], [np.mean(y)], marker=marker, color=color, markeredgecolor="white", markeredgewidth=0.5,
                        markersize=5.5, linestyle="none", zorder=3, label=label if (i == 0 and row == 0) else None)
        for row, d in ((1, 1200), (2, 2000)):
            axes[row].hlines(rand[g][d]["error_vs_hci_by_round_mha"][0], i - 0.42, i + 0.42, colors=MUTED, linestyles="--",
                             linewidth=1.0, zorder=1)
    vf = 100 * rand["1.0"][1200]["random_valid_fraction"]
    mant, expo = f"{vf:.1e}".split("e")
    axes[0].axhline(vf, color=MUTED, linestyle="--", linewidth=1.0, label=rf"uniform random (a: {mant}$\times$10$^{{{int(expo)}}}$%;")
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-5, 5)
    axes[0].set_ylabel("Valid fraction (%)")
    for row, d in ((1, "1.44M"), (2, "4M")):
        axes[row].set_yscale("log")
        axes[row].set_ylim(20, 5000)
        axes[row].set_ylabel(f"Before-recovery\nerror (mHa), d = {d}")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(STYLE[k][0]) for k in "TEGF"] + [len(l) - 1]
    h, l = [h[i] for i in order], [l[i] for i in order]
    l[-1] += " b, c: HF-seeded recovery)"
    axes[0].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=2, columnspacing=1.0, handletextpad=0.3)
    for ax, t in zip(axes, "abc"):
        panel_label(ax, t)
        geom_axis(ax)
    axes[-1].set_xlabel(r"N$_2$ bond length (Å), 52 qubits")
    save(fig, out, "fig1_valid_fraction_pre_recovery")


# ---------- Fig 2 ----------

def fig2(out: Path, res: dict) -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(COLUMN_IN, 4.0))
    keys = {"1.0": "H1 (primary) R=1.0", "1.2": "H1 (primary) R=1.2",
            "1.8": "Exploratory A vs B R=1.8", "2.4": "Exploratory A vs B R=2.4"}
    caps = np.array([int(c) for c in CAPS_32])
    a.axhspan(-10, 10, color=GRIDLINE, alpha=0.7, linewidth=0, zorder=0)
    a.axhline(0, color=MUTED, linewidth=0.6)
    clip = 30
    for j, (r, key) in enumerate(keys.items()):
        t = [res["32q"][key]["caps"][c] for c in CAPS_32]
        x = caps * (1 + 0.035 * (j - 1.5))
        primary = r in ("1.0", "1.2")
        mean = np.array([v["mean"] for v in t])
        lo, hi = np.array([v["ci90"][0] for v in t]), np.array([v["ci90"][1] for v in t])
        shown = np.clip(mean, -clip, clip)
        a.errorbar(x, shown, yerr=[shown - np.clip(lo, -clip, clip), np.clip(hi, -clip, clip) - shown], color=GEOM_INK[r],
                   marker=GEOM_MARK[r], markerfacecolor=GEOM_INK[r] if primary else "white", markeredgewidth=0.8,
                   linewidth=0.9, elinewidth=0.7, label=f"{r} Å" + ("" if primary else " (exploratory)"))
        for xi, c, m, l, h in zip(x, caps, mean, lo, hi):
            if abs(m) > clip:
                a.annotate(f"{r} Å, d = {c}: {m:+.1f} [{l:+.1f}, {h:+.1f}], off scale", (xi, np.sign(m) * clip),
                           xytext=(44, np.sign(m) * 22), textcoords="data", ha="left", va="center", fontsize=6,
                           color="#52514e", arrowprops={"arrowstyle": "-", "color": "#898781", "linewidth": 0.5})
    a.set_xscale("log")
    a.set_xticks(caps, [str(c) for c in caps])
    a.minorticks_off()
    a.set_ylim(-clip - 2, clip + 2)
    a.set_xlabel("Strings per spin sector, d (subspace d$^2$), 32 qubits")
    a.set_ylabel("A minus B, final error (mHa)")
    a.legend(loc="lower right", ncol=2, columnspacing=0.8, handletextpad=0.3)
    a.text(0.99, 0.98, "±10 mHa equivalence band", transform=a.transAxes, ha="right", va="top", fontsize=6.5, color="#52514e")

    key = "H5: E minus F (grid 1.0 tested; 1.5 and 2.2 exploratory)"
    b.axhspan(-5, 5, color=GRIDLINE, alpha=0.7, linewidth=0, zorder=0)
    b.axhline(0, color=MUTED, linewidth=0.6)
    ink = dict(zip(GRIDS, ("#0b0b0b", "#52514e", "#898781")))
    mark = dict(zip(GRIDS, ("o", "s", "^")))
    for j, g in enumerate(GRIDS):
        t = [res["52q"][key][g]["caps"][str(d)] for d in CAPS_52]
        x = np.arange(2) + 0.12 * (j - 1)
        primary = g == "1.0"
        b.errorbar(x, [v["mean"] for v in t], yerr=np.hstack([ci(v) for v in t]), color=ink[g], marker=mark[g],
                   markerfacecolor=ink[g] if primary else "white", markeredgewidth=0.8, linewidth=0.9, elinewidth=0.7,
                   label=f"{ANGSTROM_52[g]:.3f} Å" + ("" if primary else " (exploratory)"))
    b.set_xticks([0, 1], ["1200 (d$^2$ = 1.44M)", "2000 (d$^2$ = 4M)"])
    b.set_xlim(-0.5, 1.5)
    b.grid(False, axis="x")
    b.set_ylim(-9, 9)
    b.set_xlabel("Strings per spin sector, d, 52 qubits")
    b.set_ylabel("E minus F, final error (mHa)")
    b.legend(loc="lower left", ncol=1, handletextpad=0.3)
    b.text(0.99, 0.98, "±5 mHa equivalence band", transform=b.transAxes, ha="right", va="top", fontsize=6.5, color="#52514e")
    panel_label(a, "a")
    panel_label(b, "b")
    fig.tight_layout(h_pad=0.8)
    save(fig, out, "fig2_paired_differences")


# ---------- Fig 3 ----------

def fig3(out: Path, res: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_IN, 3.6), sharex=True)
    sc = res["shot_scaling"]
    curves = {"E": ("phoenix E (pooled, 10M)", (100_000, 300_000, 1_000_000, 3_000_000, 10_000_000)),
              "G": ("kingston G (1M)", (100_000, 300_000, 1_000_000))}
    for i, g in enumerate(("1.0", "2.2")):
        for j, d in enumerate(CAPS_52):
            ax = axes[i, j]
            for k, (title, sizes) in curves.items():
                _, color, marker = STYLE[k]
                pts = [(n, sc[f"{title} G{g} n{n} d{d}"]) for n in sizes if f"{title} G{g} n{n} d{d}" in sc]
                if not pts and k == "G":  # no kingston shot-scaling run at grid 1.0: the study repeats at 1e5 shots
                    runs = list(arm_runs("G", g, d).values())
                    pts = [(100_000, {"before": [r["error_vs_hci_by_round_mha"][0] for r in runs],
                                      "final": [r["error_vs_hci_mha"] for r in runs]})]
                n = np.array([p[0] for p in pts])
                for field, ls, fill in (("final", "-", color), ("before", "--", "white")):
                    m = np.array([np.mean(p[1][field]) for p in pts])
                    s = np.array([np.std(p[1][field], ddof=1) if len(p[1][field]) > 1 else 0 for p in pts])
                    ax.errorbar(n, m, yerr=s, color=color, linestyle=ls, marker=marker, markerfacecolor=fill,
                                markeredgewidth=0.8, linewidth=1.0, elinewidth=0.6, markersize=3.5)
            for nr, ls in ((1_040_000, "-"),):
                x = load_json(f"sqd_random_hfinit_seed101_n{nr}_R{g}_d{d}.json")
                ax.axhline(x["error_vs_hci_mha"], color=MUTED, linestyle=ls, linewidth=0.9, zorder=1)
                ax.axhline(x["error_vs_hci_by_round_mha"][0], color=MUTED, linestyle=":", linewidth=1.1, zorder=1)
            ax.set_xscale("log")
            ax.set_xticks([1e5, 1e6, 1e7], ["$10^5$", "$10^6$", "$10^7$"])
            ax.set_xlim(6e4, 1.6e7)
            ax.set_ylim(0, 140 if g == "2.2" else 70)
            ax.set_title(f"{ANGSTROM_52[g]:.3f} Å, d$^2$ = {'1.44M' if d == 1200 else '4M'}", pad=2)
            panel_label(ax, "abcd"[2 * i + j])
            if j == 0:
                ax.set_ylabel("Error vs HCI (mHa)")
            if i == 1:
                ax.set_xlabel("Shots")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=STYLE["E"][1], marker="s", label="E phoenix, square"),
               Line2D([], [], color=STYLE["G"][1], marker="^", label="G kingston, heavy-hex"),
               Line2D([], [], color=INK, linestyle="-", marker="o", markersize=3, label="final"),
               Line2D([], [], color=INK, linestyle="--", marker="o", markerfacecolor="white", markersize=3, label="before recovery"),
               Line2D([], [], color=MUTED, linestyle="-", label="random_hfinit final"),
               Line2D([], [], color=MUTED, linestyle=":", linewidth=1.1, label="random_hfinit before")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, out, "fig3_shot_scaling")


# ---------- Fig 4 ----------

def fig4(out: Path, res: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_IN, 3.9))
    data = load_32()
    caps = np.array([int(c) for c in CAPS_32])
    for ax, r, t in ((axes[0, 0], "1.0", "a"), (axes[0, 1], "2.4", "b")):
        block = data[r]
        rnd = np.array([[x["caps"][c]["error_mha_final"] for c in CAPS_32] for x in block["random"][1]])
        ax.errorbar(caps, rnd.mean(0), yerr=rnd.std(0, ddof=1), color=MUTED, linestyle="--", marker="o", markersize=3,
                    linewidth=0.9, elinewidth=0.6, label="uniform random (3 seeds)")
        for k in ("A", "B", "D"):
            label, color, marker = STYLE[k]
            runs = list(block["arms"][ARM_32[k]].values())
            v = np.array([[x["caps"][c]["error_mha_final"] for c in CAPS_32] for x in runs])
            ax.errorbar(caps, v.mean(0), yerr=v.std(0, ddof=1), color=color, marker=marker, markersize=3.5, linewidth=1.0,
                        elinewidth=0.6, label=label)
        day1 = json.loads((ROOT / "results/hardware/study_20261022T2126Z/analysis_day1.json").read_text())[r]["baselines"]
        ax.plot(caps, [day1["hci"][c] for c in CAPS_32], color=INK, linewidth=1.0, label="HCI")
        ax.plot(caps, [day1["best_fci"][c] for c in CAPS_32], color=INK, linestyle="none", marker="o", markersize=4.5,
                markerfacecolor="none", markeredgewidth=0.7, label="best FCI subspace")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(caps, [str(c) for c in caps])
        ax.minorticks_off()
        ax.set_ylim(3, 900)
        ax.set_title(f"{r} Å, 32 qubits", pad=2)
        ax.set_xlabel("Strings per spin, d")
        panel_label(ax, t)
    axes[0, 0].set_ylabel("Final error vs FCI (mHa)")
    axes[0, 1].tick_params(labelleft=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index(STYLE[k][0]) for k in "ABD"] + [l.index(x) for x in ("uniform random (3 seeds)", "HCI", "best FCI subspace")]
    h, l = [h[i] for i in order[0:6:3] + order[1:6:3] + order[2:6:3]], [l[i] for i in order[0:6:3] + order[1:6:3] + order[2:6:3]]

    offsets = dict(zip("EFG", (-0.2, 0.0, 0.2)))
    specs = ((axes[1, 0], "c", "random_hfinit", lambda k: f"H6 {k} minus random_hfinit ({'1,040,000' if k != 'G' else '100,000'} shots)",
              "Arm minus random_hfinit (mHa)"),
             (axes[1, 1], "d", "T", lambda k: f"H6 {k} (subsampled to T's valid shots) minus T", "Arm minus T (mHa)"))
    for ax, t, _, key, ylabel in specs:
        ax.axhspan(-5, 5, color=GRIDLINE, alpha=0.7, linewidth=0, zorder=0)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        for k in "EFG":
            label, color, marker = STYLE[k]
            for i, g in enumerate(GRIDS):
                for d, dx, fill in ((1200, -0.045, color), (2000, 0.045, "white")):
                    v = res["52q"][key(k)][g]["caps"][str(d)]
                    ax.errorbar([i + offsets[k] + dx], [v["mean"]], yerr=ci(v), color=color, marker=marker, markersize=3.5,
                                markerfacecolor=fill, markeredgewidth=0.8, linestyle="none", elinewidth=0.7)
        geom_axis(ax)
        ax.set_ylim(-32, 8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Bond length (Å), 52 qubits")
        panel_label(ax, t)
    from matplotlib.lines import Line2D
    h52 = [Line2D([], [], color=STYLE[k][1], marker=STYLE[k][2], linestyle="none", markersize=3.5, label=STYLE[k][0]) for k in "EFG"]
    h52 += [Line2D([], [], color=INK, marker="o", linestyle="none", markersize=3.5, label="d$^2$ = 1.44M"),
            Line2D([], [], color=INK, marker="o", markerfacecolor="white", linestyle="none", markersize=3.5, label="d$^2$ = 4M")]
    fig.legend(handles=h, labels=l, loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.legend(handles=h52, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, columnspacing=0.8, handletextpad=0.2)
    save(fig, out, "fig4_baselines")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results/figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((ROOT / "results/hardware/analysis_preregistered.json").read_text())
    fig1(out)
    fig2(out, res)
    fig3(out, res)
    fig4(out, res)


if __name__ == "__main__":
    main()
