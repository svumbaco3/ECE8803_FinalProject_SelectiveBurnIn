"""Lot 5 cost per wafer vs. % burned in at other burn-in costs, same format as notebook 05's lot5_cost_vs_burnin.png.

python tools/cost_figures.py [--sim sim_full] [--c-escape 100] [--c-burnin 1 2]
    -> results/<sim>/figures/lot5_cost_vs_burnin_escape<c_escape>.png
Uses the saved Lot 5 scores (results/<sim>/lot5_scores.npz), so no model is re-run.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from burnin import Paths
from burnin.config import TEST_LOT
from burnin.data import attach_hidden, load_visible
from burnin.metrics import screening_curve

METHODS = ["U-Net (pre-trained)", "Feature GBM", "Distance to fail", "Random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="sim_full")
    ap.add_argument("--c-escape", type=float, default=100)
    ap.add_argument("--c-burnin", type=float, nargs="+", default=[1.0, 2.0])
    a = ap.parse_args()
    P = Paths(ROOT, sim=a.sim)
    test = attach_hidden(load_visible(P.visible), P.hidden).lots([TEST_LOT])
    sc = np.load(P.results / "lot5_scores.npz")
    assert (sc["wafer"] == test.wafer).all()
    curves = {k: screening_curve(sc[k], test.latent, test.passing) for k in METHODS}
    fr = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(1, len(a.c_burnin), figsize=(5 * len(a.c_burnin), 4.2), squeeze=False)
    ax = ax[0]
    for axi, c_burn in zip(ax, a.c_burnin):
        for k in METHODS:
            y = np.array([curves[k].cost_per_wafer(f, c_burn, a.c_escape) for f in fr])
            axi.plot(100 * fr, y, label=k, lw=2 if "U-Net" in k else 1.2)
            if k == METHODS[0]:
                j = int(np.argmin(y))
                print(f"c_burnin ${c_burn:.2f}, c_escape ${a.c_escape:,.0f}: best {100 * fr[j]:.1f}% burned in, "
                      f"${y[j]:,.0f}/wafer vs. ${y[-1]:,.0f} burn-in-all ({100 * (1 - y[j] / y[-1]):.1f}% saving)")
        axi.set_title(f"c_burnin \\${c_burn:.2f}, c_escape \\${a.c_escape:,.0f}", fontsize=10)
        axi.set_xlabel("% of passing dies burned in")
        axi.grid(alpha=0.3)
    ax[0].set_ylabel("cost per wafer ($)")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    out = P.results / "figures" / f"lot5_cost_vs_burnin_escape{a.c_escape:.0f}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
