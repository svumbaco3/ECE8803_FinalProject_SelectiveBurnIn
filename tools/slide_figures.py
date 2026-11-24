"""Slide 6 figures: WM811K pre-training examples and the U-Net input channels of one Lot 5 wafer.

python tools/slide_figures.py [--sim sim_full]   -> results/<sim>/figures/slide6_wm811k.png, slide6_inputs.png, slide6_risk_output.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from burnin import Paths
from burnin.config import TEST_LOT, TRAIN_LOTS
from burnin.data import Normalizer, load_visible
from burnin.plots import WAFER_CMAP, show_failmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="sim_full")
    a = ap.parse_args()
    P = Paths(ROOT, sim=a.sim)
    fig_dir = P.results / "figures"
    fig_dir.mkdir(exist_ok=True)

    d = np.load(P.wm811k, allow_pickle=True)
    X, y = d["X"], d["y"].astype(str)
    pre = np.load(P.wm811k_split)["pretrain_idx"]
    fig, ax = plt.subplots(1, 4, figsize=(5.6, 1.9))
    for k, c in zip(ax, ["Edge-Ring", "Center", "Loc", "Scratch"]):
        k.imshow(X[pre[np.flatnonzero(y[pre] == c)[0]]], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        k.set_xticks([])
        k.set_yticks([])
        k.set_title(c, fontsize=9)
    fig.suptitle("WM811K pre-training patterns (fail = red)", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "slide6_wm811k.png", dpi=200, bbox_inches="tight")
    plt.close()

    ws = load_visible(P.visible)
    norm = Normalizer().fit(ws.lots(TRAIN_LOTS))
    test = ws.lots([TEST_LOT])
    order = np.argsort(-test.fail.sum((1, 2)) / test.die.sum((1, 2)))
    i = order[len(order) // 2]
    z = norm.z(test.subset(np.arange(len(test)) == i))[0]
    fig, ax = plt.subplots(1, 3, figsize=(5.6, 3.25))
    for k, c in zip(ax[:2], [0, 2]):
        img = np.where(test.die[i], z[c], np.nan)
        k.imshow(img, cmap="coolwarm", vmin=-3, vmax=3, origin="lower", interpolation="nearest")
        k.set_xticks([])
        k.set_yticks([])
        k.set_title(f"parameter_{c + 1} (z)", fontsize=9)
    show_failmap(ax[2], test, i, "Wafer-Test Fail map")
    fig.suptitle(f"U-Net inputs, wafer {test.wafer[i]} (2 of 5 parameters shown)", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "slide6_inputs.png", dpi=200, bbox_inches="tight")
    plt.close()

    sc = np.load(P.results / "lot5_scores.npz")
    risk = sc["U-Net (pre-trained)"][list(sc["wafer"]).index(test.wafer[i])]
    img = np.where(test.die[i] & ~test.fail[i], risk, np.nan)
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    im = ax.imshow(img, cmap="YlOrRd", norm=PowerNorm(0.5, vmin=0, vmax=1), origin="lower", interpolation="nearest")
    ax.imshow(np.where(test.fail[i], 1.0, np.nan), cmap=ListedColormap(["#202020"]), origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03, ticks=[0, 0.05, 0.2, 0.5, 1])
    cb.ax.tick_params(labelsize=6)
    cb.set_label("risk score (0–1)", fontsize=7)
    ax.legend(handles=[Patch(color="#202020", label="Wafer-Test Fail\n(not scored)")], fontsize=6, frameon=False,
              loc="lower right", bbox_to_anchor=(0.02, 0.0), handlelength=1)
    ax.set_title(f"U-Net risk map, wafer {test.wafer[i]}", fontsize=8)
    plt.savefig(fig_dir / "slide6_risk_output.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", fig_dir / "slide6_wm811k.png", ",", fig_dir / "slide6_inputs.png", "and", fig_dir / "slide6_risk_output.png")


if __name__ == "__main__":
    main()
