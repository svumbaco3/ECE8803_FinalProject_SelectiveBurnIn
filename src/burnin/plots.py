import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PASS, FAIL, OUT = "#4A7FB5", "#E8503A", "#1a1a1a"
WAFER_CMAP = ListedColormap([OUT, PASS, FAIL])


def _clean(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def show_failmap(ax, ws, i, title=None):
    img = np.where(ws.die[i], np.where(ws.fail[i], 2, 1), 0)
    ax.imshow(img, cmap=WAFER_CMAP, vmin=0, vmax=2, origin="lower", interpolation="nearest")
    _clean(ax, title or f"wafer {ws.wafer[i]}: wafer-test fails")


def show_latent(ax, ws, i, title=None):
    img = np.where(ws.passing[i], 0.0, np.nan)
    img[ws.latent[i] & ws.passing[i]] = 1.0
    ax.imshow(img, cmap=ListedColormap(["#EEEEEE", "#B00020"]), vmin=0, vmax=1, origin="lower", interpolation="nearest")
    _clean(ax, title or f"latent defective ({int((ws.latent[i] & ws.passing[i]).sum())} dies)")


def show_risk(ax, ws, i, score, title="model risk"):
    img = np.where(ws.passing[i], score[i], np.nan)
    lo, hi = np.nanpercentile(img, [1, 99])
    im = ax.imshow(img, cmap="magma", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    _clean(ax, title)
    return im


def plot_curves(curves, ax=None, title="Burn-in fraction vs. latent-defect recall", ceiling=None):
    ax = ax or plt.subplots(figsize=(6.5, 5))[1]
    for name, c in curves.items():
        ax.plot(100 * c.burn_frac, 100 * c.recall, lw=2 if "U-Net" in name else 1.4,
                ls="--" if name == "Random" else "-", label=f"{name} (AUC {c.auc():.3f})")
    if ceiling is not None:
        ax.axhline(100 * ceiling, color="gray", ls=":", lw=1, label=f"recall ceiling {100 * ceiling:.0f}%")
    ax.axvline(10, color="k", lw=0.6, ls=":")
    ax.set_xlabel("% of passing dies burned in")
    ax.set_ylabel("% of latent defects caught (recall)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    return ax
