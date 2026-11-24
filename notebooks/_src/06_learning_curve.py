# %% [markdown]
# # 06 · Learning curve: would more wafers help?
# **Burn In Only What Matters** · ECE 8803-AI4 · Fall 2026
#
# Each model is trained on a subset of the training lots, with leave-one-lot-out on Lots 1–4 as in
# notebooks 02 and 04, using a random subset of each training lot. Each model is scored out-of-fold on the
# full held-out lot.
#
# * `sim_dev` (10 wafers per lot): 2, 4, 7 or 10 wafers per lot, i.e. 6, 12, 21 or 30 training wafers per fold.
# * `sim_full` (200 wafers per lot): 10, 25, 50 or 100 wafers per lot (30-300 per fold). The 200-per-lot point
#   (600 per fold) is taken from the out-of-fold scores of notebooks 02 and 04 rather than retrained.
#
# If a model's recall is still rising at the largest size, more data should help it. If it has flattened,
# it is limited by the information in the data, not by the amount of data.
#
# * `sim_dev`: the U-Net's epoch count is scaled so every size gets roughly the same number of gradient steps.
#   `sim_full`: the same `auto_epochs` rule as notebook 04. Early stopping picks the best epoch on the held-out lot.
# * Small subsets are repeated with different random wafer draws.
# * Only the no-pre-training U-Net is run; notebook 04 showed pre-training gives no measurable gain.
# * Results are cached in `results/<sim>/learning_curve.csv`; set `RERUN = True` to recompute
#   (~15 min on GPU for `sim_dev`, ~45 min for `sim_full`).
# * Lot 5 is not used.

# %% [markdown]
# ## 0 · Setup

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
COLAB_ROOT = "/content/drive/MyDrive/BurnIn_Project"
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path(COLAB_ROOT)
else:
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "burnin").is_dir())
sys.path.insert(0, str(ROOT / "src"))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from burnin import Paths
from burnin.config import TRAIN_LOTS
from burnin.data import load_visible, attach_hidden
from burnin.metrics import screening_curve
from burnin.cv import UNET_DEFAULTS, auto_epochs, fit_baselines, score_baselines, unet_inputs
from burnin.train import train_unet, predict_unet

SIM = os.environ.get("BURNIN_SIM", "sim_dev")
P = Paths(ROOT, sim=SIM)
print("dataset:", P.sim_dir)
ws = attach_hidden(load_visible(P.visible), P.hidden).lots(TRAIN_LOTS)
FULL = SIM != "sim_dev"
PER_LOT = [10, 25, 50, 100] if FULL else [2, 4, 7, 10]
REPEATS = {k: 1 for k in PER_LOT} if FULL else {2: 2, 4: 2, 7: 1, 10: 1}
MAX_EPOCHS = 150
RERUN = False
BASE_EPOCHS = 60

# %% [markdown]
# ## 1 · Run

# %%
def subsample(train_ws, k, rng):
    keep = np.zeros(len(train_ws), bool)
    for lot in np.unique(train_ws.lot):
        idx = np.flatnonzero(train_ws.lot == lot)
        keep[rng.choice(idx, k, replace=False)] = True
    return train_ws.subset(keep)

cfg = {k: v for k, v in UNET_DEFAULTS.items() if k not in ("derived", "epochs", "eval_every")}
cfg["eval_every"] = 5
rows = []
t0 = time.time()
CACHE = P.results / "learning_curve.csv"
for k in (PER_LOT if RERUN or not CACHE.exists() else []):
    epochs = auto_epochs(3 * k) if FULL else min(MAX_EPOCHS, int(round(BASE_EPOCHS * 10 / k)))
    for rep in range(REPEATS[k]):
        rng = np.random.default_rng(100 * k + rep)
        oof = {}
        for val in TRAIN_LOTS:
            vm = np.isin(ws.lot, [val])
            va = ws.subset(vm)
            tr = subsample(ws.lots([l for l in TRAIN_LOTS if l != val]), k, rng)
            b = score_baselines(fit_baselines(tr, seed=rep), va, seed=rep)
            for name in ["Feature GBM", "Hotelling T2 (PCA)", "Distance to fail"]:
                oof.setdefault(name, np.zeros(ws.fail.shape, np.float32))[vm] = b[name]
            _, (Xa, Xb) = unet_inputs(tr, [tr, va], UNET_DEFAULTS["derived"])
            for name, enc in [("U-Net (no pre-training)", None)]:
                net, _, _ = train_unet(Xa, tr.latent, tr.passing, Xb, va.latent, va.passing, encoder_state=enc,
                                       epochs=epochs, seed=rep, log=lambda s: None, **cfg)
                oof.setdefault(name, np.zeros(ws.fail.shape, np.float32))[vm] = np.where(va.passing, predict_unet(net, Xb), 0)
        for name, s in oof.items():
            c = screening_curve(s, ws.latent, ws.passing)
            rows.append({"wafers/fold": 3 * k, "repeat": rep, "method": name,
                         "recall@10%": c.recall_at(0.10), "AP": c.average_precision(), "AUC": c.auc()})
        print(f"{3 * k:2d} wafers/fold, repeat {rep}: done ({time.time() - t0:.0f}s)", flush=True)
if rows and FULL:
    b, u = np.load(P.results / "oof_baselines.npz"), np.load(P.results / "oof_unet.npz")
    assert np.array_equal(b["wafer"], ws.wafer) and np.array_equal(u["wafer"], ws.wafer)
    full = {"Feature GBM": b["Feature GBM"], "Hotelling T2 (PCA)": b["Hotelling T2 (PCA)"],
            "Distance to fail": b["Distance to fail"], "U-Net (no pre-training)": u["scratch"]}
    for name, s in full.items():
        c = screening_curve(s, ws.latent, ws.passing)
        rows.append({"wafers/fold": len(ws) * 3 // 4, "repeat": 0, "method": name,
                     "recall@10%": c.recall_at(0.10), "AP": c.average_precision(), "AUC": c.auc()})
if rows:
    lc = pd.DataFrame(rows)
    lc.to_csv(CACHE, index=False)
else:
    lc = pd.read_csv(CACHE)
    print(f"loaded cached {CACHE.name}")

# %% [markdown]
# ## 2 · Results (mean ± std over repeats)

# %%
summ = lc.groupby(["method", "wafers/fold"])[["recall@10%", "AP"]].agg(["mean", "std"]).round(3)
summ

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for name, g in lc.groupby("method"):
    m = g.groupby("wafers/fold")[["recall@10%", "AP"]].agg(["mean", "std"])
    x = m.index.values
    for a, col in zip(ax, ["recall@10%", "AP"]):
        a.errorbar(x, m[(col, "mean")], yerr=m[(col, "std")].fillna(0), marker="o", capsize=3,
                   lw=2 if "U-Net" in name else 1.2, label=name)
ax[0].set_ylabel("recall at 10% burn-in (out-of-fold, Lots 1-4)")
ax[1].set_ylabel("average precision")
for a in ax:
    a.set_xlabel("training wafers per fold")
    a.set_xscale("log")
    a.set_xticks(sorted(lc["wafers/fold"].unique()))
    a.set_xticklabels(sorted(lc["wafers/fold"].unique()))
    a.minorticks_off()
    a.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.suptitle("Learning curve: does more training data help?", fontsize=11)
plt.tight_layout()
plt.savefig(P.results / "figures" / "learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3 · Slope over the last doubling
# Gain in recall@10% per doubling of training data, from a log-linear fit on the three largest sizes.
# A clearly positive slope means the model is still data-limited.

# %%
slope = {}
for name, g in lc.groupby("method"):
    m = g.groupby("wafers/fold")["recall@10%"].mean()
    m = m.iloc[-3:]
    slope[name] = np.polyfit(np.log2(m.index.values), m.values, 1)[0]
pd.Series(slope, name="recall@10% gain per doubling of wafers").round(3).sort_values(ascending=False)
