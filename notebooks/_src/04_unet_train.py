# %% [markdown]
# # 04 · Train the U-Net risk-map model
# **Burn In Only What Matters** · ECE 8803-AI4 · Fall 2026
#
# CRISP-DM phases covered: **Modeling** and **Evaluation on Lots 1–4**.
#
# **Model (slide 6).** A per-wafer image-to-image U-Net:
# * input: one wafer as a 16-channel image on the 193×115 KAI grid:
#   * one-hot outside / pass / fail (channels 0–2), the channels the encoder was pre-trained on,
#   * parameter_1–5 as z-scores (channels 3–7),
#   * derived features: abs z (channels 8–12) and distance to the nearest fail, clipped at 20 dies (channel 13),
#   * x and y coordinates (channels 14–15);
# * output: one risk score per die.
#
# Normalisation statistics are fitted on the training lots of each fold.
#
# **Architecture.** Encoder blocks of 32→64→128→128 channels, two 3×3 conv + BatchNorm + ReLU layers per
# block, with max-pool between blocks (Module 3 blocks, doubled). The decoder uses bilinear upsampling with
# skip connections, and the input is also skipped straight to a 1×1 head so the model can use die-level
# parametric clues directly.
#
# **Loss.** Binary cross-entropy on **passing dies only**, with `pos_weight` = 10 for the ~1.2% positives.
#
# **Threshold.** Swept, not learned. Every result is a full burn-in-vs-recall curve.
#
# **Validation.** Leave-one-lot-out over Lots 1–4. Training uses flip augmentation; the coordinate
# channels are kept fixed under flips. The best epoch is chosen by validation average precision, and the
# final model trains on all 4 lots for the median best epoch.
#
# **Ablation (slide 7).** The same model with no WM811K pre-training.

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

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from burnin import Paths
from burnin.config import TRAIN_LOTS
from burnin.data import load_visible, attach_hidden, channel_names
from burnin.metrics import compare, screening_curve, recall_ceiling, bootstrap_ci
from burnin.cv import UNET_DEFAULTS, lolo_unet, fit_final_unet
from burnin.train import get_device
from burnin.plots import show_failmap, show_latent, show_risk, plot_curves

SIM = os.environ.get("BURNIN_SIM", "sim_dev")
P = Paths(ROOT, sim=SIM)
print("dataset:", P.sim_dir)
DEVICE = get_device()
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "(CPU: expect ~20x slower)")
SEED = 0
CFG = dict(UNET_DEFAULTS)
if os.environ.get("BURNIN_UNET_EPOCHS"):
    CFG["epochs"] = int(os.environ["BURNIN_UNET_EPOCHS"])
print("U-Net config:", CFG)
print("input channels:", channel_names(CFG["derived"]))

# %% [markdown]
# ## 1 · Data (Lots 1–4) and the pre-trained encoder from notebook 03

# %%
ws = attach_hidden(load_visible(P.visible), P.hidden, analysis=True).lots(TRAIN_LOTS)
ENC = torch.load(P.models / "encoder_wm811k.pt", map_location="cpu")
CEILING = recall_ceiling(ws)
print(f"{len(ws)} wafers, recall ceiling {CEILING:.1%}")

# %% [markdown]
# ## 2 · Leave-one-lot-out CV: pre-trained encoder

# %%
oof_pre, ep_pre, hist_pre = lolo_unet(ws, encoder_state=ENC, seed=SEED, **CFG)
print("best epoch per fold:", ep_pre)

# %% [markdown]
# ## 3 · Leave-one-lot-out CV: no pre-training (ablation)

# %%
oof_scr, ep_scr, hist_scr = lolo_unet(ws, encoder_state=None, seed=SEED, **CFG)
print("best epoch per fold:", ep_scr)

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for a, (name, H) in zip(ax, [("pre-trained", hist_pre), ("no pre-training", hist_scr)]):
    for lot, h in H.items():
        h = pd.DataFrame(h).dropna(subset=["val_ap"])
        a.plot(h["epoch"], h["val_ap"], marker=".", label=f"val {lot}")
    a.set_title(f"U-Net {name}: validation AP by epoch", fontsize=10)
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].set_ylabel("average precision"); ax[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4 · Comparison with the baselines (all out-of-fold on Lots 1–4)
# The baseline scores come from notebook 02 (`results/sim_dev/oof_baselines.npz`).

# %%
b = np.load(P.results / "oof_baselines.npz")
assert np.array_equal(b["wafer"], ws.wafer)
scores = {k: b[k] for k in b.files if k != "wafer"}
scores["U-Net (no pre-training)"] = oof_scr
scores["U-Net (pre-trained)"] = oof_pre
table, curves = compare(scores, ws)
table.round(3)

# %%
plot_curves(curves, ceiling=CEILING, title="Lots 1-4 out-of-fold: U-Net vs. baselines")
plt.show()

# %% [markdown]
# Wafer-bootstrap 95% CI for recall at 10% burn-in (resampling wafers, not dies, because dies on one
# wafer are correlated):

# %%
ci = {k: bootstrap_ci(scores[k], ws, n_boot=300) for k in ["Feature GBM", "U-Net (no pre-training)", "U-Net (pre-trained)", "Distance to fail"]}
pd.DataFrame(ci, index=["2.5%", "97.5%"]).T.assign(point=[table.loc[k, "recall@10%burn"] for k in ci]).round(3)

# %%
for i in [0, 20]:
    fig, ax = plt.subplots(1, 4, figsize=(13, 5))
    show_failmap(ax[0], ws, i)
    show_latent(ax[1], ws, i)
    show_risk(ax[2], ws, i, oof_pre, "U-Net risk (pre-trained, out-of-fold)")
    show_risk(ax[3], ws, i, scores["Feature GBM"], "Feature GBM risk (out-of-fold)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5 · Train the final models on all of Lots 1–4
# Epoch count = median of the per-fold best epochs. No validation lot is used here; Lot 5 stays untouched
# until notebook 05.

# %%
final = {}
for name, enc, eps in [("unet_pretrained", ENC, ep_pre), ("unet_scratch", None, ep_scr)]:
    E = int(np.median(list(eps.values())))
    net, norm, derived, _ = fit_final_unet(ws, E, encoder_state=enc, seed=SEED, **{k: v for k, v in CFG.items() if k != "epochs"})
    torch.save({"state": net.state_dict(), "mean": norm.mean.tolist(), "std": norm.std.tolist(), "derived": derived,
                "in_ch": len(channel_names(derived)), "epochs": E, "cfg": CFG}, P.models / f"{name}_{P.sim}.pt")
    final[name] = E
    print(f"saved {name}: {E} epochs")
np.savez_compressed(P.results / "oof_unet.npz", wafer=ws.wafer, pretrained=oof_pre, scratch=oof_scr)
table.to_csv(P.results / "oof_all_table.csv")
(P.results / "unet_cv.json").write_text(json.dumps({"best_epochs_pretrained": ep_pre, "best_epochs_scratch": ep_scr,
                                                    "final_epochs": final, "cfg": CFG}, indent=2))
