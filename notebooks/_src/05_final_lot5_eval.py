# %% [markdown]
# # 05 · Final evaluation on Lot 5 (touched once)
# **Burn In Only What Matters** · ECE 8803-AI4 · Fall 2026
#
# CRISP-DM phase: **Evaluation**. Every method is fitted on Lots 1–4 only and scored once on Lot 5:
# * baselines are re-fitted here on Lots 1–4;
# * the U-Nets are the final models saved by notebook 04.
#
# Nothing below changes a model or chooses a hyper-parameter. Outputs:
# * **slide 8:** results table with wafer-bootstrap 95% CIs, plus the recall ceiling;
# * **slide 9:** cost model `C = c_burnin × #burned + c_escape × #escaped`, cost vs. burn-in fraction,
#   and its sensitivity to the two costs;
# * **slide 10:** corner cases by radial zone, wafer fail level, clue status and distance to the cluster.

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
from matplotlib.colors import ListedColormap
from burnin import Paths
from burnin.config import TRAIN_LOTS, TEST_LOT
from burnin.data import load_visible, attach_hidden
from burnin.metrics import compare, screening_curve, recall_ceiling, bootstrap_ci, per_group_recall
from burnin.cv import fit_baselines, score_baselines, load_unet, score_unet
from burnin.baselines import radius_map, wafer_fail_fraction
from burnin.plots import show_failmap, show_latent, show_risk, plot_curves

SIM = os.environ.get("BURNIN_SIM", "sim_dev")
P = Paths(ROOT, sim=SIM)
print("dataset:", P.sim_dir)
FIG = P.results / "figures"
FIG.mkdir(exist_ok=True)
SEED = 0
BURN_BUDGET = 0.10

# %% [markdown]
# ## 1 · Data

# %%
ws_all = attach_hidden(load_visible(P.visible), P.hidden, analysis=True)
train, test = ws_all.lots(TRAIN_LOTS), ws_all.lots([TEST_LOT])
p = test.passing
print(f"train: {len(train)} wafers (Lots 1-4) | test: {len(test)} wafers ({TEST_LOT})")
print(f"Lot 5 wafer-test fail rate {test.fail.sum() / test.die.sum():.2%}, "
      f"latent rate {(test.latent & p).sum() / p.sum():.2%} of passing dies")
CEILING = recall_ceiling(test)
print(f"Lot 5 recall ceiling: {CEILING:.1%}")

# %% [markdown]
# ## 2 · Score every method on Lot 5

# %%
scores = score_baselines(fit_baselines(train, SEED), test, SEED)
unet_meta = {}
for name, f in [("U-Net (no pre-training)", "unet_scratch"), ("U-Net (pre-trained)", "unet_pretrained")]:
    net, norm, derived, ck = load_unet(P.models / f"{f}_{P.sim}.pt")
    scores[name] = score_unet(net, norm, derived, test)
    unet_meta[name] = ck["epochs"]
print("U-Net final epochs:", unet_meta)
table, curves = compare(scores, test)
table.round(3)

# %%
ax = plot_curves(curves, ceiling=CEILING, title="Lot 5 (held out): burn-in fraction vs. recall")
plt.savefig(FIG / "lot5_recall_vs_burnin.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3 · Slide 8 table, with wafer-bootstrap 95% CIs

# %%
SLIDE8 = ["U-Net (pre-trained)", "Distance to fail", "Neighbor z-score", "Random", "U-Net (no pre-training)",
          "Feature GBM", "Hotelling T2 (PCA)", "Wafer-level fallback"]
rows = []
for k in SLIDE8:
    r10, b90 = bootstrap_ci(scores[k], test, [lambda c: c.recall_at(BURN_BUDGET), lambda c: c.burn_at(0.90)],
                            n_boot=500, seed=SEED)
    rows.append({"method": k,
                 "Recall @ 10% burn-in": f"{100 * table.loc[k, 'recall@10%burn']:.1f}% [{100 * r10[0]:.1f}, {100 * r10[1]:.1f}]",
                 "Burn-in % @ 90% recall": f"{100 * table.loc[k, 'burn%@90%recall']:.1f}% [{100 * b90[0]:.1f}, {100 * b90[1]:.1f}]",
                 "DPPM @ 10% burn-in": round(table.loc[k, "DPPM@10%burn"]),
                 "AP": round(table.loc[k, "AP"], 3)})
slide8 = pd.DataFrame(rows).set_index("method")
slide8.loc["Burn in all passing dies"] = ["100.0% (burns 100%)", "100.0%", 0, np.nan]
slide8.loc["Recall ceiling (clued + near-cluster)"] = [f"{100 * CEILING:.1f}%", "-", np.nan, np.nan]
slide8.to_csv(P.results / "lot5_slide8_table.csv")
slide8

# %%
u, ref = table.loc["U-Net (pre-trained)"], table.loc["Distance to fail"]
best_base = table.drop(index=[k for k in table.index if k.startswith("U-Net")])["recall@10%burn"].idxmax()
print(f"Headline: at a 10% burn-in budget the pre-trained U-Net catches {u['recall@10%burn']:.0%} of latent defects "
      f"vs. {ref['recall@10%burn']:.0%} for distance-to-fail and {table.loc[best_base, 'recall@10%burn']:.0%} for the best "
      f"baseline ({best_base}); reaching 90% recall takes {u['burn%@90%recall']:.0%} burn-in vs. {ref['burn%@90%recall']:.0%}.")

# %% [markdown]
# ## 4 · Slide 9: cost model
# `C = c_burnin × #burned + c_escape × #escaped`, per wafer. Burn-in costs $0.50/die (sensitivity
# $0.05–$5); an escape costs $100–$10,000/die (slide 9; the source is Test & Qualification Lectures 1–2).
# "Burn in all" means burned = all passing dies and escaped = 0. "Burn in none" means escaped = all latent dies.

# %%
fr = np.linspace(0, 1, 401)
C_BURN = 0.50
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cost_rows = []
for a, c_esc in zip(ax, [100, 1000, 10000]):
    for k in ["U-Net (pre-trained)", "Feature GBM", "Distance to fail", "Random"]:
        c = curves[k]
        y = np.array([c.cost_per_wafer(f, C_BURN, c_esc) for f in fr])
        a.plot(100 * fr, y, label=k, lw=2 if "U-Net" in k else 1.2)
        j = int(np.argmin(y))
        cost_rows.append({"c_escape": c_esc, "method": k, "best burn-in %": 100 * fr[j], "cost/wafer at best": y[j],
                          "recall at best": c.recall_at(fr[j]), "DPPM at best": c.dppm_at(fr[j]),
                          "burn-in-all cost/wafer": y[-1], "saving vs burn-in-all %": 100 * (1 - y[j] / y[-1])})
    a.set_title(f"c_burnin \\${C_BURN:.2f}, c_escape \\${c_esc:,}", fontsize=10)
    a.set_xlabel("% of passing dies burned in"); a.grid(alpha=0.3)
ax[0].set_ylabel("cost per wafer ($)"); ax[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "lot5_cost_vs_burnin.png", dpi=150, bbox_inches="tight")
plt.show()
cost_table = pd.DataFrame(cost_rows).set_index(["c_escape", "method"])
cost_table.to_csv(P.results / "lot5_cost_table.csv")
cost_table.round(1)

# %% [markdown]
# Slide-sized version at the low end of the escape-cost range ($100/die), with each method's cost-optimal
# point marked. A die is worth burning in when its latent probability exceeds c_burnin / c_escape
# (0.5% at $0.50/$100, 0.05% at $0.50/$1,000). If even the lowest-risk dies are above that break-even,
# burning in everything is optimal and selective burn-in cannot save money.

# %%
lat_far = test.latent[test.passing & ~np.isfinite(test.extra["dist"])].mean()
print(f"Lot 5 latent rate on wafers with no cluster: {lat_far:.2%}")
for e in [100, 300, 1000, 10000]:
    print(f"  break-even latent probability at c_escape ${e:,}: {C_BURN / e:.3%}")
C_ESC_SLIDE = 100
fig, a = plt.subplots(figsize=(7, 4.1))
for k in ["U-Net (pre-trained)", "Feature GBM", "Distance to fail", "Random"]:
    y = np.array([curves[k].cost_per_wafer(f, C_BURN, C_ESC_SLIDE) for f in fr])
    j = int(np.argmin(y))
    line, = a.plot(100 * fr, y, label=k, lw=2.2 if "U-Net" in k else 1.2)
    a.plot(100 * fr[j], y[j], "o", color=line.get_color(), ms=7)
    if "U-Net" in k:
        a.annotate(f"optimum: {100 * fr[j]:.0f}% burned in\n\\${y[j]:,.0f}/wafer", (100 * fr[j], y[j]),
                   xytext=(100 * fr[j] + 12, y[j] + 0.25 * (y.max() - y.min())), fontsize=9,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
a.axhline(y[-1], color="gray", ls=":", lw=1)
a.text(1, y[-1], " burn in all", va="bottom", fontsize=8, color="gray")
a.set_xlabel("% of passing dies burned in (threshold swept)")
a.set_ylabel("cost per wafer ($)")
a.set_title(f"Lot 5 cost per wafer: burn-in \\${C_BURN:.2f}/die, escape \\${C_ESC_SLIDE:,}/die", fontsize=10)
a.grid(alpha=0.3)
a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "slide9_cost_curve.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# Sensitivity: the U-Net's saving at its cost-optimal burn-in fraction, relative to burning in every
# passing die, over a grid of the two costs.

# %%
cb = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0]
ce = [100, 300, 1000, 3000, 10000]
c = curves["U-Net (pre-trained)"]
sens = pd.DataFrame(index=pd.Index(cb, name="c_burnin $"), columns=pd.Index(ce, name="c_escape $"), dtype=float)
for b in cb:
    for e in ce:
        y = np.array([c.cost_per_wafer(f, b, e) for f in fr])
        sens.loc[b, e] = 100 * (1 - y.min() / y[-1])
sens.round(1)

# %% [markdown]
# ## 5 · Slide 10: corner cases at the 10% burn-in budget
# The groups below use simulator internals (clue flag, distance to the cluster), **for analysis only**.
# They are never model inputs.

# %%
lat = test.latent & test.passing
rad = np.broadcast_to(radius_map(), test.fail.shape)
zones = np.where(rad < 0.5, "center (r<0.5)", np.where(rad < 0.85, "middle (0.5-0.85)", "edge (r>=0.85)"))
wf = wafer_fail_fraction(test)
fail_lvl = np.where(wf < 0.03, "low fail wafer (<3%)", np.where(wf < 0.10, "mid fail wafer (3-10%)", "high fail wafer (>=10%)"))
clue = np.where(test.extra["clued"], "clued", "unclued")
d = test.extra["dist"]
near = np.where(d <= 2, "near cluster (<=2)", np.where(np.isfinite(d) & (d <= 10), "2-10 from cluster", "far / no cluster"))
groups = {"radial zone": zones, "wafer fail level": fail_lvl, "clue": clue, "distance to cluster": near}
rows = []
for gname, g in groups.items():
    for k in ["U-Net (pre-trained)", "U-Net (no pre-training)", "Feature GBM", "Distance to fail"]:
        for grp, (rec, n) in per_group_recall(scores[k], test, BURN_BUDGET, g).items():
            rows.append({"grouping": gname, "group": grp, "method": k, "recall": rec, "n latent": n})
corner = pd.DataFrame(rows).pivot_table(index=["grouping", "group", "n latent"], columns="method", values="recall")
corner.to_csv(P.results / "lot5_corner_cases.csv")
corner.round(3)

# %%
order = np.argsort(-test.fail.sum((1, 2)) / test.die.sum((1, 2)))
for i in [order[0], order[len(order) // 2], order[-1]]:
    fig, ax = plt.subplots(1, 4, figsize=(13, 5))
    show_failmap(ax[0], test, i)
    show_latent(ax[1], test, i)
    show_risk(ax[2], test, i, scores["U-Net (pre-trained)"], "U-Net risk (pre-trained)")
    show_risk(ax[3], test, i, scores["Distance to fail"], "distance-to-fail score")
    plt.tight_layout()
    plt.savefig(FIG / f"lot5_riskmap_wafer{test.wafer[i]}.png", dpi=130, bbox_inches="tight")
    plt.show()

# %%
i = order[len(order) // 2]
fig, ax = plt.subplots(1, 3, figsize=(5.6, 3.25))
show_failmap(ax[0], test, i, f"wafer {test.wafer[i]}: test fails")
show_risk(ax[1], test, i, scores["U-Net (pre-trained)"], "U-Net risk map")
flag = scores["U-Net (pre-trained)"] >= np.quantile(scores["U-Net (pre-trained)"][test.passing], 1 - BURN_BUDGET)
img = np.where(test.passing[i], np.where(flag[i], 2, 1), np.where(test.die[i], 3, 0))
ax[2].imshow(img, cmap=ListedColormap(["#1a1a1a", "#DDDDDD", "#F2A900", "#E8503A"]), vmin=0, vmax=3, origin="lower",
             interpolation="nearest")
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title("flagged at 10% budget", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "slide6_riskmap.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6 · Save

# %%
table.to_csv(P.results / "lot5_all_metrics.csv")
np.savez_compressed(P.results / "lot5_scores.npz", wafer=test.wafer, **{k: v.astype(np.float32) for k, v in scores.items()})
(P.results / "lot5_summary.json").write_text(json.dumps({
    "recall_ceiling": CEILING, "burn_budget": BURN_BUDGET, "unet_epochs": unet_meta,
    "latent_rate_no_cluster": float(lat_far), "c_burnin": C_BURN, "c_escape_slide": C_ESC_SLIDE,
    "n_test_wafers": int(len(test)),
    "metrics": table.round(4).to_dict(orient="index")}, indent=2))
print("saved to", P.results)

# %% [markdown]
# ## 7 · Per-die export (same row layout as the simulator's `visible.npz`)
# One row per Lot 5 die site. `burn_in_10pct` flags the top 10% of passing dies by U-Net risk, ranked
# across Lot 5 (the 10% budget applied to the lot). Failing dies are rejected at wafer test, so their risk is 0.

# %%
EXPORT = {"risk_unet": "U-Net (pre-trained)", "risk_unet_scratch": "U-Net (no pre-training)",
          "risk_gbm": "Feature GBM", "risk_dist_to_fail": "Distance to fail"}
wi, yi, xi = np.nonzero(test.die)
thr = np.quantile(scores["U-Net (pre-trained)"][test.passing], 1 - BURN_BUDGET)
per_die = pd.DataFrame({"lot": TEST_LOT, "wafer": test.wafer[wi], "x": xi + 1, "y": yi + 1,
                        "wafer_test_fail": test.fail[wi, yi, xi].astype(np.int8),
                        **{c: scores[k][wi, yi, xi].round(6) for c, k in EXPORT.items()}})
per_die["burn_in_10pct"] = ((per_die["wafer_test_fail"] == 0) & (per_die["risk_unet"] >= thr)).astype(np.int8)
per_die.to_csv(P.results / "lot5_risk_per_die.csv.gz", index=False)
print(f"{len(per_die):,} rows, {per_die['burn_in_10pct'].mean():.1%} of all die sites flagged -> lot5_risk_per_die.csv.gz")
