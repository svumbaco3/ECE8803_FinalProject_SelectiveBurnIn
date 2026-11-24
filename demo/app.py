"""Selective Burn-In Screening: Gradio demo.

python demo/app.py [--share]
Pick a Lot 5 wafer, a screening condition (risk threshold or burn-in budget) and cost assumptions; see the
U-Net risk map, which dies are Flagged for burn-in, and the resulting recall, escapes, DPPM and cost.
Data: demo/lot5_demo.npz (built by demo/build_demo_data.py from the saved Lot 5 results).
"""
import argparse
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

D = np.load(Path(__file__).resolve().parent / "lot5_demo.npz")
WAFERS, PATTERN = D["wafer"], D["pattern"]
DIE, FAIL, LAT, CATCH, RISK = D["die"], D["fail"], D["latent"], D["catchable"], D["risk"]
PASS = DIE & ~FAIL
LOT_RISK, LOT_LAT = RISK[PASS], LAT[PASS]
LOT_SORTED = np.sort(LOT_RISK)[::-1]
N_W = len(WAFERS)
CEILING = CATCH.sum() / LAT.sum()
_cum = np.cumsum(LOT_LAT[np.argsort(-LOT_RISK, kind="stable")])
CURVE_BURN = np.linspace(0, 1, 501)
CURVE_REC = np.r_[0, _cum][np.round(CURVE_BURN * len(_cum)).astype(int)] / LOT_LAT.sum()
COST_METHODS, COST_FRAC, COST_RECALL = D["curve_methods"].astype(str), D["curve_frac"], D["curve_recall"]

SCOPE = {"threshold": (0.0, 1.0), "budget": (0.0, 100.0), "c_burn": (0.05, 5.0), "c_esc": (100.0, 10000.0)}
LABEL = {"threshold": "Risk threshold", "budget": "Burn-in budget (%)", "c_burn": "Burn-in cost ($/die)",
         "c_esc": "Escape cost ($/die)"}
FAIL_C, SKIP_C, WASTE_C, CATCH_C, ESC_C = "#202020", "#e3e8ee", "#fdae61", "#1a9850", "#d62728"
WAFER_CHOICES = [(f"{w}  ({p})", int(w)) for w, p in zip(WAFERS, PATTERN)]

SCOPE_MD = f"""
### Scope and units
| Input | Valid range | Unit |
|---|---|---|
| Wafer | Lot 5, wafers {WAFERS.min()}–{WAFERS.max()} (held-out test lot, never used in training) | wafer number |
| Risk threshold | {SCOPE['threshold'][0]:.0f}–{SCOPE['threshold'][1]:.0f} | U-Net risk score (unitless) |
| Burn-in budget | {SCOPE['budget'][0]:.0f}–{SCOPE['budget'][1]:.0f} | % of all Lot 5 passing dies |
| Burn-in cost | \\${SCOPE['c_burn'][0]:.2f}–\\${SCOPE['c_burn'][1]:.2f} | \\$ per die burned in |
| Escape cost | \\${SCOPE['c_esc'][0]:,.0f}–\\${SCOPE['c_esc'][1]:,.0f} | \\$ per latent defect shipped |

* **Prediction:** a U-Net risk score from 0 to 1 for every die that passed wafer test. It ranks dies by risk; it is
  not a calibrated probability. One fixed threshold is applied to the whole lot; dies at or above it are **Flagged**
  for burn-in.
* **Model inputs:** only parameter_1–5, die x/y and the Wafer-Test Fail map. The Latent Defective labels used to
  score caught/escaped dies come from the simulator's hidden answer key and are never model inputs.
* **Data:** Gaussian parametric values on the KAI 17,509-die grid, real WM811K defect patterns, simulated burn-in
  labels; test limits = mean ± 6σ from Lots 1–4. Results describe this simulator, not a real product.
* **Assumptions:** burn-in catches 100% of latent defects; every wafer has at least ~0.5% Latent Defective dies
  (simulator risk floor); about {CEILING:.0%} of latent dies are catchable at all (clued, or within 2 dies of a
  cluster), so recall near 100% at low burn-in would indicate leakage.
"""


def check(name, v):
    lo, hi = SCOPE[name]
    if v is None or not np.isfinite(v) or not lo <= v <= hi:
        raise gr.Error(f"{LABEL[name]} = {v} is outside the validated range {lo:g}–{hi:g}. See 'Scope and units'.")


def stats(passing, risk, lat, t, n_w, c_burn, c_esc):
    flag = passing & (risk >= t)
    n_pass, n_flag, n_lat = int(passing.sum()), int(flag.sum()), int((lat & passing).sum())
    caught = int((flag & lat).sum())
    esc = n_lat - caught
    return {
        "Passing dies": f"{n_pass:,}",
        "Flagged for burn-in": f"{n_flag:,} ({n_flag / n_pass:.1%} of passing)",
        "Latent Defective (simulator)": f"{n_lat:,} ({n_lat / n_pass:.2%} of passing)",
        "Caught (recall)": f"{caught:,} ({caught / max(n_lat, 1):.1%})",
        "Escapes": f"{esc:,} ({esc / max(n_pass - caught, 1) * 1e6:,.0f} DPPM)",
        "Precision of burn-in": f"{caught / max(n_flag, 1):.1%}",
        "Cost, selective burn-in": f"${(n_flag * c_burn + esc * c_esc) / n_w:,.0f} per wafer",
        "Cost, burn in all": f"${n_pass * c_burn / n_w:,.0f} per wafer",
        "Cost, burn in none": f"${n_lat * c_esc / n_w:,.0f} per wafer",
        "Saving vs. burn in all (negative = costs more)": f"{1 - (n_flag * c_burn + esc * c_esc) / (n_pass * c_burn):.1%}",
    }


def table(w, lot):
    rows = "\n".join(f"| {k} | {w[k]} | {lot[k]} |" for k in w)
    return "| | This wafer | All Lot 5 (200 wafers) |\n|---|---|---|\n" + rows


def wafer_figure(i, t):
    die, fail, passing, lat = DIE[i], FAIL[i], PASS[i], LAT[i]
    fig, ax = plt.subplots(1, 3, figsize=(12, 5.4))
    sort_map = np.where(die, np.where(fail, 2, 1), np.nan)
    ax[0].imshow(sort_map, cmap=ListedColormap(["#4A7FB5", "#E8503A"]), vmin=1, vmax=2, origin="lower", interpolation="nearest")
    ax[0].set_title(f"Wafer-Test Fail map (model input)\n{int(fail.sum()):,} fails of {int(die.sum()):,} dies", fontsize=10)
    ax[0].legend(handles=[Patch(color="#4A7FB5", label="pass"), Patch(color="#E8503A", label="Wafer-Test Fail")],
                 fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    im = ax[1].imshow(np.where(passing, RISK[i], np.nan), cmap="YlOrRd", norm=PowerNorm(0.5, vmin=0, vmax=1),
                      origin="lower", interpolation="nearest")
    ax[1].imshow(np.where(fail, 1.0, np.nan), cmap=ListedColormap([FAIL_C]), origin="lower", interpolation="nearest")
    ax[1].set_title("U-Net risk map (prediction)\nscore 0–1 per passing die", fontsize=10)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.07, pad=0.02, ticks=[0, 0.05, 0.2, 0.5, 1])
    cb.ax.tick_params(labelsize=8)
    if 0 < t <= 1:
        cb.ax.axhline(t, color="k", lw=2)
        cb.set_label(f"black line = threshold t = {t:.3f}", fontsize=8)
    flag = passing & (RISK[i] >= t)
    dec = np.where(die, np.where(fail, 0, np.where(flag, 2, 1)), np.nan)
    ax[2].imshow(dec, cmap=ListedColormap([FAIL_C, SKIP_C, WASTE_C]), vmin=0, vmax=2, origin="lower", interpolation="nearest")
    yy, xx = np.nonzero(lat & flag)
    ax[2].scatter(xx, yy, s=9, c=CATCH_C, marker="o", linewidths=0)
    yy, xx = np.nonzero(lat & passing & ~flag)
    ax[2].scatter(xx, yy, s=14, c=ESC_C, marker="x", linewidths=1)
    ax[2].set_title(f"Burn-in decision at t = {t:.3f}\n{int(flag.sum()):,} Flagged ({flag.sum() / passing.sum():.1%} of passing)", fontsize=10)
    ax[2].legend(handles=[Patch(color=WASTE_C, label="Flagged"), Patch(color=SKIP_C, label="skipped"),
                          Patch(color=FAIL_C, label="Wafer-Test Fail"),
                          Line2D([], [], color=CATCH_C, marker="o", ls="", label="latent, caught"),
                          Line2D([], [], color=ESC_C, marker="x", ls="", label="latent, escaped")],
                 fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Lot 5 wafer {WAFERS[i]} (simulated defect pattern: {PATTERN[i]})", fontsize=12)
    fig.tight_layout()
    return fig


def curve_figure(t):
    burn = (LOT_RISK >= t).mean()
    rec = (LOT_LAT & (LOT_RISK >= t)).sum() / LOT_LAT.sum()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(100 * CURVE_BURN, 100 * CURVE_REC, lw=2, label="U-Net (pre-trained)")
    ax.plot([0, 100], [0, 100], "--", color="tab:red", lw=1.2, label="random selection")
    ax.axhline(100 * CEILING, color="gray", ls=":", lw=1, label=f"recall ceiling {CEILING:.0%}")
    ax.plot(100 * burn, 100 * rec, "o", color="k", ms=8, label=f"this threshold: {burn:.1%} burned, {rec:.1%} caught")
    ax.set_xlabel("% of Lot 5 passing dies burned in")
    ax.set_ylabel("% of Latent Defective dies caught (recall)")
    ax.set_title("Lot 5 trade-off curve", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def cost_figure(t, c_burn, c_esc):
    n_pass, n_lat = len(LOT_RISK), int(LOT_LAT.sum())
    fig, a = plt.subplots(figsize=(5.5, 4.2))
    for k, rec in zip(COST_METHODS, COST_RECALL):
        y = (COST_FRAC * n_pass * c_burn + n_lat * (1 - rec) * c_esc) / N_W
        j = int(np.argmin(y))
        line, = a.plot(100 * COST_FRAC, y, label=k, lw=2.2 if "U-Net" in k else 1.2)
        a.plot(100 * COST_FRAC[j], y[j], "o", color=line.get_color(), ms=7)
        if "U-Net" in k:
            x0 = 100 * COST_FRAC[j]
            a.annotate(f"optimum: {x0:.0f}% burned in\n\\${y[j]:,.0f}/wafer", (x0, y[j]),
                       xytext=(x0 + 12 if x0 < 60 else x0 - 50, y[j] + 0.25 * (y.max() - y.min())), fontsize=9,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
    a.axhline(y[-1], color="gray", ls=":", lw=1)
    a.text(1, y[-1], " burn in all", va="bottom", fontsize=8, color="gray")
    flag = LOT_RISK >= t
    caught = int((LOT_LAT & flag).sum())
    y_t = (flag.sum() * c_burn + (n_lat - caught) * c_esc) / N_W
    a.plot(100 * flag.mean(), y_t, "o", color="k", ms=8, label=f"this threshold: \\${y_t:,.0f}/wafer")
    a.set_xlabel("% of passing dies burned in (threshold swept)")
    a.set_ylabel("cost per wafer ($)")
    a.set_title(f"Lot 5 cost per wafer: burn-in \\${c_burn:.2f}/die, escape \\${c_esc:,.0f}/die", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run(wafer, mode, threshold, budget, c_burn, c_esc):
    if wafer is None or int(wafer) not in WAFERS:
        raise gr.Error(f"Wafer {wafer} is not in Lot 5 ({WAFERS.min()}–{WAFERS.max()}).")
    check("c_burn", c_burn)
    check("c_esc", c_esc)
    if mode == "Burn-in budget":
        check("budget", budget)
        k = int(round(budget / 100 * len(LOT_SORTED)))
        t = float(LOT_SORTED[k - 1]) if k > 0 else np.inf
    else:
        check("threshold", threshold)
        t = float(threshold)
    i = int(np.flatnonzero(WAFERS == int(wafer))[0])
    w = stats(PASS[i], RISK[i], LAT[i], t, 1, c_burn, c_esc)
    lot = stats(PASS, RISK, LAT, t, N_W, c_burn, c_esc)
    head = f"**Risk threshold t = {t:.4f}** (dies with U-Net risk ≥ t are Flagged for burn-in)" if np.isfinite(t) \
        else "**Budget 0%: no dies are Flagged**"
    return wafer_figure(i, t), curve_figure(t), cost_figure(t, c_burn, c_esc), head + "\n\n" + table(w, lot)


def toggle(mode):
    return gr.update(visible=mode == "Risk threshold"), gr.update(visible=mode == "Burn-in budget")


with gr.Blocks(title="Selective Burn-In Screening") as demo:
    gr.Markdown("# Selective Burn-In Screening: U-Net Risk Map Demo\n"
                "ECE 8803-AI4, Fall 2026 · Nick Saba & Sean Vumbaco. Choose a wafer and a screening condition; "
                "the model's risk map decides which passing dies are Flagged for burn-in.")
    with gr.Row():
        with gr.Column(scale=1):
            wafer = gr.Dropdown(WAFER_CHOICES, value=889, label="Lot 5 wafer (simulated defect pattern)")
            mode = gr.Radio(["Burn-in budget", "Risk threshold"], value="Burn-in budget", label="Screening condition")
            budget = gr.Slider(0, 100, value=10, step=0.5, label="Burn-in budget (% of Lot 5 passing dies)")
            threshold = gr.Slider(0, 1, value=0.22, step=0.005, label="Risk threshold t (unitless, 0–1)", visible=False)
            c_burn = gr.Number(0.50, label="Burn-in cost ($ per die, 0.05–5.00)")
            c_esc = gr.Number(100, label="Escape cost ($ per latent defect shipped, 100–10,000)")
            btn = gr.Button("Update", variant="primary")
        with gr.Column(scale=3):
            wafer_plot = gr.Plot(label="Wafer maps")
    with gr.Row():
        with gr.Column(scale=3):
            metrics = gr.Markdown()
        with gr.Column(scale=2):
            curve_plot = gr.Plot(label="Lot 5 trade-off")
            cost_plot = gr.Plot(label="Lot 5 cost per wafer")
    gr.Markdown(SCOPE_MD)
    inputs = [wafer, mode, threshold, budget, c_burn, c_esc]
    outputs = [wafer_plot, curve_plot, cost_plot, metrics]
    mode.change(toggle, mode, [threshold, budget])
    gr.on([btn.click, wafer.change, mode.change, budget.release, threshold.release, c_burn.submit, c_esc.submit],
          run, inputs, outputs)
    demo.load(run, inputs, outputs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also create a temporary public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port)
