# Gradio demo

```
python demo/app.py            # opens http://127.0.0.1:7860
python demo/app.py --share    # also prints a temporary public gradio.live link (e.g. for the presentation)
```

**Inputs (validated; out-of-range values are rejected with a scope message)**
* Lot 5 wafer (801–1000), labelled with its simulated defect pattern.
* Screening condition: a burn-in budget (% of all Lot 5 passing dies) or a risk threshold t (0–1).
  One threshold is applied to the whole lot, as in the project's evaluation.
* Burn-in cost (\$0.05–\$5.00 per die) and escape cost (\$100–\$10,000 per latent defect shipped).

**Outputs**
* Wafer-Test Fail map (model input), U-Net risk map (prediction), and the burn-in decision map: Flagged / skipped
  dies, with Latent Defective dies marked caught or escaped.
* Metrics with units for the wafer and for all of Lot 5: dies Flagged (% of passing), recall, escapes (DPPM),
  precision, and cost per wafer for selective burn-in, burn-in-all and burn-in-none.
* Lot 5 burn-in % vs. recall curve with the current operating point, and a "Scope and units" panel.

**Data:** `demo/lot5_demo.npz` (5.8 MB) holds the saved Lot 5 U-Net scores, die/fail maps and the simulator's latent
labels (used only to score the result). The model is not re-run. Rebuild it with
`python demo/build_demo_data.py --sim sim_full` after re-running notebook 05.
