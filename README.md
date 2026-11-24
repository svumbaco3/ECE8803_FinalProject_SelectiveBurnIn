# Burn In Only What Matters: Spatial Risk Mapping of Wafer-Sort Data for Selective Burn-In Screening

ECE 8803-AI4, Fall 2026: Nick Saba, Sean Vumbaco

We train a per-wafer **U-Net** that turns wafer-sort data into a risk map: for each passing die, the risk that it is
a *latent* defect that would fail burn-in. Only the riskiest dies are then burned in. Inputs:

* the five parametric test values;
* the Wafer-Test Fail map;
* die position;
* derived neighbour features.

The encoder is pre-trained on WM811K wafer-map patterns. The data is the KAI wafer grid with Gaussian parametric
values. Our simulator (notebook 01) injects defect clusters, wafer-test fails, parametric "clues" and the hidden
latent labels.

## Results: Lot 5 (200 wafers, held out, evaluated once)

All models were trained and tuned on Lots 1–4 only, with leave-one-lot-out cross-validation. Brackets are 95%
wafer-bootstrap confidence intervals.

| Method | Recall @ 10% burn-in | Burn-in % @ 90% recall | AP |
|---|---|---|---|
| **U-Net (WM811K pre-trained)** | **59.6%** [58.7, 60.6] | **45.4%** [42.9, 47.8] | 0.296 |
| U-Net (no pre-training) | 59.7% [58.7, 60.7] | 45.2% [43.1, 47.6] | 0.297 |
| Feature GBM (13 hand-built per-die features) | 59.7% [58.7, 60.7] | 46.0% [43.7, 48.3] | 0.297 |
| Hotelling T² (PCA) | 48.8% [48.1, 49.4] | 74.9% [74.1, 75.5] | 0.232 |
| Neighbor z-score | 46.2% [45.5, 46.9] | 75.1% [74.4, 75.8] | 0.169 |
| Distance to nearest fail | 32.3% [30.0, 34.4] | 73.1% [70.8, 75.4] | 0.037 |
| Wafer-level fallback | 27.8% [23.9, 30.6] | 74.4% [72.1, 76.2] | 0.033 |
| Random | 10.0% | 89.9% | 0.015 |
| Burn in all passing dies | 100% (burns 100%) | 100% | – |

**Detection**
* Burning in 10% of passing dies catches about 60% of latent defects, 6× random selection.
* Catching 90% needs about 45% burn-in instead of 90%.
* The U-Net matches the hand-engineered Feature GBM but does not beat it.
* WM811K pre-training gives no measurable gain at 600 training wafers.

**Ceiling and failure modes**
* About 86% of latent dies are catchable at all (clued, or within 2 dies of a cluster). The rest sets a floor on escapes.
* Clued latent dies: 72% caught. Unclued latent dies: 30% caught.

**Cost, `C = c_burnin·#burned + c_escape·#escaped`**
* At \$0.50/die burn-in and \$100/escape, the optimum burns in 40% of dies, about 25% cheaper than burning in everything.
* At an escape cost of \$1,000 or more, burning in everything is optimal.
* The reason: a die is worth skipping only if its latent probability is below c_burnin/c_escape (0.05% at \$1,000).
  Even cluster-free wafers have about 0.5% latent dies in our simulator.

Lots 1–4 cross-validation gives the same picture (U-Net 59.4% recall @ 10%, GBM 59.3%).
See `results/sim_full/oof_all_table.csv`.

Training the U-Net for 40 epochs instead of 20 changes cross-validation recall @ 10% by less than 0.1 points
(`results/sim_full/unet_epochs_ablation.csv`). The 20-epoch models are the final ones.

## Repository layout

```
notebooks/            01-06 Jupyter notebooks (executed, outputs included); _src/*.py are their sources
  02_baselines_eval     data summary, evaluation harness, 6 baselines (LOLO CV, Lots 1-4)
  03_pretrain_encoder   WM811K 9-class CNN (pretrain_idx maps only) -> encoder weights
  04_unet_train         U-Net LOLO CV (pre-trained vs. scratch), final models on Lots 1-4
  05_final_lot5_eval    the one Lot 5 test: slide 8 table, cost model, corner cases, risk maps, per-die CSV
  06_learning_curve     recall vs. number of training wafers
src/burnin/           shared code: config, data loader, baselines, U-Net model, training, CV, metrics, plots
demo/                 Gradio demo: app.py (python demo/app.py), build_demo_data.py, lot5_demo.npz; see demo/README.md
tools/                build_notebooks.py (rebuild/execute notebooks), fill_deck.py (slides from results),
                      slide_figures.py, split_large_files.py, make_submission.py
cache/sim_dev/        10-wafers-per-lot dev subset (quick runs)
cache/sim_full/       full 200-wafers-per-lot simulator output (visible.npz is split into .partNNN files)
cache/models/         trained weights: encoder_wm811k.pt, unet_{pretrained,scratch}_sim_full.pt
WM811K Dataset/       wm811k_dev.npz (course file)
results/sim_full/     metric tables (.csv/.json), figures/, lot5_scores.npz (risk maps), lot5_risk_per_die.csv.gz
docs/                 project brief, simulator contract, presentation (.pptx/.pdf)
```

Notebook `notebooks/01_simulator.ipynb` (Sean) generates `cache/sim_full/` (`DATASET = "full"`) or `cache/sim_dev/`
(`DATASET = "dev"`). It is seeded and reproduces the committed `sim_full` files exactly.

### Risk-map formats

`results/sim_full/lot5_scores.npz` has one array per method, shaped `[wafer, y-1, x-1]` = 200 × 193 × 115, float32.
Off-wafer and wafer-test-fail sites are 0. The key `wafer` lists the wafer numbers.

`lot5_risk_per_die.csv.gz` has one row per Lot 5 die:

* `lot, wafer, x, y, wafer_test_fail`;
* `risk_unet, risk_unet_scratch, risk_gbm, risk_dist_to_fail` (higher = riskier);
* `burn_in_10pct`.

## Rules the code enforces

* **Allowed model inputs:** `parameter_1..5`, x/y, `wafer_test_fail` and features derived from them.
  `hidden.npz` supplies only the `latent` label for training. Its other fields are used only for analysis
  (recall ceiling, corner cases).
* **No leakage:** normalisation, PCA and model fitting happen inside each fold. Lot 5 is scored once, by notebook 05.
* **Training and scoring:** the loss and metrics use passing dies only. The burn-in threshold is swept, not learned.

## Setup

**Local (Windows/Linux, Python 3.12, optional NVIDIA GPU)**
```
python -m venv burnin_env
burnin_env/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU
burnin_env/Scripts/python -m pip install -r requirements.txt
```
Without a GPU, install plain `torch`. Then open a notebook and select this environment as the kernel.

**Google Colab**
1. Copy this folder to Google Drive.
2. Set `COLAB_ROOT` in the first cell of each notebook.
3. Choose Runtime → GPU.

The device (CUDA or CPU) is detected automatically.

**Data selection**
* All published results use `sim_full` (200 wafers per lot). The code default is still `sim_dev`, so set the
  environment variable `BURNIN_SIM=sim_full`, or edit `SIM = ...` in the setup cell.
* On first use the loader reassembles `cache/sim_full/visible.npz` from its parts and checks its SHA-256.
  To do it by hand: `python tools/split_large_files.py join cache/sim_full/visible.npz`.

**Run order:** 02 → 03 → 04 → 05 → 06.
* Notebook 03 depends only on WM811K. Its weights are included, so you can skip it.
* Full-data run times on an RTX A2000 laptop GPU: 02 ≈ 15 min, 04 ≈ 45–60 min, 05 ≈ 15 min.
* To execute a notebook from the command line:
  `BURNIN_SIM=sim_full python tools/build_notebooks.py 04_unet_train --execute`

**Not included**
* `results/sim_full/oof_baselines.npz` (217 MB). Notebook 02 regenerates it; notebook 06 needs it for its
  600-wafer point.
