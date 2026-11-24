# Model notebooks

Notebook `01_simulator` (Sean) generates the simulator data in `cache/sim_full/` (`DATASET = "full"`, 200 wafers per lot) or
`cache/sim_dev/` (`DATASET = "dev"`, first 10 wafers per lot). It finds the project root itself, so it can be run from
`notebooks/`. All published results use `sim_full`. Run the others in order:

| Notebook | Purpose | Main outputs |
|---|---|---|
| `02_baselines_eval` | Data summary and all baselines, Lots 1–4, leave-one-lot-out | `results/sim_full/oof_baselines.npz`, `oof_baselines_table.csv` |
| `03_pretrain_encoder` | WM811K 9-class pre-training of the U-Net encoder (`pretrain_idx` only) | `cache/models/encoder_wm811k.pt` |
| `04_unet_train` | U-Net LOLO CV (pre-trained and no-pre-training ablation), final models on Lots 1–4 | `cache/models/unet_*_sim_full.pt`, `results/sim_full/unet_cv.json` |
| `05_final_lot5_eval` | Single Lot 5 evaluation: slide 8 table, slide 9 cost model, slide 10 corner cases | `results/sim_full/lot5_*`, `results/sim_full/figures/` |
| `06_learning_curve` | Recall vs. training wafers per fold on Lots 1–4; loads the cached CSV unless `RERUN = True` (~45 min GPU on `sim_full`) | `results/sim_full/learning_curve.csv`, `figures/learning_curve.png` |

Approximate full-data (`sim_full`) run time on an RTX A2000 laptop GPU: 02 ≈ 15 min, 03 ≈ 5 min, 04 ≈ 45–60 min,
05 ≈ 15 min. CPU-only runs take much longer.

Note: the saved outputs of `06_learning_curve` are still from a `sim_dev` run; `results/sim_full/learning_curve.csv`
does not exist yet.

## Layout

- `src/burnin/`: shared code.
  - `data.py`: loader, input channels, normaliser.
  - `metrics.py`: recall vs. burn-in curve, DPPM, cost, bootstrap.
  - `baselines.py`, `models.py`, `train.py`, `cv.py`, `plots.py`.
- `cache/sim_full/`: `visible.npz` (stored as `.partNNN` files), `hidden.npz`, `wafers.csv`, `limits.json`, `config.json`,
  `wm811k_split.npz` from the full simulator run. `cache/sim_dev/` holds the same files for the 10-wafer dev subset.
- `WM811K Dataset/wm811k_dev.npz`: course WM811K file.
- `notebooks/_src/*.py`: sources the `.ipynb` files are generated from.

## Running

- **Locally:** select the kernel `C:\burnin_env\Scripts\python.exe` and run all cells.
- **Google Colab:** copy the project folder to Google Drive, set `COLAB_ROOT` in the first cell, and pick
  Runtime → GPU. The device is detected automatically.
- **Rebuild or re-execute notebooks from the sources:**
  `C:\burnin_env\Scripts\python.exe tools\build_notebooks.py [names] [--execute]`.
  Editing an `.ipynb` directly is fine, but rebuilding overwrites it.
## Full 200-wafer data

1. `cache/sim_full/` already contains the full-run simulator output. To regenerate it, run `01_simulator` with
   `DATASET = "full"`; the output is seeded and reproduces the committed files exactly.
2. Select the dataset with the environment variable `BURNIN_SIM=sim_full`, or edit the `SIM = ...` line in each
   notebook's setup cell. The code default is still `sim_dev`, so set this for every run.
3. Run 02 → 04 → 05. Notebook 03 (WM811K pre-training) does not depend on the simulator data and does not
   need to be rerun. Results go to `results/sim_full/`; models are saved as `cache/models/unet_*_sim_full.pt`.

## Sharing and GitHub

- `tools/make_submission.py` builds `submission/ECE8803_FinalProject_SelectiveBurnIn/`, the shareable repo.
  It contains the README (from `SUBMISSION_README.md`), code, executed notebooks, models, the `sim_full` results
  and the v3 deck.
- Files over GitHub's 100 MB limit are split into 45 MB `.partNNN` files plus a `.parts.json` manifest
  (`tools/split_large_files.py`). This applies to `cache/sim_full/visible.npz`. `Paths.visible` and `Paths.hidden`
  reassemble the file automatically on first use and verify its SHA-256.
- `tools/fill_deck.py v2.pptx v3.pptx --sim sim_full` fills slides 6 and 8–11 from `results/<sim>/`.
  It needs `python-pptx`. `tools/slide_figures.py` makes the slide 6 input figures.

The U-Net epoch count scales with training size (`auto_epochs`: 60 epochs at 30 wafers, 20 epochs at
≥90 wafers). Training sets over 1 GB stay in CPU memory and are sent to the GPU one batch at a time.
