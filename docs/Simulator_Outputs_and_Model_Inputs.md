# Simulator Outputs and Model Inputs

How the files written by `01_simulator.ipynb` (in `ECE8803_FinalProject_SelectiveBurnIn/cache/sim_dev/`) are used by the model.

## Files

| File | Used for | Contents used |
|---|---|---|
| `visible.npz` | Model **inputs** (training and test) | `parameter_1`–`parameter_5`, `wafer_test_fail`, `x`, `y` (plus `lot`, `wafer` for grouping) |
| `hidden.npz` | Training **target** and evaluation only | `latent` (Latent Defective yes/no), on **passing dies only** |
| `hidden.npz` | **Never** used by the model | `risk_p`, `cluster`, `dist`, `clued`, `injected_fail`, `fail_params`, `clue_params` (simulator internals, kept for analysis and plots) |
| `WM811K Dataset/wm811k_dev.npz` | Encoder **pre-training** only | Maps listed in `pretrain_idx` of `cache/sim_dev/wm811k_split.npz` (never the overlay maps) |

## How the model notebook uses them

1. Load `visible.npz` and reshape each wafer into a 115 × 193 image with channels: 5 parameters, Wafer-Test Fail map, wafer-outline mask, and x/y.
2. Load only `latent` from `hidden.npz` as the per-wafer target map.
3. Compute the training loss only on dies that passed wafer test.
4. Score Lot 5 with the same `latent` labels at the end.

## Why two files

Keeping `visible.npz` (what a fab would see) separate from `hidden.npz` (the simulator's answer key) is a simple, checkable guarantee that the model never sees simulator internals. The simulator notebook asserts that `visible.npz` contains only the allowed columns.
