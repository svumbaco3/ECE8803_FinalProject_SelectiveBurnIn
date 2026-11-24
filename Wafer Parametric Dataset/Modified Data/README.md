# Simulated Wafer Test Data - First 10 Wafers per Lot

Subset of a larger simulated wafer data set: the first 10 wafers of each of the 5 lots.

## Files

| File | Lot | Wafers | Role |
|---|---|---|---|
| training_1_first10.csv | Lot_1 | 1-10 | Training |
| training_2_first10.csv | Lot_2 | 201-210 | Training |
| training_3_first10.csv | Lot_3 | 401-410 | Training |
| training_4_first10.csv | Lot_4 | 601-610 | Training |
| test_first10.csv | Lot_5 | 801-810 | Test |

`subset_summary.json` lists the exact wafers and row count per file.

## Columns

| Column | Description |
|---|---|
| lot | Lot ID |
| wafer | Wafer number (unique across all lots) |
| x | Die x coordinate, 1-115 |
| y | Die y coordinate, 1-193 |
| parameter_1 ... parameter_5 | Simulated parametric test values |

- Comma-separated, dot decimal, one row per die (about 17,509 die per wafer).
- A regular grid of coordinates inside the wafer outline has no row (no die at those sites).

## How the parameters were generated

Each parameter is an independent normal distribution, `value = mean + std * N(0,1)`, drawn per die with a fixed random seed.
The mean and std of each parameter match those of a column in the source data set
("Simulated Analog Wafer Test Data for Pattern Recognition", KAI, 2018), computed over the full lot (200 wafers).
There is no spatial pattern: die near or beyond 4-5 sigma occur at random locations on the wafer.

Full-lot mean / std per parameter:

| Lot | parameter_1 | parameter_2 | parameter_3 | parameter_4 | parameter_5 |
|---|---|---|---|---|---|
| Lot_1 | 0.005732 / 0.01215 | 0.0002654 / 0.0007226 | 0.9212 / 0.7625 | 0.004547 / 0.007132 | 0.05234 / 0.1037 |
| Lot_2 | 0.005588 / 0.01192 | 0.0002684 / 0.0007170 | 0.9898 / 0.8114 | 0.004536 / 0.006937 | 0.05286 / 0.1066 |
| Lot_3 | 0.005334 / 0.01127 | 0.0002659 / 0.0007166 | 1.0452 / 0.7591 | 0.004543 / 0.007060 | 0.05242 / 0.1039 |
| Lot_4 | 0.005224 / 0.01100 | 0.0002667 / 0.0006954 | 0.9380 / 0.8070 | 0.004547 / 0.007102 | 0.05311 / 0.1079 |
| Lot_5 | 0.005354 / 0.01074 | 0.0002665 / 0.0007069 | 1.0289 / 0.7756 | 0.004542 / 0.007020 | 0.05252 / 0.1046 |

A 10-wafer subset will show sample mean/std close to, but not exactly equal to, these values.
