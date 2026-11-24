"""Pack the Lot 5 data the Gradio demo needs into one small file.

python demo/build_demo_data.py [--sim sim_full]   -> demo/lot5_demo.npz
Contents per Lot 5 wafer ([wafer, y-1, x-1] grids): die site, Wafer-Test Fail, saved U-Net risk score,
and, for scoring only, the simulator's Latent Defective label and whether each latent die is catchable
(clued, or within 2 dies of a cluster). The model inputs are not needed because the scores are precomputed.
Also stores the Lot 5 recall vs. burn-in curves of the methods on the cost slides, so the demo can redraw
cost per wafer for any burn-in / escape cost.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from burnin import Paths
from burnin.config import TEST_LOT
from burnin.data import attach_hidden, load_visible
from burnin.metrics import screening_curve

NEAR_DIST = 2.0
COST_METHODS = ["U-Net (pre-trained)", "Feature GBM", "Distance to fail", "Random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="sim_full")
    a = ap.parse_args()
    P = Paths(ROOT, sim=a.sim)
    test = attach_hidden(load_visible(P.visible), P.hidden, analysis=True).lots([TEST_LOT])
    sc = np.load(P.results / "lot5_scores.npz")
    assert (sc["wafer"] == test.wafer).all()
    risk = np.where(test.passing, sc["U-Net (pre-trained)"], 0).astype(np.float32)
    catchable = test.latent & (test.extra["clued"].astype(bool) | (test.extra["dist"] <= NEAR_DIST))
    frac = np.linspace(0, 1, 401)
    curves = {k: screening_curve(sc[k], test.latent, test.passing) for k in COST_METHODS}
    recall = np.array([[curves[k].recall_at(f) for f in frac] for k in COST_METHODS])
    wafers = pd.read_csv(P.sim_dir / "wafers.csv").set_index("wafer")
    out = ROOT / "demo" / "lot5_demo.npz"
    np.savez_compressed(out, wafer=test.wafer, pattern=wafers.loc[test.wafer, "pattern"].to_numpy().astype(str),
                        die=test.die, fail=test.fail, latent=test.latent, catchable=catchable, risk=risk,
                        curve_methods=np.array(COST_METHODS), curve_frac=frac, curve_recall=recall)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(test)} wafers)")


if __name__ == "__main__":
    main()
