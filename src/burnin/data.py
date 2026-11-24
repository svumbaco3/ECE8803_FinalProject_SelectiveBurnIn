from dataclasses import dataclass, field
import numpy as np
from .config import GRID_H, GRID_W, PARAMS

VISIBLE_KEYS = {"lot", "wafer", "x", "y", "wafer_test_fail", *PARAMS}
ANALYSIS_KEYS = {"cluster", "dist", "risk_p", "clued", "injected_fail", "fail_params", "clue_params"}
CHANNELS = ["outside", "pass", "fail", *[f"z_{p}" for p in PARAMS], "x_coord", "y_coord"]


@dataclass
class WaferSet:
    wafer: np.ndarray
    lot: np.ndarray
    params: np.ndarray
    fail: np.ndarray
    die: np.ndarray
    latent: np.ndarray = None
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.wafer)

    @property
    def passing(self):
        return self.die & ~self.fail

    def subset(self, mask):
        idx = np.flatnonzero(mask)
        return WaferSet(self.wafer[idx], self.lot[idx], self.params[idx], self.fail[idx], self.die[idx],
                        None if self.latent is None else self.latent[idx],
                        {k: v[idx] for k, v in self.extra.items()})

    def lots(self, names):
        return self.subset(np.isin(self.lot, list(names)))


def _grid_index(wafer_col, x, y):
    wafers = np.unique(wafer_col)
    w_idx = np.searchsorted(wafers, wafer_col)
    return wafers, w_idx, y - 1, x - 1


def load_visible(path):
    v = np.load(path)
    extra_cols = set(v.files) - VISIBLE_KEYS
    if extra_cols:
        raise ValueError(f"visible file has non-allowed columns: {sorted(extra_cols)}")
    wafers, wi, yi, xi = _grid_index(v["wafer"], v["x"], v["y"])
    n = len(wafers)
    params = np.full((n, len(PARAMS), GRID_H, GRID_W), np.nan, np.float32)
    for k, p in enumerate(PARAMS):
        params[wi, k, yi, xi] = v[p]
    fail = np.zeros((n, GRID_H, GRID_W), bool)
    fail[wi, yi, xi] = v["wafer_test_fail"]
    die = np.zeros((n, GRID_H, GRID_W), bool)
    die[wi, yi, xi] = True
    lot = np.empty(n, dtype=v["lot"].dtype)
    lot[wi] = v["lot"]
    return WaferSet(wafers, lot, params, fail, die)


def attach_hidden(ws, path, analysis=False):
    h = np.load(path)
    wafers, wi, yi, xi = _grid_index(h["wafer"], h["x"], h["y"])
    if not np.array_equal(wafers, ws.wafer):
        raise ValueError("hidden and visible wafer lists differ")
    lat = np.zeros_like(ws.fail)
    lat[wi, yi, xi] = h["latent"]
    if (lat & ws.fail).any():
        raise ValueError("latent label set on a wafer-test fail die")
    ws.latent = lat
    if analysis:
        for k in ANALYSIS_KEYS & set(h.files):
            fill = np.inf if k == "dist" else 0
            a = np.full(ws.fail.shape, fill, h[k].dtype)
            a[wi, yi, xi] = h[k]
            ws.extra[k] = a
    return ws


class Normalizer:
    def __init__(self, clip=8.0):
        self.clip = clip
        self.mean = None
        self.std = None

    def fit(self, ws):
        m = ws.passing
        vals = np.moveaxis(ws.params, 1, -1)[m]
        self.mean = vals.mean(0).astype(np.float32)
        self.std = vals.std(0).astype(np.float32)
        return self

    def z(self, ws):
        z = (ws.params - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        z = np.where(ws.die[:, None], z, 0.0)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


def coord_channels():
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W].astype(np.float32)
    return np.stack([2 * xx / (GRID_W - 1) - 1, 2 * yy / (GRID_H - 1) - 1])


DERIVED = [f"abs_z_{p}" for p in PARAMS] + ["dist_to_fail"]


def channel_names(derived=False):
    return CHANNELS[:8] + (DERIVED if derived else []) + CHANNELS[8:]


def build_inputs(ws, norm, derived=False, dist_clip=20.0):
    from .baselines import fail_distance
    names = channel_names(derived)
    out = np.empty((len(ws), len(names), GRID_H, GRID_W), np.float32)
    out[:, 0] = ~ws.die
    out[:, 1] = ws.passing
    out[:, 2] = ws.fail
    z = norm.z(ws)
    out[:, 3:8] = z
    if derived:
        out[:, 8:13] = np.abs(z)
        out[:, 13] = np.where(ws.die, np.minimum(fail_distance(ws), dist_clip) / dist_clip, 0)
    out[:, -2:] = coord_channels()[None]
    return out
