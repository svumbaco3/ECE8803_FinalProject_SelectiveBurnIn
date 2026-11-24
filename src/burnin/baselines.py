import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import GRID_H, GRID_W

FAR = 1000.0


def random_score(ws, seed=0):
    return np.random.default_rng(seed).random(ws.fail.shape)


def fail_distance(ws):
    d = np.full(ws.fail.shape, FAR, np.float32)
    for i in range(len(ws)):
        if ws.fail[i].any():
            d[i] = ndimage.distance_transform_edt(~ws.fail[i])
    return d


def distance_to_fail(ws):
    return -fail_distance(ws)


def wafer_fail_fraction(ws):
    f = ws.fail.sum((1, 2)) / ws.die.sum((1, 2))
    return np.broadcast_to(f[:, None, None], ws.fail.shape).astype(np.float32)


def wafer_fallback(ws):
    return wafer_fail_fraction(ws)


def _disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2 <= r * r).astype(np.float32)


def fail_density(ws, r):
    k = _disk(r)
    num = ndimage.convolve(ws.fail.astype(np.float32), k[None], mode="constant")
    den = ndimage.convolve(ws.die.astype(np.float32), k[None], mode="constant")
    return num / np.maximum(den, 1)


def neighbor_residual(ws, norm, size=5):
    z = norm.z(ws)
    m = ws.passing.astype(np.float32)
    k = np.ones((size, size), np.float32)
    k[size // 2, size // 2] = 0
    den = ndimage.convolve(m, k[None], mode="constant")
    r = np.empty_like(z)
    for c in range(z.shape[1]):
        num = ndimage.convolve(z[:, c] * m, k[None], mode="constant")
        r[:, c] = z[:, c] - num / np.maximum(den, 1)
    out = np.zeros(ws.fail.shape, np.float32)
    for i in range(len(ws)):
        p = ws.passing[i]
        for c in range(z.shape[1]):
            rc = r[i, c]
            med = np.median(rc[p])
            mad = 1.4826 * np.median(np.abs(rc[p] - med))
            out[i] = np.maximum(out[i], np.abs(rc - med) / max(mad, 1e-6))
    return np.where(ws.passing, out, 0)


class Hotelling:
    def fit(self, ws, norm):
        z = np.moveaxis(norm.z(ws), 1, -1)[ws.passing]
        self.mu = z.mean(0)
        self.inv = np.linalg.inv(np.cov(z, rowvar=False))
        return self

    def score(self, ws, norm):
        z = np.moveaxis(norm.z(ws), 1, -1) - self.mu
        return np.einsum("...i,ij,...j->...", z, self.inv, z).astype(np.float32)


def radius_map():
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W].astype(np.float32)
    return np.sqrt(((xx - (GRID_W - 1) / 2) / (GRID_W / 2)) ** 2 + ((yy - (GRID_H - 1) / 2) / (GRID_H / 2)) ** 2)


FEATURES = ["dist_to_fail", "fail_dens_r1", "fail_dens_r2", "fail_dens_r4", "fail_dens_r8", "wafer_fail_frac",
            "radius", "abs_z_1", "abs_z_2", "abs_z_3", "abs_z_4", "abs_z_5", "T2", "neighbor_residual"]


def die_features(ws, norm, hot):
    z = np.abs(norm.z(ws))
    cols = [np.minimum(fail_distance(ws), 50), *[fail_density(ws, r) for r in (1, 2, 4, 8)],
            wafer_fail_fraction(ws), np.broadcast_to(radius_map(), ws.fail.shape),
            *[z[:, c] for c in range(z.shape[1])], hot.score(ws, norm), neighbor_residual(ws, norm)]
    return np.stack(cols, -1).astype(np.float32)


class FeatureGBM:
    def __init__(self, seed=0, **kw):
        params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                      early_stopping=True, validation_fraction=0.15, random_state=seed)
        params.update(kw)
        self.clf = HistGradientBoostingClassifier(**params)

    def fit(self, ws, norm, hot):
        F = die_features(ws, norm, hot)[ws.passing]
        self.clf.fit(F, ws.latent[ws.passing])
        return self

    def score(self, ws, norm, hot):
        F = die_features(ws, norm, hot)
        out = np.zeros(ws.fail.shape, np.float32)
        out[ws.passing] = self.clf.predict_proba(F[ws.passing])[:, 1]
        return out
