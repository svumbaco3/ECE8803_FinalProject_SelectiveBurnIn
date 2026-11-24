from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Curve:
    burn_frac: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    threshold: np.ndarray
    n_pass: int
    n_latent: int
    n_wafers: int

    def recall_at(self, frac):
        return float(np.interp(frac, self.burn_frac, self.recall))

    def burn_at(self, recall):
        return float(np.interp(recall, self.recall, self.burn_frac))

    def auc(self):
        return float(np.trapezoid(self.recall, self.burn_frac))

    def average_precision(self):
        r = self.recall
        return float(np.sum(np.diff(r) * self.precision[1:]))

    def dppm_at(self, frac):
        caught = self.recall_at(frac) * self.n_latent
        esc = self.n_latent - caught
        return float(esc / (self.n_pass - caught) * 1e6)

    def cost_per_wafer(self, frac, c_burn, c_escape):
        esc = self.n_latent * (1 - self.recall_at(frac))
        return float((frac * self.n_pass * c_burn + esc * c_escape) / self.n_wafers)


def screening_curve(score, latent, mask):
    s = np.asarray(score)[mask].astype(np.float64)
    y = np.asarray(latent)[mask].astype(bool)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    n, npos = len(s), int(y.sum())
    k = last + 1
    burn = np.r_[0.0, k / n]
    rec = np.r_[0.0, tp[last] / max(npos, 1)]
    prec = np.r_[1.0, tp[last] / k]
    thr = np.r_[np.inf, s[last]]
    return Curve(burn, rec, prec, thr, n, npos, int(np.asarray(mask).shape[0]))


def summarize(curve, name, fracs=(0.05, 0.10, 0.20), recalls=(0.8, 0.9)):
    row = {"method": name}
    for f in fracs:
        row[f"recall@{int(f*100)}%burn"] = curve.recall_at(f)
    for r in recalls:
        row[f"burn%@{int(r*100)}%recall"] = curve.burn_at(r)
    row["AUC"] = curve.auc()
    row["AP"] = curve.average_precision()
    row["DPPM@10%burn"] = curve.dppm_at(0.10)
    return row


def compare(scores, ws, **kw):
    rows, curves = [], {}
    for name, s in scores.items():
        c = screening_curve(s, ws.latent, ws.passing)
        curves[name] = c
        rows.append(summarize(c, name, **kw))
    return pd.DataFrame(rows).set_index("method"), curves


def bootstrap_ci(score, ws, stat=lambda c: c.recall_at(0.10), n_boot=500, seed=0, alpha=0.05, levels=20000):
    """Wafer bootstrap. Each resample reweights wafers by their draw count; scores are grouped into at most
    `levels` quantile bins (exact when there are fewer distinct scores). `stat` may be a list of callables."""
    stats = stat if isinstance(stat, (list, tuple)) else [stat]
    rng = np.random.default_rng(seed)
    n = len(ws)
    mask = np.asarray(ws.passing)
    wid = np.broadcast_to(np.arange(n)[:, None, None], mask.shape)[mask]
    s = np.asarray(score)[mask].astype(np.float64)
    y = np.asarray(ws.latent)[mask].astype(bool)
    u = np.unique(s)
    edges = u if len(u) <= levels else np.unique(np.quantile(s, np.linspace(0, 1, levels + 1)[1:-1]))
    b = len(edges) - np.searchsorted(edges, s, side="left") if len(u) <= levels else \
        len(edges) - np.searchsorted(edges, s, side="right")
    L = int(b.max()) + 1
    cnt = np.bincount(wid * L + b, minlength=n * L).reshape(n, L).astype(np.float64)
    pos = np.bincount(wid * L + b, weights=y, minlength=n * L).reshape(n, L)
    vals = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        k, tp = np.cumsum(w @ cnt), np.cumsum(w @ pos)
        keep = np.r_[True, np.diff(k) > 0] & (k > 0)
        k, tp = k[keep], tp[keep]
        tot, npos = k[-1], tp[-1]
        c = Curve(np.r_[0.0, k / tot], np.r_[0.0, tp / max(npos, 1)], np.r_[1.0, tp / k], np.full(len(k) + 1, np.nan),
                  int(tot), int(npos), n)
        vals.append([f(c) for f in stats])
    q = np.quantile(np.asarray(vals), [alpha / 2, 1 - alpha / 2], axis=0)
    out = [(float(q[0, i]), float(q[1, i])) for i in range(len(stats))]
    return out if isinstance(stat, (list, tuple)) else out[0]


def recall_ceiling(ws, near=2.0):
    lat = ws.latent & ws.passing
    catchable = lat & (ws.extra["clued"] | (ws.extra["dist"] <= near))
    return float(catchable.sum() / max(lat.sum(), 1))


def per_group_recall(score, ws, frac, groups):
    c = screening_curve(score, ws.latent, ws.passing)
    i = min(np.searchsorted(c.burn_frac, frac), len(c.threshold) - 1)
    flagged = (score >= c.threshold[i]) & ws.passing
    lat = ws.latent & ws.passing
    out = {}
    for g in np.unique(groups[lat]):
        m = lat & (groups == g)
        out[g] = (float((flagged & m).sum() / m.sum()), int(m.sum()))
    return out
