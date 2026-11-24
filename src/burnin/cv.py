import numpy as np
from . import baselines as B
from .config import TRAIN_LOTS
from .data import Normalizer, build_inputs
from .train import train_unet, predict_unet

UNET_DEFAULTS = dict(derived=True, epochs=None, lr=2e-3, pos_weight=10.0, dropout=0.2, weight_decay=1e-2,
                     batch=4, eval_every=None)


def auto_epochs(n_train_wafers):
    return int(np.clip(round(1800 / n_train_wafers), 20, 60))


def fit_baselines(train_ws, seed=0):
    norm = Normalizer().fit(train_ws)
    hot = B.Hotelling().fit(train_ws, norm)
    gbm = B.FeatureGBM(seed=seed).fit(train_ws, norm, hot)
    return {"norm": norm, "hot": hot, "gbm": gbm}


def score_baselines(fitted, ws, seed=0):
    norm, hot = fitted["norm"], fitted["hot"]
    return {
        "Random": B.random_score(ws, seed),
        "Distance to fail": B.distance_to_fail(ws),
        "Wafer-level fallback": B.wafer_fallback(ws),
        "Neighbor z-score": B.neighbor_residual(ws, norm),
        "Hotelling T2 (PCA)": hot.score(ws, norm),
        "Feature GBM": fitted["gbm"].score(ws, norm, hot),
    }


def _empty_like(ws):
    return np.zeros(ws.fail.shape, np.float32)


def lolo_baselines(ws, lots=TRAIN_LOTS, seed=0, log=print):
    oof = {}
    for val in lots:
        tr, va = ws.lots([l for l in lots if l != val]), np.isin(ws.lot, [val])
        sc = score_baselines(fit_baselines(tr, seed), ws.subset(va), seed)
        for k, v in sc.items():
            oof.setdefault(k, _empty_like(ws))[va] = v
        log(f"fold {val} done")
    return oof


def unet_inputs(train_ws, ws_list, derived):
    norm = Normalizer().fit(train_ws)
    return norm, [build_inputs(w, norm, derived) for w in ws_list]


def lolo_unet(ws, encoder_state=None, lots=TRAIN_LOTS, seed=0, log=print, **cfg):
    c = {**UNET_DEFAULTS, **cfg}
    derived = c.pop("derived")
    oof, best_epochs, hists = _empty_like(ws), {}, {}
    for val in lots:
        tr, vm = ws.lots([l for l in lots if l != val]), np.isin(ws.lot, [val])
        va = ws.subset(vm)
        _, (Xa, Xb) = unet_inputs(tr, [tr, va], derived)
        ep = c["epochs"] or auto_epochs(len(tr))
        ev = c["eval_every"] or max(1, ep // 12)
        log(f"--- fold: validate on {val} ({len(tr)} train wafers, {ep} epochs)")
        net, h, be = train_unet(Xa, tr.latent, tr.passing, Xb, va.latent, va.passing,
                                encoder_state=encoder_state, seed=seed, log=log,
                                **{**c, "epochs": ep, "eval_every": ev})
        oof[vm] = np.where(va.passing, predict_unet(net, Xb), 0)
        best_epochs[val], hists[val] = be, h
    return oof, best_epochs, hists


def fit_final_unet(train_ws, epochs, encoder_state=None, seed=0, log=print, **cfg):
    c = {**UNET_DEFAULTS, **cfg, "epochs": epochs}
    derived = c.pop("derived")
    c.pop("eval_every", None)
    norm, (Xa,) = unet_inputs(train_ws, [train_ws], derived)
    net, h, _ = train_unet(Xa, train_ws.latent, train_ws.passing, encoder_state=encoder_state, seed=seed,
                           log=log, **c)
    return net, norm, derived, h


def load_unet(path):
    import torch
    from .models import UNet
    from .train import get_device
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    net = UNet(ck["in_ch"], dropout=cfg.get("dropout", 0.0), input_skip=cfg.get("input_skip", True))
    net.load_state_dict(ck["state"])
    norm = Normalizer()
    norm.mean = np.array(ck["mean"], np.float32)
    norm.std = np.array(ck["std"], np.float32)
    return net.to(get_device()), norm, ck["derived"], ck


def score_unet(net, norm, derived, ws):
    return np.where(ws.passing, predict_unet(net, build_inputs(ws, norm, derived)), 0)
