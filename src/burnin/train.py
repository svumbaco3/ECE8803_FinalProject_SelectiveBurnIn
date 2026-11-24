import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from .config import GRID_H, GRID_W, WM811K_CLASSES
from .models import WaferClassifier, UNet
from .metrics import screening_curve


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wm_to_kai(maps_u8):
    oh = F.one_hot(maps_u8.long(), 3).permute(0, 3, 1, 2).float()
    return F.interpolate(oh, size=(GRID_H, GRID_W), mode="nearest")


def random_flips(*tensors, gen=None):
    out = list(tensors)
    if torch.rand(1, generator=gen).item() < 0.5:
        out = [t.flip(-1) for t in out]
    if torch.rand(1, generator=gen).item() < 0.5:
        out = [t.flip(-2) for t in out]
    return out


def pretrain_encoder(X, y, tr_idx, va_idx, epochs=12, batch=64, lr=1e-3, seed=0, log=print):
    seed_all(seed)
    dev = get_device()
    classes = WM811K_CLASSES
    yi = np.array([classes.index(c) for c in y])
    Xt = torch.tensor(X, dtype=torch.uint8, device=dev)
    yt = torch.tensor(yi, device=dev)
    counts = np.bincount(yi[tr_idx], minlength=len(classes))
    wts = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    net = WaferClassifier(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * int(np.ceil(len(tr_idx) / batch)))
    gen = torch.Generator().manual_seed(seed)
    hist, best = [], (-1, None)
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        perm = torch.tensor(tr_idx)[torch.randperm(len(tr_idx), generator=gen)]
        tot = 0.0
        for k in range(0, len(perm), batch):
            b = perm[k:k + batch].to(dev)
            xb = wm_to_kai(Xt[b])
            (xb,) = random_flips(xb, gen=gen)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(net(xb), yt[b], weight=wts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        pred = predict_classes(net, Xt, va_idx, batch)
        f1 = f1_score(yi[va_idx], pred, average="macro")
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr_idx), "val_macro_f1": f1})
        log(f"epoch {ep + 1:2d}  loss {tot / len(tr_idx):.3f}  val macro-F1 {f1:.3f}  ({time.time() - t0:.0f}s)")
        if f1 > best[0]:
            best = (f1, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])
    return net, hist


@torch.no_grad()
def predict_classes(net, Xt, idx, batch=128):
    net.eval()
    dev = Xt.device
    out = []
    for k in range(0, len(idx), batch):
        b = torch.tensor(idx[k:k + batch], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(net(wm_to_kai(Xt[b])).argmax(1).cpu())
    return torch.cat(out).numpy()


def _val_metric(score, latent, mask):
    c = screening_curve(score, latent, mask)
    return c.average_precision(), c.recall_at(0.10)


def train_unet(X_tr, y_tr, m_tr, X_va=None, y_va=None, m_va=None, encoder_state=None, epochs=150, batch=4,
               lr=1e-3, pos_weight=10.0, weight_decay=1e-4, dropout=0.0, input_skip=True, z_noise=0.0,
               seed=0, eval_every=5, gpu_resident_gb=1.0, log=print):
    seed_all(seed)
    dev = get_device()
    net = UNet(X_tr.shape[1], dropout=dropout, input_skip=input_skip).to(dev)
    if encoder_state is not None:
        net.load_pretrained_encoder(encoder_state)
    store = dev if X_tr.nbytes / 2**30 <= gpu_resident_gb else torch.device("cpu")
    Xt = torch.tensor(X_tr, device=store)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=store)
    mt = torch.tensor(m_tr, dtype=torch.float32, device=store)
    if store.type == "cpu" and dev.type == "cuda":
        Xt, yt, mt = Xt.pin_memory(), yt.pin_memory(), mt.pin_memory()
    coords = Xt[:1, -2:].to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
    steps = epochs * int(np.ceil(len(Xt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    pw = torch.tensor(pos_weight, device=dev)
    gen = torch.Generator().manual_seed(seed)
    hist, best = [], {"ap": -1, "epoch": epochs, "state": None}
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = 0.0
        for k in range(0, len(perm), batch):
            b = perm[k:k + batch].to(store)
            xb, yb, mb = (t[b].to(dev, non_blocking=True) for t in (Xt, yt, mt))
            xb, yb, mb = random_flips(xb, yb, mb, gen=gen)
            xb = xb.clone()
            xb[:, -2:] = coords
            if z_noise > 0:
                xb[:, 3:8] += z_noise * torch.randn_like(xb[:, 3:8]) * xb[:, 1:2]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = net(xb)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb, pos_weight=pw, reduction="none")
            loss = (loss * mb).sum() / mb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        row = {"epoch": ep + 1, "train_loss": tot / int(np.ceil(len(Xt) / batch))}
        if X_va is not None and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            ap, r10 = _val_metric(predict_unet(net, X_va), y_va, m_va)
            row.update(val_ap=ap, val_recall10=r10)
            if ap > best["ap"]:
                best = {"ap": ap, "epoch": ep + 1, "state": copy.deepcopy(net.state_dict())}
            log(f"epoch {ep + 1:3d}  loss {row['train_loss']:.4f}  val AP {ap:.3f}  val recall@10% {r10:.3f}  ({time.time() - t0:.0f}s)")
        hist.append(row)
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    return net, hist, best["epoch"]


@torch.no_grad()
def predict_unet(net, X, batch=8):
    net.eval()
    dev = next(net.parameters()).device
    out = []
    for k in range(0, len(X), batch):
        xb = torch.tensor(X[k:k + batch], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(net(xb).float()).cpu())
    return torch.cat(out).numpy()
