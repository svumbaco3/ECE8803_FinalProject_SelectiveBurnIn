import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GRID_H, GRID_W

PAD_H = (-GRID_H) % 8
PAD_W = (-GRID_W) % 8


def pad_wafer(x):
    x = F.pad(x, (0, PAD_W, 0, PAD_H))
    x[:, 0, GRID_H:, :] = 1
    x[:, 0, :, GRID_W:] = 1
    return x


def double_conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, in_ch=3, widths=(32, 64, 128, 128)):
        super().__init__()
        self.widths = widths
        chans = [in_ch, *widths]
        self.stages = nn.ModuleList(double_conv(chans[k], chans[k + 1]) for k in range(len(widths)))

    def forward(self, x):
        feats = []
        for k, s in enumerate(self.stages):
            x = s(x if k == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        return feats


class WaferClassifier(nn.Module):
    def __init__(self, n_classes=9, in_ch=3):
        super().__init__()
        self.encoder = Encoder(in_ch)
        self.head = nn.Linear(self.encoder.widths[-1], n_classes)

    def forward(self, x):
        f = self.encoder(pad_wafer(x))[-1]
        return self.head(f.mean((2, 3)))


class UNet(nn.Module):
    def __init__(self, in_ch=10, dropout=0.0, input_skip=True):
        super().__init__()
        self.encoder = Encoder(in_ch)
        self.input_skip = input_skip
        w = self.encoder.widths
        self.dec = nn.ModuleList(double_conv(w[k + 1] + w[k], w[k]) for k in reversed(range(len(w) - 1)))
        self.drop = nn.Dropout2d(dropout)
        h = w[0] + (in_ch if input_skip else 0)
        self.out = nn.Sequential(nn.Conv2d(h, w[0], 1), nn.ReLU(inplace=True), nn.Conv2d(w[0], 1, 1))

    def forward(self, x):
        xp = pad_wafer(x)
        feats = self.encoder(xp)
        h = feats[-1]
        for d, skip in zip(self.dec, reversed(feats[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = d(torch.cat([self.drop(h), skip], 1))
        if self.input_skip:
            h = torch.cat([h, xp], 1)
        return self.out(h)[:, 0, :GRID_H, :GRID_W]

    def load_pretrained_encoder(self, state, n_pre_ch=3):
        own = self.encoder.state_dict()
        for k, v in state.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v
            elif v.dim() == 4 and own[k].shape[1] > v.shape[1]:
                w = own[k].clone()
                w[:, :n_pre_ch] = v
                own[k] = w
        self.encoder.load_state_dict(own)
