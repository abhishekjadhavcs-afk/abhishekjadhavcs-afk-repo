"""Binary landslide-class metrics, accumulated globally and per region.

Besides thresholded F1/IoU, average precision (AP) is reported: landslide pixel
share ranges from 1% to 14% between regions, so a threshold tuned on training
regions can be badly calibrated for a held-out one. AP is computed from
probability histograms (1000 bins), which keeps memory constant.
"""
from collections import defaultdict

import torch

from .folds import INVENTORIES

N_BINS = 1000


class BinaryMetrics:
    def __init__(self, threshold=0.5):
        self.t = threshold
        self.counts = defaultdict(lambda: torch.zeros(3, dtype=torch.long))  # tp, fp, fn
        self.hist = defaultdict(lambda: torch.zeros(2, N_BINS, dtype=torch.long))  # neg, pos

    @torch.no_grad()
    def update(self, logits, target, region):
        prob = torch.sigmoid(logits.squeeze(1).float())
        pred = prob > self.t
        tgt = target.bool()
        bins = (prob * N_BINS).long().clamp_(0, N_BINS - 1)
        for r in region.unique():
            m = region == r
            p, t, b = pred[m], tgt[m], bins[m]
            c = torch.stack([(p & t).sum(), (p & ~t).sum(), (~p & t).sum()]).cpu()
            h = torch.stack([torch.bincount(b[~t], minlength=N_BINS),
                             torch.bincount(b[t], minlength=N_BINS)]).cpu()
            for key in (INVENTORIES[int(r)], "all"):
                self.counts[key] += c
                self.hist[key] += h

    @staticmethod
    def _ap(h):
        neg, pos = h[0].flip(0).cumsum(0).double(), h[1].flip(0).cumsum(0).double()
        if pos[-1] == 0:
            return 0.0
        precision = pos / (pos + neg).clamp_min(1)
        recall_step = h[1].flip(0).double() / pos[-1]
        return float((precision * recall_step).sum())

    def _scores(self, key):
        tp, fp, fn = (float(v) for v in self.counts[key])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        return {"f1": f1, "iou": iou, "precision": prec, "recall": rec, "ap": self._ap(self.hist[key])}

    def compute(self):
        return {k: self._scores(k) for k in sorted(self.counts)}
