"""Binary landslide-class metrics, accumulated globally and per region."""
from collections import defaultdict

import torch

from .folds import INVENTORIES


class BinaryMetrics:
    def __init__(self, threshold=0.5):
        self.t = threshold
        self.counts = defaultdict(lambda: torch.zeros(3, dtype=torch.long))  # tp, fp, fn

    @torch.no_grad()
    def update(self, logits, target, region):
        pred = torch.sigmoid(logits.squeeze(1)) > self.t
        tgt = target.bool()
        for r in region.unique():
            m = region == r
            p, t = pred[m], tgt[m]
            c = torch.stack([(p & t).sum(), (p & ~t).sum(), (~p & t).sum()]).cpu()
            self.counts[INVENTORIES[int(r)]] += c
            self.counts["all"] += c

    @staticmethod
    def _scores(c):
        tp, fp, fn = (float(v) for v in c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        return {"f1": f1, "iou": iou, "precision": prec, "recall": rec}

    def compute(self):
        return {k: self._scores(c) for k, c in sorted(self.counts.items())}
