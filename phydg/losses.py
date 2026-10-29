"""Segmentation losses and the cross-region invariance penalty."""
import torch
import torch.nn.functional as F


def focal_tversky(logits, target, alpha=0.3, beta=0.7, gamma=0.75, dim=(1, 2, 3)):
    """Per-sample Focal-Tversky loss; beta > alpha penalises missed landslide pixels more."""
    p = torch.sigmoid(logits)
    t = target.float()
    tp = (p * t).sum(dim)
    fp = (p * (1 - t)).sum(dim)
    fn = ((1 - p) * t).sum(dim)
    ti = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return (1 - ti) ** gamma


def per_sample_loss(logits, target, pos_weight=5.0):
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, t, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction="none"
    ).mean(dim=(1, 2, 3))
    return bce + focal_tversky(logits, target)


def vrex_penalty(sample_losses, region):
    """V-REx (Krueger et al., 2021): variance of the mean loss across regions in the batch.

    Pushes the model towards solutions whose risk is equal in every training region,
    i.e. that rely on cues that hold across regions rather than region-specific ones.
    """
    risks = torch.stack([sample_losses[region == r].mean() for r in region.unique()])
    return risks.var(unbiased=False) if len(risks) > 1 else sample_losses.new_zeros(())


def total_loss(out, target, region, vrex_weight=0.0, prior_weight=0.1):
    target = target.unsqueeze(1)
    losses = per_sample_loss(out["logits"], target)
    loss = losses.mean()
    logs = {"seg": loss.item()}
    if "prior_logits" in out and prior_weight > 0:
        # keep the terrain expert a calibrated plausibility map on its own
        prior = F.binary_cross_entropy_with_logits(out["prior_logits"], target.float())
        loss = loss + prior_weight * prior
        logs["prior"] = prior.item()
    if vrex_weight > 0:
        pen = vrex_penalty(losses, region)
        loss = loss + vrex_weight * pen
        logs["vrex"] = pen.item()
    return loss, logs
