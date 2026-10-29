"""Train and evaluate one model on one leave-one-cluster-out fold.

Model selection uses validation patches from the *training* regions only; the
held-out cluster is touched once, after training, with the selected checkpoint.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phydg.data import RegionBalancedSampler, Sen12LS  # noqa: E402
from phydg.losses import total_loss  # noqa: E402
from phydg.metrics import BinaryMetrics  # noqa: E402
from phydg.models import build_model  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="splits/loco/loco_<cluster>.json")
    ap.add_argument("--data-root", required=True, help="folder containing s2/<region>_s2_<id>.nc")
    ap.add_argument("--model", choices=["unet", "phydg"], default="phydg")
    for c in ("style", "ibn", "film", "poe"):
        ap.add_argument(f"--no-{c}", action="store_true", help=f"ablate PhyDG component: {c}")
    ap.add_argument("--vrex", type=float, default=1.0, help="V-REx weight (0 disables)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--regions-per-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=None, help="folder of precomputed composites (scripts/build_cache.py)")
    ap.add_argument("--batches-per-epoch", type=int, default=None, help="default: one pass over train size")
    ap.add_argument("--max-batches", type=int, default=None, help="limit train and eval batches (smoke tests)")
    ap.add_argument("--out", default="runs")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    m = BinaryMetrics()
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["pre"], b["post"], b["dem"])
        m.update(out["logits"], b["msk"], b["region"])
    return m.compute()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    fold = Path(args.split).stem
    tag = args.model if args.model == "unet" else "phydg" + "".join(
        f"-no{c}" for c in ("style", "ibn", "film", "poe") if getattr(args, f"no_{c}"))
    tag += f"_vrex{args.vrex:g}_s{args.seed}"
    run_dir = Path(args.out) / fold / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = (Sen12LS.from_split(args.split, p, args.data_root, cache=args.cache)
                                 for p in ("train", "val", "test"))
    sampler = RegionBalancedSampler(train_ds.regions, args.batch_size, args.regions_per_batch,
                                    n_batches=args.max_batches or args.batches_per_epoch, seed=args.seed)
    dl = dict(num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    train_dl = DataLoader(train_ds, batch_sampler=sampler, **dl)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, **dl)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, **dl)

    kw = {} if args.model == "unet" else {c: not getattr(args, f"no_{c}") for c in ("style", "ibn", "film", "poe")}
    model = build_model(args.model, **kw).to(args.device)
    vrex = args.vrex if args.model == "phydg" else 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(sampler))
    print(f"{fold} {tag}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M device={args.device}")

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, logs = time.time(), {}
        for b in train_dl:
            b = {k: v.to(args.device) for k, v in b.items()}
            out = model(b["pre"], b["post"], b["dem"], b["region"])
            # V-REx is warmed up so the model first learns the task, then equalises risks
            loss, lg = total_loss(out, b["msk"], b["region"], vrex_weight=vrex if epoch >= args.epochs // 5 else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            for k, v in lg.items():
                logs[k] = logs.get(k, 0.0) + v / len(sampler)
        val = evaluate(model, val_dl, args.device, args.max_batches)["all"]
        history.append({"epoch": epoch, **logs, **{f"val_{k}": v for k, v in val.items()}})
        print(f"ep {epoch:3d} {' '.join(f'{k}={v:.4f}' for k, v in logs.items())} "
              f"val_f1={val['f1']:.4f} val_ap={val['ap']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        # select on AP: threshold-free, so not tied to the training regions' calibration
        if val["ap"] > best:
            best = val["ap"]
            torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=args.device))
    test = evaluate(model, test_dl, args.device, args.max_batches)
    json.dump({"args": vars(args), "best_val_ap": best, "test": test, "history": history},
              open(run_dir / "results.json", "w"), indent=1)
    print("held-out test:", json.dumps(test["all"]))


if __name__ == "__main__":
    main()
