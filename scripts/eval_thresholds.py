"""Re-evaluate saved checkpoints with a decision threshold chosen on validation data.

The fixed 0.5 threshold is badly calibrated in unseen regions. This picks the
threshold that maximises F1 on the validation split (training regions only, so
no held-out information is used) and reports held-out F1 at that threshold.
The oracle F1 (best threshold on the test set itself) is printed as a
diagnostic upper bound only; it must not be reported as a result.

    python scripts/eval_thresholds.py --runs runs_ablate --cache data/cache
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phydg.data import Sen12LS  # noqa: E402
from phydg.metrics import N_BINS, BinaryMetrics  # noqa: E402
from phydg.models import build_model  # noqa: E402


def f1_curve(h):
    """h [2, N_BINS] (neg, pos) -> F1 for threshold at each bin's lower edge."""
    neg = h[0].flip(0).cumsum(0).flip(0).double()  # predicted positive if bin >= t
    pos = h[1].flip(0).cumsum(0).flip(0).double()
    fn = h[1].sum().double() - pos
    return 2 * pos / (2 * pos + neg + fn).clamp_min(1)


@torch.no_grad()
def hist(model, ds, device, bs=16):
    m = BinaryMetrics()
    for b in DataLoader(ds, batch_size=bs, num_workers=2):
        b = {k: v.to(device) for k, v in b.items()}
        m.update(model(b["pre"], b["post"], b["dem"])["logits"], b["msk"], b["region"])
    return m.hist["all"], m.compute()["all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--data-root", default="data/data_harmonized")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows = []
    for res in sorted(Path(a.runs).glob("*/*/results.json")):
        ckpt = res.parent / "best.pt"
        if not ckpt.exists():
            continue
        args = json.loads(res.read_text())["args"]
        kw = {} if args["model"] == "unet" else {c: not args[f"no_{c}"] for c in ("style", "ibn", "film", "poe")}
        model = build_model(args["model"], **kw).to(a.device)
        model.load_state_dict(torch.load(ckpt, map_location=a.device))
        model.eval()
        split = ROOT / "splits/loco" / f"{res.parent.parent.name}.json"
        hv, _ = hist(model, Sen12LS.from_split(split, "val", a.data_root, cache=a.cache), a.device)
        ht, base = hist(model, Sen12LS.from_split(split, "test", a.data_root, cache=a.cache), a.device)
        k = int(f1_curve(hv).argmax())
        ft = f1_curve(ht)
        row = {"fold": res.parent.parent.name, "variant": res.parent.name,
               "f1@0.5": base["f1"], "ap": base["ap"], "val_threshold": k / N_BINS,
               "f1@val_threshold": float(ft[k]), "oracle_f1(diagnostic)": float(ft.max()),
               "oracle_threshold": int(ft.argmax()) / N_BINS}
        rows.append(row)
        (res.parent / "threshold_eval.json").write_text(json.dumps(row, indent=1))
        print(json.dumps(row), flush=True)

    print(f"\n{'fold':16s} {'variant':26s} {'F1@0.5':>7s} {'thr(val)':>8s} {'F1@thr':>7s} {'AP':>6s} {'oracleF1':>9s}")
    for r in rows:
        print(f"{r['fold']:16s} {r['variant']:26s} {r['f1@0.5']:7.3f} {r['val_threshold']:8.3f} "
              f"{r['f1@val_threshold']:7.3f} {r['ap']:6.3f} {r['oracle_f1(diagnostic)']:9.3f}")


if __name__ == "__main__":
    main()
