"""Write one leave-one-cluster-out split file per region cluster."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phydg.splits import write_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--official", default="splits/official_s2/splits.json")
ap.add_argument("--out", default="splits/loco")
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

for p in write_all(args.official, args.out, args.val_frac, args.seed):
    d = json.load(open(p))
    sizes = {k: len(v) for k, v in d.items()}
    held = sorted(Counter(e["inventory"] for e in d["test"]))
    print(f"{p.name:28s} {sizes}  test inventories={held}")
