"""Build leave-one-cluster-out splits from the official S12LS-LD splits.json.

Output files use the official splits.json format ({"train": [...], "val": [...],
"test": [...]}, each entry with "id", "s2", "inventory"), so the official
Sen12Landslides training pipeline can also run on them for baselines.
"""
import json
import random
from pathlib import Path

from .folds import CLUSTERS, INVENTORY_TO_CLUSTER


def load_entries(official_splits: str) -> list[dict]:
    with open(official_splits) as f:
        d = json.load(f)
    return [e for part in ("train", "val", "test") for e in d[part]]


def loco_split(entries: list[dict], held_out: str, val_frac: float = 0.15, seed: int = 42) -> dict:
    """Held-out cluster -> test. Remaining clusters -> train/val, split per inventory."""
    if held_out not in CLUSTERS:
        raise KeyError(f"unknown cluster {held_out!r}; choose from {sorted(CLUSTERS)}")
    unknown = {e["inventory"] for e in entries} - set(INVENTORY_TO_CLUSTER)
    if unknown:
        raise ValueError(f"inventories without a cluster: {sorted(unknown)}")

    rng = random.Random(seed)
    test = [e for e in entries if INVENTORY_TO_CLUSTER[e["inventory"]] == held_out]
    train, val = [], []
    for inv in sorted({e["inventory"] for e in entries} - set(CLUSTERS[held_out])):
        group = sorted((e for e in entries if e["inventory"] == inv), key=lambda e: e["id"])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac))
        val += group[:n_val]
        train += group[n_val:]
    return {"train": train, "val": val, "test": test}


def write_all(official_splits: str, out_dir: str, val_frac: float = 0.15, seed: int = 42) -> list[Path]:
    entries = load_entries(official_splits)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for cluster in CLUSTERS:
        p = out / f"loco_{cluster}.json"
        with open(p, "w") as f:
            json.dump(loco_split(entries, cluster, val_frac, seed), f, indent=1)
        paths.append(p)
    return paths
