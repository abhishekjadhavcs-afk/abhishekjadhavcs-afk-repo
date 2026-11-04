"""Run the experiment grid (folds x variants x seeds) and summarise results.

Skips runs whose results.json already exists, so it can be resumed on Colab/Kaggle
after a disconnect. `--summary-only` just prints the table.

    python scripts/run_grid.py --data-root data/data_harmonized --cache data/cache
    python scripts/run_grid.py --variants unet phydg --seeds 42 --folds oceania africa
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phydg.folds import CLUSTERS  # noqa: E402

VARIANTS = {
    "unet": ["--model", "unet", "--vrex", "0"],
    "phydg": ["--model", "phydg"],
    "phydg-nostyle": ["--model", "phydg", "--no-style"],
    "phydg-noibn": ["--model", "phydg", "--no-ibn"],
    "phydg-nofilm": ["--model", "phydg", "--no-film"],
    "phydg-nopoe": ["--model", "phydg", "--no-poe"],
    "phydg-novrex": ["--model", "phydg", "--vrex", "0"],
}


def run_dir_tag(args_list, seed):
    model = args_list[args_list.index("--model") + 1]
    vrex = args_list[args_list.index("--vrex") + 1] if "--vrex" in args_list else "1"
    if model == "unet":
        tag = "unet"
    else:
        tag = "phydg" + "".join(f"-no{c}" for c in ("style", "ibn", "film", "poe") if f"--no-{c}" in args_list)
    return f"{tag}_vrex{float(vrex):g}_s{seed}"


def summarise(out, folds, variants, seeds):
    rows = []
    for v in variants:
        cells = []
        for f in folds:
            vals = []
            for s in seeds:
                p = Path(out) / f"loco_{f}" / run_dir_tag(VARIANTS[v], s) / "results.json"
                if p.exists():
                    vals.append(json.loads(p.read_text())["test"]["all"])
            cells.append(vals)
        rows.append((v, cells))
    # folds that every listed variant has at least one finished run for: the only fair mean
    common = [i for i in range(len(folds)) if all(cells[i] for _, cells in rows)]
    for metric in ("f1", "ap"):
        print(f"\nheld-out {metric.upper()} (mean over available seeds; n = finished seeds)")
        print(f"{'variant':16s}" + "".join(f"{f[:12]:>13s}" for f in folds)
              + f"{'mean(common)':>14s}")
        for v, cells in rows:
            means = [sum(c[metric] for c in vals) / len(vals) if vals else None for vals in cells]
            line = "".join(f"{m:9.3f} n={len(vals)}" if m is not None else f"{'-':>13s}"
                           for m, vals in zip(means, cells))
            cm = [means[i] for i in common]
            print(f"{v:16s}{line}{(sum(cm) / len(cm)) if cm else float('nan'):14.3f}")
    print(f"\nmean(common) uses the {len(common)} fold(s) finished by every variant: "
          f"{[folds[i] for i in common]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/data_harmonized")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--folds", nargs="+", default=list(CLUSTERS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="stop launching runs when the next one would not finish in time (e.g. Kaggle's 12 h limit)")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="passed to train.py after --")
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra

    start, longest = time.time(), 0.0
    if not a.summary_only:
        for f in a.folds:
            for v in a.variants:
                for s in a.seeds:
                    done = Path(a.out) / f"loco_{f}" / run_dir_tag(VARIANTS[v], s) / "results.json"
                    if done.exists():
                        continue
                    elapsed = time.time() - start
                    if a.time_budget_hours and elapsed + 1.2 * longest > a.time_budget_hours * 3600:
                        print(f"time budget reached after {elapsed / 3600:.1f} h; stopping. "
                              "Re-run to continue with the remaining runs.", flush=True)
                        summarise(a.out, a.folds, a.variants, a.seeds)
                        return
                    t0 = time.time()
                    cmd = [sys.executable, str(ROOT / "scripts/train.py"),
                           "--split", str(ROOT / f"splits/loco/loco_{f}.json"),
                           "--data-root", a.data_root, "--cache", a.cache, "--out", a.out,
                           "--seed", str(s), *VARIANTS[v], *extra]
                    print(">>", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)
                    longest = max(longest, time.time() - t0)
    summarise(a.out, a.folds, a.variants, a.seeds)


if __name__ == "__main__":
    main()
