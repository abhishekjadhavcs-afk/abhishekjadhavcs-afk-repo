"""Precompute cloud-masked pre/post composites + DEM + mask for every S12LS-LD patch.

Building composites (per-pixel nanmedian over up to 15 frames) dominates data
loading; caching them makes each epoch I/O-light. ~0.75 MB per patch, ~3.7 GB total.

    python scripts/build_cache.py --data-root data/data_harmonized --out data/cache
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phydg.data import read_patch, save_cached  # noqa: E402


def build(job):
    src, dst = job
    if Path(dst).exists():
        return 0
    tmp = dst.replace(".npz", ".tmp.npz")
    save_cached(tmp, read_patch(src))
    Path(tmp).rename(dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/data_harmonized")
    ap.add_argument("--splits", default="splits/official_s2/splits.json")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(args.splits))
    jobs = [(str(Path(args.data_root) / e["s2"]), str(out / f"{e['id']}.npz"))
            for part in d.values() for e in part]
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(build, jobs, chunksize=16), 1):
            done += n
            if i % 500 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} ({done} new)", flush=True)


if __name__ == "__main__":
    main()
