"""Stream the Sen12Landslides harmonised S2 archives and keep only the S12LS-LD patches.

The full S2 release is ~39 GB compressed (13,628 patches). S12LS-LD uses 4,988 of
them, so each archive is streamed (never stored) and only the needed .nc files
are written to <out>/s2/. Already-downloaded files are skipped, so the script
can be re-run after an interruption.

    python scripts/download_s2.py --out data/data_harmonized
"""
import argparse
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/paulhoehn/Sen12Landslides/resolve/main/data_harmonized/s2"
N_PARTS = 28

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/data_harmonized")
ap.add_argument("--splits", default="splits/official_s2/splits.json")
ap.add_argument("--parts", default=f"1-{N_PARTS}", help="e.g. 1-28 or 3-5")
args = ap.parse_args()

d = json.load(open(args.splits))
wanted = {Path(e["s2"]).name for part in d.values() for e in part}
out = Path(args.out) / "s2"
out.mkdir(parents=True, exist_ok=True)
lo, hi = (int(x) for x in args.parts.split("-"))

def stream_part(url, missing):
    kept = 0
    with urllib.request.urlopen(url, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            # archives are untrusted: only regular files, written by basename only
            if not m.isfile() or name not in missing:
                continue
            src = tar.extractfile(m)
            tmp = out / (name + ".part")
            with open(tmp, "wb") as f:
                shutil.copyfileobj(src, f)
            tmp.rename(out / name)
            kept += 1
    return kept


for p in out.glob("*.part"):  # leftovers from an interrupted run
    p.unlink()

for i in range(lo, hi + 1):
    url = f"{BASE}/s2_part{i:02d}.tar.gz"
    for attempt in range(5):
        missing = wanted - {p.name for p in out.glob("*.nc")}
        if not missing:
            break
        try:
            kept = stream_part(url, missing)
            break
        except (OSError, tarfile.TarError) as e:  # connection resets, truncated streams
            for p in out.glob("*.part"):
                p.unlink()
            wait = 2 ** (attempt + 1)
            print(f"part {i:02d}: {type(e).__name__}: {e}; retry {attempt + 1}/5 in {wait}s", flush=True)
            time.sleep(wait)
    else:
        print(f"part {i:02d}: giving up after 5 attempts", flush=True)
        continue
    if not missing:
        break
    have = len({p.name for p in out.glob("*.nc")} & wanted)
    print(f"part {i:02d}: kept {kept}, total {have}/{len(wanted)}", flush=True)

have = {p.name for p in out.glob("*.nc")} & wanted
print(f"done: {len(have)}/{len(wanted)} S12LS-LD patches in {out}")
sys.exit(0 if len(have) == len(wanted) else 1)
