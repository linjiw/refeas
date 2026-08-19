#!/bin/bash
# Feasibility screen over a clip list: brief per-clip JSON -> one CSV. Usage: screen_feasibility.sh <clips.txt> <outdir> [workers]
set -u
R=/data/robotixx/climb; LIST=$1; OUT=$2; W=${3:-8}
mkdir -p "$OUT/json"
cat "$LIST" | CUDA_VISIBLE_DEVICES="" nice -n 10 xargs -P "$W" -I{} sh -c "[ -f '$OUT/json/{}.json' ] || $R/bridge/.venv/bin/python $R/tools/n1_knee_id.py --clip '{}' --t0 0 --t1 1e9 --gap 0.06 --brief --out '$OUT/json/{}.json' > /dev/null 2>&1"
$R/bridge/.venv/bin/python - "$OUT" <<'PY'
import sys, json, glob, csv, os
out = sys.argv[1]; rows = []
for p in sorted(glob.glob(os.path.join(out, "json", "*.json"))):
    try: rows.append(json.load(open(p)))
    except Exception: pass
if rows:
    with open(os.path.join(out, "feasibility.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("clips:", len(rows), "->", os.path.join(out, "feasibility.csv"))
PY
