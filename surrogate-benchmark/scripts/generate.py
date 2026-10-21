"""Generate the reference dataset (20 cells per diameter) for one split.

python scripts/generate.py --split train --batch 4 --threads 2
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.data import run, SPLITS, D_REF

p = argparse.ArgumentParser()
p.add_argument("--split", required=True)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--d", type=int, default=D_REF)
p.add_argument("--out", default="data")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
res = SPLITS[a.split]
for i in range(0, len(res), a.batch):
    chunk = res[i:i + a.batch]
    todo = [r for r in chunk if not os.path.exists(f"{a.out}/d{a.d}_re{r:g}.npz")]
    if not todo:
        continue
    out = run(a.d, todo, threads=a.threads)
    for j, r in enumerate(todo):
        np.savez(f"{a.out}/d{a.d}_re{r:g}.npz", frames=out["frames"][j], coef=out["coef"][j],
                 t_frames=out["t_frames"], t_coef=out["t_coef"], re=r, d=a.d,
                 wall_batch=out["wall"], batch=len(todo), steps=out["steps"])
    print(f"{a.split} d={a.d} Re={todo} wall {out['wall']:.0f}s", flush=True)
